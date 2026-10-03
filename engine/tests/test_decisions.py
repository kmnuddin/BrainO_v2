from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError
from typer.testing import CliRunner

from braino.cli import app
from braino.provenance import ProposalNotFoundError, load_run
from braino.tools import (
    MissingContextError,
    PendingProposal,
    ProposalAlreadyDecidedError,
    Risk,
    RunContext,
    ToolDefinitionError,
    ToolRegistry,
    decide,
    make_tool,
)


class DetectInput(BaseModel):
    threshold: float = 3.0


class BadChannels(BaseModel):
    channels: list[str] = Field(description="Channels proposed for removal")
    reason: str = ""


class Applied(BaseModel):
    dropped: list[str]


def apply_bad_channels(proposal: BadChannels, ctx: RunContext) -> Applied:
    out = ctx.record_output(ctx.output_dir("decisions") / "bad_channels.json")
    out.write_text(json.dumps(proposal.channels), encoding="utf-8")
    return Applied(dropped=proposal.channels)


def detect(params: DetectInput, ctx: RunContext) -> BadChannels:
    """Find noisy channels."""
    return BadChannels(channels=["Fp1", "T7"], reason=f"z > {params.threshold}")


reg = ToolRegistry()
reg.tool(name="eeg.fake_bad_channels", risk=Risk.DECISION, apply=apply_bad_channels)(detect)


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    return RunContext(dataset_root=tmp_path, run_id="r1")


def _propose(ctx: RunContext) -> PendingProposal:
    result = reg.get("eeg.fake_bad_channels").run({"threshold": 2.5}, ctx)
    assert isinstance(result, PendingProposal)
    return result


def test_running_a_decision_tool_only_proposes(ctx: RunContext) -> None:
    pending = _propose(ctx)
    assert pending.proposal_id == "r1-1"
    assert pending.proposal == {"channels": ["Fp1", "T7"], "reason": "z > 2.5"}

    record = ctx.proposals.get("r1-1")
    assert record.status == "pending"
    assert record.arguments == {"threshold": 2.5}
    assert not (ctx.derivatives_root / "decisions").exists()
    assert ctx.provenance.calls[0].proposal_id == "r1-1"


def test_approve_applies_and_records(ctx: RunContext) -> None:
    _propose(ctx)
    record = decide(
        "r1-1", ctx, approve=True, decided_by="user:test", tools=reg, note="looks right"
    )

    assert record.status == "approved"
    decision = record.decision
    assert decision is not None
    assert (decision.decided_by, decision.note, decision.edited) == (
        "user:test",
        "looks right",
        False,
    )
    assert decision.apply_result == {"dropped": ["Fp1", "T7"]}
    assert decision.applied_in == "r1-2"
    assert ctx.proposals.get("r1-1") == record

    [_, apply_call] = load_run(ctx.derivatives_root, "r1").calls
    assert apply_call.tool == "eeg.fake_bad_channels.apply"
    assert apply_call.proposal_id == "r1-1"
    assert apply_call.outputs[0].path == "derivatives/braino/decisions/bad_channels.json"


def test_approve_with_edits(ctx: RunContext) -> None:
    _propose(ctx)
    record = decide(
        "r1-1", ctx, approve=True, decided_by="user:test", tools=reg, edits={"channels": ["T7"]}
    )
    assert record.decision is not None
    assert record.decision.edited
    assert record.decision.final == {"channels": ["T7"], "reason": "z > 2.5"}
    assert record.decision.apply_result == {"dropped": ["T7"]}
    assert record.proposal["channels"] == ["Fp1", "T7"]  # the original is kept


def test_invalid_edits_leave_proposal_pending(ctx: RunContext) -> None:
    _propose(ctx)
    with pytest.raises(ValidationError):
        decide(
            "r1-1", ctx, approve=True, decided_by="user:test", tools=reg, edits={"channels": "T7x"}
        )
    assert ctx.proposals.get("r1-1").status == "pending"


def test_reject_applies_nothing(ctx: RunContext) -> None:
    _propose(ctx)
    record = decide("r1-1", ctx, approve=False, decided_by="user:test", tools=reg, note="keep all")
    assert record.status == "rejected"
    assert record.decision is not None
    assert record.decision.final is None
    assert len(ctx.provenance.calls) == 1
    assert not (ctx.derivatives_root / "decisions").exists()


def test_cannot_decide_twice(ctx: RunContext) -> None:
    _propose(ctx)
    decide("r1-1", ctx, approve=False, decided_by="user:test", tools=reg)
    with pytest.raises(ProposalAlreadyDecidedError):
        decide("r1-1", ctx, approve=True, decided_by="user:test", tools=reg)


def test_decide_in_a_later_run(tmp_path: Path) -> None:
    _propose(RunContext(dataset_root=tmp_path, run_id="r1"))
    later = RunContext(dataset_root=tmp_path, run_id="r2")
    record = decide("r1-1", later, approve=True, decided_by="user:test", tools=reg)
    assert record.decision is not None
    assert record.decision.applied_in == "r2-1"


def test_proposals_in_memory_without_dataset() -> None:
    ctx = RunContext(run_id="mem")
    assert isinstance(reg.get("eeg.fake_bad_channels").run({}, ctx), PendingProposal)
    assert [r.proposal_id for r in ctx.proposals.list()] == ["mem-1"]


def test_decision_tool_needs_context() -> None:
    with pytest.raises(MissingContextError):
        reg.get("eeg.fake_bad_channels").run({})


@pytest.mark.parametrize("bad_id", ["missing", "../escape", "a/b"])
def test_unknown_or_invalid_proposal_ids(ctx: RunContext, bad_id: str) -> None:
    with pytest.raises(ProposalNotFoundError):
        ctx.proposals.get(bad_id)


def test_llm_schema_mentions_approval() -> None:
    description = reg.get("eeg.fake_bad_channels").llm_schema()["function"]["description"]
    assert "approves" in description


def test_decision_tool_requires_apply() -> None:
    with pytest.raises(ToolDefinitionError, match="needs an apply"):
        make_tool(detect, name="eeg.x", risk=Risk.DECISION)


def test_only_decision_tools_take_apply() -> None:
    with pytest.raises(ToolDefinitionError, match="only decision tools"):
        make_tool(detect, name="eeg.x", risk=Risk.COMPUTE, apply=apply_bad_channels)


def test_apply_signature_is_checked() -> None:
    def wrong_apply(proposal: DetectInput, ctx: RunContext) -> Applied:
        return Applied(dropped=[])

    with pytest.raises(ToolDefinitionError, match="apply must be annotated"):
        make_tool(detect, name="eeg.x", risk=Risk.DECISION, apply=wrong_apply)


def test_cli_propose_then_approve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("braino.cli.registry", reg)
    monkeypatch.setattr("braino.tools.decisions.registry", reg)
    runner = CliRunner()
    ds = str(tmp_path)

    no_dataset = runner.invoke(app, ["tools", "run", "eeg.fake_bad_channels"])
    assert no_dataset.exit_code == 1

    proposed = runner.invoke(app, ["tools", "run", "eeg.fake_bad_channels", "--dataset", ds])
    assert proposed.exit_code == 0, proposed.output
    proposal_id = json.loads(proposed.stdout)["proposal_id"]
    assert "awaits your decision" in proposed.stderr

    listed = runner.invoke(app, ["proposals", "list", "--pending", "--dataset", ds])
    assert proposal_id in listed.stdout

    approved = runner.invoke(
        app,
        ["proposals", "approve", proposal_id, "--dataset", ds, "--edit", '{"channels": ["T7"]}'],
    )
    assert approved.exit_code == 0, approved.output
    assert json.loads(approved.stdout)["decision"]["final"]["channels"] == ["T7"]

    again = runner.invoke(app, ["proposals", "reject", proposal_id, "--dataset", ds])
    assert again.exit_code == 1
    assert "already approved" in again.stderr

    shown = runner.invoke(app, ["proposals", "show", proposal_id, "--dataset", ds])
    assert json.loads(shown.stdout)["status"] == "approved"
