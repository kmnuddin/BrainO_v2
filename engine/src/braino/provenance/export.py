"""Exporting an analysis as a standalone Python script, from its provenance record.

The script re-runs every tool call with the recorded arguments, makes every recorded decision
again (approval or rejection, with the reviewer's edits), and checks that each result matches
the recorded one (see :mod:`braino.provenance.replay`). It needs only BrainO installed and the
BIDS dataset::

    python analysis.py path/to/bids

It first checks that the input recordings have the recorded SHA-256 hashes. Outputs are written
to the dataset's derivatives, as in the original run; run it on a copy to keep those.

A script is made from one run (``braino runs export``) or from a pipeline's current results
(``braino pipeline export``), which may come from several runs when steps were reused.
"""

from __future__ import annotations

import pprint
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.provenance.models import ProposalRecord, RunRecord, ToolCallRecord
from braino.provenance.proposals import ProposalNotFoundError, ProposalStore
from braino.provenance.store import load_run

DERIVATIVES_PREFIX = "derivatives/"
APPLY_SUFFIX = ".apply"


class ExportedDecision(BaseModel):
    approve: bool
    edits: dict[str, Any] = Field(description="Fields the reviewer changed")
    decided_by: str
    decided_at: datetime
    note: str | None
    proposal: dict[str, Any]
    apply_result: dict[str, Any] | None


class ExportedStep(BaseModel):
    label: str = Field(description="What the step is, e.g. 'sub-001 preprocess'")
    source: str = Field(description="'<run_id>-<call_index>' of the recorded call")
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    decision: ExportedDecision | None = None
    pending: bool = Field(default=False, description="A proposal not yet decided")


class ExportError(RuntimeError):
    pass


def steps_from_run(
    derivatives_root: Path, run_id: str, proposals: ProposalStore
) -> tuple[list[ExportedStep], list[ToolCallRecord], RunRecord]:
    """The steps of one run, in order (failed calls and calls of tools that are not analysis
    tools, such as imports, are left out). Also returns the calls and the run's environment."""
    log = load_run(derivatives_root, run_id)
    calls = [c for c in log.calls if c.status == "ok" and not c.tool.endswith(APPLY_SUFFIX)]
    steps = [_step(call, proposals, label=_label(call)) for call in calls]
    return steps, calls, log.run


def steps_from_pipeline(
    derivatives_root: Path, state: Any, spec: Any, proposals: ProposalStore
) -> tuple[list[ExportedStep], list[ToolCallRecord], RunRecord]:
    """The steps behind a pipeline's current results (from its state), subject by subject and
    then the group steps. Also returns the calls and the environment of the latest run."""
    runs: dict[str, Any] = {}

    def call_of(run_id: str, index: int) -> ToolCallRecord:
        if run_id not in runs:
            runs[run_id] = load_run(derivatives_root, run_id)
        for call in runs[run_id].calls:
            if call.call_index == index:
                return call  # type: ignore[no-any-return]
        raise ExportError(f"call {run_id}-{index} is missing from the provenance record")

    subjects = sorted({r.subject for r in state.records.values() if r.subject != "group"})
    order = [(s, step.id) for s in subjects for step in spec.steps]
    order += [("group", step.id) for step in spec.group]
    steps, calls = [], []
    for subject, step_id in order:
        record = state.records.get(f"{subject}/{step_id}")
        if record is None:
            continue
        call = call_of(record.call.run_id, record.call.call_index)
        where = "group" if subject == "group" else f"sub-{subject}"
        steps.append(_step(call, proposals, label=f"{where} {step_id}"))
        calls.append(call)
    if not steps:
        raise ExportError(f"pipeline {spec.name} has no results on this dataset")
    latest = max(runs.values(), key=lambda log: log.run.started_at)
    return steps, calls, latest.run


def original_inputs(calls: list[ToolCallRecord]) -> dict[str, str]:
    """The dataset's own files (not derivatives) that the calls read: path -> SHA-256."""
    out: dict[str, str] = {}
    for call in calls:
        for f in call.inputs:
            if f.path.startswith((DERIVATIVES_PREFIX, "<")) or Path(f.path).is_absolute():
                continue
            out.setdefault(f.path, f.sha256)
    return dict(sorted(out.items()))


def render_script(
    steps: list[ExportedStep],
    *,
    title: str,
    environment: RunRecord,
    inputs: dict[str, str],
) -> str:
    """The Python script that replays ``steps``."""
    exported = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    packages = ", ".join(f"{k} {v}" for k, v in sorted(environment.packages.items()))
    pending = [s for s in steps if s.pending]
    origin = textwrap.fill(
        f"Exported from BrainO's provenance record on {exported}. Recorded with BrainO "
        f"{environment.braino_version}, Python {environment.python_version} on "
        f"{environment.platform}; {packages}.",
        width=92,
    )
    lines = [
        "#!/usr/bin/env python",
        f'"""{title}',
        "",
        *origin.splitlines(),
        "",
        "Usage: python <this file> path/to/bids [--no-verify]",
        "",
        "Re-runs every step with the recorded arguments and makes the recorded decisions again,",
        "then reports whether every result matches the recorded one. Outputs are written to the",
        "dataset's derivatives/braino/, as in the original run: run it on a copy to keep those.",
    ]
    if pending:
        lines += [
            "",
            f"{len(pending)} proposals were still undecided when this was exported; the script",
            "stops at the first one.",
        ]
    lines += [
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "",
        "from braino.provenance.replay import Replay",
        "",
        "",
        "def main(argv: list[str] | None = None) -> int:",
        "    replay = Replay.from_command_line(",
        "        argv,",
        f"        description={title!r},",
        "        recorded=RECORDED,",
        "        inputs=INPUTS,",
        "        environment=ENVIRONMENT,",
        "    )",
    ]
    recorded: dict[int, dict[str, Any]] = {}
    for number, step in enumerate(steps, start=1):
        lines.append("")
        lines.append(f"    # {number}. {step.label}  (recorded as call {step.source})")
        arguments = _literal(step.arguments, indent=8)
        is_decision = step.decision is not None or step.pending
        target = "pending = " if is_decision else ""
        lines.append(f"    {target}replay.run(")
        lines.append(f"        {step.tool!r},")
        lines.append(f"        {arguments},")
        lines.append(f"        step={number},")
        lines.append("    )")
        entry: dict[str, Any] = {"result": step.result}
        if step.pending:
            lines.append("    print('stopping: this proposal was undecided when exported')")
            lines.append("    return replay.finish()")
            recorded[number] = entry
            break
        if step.decision is not None:
            d = step.decision
            entry = {"proposal": d.proposal, "apply_result": d.apply_result}
            verb = "approved" if d.approve else "rejected"
            when = d.decided_at.strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"    # {verb} by {d.decided_by} on {when}" + (
                f": {d.note}" if d.note else ""
            ))  # fmt: skip
            lines.append("    replay.decide(")
            lines.append("        pending,")
            lines.append(f"        step={number},")
            lines.append(f"        approve={d.approve},")
            if d.edits:
                lines.append(f"        edits={_literal(d.edits, indent=8)},")
            lines.append(f"        decided_by={d.decided_by!r},")
            if d.note:
                lines.append(f"        note={d.note!r},")
            lines.append("    )")
        recorded[number] = entry
    lines += [
        "",
        "    return replay.finish()",
        "",
        "",
        "# What the original run used and found.",
        f"ENVIRONMENT = {_literal(environment.model_dump(mode='json'), indent=0)}",
        "",
        "# The dataset's files the analysis read, with their SHA-256 hashes.",
        f"INPUTS = {_literal(inputs, indent=0)}",
        "",
        "# The result of every step, as recorded, to compare the replay with.",
        f"RECORDED = {_literal(recorded, indent=0)}",
        "",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(main())",
        "",
    ]
    return "\n".join(lines)


def _step(call: ToolCallRecord, proposals: ProposalStore, *, label: str) -> ExportedStep:
    step = ExportedStep(
        label=label,
        source=f"{call.run_id}-{call.call_index}",
        tool=call.tool,
        arguments=call.arguments,
        result=call.result,
    )
    if call.proposal_id is None:
        return step
    try:
        record = proposals.get(call.proposal_id)
    except ProposalNotFoundError:
        raise ExportError(f"proposal {call.proposal_id} is missing") from None
    if record.status == "pending" or record.decision is None:
        return step.model_copy(update={"pending": True})
    return step.model_copy(update={"decision": _decision(record)})


def _decision(record: ProposalRecord) -> ExportedDecision:
    decision = record.decision
    assert decision is not None
    final = decision.final or {}
    edits = {k: v for k, v in final.items() if record.proposal.get(k) != v}
    return ExportedDecision(
        approve=decision.approved,
        edits=edits,
        decided_by=decision.decided_by,
        decided_at=decision.decided_at,
        note=decision.note,
        proposal=record.proposal,
        apply_result=decision.apply_result,
    )


def _label(call: ToolCallRecord) -> str:
    subject = call.arguments.get("subject")
    return f"sub-{subject} {call.tool}" if subject else call.tool


def _literal(value: Any, *, indent: int) -> str:
    """``value`` as a Python literal, continuation lines indented by ``indent``."""
    text = pprint.pformat(value, width=96 - indent, sort_dicts=False)
    return text.replace("\n", "\n" + " " * indent)
