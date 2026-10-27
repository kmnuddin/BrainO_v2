"""Pipeline specs and the runner: checks, caching, re-running from a step, and decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from braino.pipelines import (
    PipelineSpec,
    SpecError,
    builtin_specs,
    load_spec,
    load_state,
    parse_spec,
    run_pipeline,
)
from braino.tools import RunContext, decide

SPEC = """
name: test-oddball
description: synthetic oddball
task: oddball
steps:
  - id: preprocess
    tool: eeg.preprocess
    args: {h_freq: 30.0}
  - id: ica
    tool: eeg.ica_clean
    args: {labeler: eog}
  - id: bad_channels
    tool: eeg.detect_bad_channels
    args: {method: qc}
  - id: epochs
    tool: eeg.epoch
    args:
      conditions: {rare: [stimulus/rare], frequent: [stimulus/frequent]}
      reject: peak_to_peak
  - id: erp
    tool: eeg.erp
    args:
      contrasts: [{name: difference, plus: rare, minus: frequent}]
      measures: [{name: P3, channels: [Pz], tmin: 0.3, tmax: 0.5}]
group:
  - id: test
    tool: stats.mass_univariate
    args: {conditions: [difference], channels: [Pz], tmin: 0.3, tmax: 0.5}
"""


# --- specs (no science stack needed) ------------------------------------------------------


def test_spec_is_checked_against_the_tools() -> None:
    spec = parse_spec(SPEC)
    assert spec.step_ids == ["preprocess", "ica", "bad_channels", "epochs", "erp", "test"]
    assert len(spec.sha256()) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("tool: eeg.erp", "tool: eeg.nope"), "no tool 'eeg.nope'"),
        (("args: {h_freq: 30.0}", "args: {h_freq: 30.0, subject: '01'}"), "runner sets subject"),
        (("args: {h_freq: 30.0}", "args: {h_freq: -1}"), "h_freq"),
        (("args: {labeler: eog}", "args: {labeler: magic}"), "labeler"),
        (("tool: stats.mass_univariate", "tool: eeg.erp"), "works on one recording"),
        (("  - id: ica\n    tool: eeg.ica_clean", "  - id: ica\n    tool: stats.mass_univariate"),
         "works on the group"),
        (("  - id: ica", "  - id: preprocess"), "repeated: preprocess"),
    ],
)  # fmt: skip
def test_spec_errors(change: tuple[str, str], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_spec(SPEC.replace(*change, 1))


def test_builtin_specs_are_valid() -> None:
    specs = builtin_specs()
    assert {"erpcore-p3", "erpcore-n170"} <= set(specs)
    for name in specs:
        spec = load_spec(name)
        assert spec.task in {"P3", "N170"}
        assert spec.step("erp").args["measures"]
    with pytest.raises(SpecError, match="not a file or a built-in spec"):
        load_spec("no-such-spec")


def test_spec_from_json(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(parse_spec(SPEC).model_dump_json(), encoding="utf-8")
    assert load_spec(path).name == "test-oddball"


# --- running (needs the science stack) ----------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    pytest.importorskip("picard")
    from braino.datasets.synthetic import write_dataset

    return write_dataset(tmp_path / "bids", ["01", "02", "03"], task="oddball", seconds=40.0)


def spec() -> PipelineSpec:
    return parse_spec(SPEC)


def statuses(report: object) -> dict[tuple[str, str], str]:
    return {(o.subject, o.step): o.status for o in report.outcomes}  # type: ignore[attr-defined]


def test_run_then_everything_is_cached(dataset: Path) -> None:
    first = run_pipeline(spec(), RunContext(dataset_root=dataset), auto_approve="user:test")
    assert first.complete, first.outcomes
    assert first.count("ran") == 3 * 5 + 1
    assert ("group", "test") in statuses(first)
    ctx = RunContext(dataset_root=dataset)
    second = run_pipeline(spec(), ctx, auto_approve="user:test")
    assert set(statuses(second).values()) == {"cached"}
    assert ctx.provenance.calls == []  # nothing ran
    state = load_state(ctx, spec())
    assert set(state.records) >= {"01/preprocess", "03/erp", "group/test"}
    record = state.records["01/ica"]
    assert record.proposal_id is not None and record.apply is not None


def test_decisions_are_recorded_as_auto_approved(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    run_pipeline(spec(), ctx, subjects=["01"], auto_approve="user:test (auto)")
    decided = [p for p in ctx.proposals.list() if p.status == "approved"]
    assert len(decided) == 2
    assert all(p.decision and p.decision.decided_by == "user:test (auto)" for p in decided)
    assert all(p.decision and "auto-approved" in (p.decision.note or "") for p in decided)


def test_rerun_from_a_step(dataset: Path) -> None:
    run_pipeline(spec(), RunContext(dataset_root=dataset), auto_approve="user:test")
    report = run_pipeline(
        spec(), RunContext(dataset_root=dataset), from_step="epochs", auto_approve="user:test"
    )
    s = statuses(report)
    assert s[("02", "preprocess")] == "cached"
    assert s[("02", "bad_channels")] == "cached"
    assert s[("02", "epochs")] == "ran"
    assert s[("02", "erp")] == "ran"
    assert s[("group", "test")] == "ran"


def test_changed_arguments_rerun_that_step_and_the_later_ones(dataset: Path) -> None:
    run_pipeline(spec(), RunContext(dataset_root=dataset), auto_approve="user:test")
    changed = parse_spec(SPEC.replace("reject: peak_to_peak", "reject: none"))
    s = statuses(run_pipeline(changed, RunContext(dataset_root=dataset), auto_approve="u:t"))
    assert s[("01", "ica")] == "cached"
    assert s[("01", "epochs")] == "ran"
    assert s[("01", "erp")] == "ran"
    assert s[("group", "test")] == "ran"


def test_a_file_changed_outside_the_pipeline_is_noticed(dataset: Path) -> None:
    run_pipeline(spec(), RunContext(dataset_root=dataset), auto_approve="user:test")
    epochs = next((dataset / "derivatives/braino/sub-01/eeg").glob("*_desc-epochs_epo.fif"))
    epochs.write_bytes(epochs.read_bytes() + b"\0")
    s = statuses(run_pipeline(spec(), RunContext(dataset_root=dataset), auto_approve="u:t"))
    assert s[("01", "epochs")] == "ran"
    assert s[("02", "epochs")] == "cached"


def test_decision_steps_wait_for_a_person(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    report = run_pipeline(spec(), ctx, subjects=["01", "02", "03"])
    s = statuses(report)
    assert s[("01", "preprocess")] == "ran"
    assert s[("01", "ica")] == "pending"
    assert s[("01", "epochs")] == "blocked"
    assert s[("group", "test")] == "blocked"
    assert not report.complete and len(report.pending) == 3

    # Running again without deciding asks again, without making new proposals.
    again = run_pipeline(spec(), RunContext(dataset_root=dataset))
    assert [o.proposal_id for o in again.pending] == [o.proposal_id for o in report.pending]

    for outcome in report.pending:
        assert outcome.proposal_id is not None
        decide(outcome.proposal_id, ctx, approve=True, decided_by="user:test")
    third = run_pipeline(spec(), RunContext(dataset_root=dataset))
    s = statuses(third)
    assert s[("01", "ica")] == "cached"
    assert s[("01", "bad_channels")] == "pending"


def test_rejected_decisions_are_skipped(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    report = run_pipeline(spec(), ctx, subjects=["01"])
    pending = report.pending[0]
    assert pending.proposal_id is not None
    decide(pending.proposal_id, ctx, approve=False, decided_by="user:test", note="keep all")
    s = statuses(run_pipeline(spec(), RunContext(dataset_root=dataset), subjects=["01"]))
    assert s[("01", "ica")] == "rejected"
    assert s[("01", "bad_channels")] == "pending"


def test_a_failing_subject_does_not_stop_the_others(dataset: Path) -> None:
    broken = parse_spec(SPEC.replace("channels: [Pz], tmin: 0.3, tmax: 0.5}]", "channels: [Pz],"
                                     " tmin: 0.3, tmax: 2.5}]"))  # fmt: skip
    report = run_pipeline(broken, RunContext(dataset_root=dataset), auto_approve="user:test")
    s = statuses(report)
    assert s[("01", "erp")] == "failed"
    assert s[("03", "erp")] == "failed"
    assert s[("group", "test")] == "blocked"
    assert "outside the epoch" in (report.failed[0].message or "")


def test_subjects_and_exclusions(dataset: Path) -> None:
    excluded = parse_spec(SPEC + "exclude_subjects: ['02']\n")
    report = run_pipeline(excluded, RunContext(dataset_root=dataset), auto_approve="user:test")
    assert report.subjects == ["01", "03"]
    with pytest.raises(SpecError, match="not in this pipeline"):
        run_pipeline(excluded, RunContext(dataset_root=dataset), subjects=["02"])
    with pytest.raises(SpecError, match="no step"):
        run_pipeline(spec(), RunContext(dataset_root=dataset), from_step="nope")
    missing = parse_spec(SPEC + "subjects: ['01', '09']\n")
    with pytest.raises(SpecError, match="no oddball recording"):
        run_pipeline(missing, RunContext(dataset_root=dataset))


def test_cli(dataset: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from braino.cli import WAITING_EXIT_CODE, app

    path = tmp_path / "spec.yaml"
    path.write_text(SPEC, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["pipeline", "validate", str(path)])
    assert result.exit_code == 0 and "OK" in result.output
    result = runner.invoke(app, ["pipeline", "run", str(path), "--dataset", str(dataset)])
    assert result.exit_code == WAITING_EXIT_CODE, result.output
    assert "decisions wait for you" in result.output
    result = runner.invoke(
        app, ["pipeline", "run", str(path), "--dataset", str(dataset), "--auto-approve"]
    )
    assert result.exit_code == 0, result.output
    assert "sub-01" in result.output
    result = runner.invoke(app, ["pipeline", "status", str(path), "--dataset", str(dataset)])
    assert "erp" in result.output
    assert runner.invoke(app, ["pipeline", "list"]).output.count("erpcore") == 2
    assert "eeg.preprocess" in runner.invoke(app, ["pipeline", "show", "erpcore-p3"]).output
