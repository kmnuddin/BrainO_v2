"""Script export from provenance: the exported script must re-run the analysis, make the same
decisions and get the same results."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from braino.provenance.replay import compare


def test_compare_numbers_within_tolerance() -> None:
    assert compare({"a": 1.0, "b": [1, 2]}, {"a": 1.0 + 1e-7, "b": [1, 2]}) == []
    assert compare({"a": 1.0}, {"a": 1.1}) == [("result.a", 1.0, 1.1)]
    assert compare({"a": [1, 2]}, {"a": [1]}) == [("result.a (length)", 2, 1)]
    assert compare({"a": "x"}, {"a": "y"}) == [("result.a", "x", "y")]
    assert compare({"proposal_id": "r1-1"}, {"proposal_id": "r2-1"}) == []
    assert compare({"a": 1}, {"b": 1}) == [("result.a", 1, None), ("result.b", None, 1)]
    assert compare({"x": float("nan")}, {"x": float("nan")}) == []


# --- end to end (needs the science stack) -------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    pytest.importorskip("picard")
    from braino.datasets.synthetic import write_dataset

    return write_dataset(tmp_path / "bids", ["01", "02", "03"], task="oddball", seconds=30.0)


def fresh_copy(dataset: Path, target: Path) -> Path:
    """The dataset without BrainO's derivatives, as someone re-running the script has it."""
    shutil.copytree(dataset, target, ignore=shutil.ignore_patterns("derivatives"))
    return target


def replay(
    script: str, dataset: Path, tmp_path: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "analysis.py"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(path), str(dataset), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=600,
    )


def run(ctx: Any, tool: str, **args: Any) -> Any:
    from braino.tools import registry

    return registry.get(tool).run({"subject": "01", "task": "oddball", **args}, ctx)


def export_run(dataset: Path, run_id: str) -> str:
    from braino.provenance.export import original_inputs, render_script, steps_from_run
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset)
    steps, calls, environment = steps_from_run(ctx.derivatives_root, run_id, ctx.proposals)
    return render_script(
        steps, title="test export", environment=environment, inputs=original_inputs(calls)
    )


def test_exported_run_reproduces_its_results(dataset: Path, tmp_path: Path) -> None:
    from braino.tools import RunContext, decide

    ctx = RunContext(dataset_root=dataset, run_id="original")
    run(ctx, "eeg.preprocess", h_freq=30.0)
    ica = run(ctx, "eeg.ica_clean", labeler="eog")
    decide(ica.proposal_id, ctx, approve=False, decided_by="user:alice", note="no ICA")
    bads = run(ctx, "eeg.detect_bad_channels", method="qc")
    decide(bads.proposal_id, ctx, approve=True, decided_by="user:alice", edits={"channels": ["T8"]})
    run(ctx, "eeg.epoch", conditions={"rare": ["stimulus/rare"]}, reject="none")
    run(ctx, "eeg.erp", measures=[{"name": "P3", "channels": ["Pz"], "tmin": 0.3, "tmax": 0.5}])

    script = export_run(dataset, "original")
    assert "approve=False" in script and "no ICA" in script
    assert "edits={'channels': ['T8']}" in script
    assert "sub-01/eeg/sub-01_task-oddball_eeg.vhdr" in script  # an input to verify
    compile(script, "analysis.py", "exec")

    copy = fresh_copy(dataset, tmp_path / "copy")
    result = replay(script, copy, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reproduced: all 5 steps" in result.stdout
    table = next((copy / "derivatives/braino/sub-01/eeg").glob("*_desc-preproc_channels.tsv"))
    assert "T8\tEEG\tbad" in table.read_text(encoding="utf-8")


def test_changed_inputs_are_refused(dataset: Path, tmp_path: Path) -> None:
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset, run_id="original")
    run(ctx, "eeg.preprocess")
    script = export_run(dataset, "original")
    copy = fresh_copy(dataset, tmp_path / "copy")
    data = next((copy / "sub-01/eeg").glob("*.eeg"))
    data.write_bytes(data.read_bytes()[:-2] + b"\x00\x01")
    result = replay(script, copy, tmp_path)
    assert result.returncode != 0
    assert "differ from the recorded ones" in result.stderr
    assert replay(script, copy, tmp_path, "--no-verify").returncode in (0, 1)


def test_differences_are_reported(dataset: Path, tmp_path: Path) -> None:
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset, run_id="original")
    run(ctx, "eeg.preprocess")
    script = export_run(dataset, "original")
    tampered = script.replace("'duration_s': ", "'duration_s': 1 + ", 1)
    result = replay(tampered, fresh_copy(dataset, tmp_path / "copy"), tmp_path)
    assert result.returncode == 1
    assert "duration_s" in result.stdout


def test_pending_proposals_stop_the_script(dataset: Path, tmp_path: Path) -> None:
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset, run_id="original")
    run(ctx, "eeg.preprocess")
    run(ctx, "eeg.detect_bad_channels", method="qc")
    script = export_run(dataset, "original")
    assert "undecided when this was exported" in script
    result = replay(script, fresh_copy(dataset, tmp_path / "copy"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "stopping" in result.stdout


def test_pipeline_export_includes_reused_steps(dataset: Path, tmp_path: Path) -> None:
    from test_pipelines import SPEC

    from braino.pipelines import load_state, parse_spec, run_pipeline
    from braino.provenance.export import original_inputs, render_script, steps_from_pipeline
    from braino.tools import RunContext

    spec = parse_spec(SPEC)
    run_pipeline(spec, RunContext(dataset_root=dataset), auto_approve="user:alice")
    later = RunContext(dataset_root=dataset)
    run_pipeline(spec, later, from_step="epochs", auto_approve="user:alice")  # 2nd run, partial
    state = load_state(later, spec)
    steps, calls, environment = steps_from_pipeline(
        later.derivatives_root, state, spec, later.proposals
    )
    assert len(steps) == 3 * 5 + 1
    assert len({s.source.rsplit("-", 1)[0] for s in steps}) == 2  # from both runs
    script = render_script(
        steps, title="pipeline", environment=environment, inputs=original_inputs(calls)
    )
    result = replay(script, fresh_copy(dataset, tmp_path / "copy"), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reproduced: all 16 steps" in result.stdout


def test_cli_export(dataset: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from braino.cli import app
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset, run_id="original")
    run(ctx, "eeg.preprocess")
    out = tmp_path / "script.py"
    result = CliRunner().invoke(
        app, ["runs", "export", "original", "--dataset", str(dataset), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "replay.run(" in out.read_text(encoding="utf-8")
    missing = CliRunner().invoke(app, ["runs", "export", "nope", "--dataset", str(dataset)])
    assert missing.exit_code == 1
