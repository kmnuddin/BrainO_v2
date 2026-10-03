from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from braino import __version__
from braino.cli import app
from braino.provenance import RunNotFoundError, list_runs, load_run
from braino.tools import Risk, RunContext, ToolRegistry


class CopyInput(BaseModel):
    source: str
    target: str


class CopyOutput(BaseModel):
    size: int


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    pass


reg = ToolRegistry()


@reg.tool(name="files.copy", risk=Risk.COMPUTE)
def copy_file(params: CopyInput, ctx: RunContext) -> CopyOutput:
    """Copy a file inside the dataset into the derivatives folder."""
    data = ctx.record_input(ctx.dataset / params.source).read_bytes()
    target = ctx.record_output(ctx.output_dir() / params.target)
    target.write_bytes(data)
    return CopyOutput(size=len(data))


@reg.tool(name="files.fail", risk=Risk.COMPUTE)
def fail(params: EmptyInput, ctx: RunContext) -> EmptyOutput:
    """Always fails."""
    raise ValueError("boom")


@reg.tool(name="files.forget_output", risk=Risk.COMPUTE)
def forget_output(params: EmptyInput, ctx: RunContext) -> EmptyOutput:
    """Declares an output but never writes it."""
    ctx.record_output(ctx.output_dir() / "never-written.txt")
    return EmptyOutput()


@reg.tool(name="files.nested", risk=Risk.COMPUTE)
def nested(params: EmptyInput, ctx: RunContext) -> EmptyOutput:
    """Tries to run another tool."""
    reg.get("files.fail").run({}, ctx)
    return EmptyOutput()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "raw.txt").write_bytes(b"signal")
    return tmp_path


def test_successful_call_is_recorded_and_written(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset, run_id="run-1")
    reg.get("files.copy").run({"source": "raw.txt", "target": "copy.txt"}, ctx)

    log = load_run(ctx.derivatives_root, "run-1")
    assert log.run.braino_version == __version__
    assert "pydantic" in log.run.packages
    assert log.run.dataset_root == str(dataset.resolve())

    [call] = log.calls
    assert call == ctx.provenance.calls[0]
    assert (call.tool, call.risk, call.status, call.call_index) == (
        "files.copy",
        "compute",
        "ok",
        1,
    )
    assert call.arguments == {"source": "raw.txt", "target": "copy.txt"}
    assert call.result == {"size": 6}
    digest = hashlib.sha256(b"signal").hexdigest()
    assert [(f.path, f.sha256, f.size) for f in call.inputs] == [("raw.txt", digest, 6)]
    assert [(f.path, f.sha256) for f in call.outputs] == [("derivatives/braino/copy.txt", digest)]
    assert call.started_at <= call.finished_at


def test_failed_call_is_recorded_and_reraised(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    with pytest.raises(ValueError, match="boom"):
        reg.get("files.fail").run({}, ctx)
    [call] = load_run(ctx.derivatives_root, ctx.run_id).calls
    assert call.status == "error"
    assert call.error == "ValueError: boom"
    assert call.result is None


def test_missing_declared_output_fails_the_call(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    with pytest.raises(FileNotFoundError):
        reg.get("files.forget_output").run({}, ctx)
    assert ctx.provenance.calls[0].status == "error"


def test_calls_are_numbered_within_a_run(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    reg.get("files.copy").run({"source": "raw.txt", "target": "a.txt"}, ctx)
    reg.get("files.copy").run({"source": "raw.txt", "target": "b.txt"}, ctx)
    calls = load_run(ctx.derivatives_root, ctx.run_id).calls
    assert [c.call_index for c in calls] == [1, 2]


def test_tools_cannot_call_tools(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    with pytest.raises(RuntimeError, match="must not call other tools"):
        reg.get("files.nested").run({}, ctx)


def test_recording_outside_a_tool_call_fails(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    with pytest.raises(RuntimeError, match="while a tool is running"):
        ctx.record_input(dataset / "raw.txt")


def test_run_without_dataset_keeps_records_in_memory() -> None:
    ctx = RunContext()
    with pytest.raises(ValueError):
        reg.get("files.fail").run({}, ctx)
    assert len(ctx.provenance.calls) == 1


def test_run_without_calls_writes_nothing(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset)
    assert not ctx.derivatives_root.exists()
    assert list_runs(ctx.derivatives_root) == []


def test_files_outside_dataset_use_absolute_paths(
    dataset: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("elsewhere") / "ext.txt"
    outside.write_bytes(b"x")
    ctx = RunContext(dataset_root=dataset)
    reg.get("files.copy").run({"source": str(outside), "target": "ext.txt"}, ctx)
    assert ctx.provenance.calls[0].inputs[0].path == outside.resolve().as_posix()


def test_list_and_load_errors(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset, run_id="b")
    reg.get("files.copy").run({"source": "raw.txt", "target": "x.txt"}, ctx)
    reg.get("files.copy").run({"source": "raw.txt", "target": "x.txt"}, RunContext(dataset, "a"))
    assert list_runs(ctx.derivatives_root) == ["a", "b"]
    with pytest.raises(RunNotFoundError):
        load_run(ctx.derivatives_root, "missing")


def test_cli_records_runs_and_shows_them(dataset: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["tools", "run", "system.info", "--dataset", str(dataset)])
    assert result.exit_code == 0
    run_id = result.stderr.strip().removeprefix("run ")

    listed = runner.invoke(app, ["runs", "list", "--dataset", str(dataset)])
    assert listed.exit_code == 0
    assert run_id in listed.stdout
    assert "system.info" in listed.stdout

    shown = runner.invoke(app, ["runs", "show", run_id, "--dataset", str(dataset)])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["calls"][0]["tool"] == "system.info"

    missing = runner.invoke(app, ["runs", "show", "nope", "--dataset", str(dataset)])
    assert missing.exit_code == 1
