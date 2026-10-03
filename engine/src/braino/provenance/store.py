"""On-disk layout of provenance records.

Each run gets a folder ``<dataset>/derivatives/braino/runs/<run_id>/`` holding ``run.json``
(the environment) and ``calls.jsonl`` (one tool call per line, appended as calls finish, so a
crash loses at most the call in progress).
"""

from __future__ import annotations

from pathlib import Path

from braino.provenance.models import RunLog, RunRecord, ToolCallRecord

RUNS_DIR = "runs"
RUN_FILE = "run.json"
CALLS_FILE = "calls.jsonl"


class RunNotFoundError(FileNotFoundError):
    pass


def write_run(run_dir: Path, run: RunRecord) -> None:
    (run_dir / RUN_FILE).write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def append_call(run_dir: Path, call: ToolCallRecord) -> None:
    with (run_dir / CALLS_FILE).open("a", encoding="utf-8") as f:
        f.write(call.model_dump_json() + "\n")


def load_run(derivatives_root: Path, run_id: str) -> RunLog:
    run_dir = derivatives_root / RUNS_DIR / run_id
    run_file = run_dir / RUN_FILE
    if not run_file.is_file():
        raise RunNotFoundError(f"no run {run_id!r} in {derivatives_root / RUNS_DIR}")
    run = RunRecord.model_validate_json(run_file.read_text(encoding="utf-8"))
    calls_file = run_dir / CALLS_FILE
    lines = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    calls = [ToolCallRecord.model_validate_json(line) for line in lines if line.strip()]
    return RunLog(run=run, calls=calls)


def list_runs(derivatives_root: Path) -> list[str]:
    """IDs of the recorded runs, oldest first (run IDs sort by start time)."""
    runs_dir = derivatives_root / RUNS_DIR
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if (p / RUN_FILE).is_file())
