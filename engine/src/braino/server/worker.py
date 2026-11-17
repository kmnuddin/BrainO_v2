"""What a job does, run in a process of its own: ``python -m braino.server.worker``.

Each job runs in a fresh process, so it can be cancelled (the process is stopped), a crash in
the science stack cannot take the service down, and the threads each job's numerical libraries
use can be limited (by the service, through the environment) so that jobs running side by side
share the processor.

The service writes the job to the process's standard input as one JSON object::

    {"kind": "pipeline", "payload": {...}, "threads": 2, "worker": null, "path": []}

and the job reports back on standard output, one JSON message per line, ``[kind, data]``:

- ``["started", {"run_id": ...}]`` once its run has an ID;
- ``["progress", {"fraction": 0..1 or null, "message": ..., "event": {...} or null}]``;
- ``["done", <result>]`` or ``["failed", <message>]``, last.

Before the job starts, everything else written to standard output (by the job or a library it
uses) is sent to standard error instead, so only these messages reach the service.

Failure messages carry only the exception's text, never a traceback: tracebacks hold file
paths, and an import's paths can identify the person recorded.

``worker`` names another function, ``"module:function"``, to run in place of :func:`main`
(tests), and ``path`` adds folders to ``sys.path`` to find it.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol, TextIO

PROGRESS_INTERVAL = 0.25
"""Seconds between progress messages that carry no event (e.g. a download's bytes)."""


class Queue(Protocol):
    def put(self, item: Any) -> None: ...


class Reporter:
    """How a job tells the service what it is doing."""

    def __init__(self, queue: Queue) -> None:
        self._queue = queue
        self._last = 0.0

    def started(self, run_id: str | None) -> None:
        self._queue.put(("started", {"run_id": run_id}))

    def progress(
        self,
        fraction: float | None,
        message: str | None = None,
        event: dict[str, Any] | None = None,
    ) -> None:
        """Report progress. Messages without an event are sent at most every
        :data:`PROGRESS_INTERVAL` seconds, so a download's many updates stay few."""
        now = time.monotonic()
        if event is None and now - self._last < PROGRESS_INTERVAL and fraction != 1.0:
            return
        self._last = now
        self._queue.put(("progress", {"fraction": fraction, "message": message, "event": event}))


class LineWriter:
    """A :class:`Queue` that writes each message as a line of JSON."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def put(self, item: Any) -> None:
        self._stream.write(json.dumps(item, default=str) + "\n")
        self._stream.flush()


def main(kind: str, payload: dict[str, Any], queue: Queue, threads: int) -> None:
    """Run one job, reporting through ``queue``."""
    reporter = Reporter(queue)
    try:
        result = KINDS[kind](payload, reporter)
    except Exception as exc:  # reported to the service, which marks the job failed
        queue.put(("failed", _message(exc)))
        return
    queue.put(("done", result))


def _message(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _process_main() -> None:
    """The process's entry point: read the job, keep standard output for messages, run it."""
    job = json.loads(sys.stdin.read())
    # Messages go to a copy of the real standard output; descriptor 1 (used by print and by
    # libraries written in C) is pointed at standard error.
    sys.stdout.flush()
    messages = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.path[:0] = [p for p in job.get("path") or [] if p not in sys.path]
    target: Callable[..., None] = main
    if job.get("worker"):
        module, _, name = job["worker"].partition(":")
        target = getattr(importlib.import_module(module), name)
    target(job["kind"], job["payload"], LineWriter(messages), int(job["threads"]))


# --- the kinds of job ---------------------------------------------------------------------


def run_tool(payload: dict[str, Any], reporter: Reporter) -> dict[str, Any]:
    """One tool call. A decision tool's result is its pending proposal."""
    from braino.tools import RunContext, registry

    tool = registry.get(payload["tool"])
    ctx = RunContext(dataset_root=payload.get("dataset"))
    reporter.started(ctx.run_id)
    result = tool.run(payload.get("arguments") or {}, ctx)
    return {"run_id": ctx.run_id, "tool": tool.name, "result": result.model_dump(mode="json")}


def run_pipeline_job(payload: dict[str, Any], reporter: Reporter) -> dict[str, Any]:
    """A pipeline over a dataset; each step's outcome is reported as it finishes."""
    from braino.pipelines import PipelineSpec, check_spec, pipeline_subjects, run_pipeline
    from braino.pipelines.runner import Status
    from braino.tools import RunContext, current_user

    spec = PipelineSpec.model_validate(payload["spec"])
    check_spec(spec)
    ctx = RunContext(dataset_root=payload["dataset"])
    reporter.started(ctx.run_id)
    subjects: list[str] | None = payload.get("subjects")
    n_subjects = len(subjects) if subjects is not None else len(pipeline_subjects(ctx, spec))
    total = max(1, n_subjects * len(spec.steps) + len(spec.group))
    done = 0

    def progress(outcome: Any) -> None:
        nonlocal done
        done += 1
        reporter.progress(
            min(done / total, 1.0),
            f"{outcome.subject} · {outcome.step}: {outcome.status}",
            outcome.model_dump(mode="json"),
        )

    approver = (
        f"{current_user()} (auto-approved in a pipeline run)"
        if payload.get("auto_approve")
        else None
    )
    report = run_pipeline(
        spec,
        ctx,
        subjects=subjects,
        from_step=payload.get("from_step"),
        auto_approve=approver,
        progress=progress,
    )
    statuses: tuple[Status, ...] = ("ran", "cached", "pending", "rejected", "failed", "blocked")
    return {
        "run_id": report.run_id,
        "name": report.name,
        "subjects": report.subjects,
        "counts": {s: report.count(s) for s in statuses},
        "complete": report.complete,
        "outcomes": [o.model_dump(mode="json") for o in report.outcomes],
    }


def run_import(payload: dict[str, Any], reporter: Reporter) -> dict[str, Any]:
    """Recordings into a dataset. Results are given by position in the request, so they
    carry no source path or original ID."""
    from pathlib import Path

    from braino.io.batch import ImportFailedError, ImportJob, import_job
    from braino.io.deidentify import SubjectKey
    from braino.tools import RunContext

    dataset = Path(payload["dataset"])
    dataset.mkdir(parents=True, exist_ok=True)
    key = SubjectKey.open(Path(payload["key"]), dataset_root=dataset)
    jobs = [ImportJob.model_validate(j) for j in payload["jobs"]]
    ctx = RunContext(dataset_root=dataset)
    reporter.started(ctx.run_id)
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        try:
            report = import_job(ctx, key, job, overwrite=bool(payload.get("overwrite")))
        except ImportFailedError as exc:
            item: dict[str, Any] = {"index": index, "ok": False, "error": str(exc)}
        else:
            item = {
                "index": index,
                "ok": True,
                "subject": report.deidentification.subject,
                "format": report.format,
                "check_by_hand": list(report.deidentification.free_text_events),
            }
        results.append(item)
        reporter.progress((index + 1) / len(jobs), f"{index + 1} of {len(jobs)}", item)
    imported = sum(r["ok"] for r in results)
    return {
        "run_id": ctx.run_id,
        "imported": imported,
        "failed": len(results) - imported,
        "results": results,
    }


def run_decision(payload: dict[str, Any], reporter: Reporter) -> dict[str, Any]:
    """A person's decision on a proposal; on approval, the tool commits it."""
    from braino.tools import RunContext, decide

    ctx = RunContext(dataset_root=payload["dataset"])
    reporter.started(ctx.run_id)
    record = decide(
        payload["proposal_id"],
        ctx,
        approve=bool(payload["approve"]),
        decided_by=payload["decided_by"],
        edits=payload.get("edits"),
        note=payload.get("note"),
    )
    return record.model_dump(mode="json")


def run_model_setup(payload: dict[str, Any], reporter: Reporter) -> dict[str, Any]:
    """Install llama.cpp and download the model (first run), reporting download progress."""
    from braino.runtime import setup

    reporter.started(None)

    def progress(label: str) -> Callable[[int, int], None]:
        def update(done: int, total: int) -> None:
            fraction = done / total if total else None
            reporter.progress(fraction, label, None)

        return update

    settings = setup(
        payload.get("model"),
        payload.get("backend"),
        benchmark=bool(payload.get("benchmark", True)),
        progress=progress,
    )
    return settings.model_dump(mode="json")


KINDS: dict[str, Callable[[dict[str, Any], Reporter], Any]] = {
    "tool": run_tool,
    "pipeline": run_pipeline_job,
    "import": run_import,
    "decision": run_decision,
    "model_setup": run_model_setup,
}


if __name__ == "__main__":
    _process_main()
