"""The job runner: long work in processes of its own, as many at once as the computer allows.

A job is queued, then runs (in a process started by :mod:`braino.server.worker`), then ends
done, failed or cancelled. Jobs that would write the same files (two runs of one pipeline on
one dataset, two imports into one dataset) are refused while one of them is unfinished.

Listeners (the service's WebSocket) are told of every change, from the runner's own threads.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""Set in each job's environment, so that the numerical libraries of jobs running side by side
share the processor instead of each using all of it."""

JobKind = Literal["tool", "pipeline", "import", "decision", "model_setup", "example"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
FINISHED: tuple[JobStatus, ...] = ("done", "failed", "cancelled")

MEMORY_PER_JOB_GB = 3.0
"""What one EEG job may need at its peak (ICA on a long, dense recording)."""
MEMORY_RESERVED_GB = 6.0
"""Left for the operating system, the app and the AI model."""
MAX_PARALLEL = 8

Listener = Callable[[dict[str, Any]], None]


class JobNotFoundError(KeyError):
    pass


class JobConflictError(RuntimeError):
    """Raised when a job would write what an unfinished job writes."""


class JobInfo(BaseModel):
    id: str
    kind: JobKind
    title: str
    dataset: str | None = None
    status: JobStatus = "queued"
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float | None = Field(default=None, description="0 to 1, when known")
    message: str | None = None
    run_id: str | None = Field(default=None, description="The provenance run of the job")
    events: list[dict[str, Any]] = Field(
        default_factory=list, description="What the job reported, e.g. each pipeline step"
    )
    result: Any = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """The job without its events and result, as sent on every change."""
        return self.model_dump(mode="json", exclude={"events", "result"})


def parallel_limit(cpu_count: int | None, memory_gb: float | None) -> int:
    """How many jobs may run at once: one per two logical processors, as far as memory
    allows (:data:`MEMORY_PER_JOB_GB` each, after :data:`MEMORY_RESERVED_GB`)."""
    by_cpu = max(1, (cpu_count or 2) // 2)
    by_memory = (
        by_cpu
        if memory_gb is None
        else max(1, int((memory_gb - MEMORY_RESERVED_GB) // MEMORY_PER_JOB_GB))
    )
    return max(1, min(by_cpu, by_memory, MAX_PARALLEL))


@dataclass
class _Job:
    info: JobInfo
    payload: dict[str, Any]
    key: str | None
    process: subprocess.Popen[str] | None = None
    cancel_requested: bool = False
    finished: threading.Event = field(default_factory=threading.Event)


class JobRunner:
    """Runs jobs in their own processes.

    ``max_parallel`` defaults to :func:`parallel_limit` of this computer; ``threads`` (per
    job) to the logical processors shared among them. ``worker``, ``"module:function"``, runs
    in place of :func:`braino.server.worker.main` (tests).
    """

    def __init__(
        self,
        *,
        max_parallel: int | None = None,
        threads: int | None = None,
        worker: str | None = None,
    ) -> None:
        cpu_count = os.cpu_count()
        if max_parallel is None:
            from braino.runtime.hardware import total_memory_gb

            max_parallel = parallel_limit(cpu_count, total_memory_gb())
        self.max_parallel = max_parallel
        self.threads = threads or max(1, (cpu_count or 1) // max_parallel)
        self._worker = worker
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._order: list[str] = []
        self._listeners: list[Listener] = []
        self._closed = False

    # --- asking --------------------------------------------------------------------------

    def submit(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        title: str,
        dataset: str | None = None,
        key: str | None = None,
    ) -> JobInfo:
        """Queue a job; it starts as soon as a slot is free. ``key`` names what it writes:
        a job whose key an unfinished job holds is refused with :class:`JobConflictError`."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the job runner has shut down")
            if key is not None:
                for other in self._jobs.values():
                    if other.key == key and other.info.status not in FINISHED:
                        raise JobConflictError(
                            f"job {other.info.id} ({other.info.title}) is already working on "
                            "this; wait for it or cancel it"
                        )
            job_id = f"job-{secrets.token_hex(4)}"
            info = JobInfo(id=job_id, kind=kind, title=title, dataset=dataset, created_at=_now())
            self._jobs[job_id] = _Job(info=info, payload=payload, key=key)
            self._order.append(job_id)
            self._emit(info)
            self._start_queued()
            return info.model_copy(deep=True)

    def get(self, job_id: str) -> JobInfo:
        with self._lock:
            return self._job(job_id).info.model_copy(deep=True)

    def list(self) -> list[JobInfo]:
        """Every job of this session, newest first."""
        with self._lock:
            return [self._jobs[i].info.model_copy(deep=True) for i in reversed(self._order)]

    def cancel(self, job_id: str) -> JobInfo:
        """Cancel a queued job, or stop a running one. Finished jobs are left as they are."""
        with self._lock:
            job = self._job(job_id)
            if job.info.status == "queued":
                self._finish(job, "cancelled", error=None)
            elif job.info.status == "running":
                job.cancel_requested = True
                job.info.message = "cancelling"
                self._emit(job.info)
                if job.process is not None:
                    job.process.terminate()
            return job.info.model_copy(deep=True)

    def wait(self, job_id: str, timeout: float | None = None) -> JobInfo:
        """Block until the job finishes (or ``timeout`` seconds pass)."""
        with self._lock:
            job = self._job(job_id)
        job.finished.wait(timeout)
        return self.get(job_id)

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call ``listener`` with ``{"type": "job", "job": {...}}`` on every change and
        ``{"type": "job_event", "job_id": ..., "event": {...}}`` for each reported event.
        Returns a function that unsubscribes."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel every unfinished job and wait for their processes to end."""
        with self._lock:
            self._closed = True
            jobs = [j for j in self._jobs.values() if j.info.status not in FINISHED]
        for job in jobs:
            self.cancel(job.info.id)
        for job in jobs:
            job.finished.wait(timeout)

    # --- running -------------------------------------------------------------------------

    def _job(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def _start_queued(self) -> None:
        running = sum(j.info.status == "running" for j in self._jobs.values())
        for job_id in self._order:
            if running >= self.max_parallel:
                return
            job = self._jobs[job_id]
            if job.info.status == "queued":
                self._start(job)
                running += 1

    def _start(self, job: _Job) -> None:
        env = {**os.environ, **{v: str(self.threads) for v in THREAD_VARIABLES}}
        env["PYTHONIOENCODING"] = "utf-8"
        job.process = subprocess.Popen(
            [sys.executable, "-m", "braino.server.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            # No console window flashes up when the service runs under the desktop app.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        job.info.status = "running"
        job.info.started_at = _now()
        self._emit(job.info)
        request = {
            "kind": job.info.kind,
            "payload": job.payload,
            "threads": self.threads,
            "worker": self._worker,
            "path": sys.path if self._worker else [],
        }
        threading.Thread(target=self._watch, args=(job, json.dumps(request)), daemon=True).start()

    def _watch(self, job: _Job, request: str) -> None:
        """Send the job to its process, then relay its messages until it ends."""
        process = job.process
        assert process is not None and process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(request)
            process.stdin.close()
        except OSError:  # the process ended already (e.g. cancelled at once)
            pass
        for line in process.stdout:
            try:
                kind, data = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                if self._handle(job, kind, data):
                    break
        code = process.wait()
        with self._lock:
            if job.info.status in FINISHED:
                return
            if job.cancel_requested:
                self._finish(job, "cancelled", error=None)
            else:
                self._finish(job, "failed", error=f"the job's process stopped (exit code {code})")

    def _handle(self, job: _Job, kind: str, data: Any) -> bool:
        """Apply one message; returns whether the job has ended."""
        if job.cancel_requested and kind in ("done", "failed"):
            self._finish(job, "cancelled", error=None)
            return True
        if kind == "started":
            job.info.run_id = data.get("run_id")
            self._emit(job.info)
        elif kind == "progress":
            if data.get("fraction") is not None:
                job.info.progress = data["fraction"]
            if data.get("message") is not None:
                job.info.message = data["message"]
            event = data.get("event")
            if event is not None:
                job.info.events.append(event)
                self._send({"type": "job_event", "job_id": job.info.id, "event": event})
            self._emit(job.info)
        elif kind == "done":
            job.info.result = data
            job.info.progress = 1.0
            self._finish(job, "done", error=None)
            return True
        elif kind == "failed":
            self._finish(job, "failed", error=str(data))
            return True
        return False

    def _finish(self, job: _Job, status: JobStatus, *, error: str | None) -> None:
        job.info.status = status
        job.info.error = error
        job.info.finished_at = _now()
        if status == "cancelled":
            job.info.message = "cancelled"
        self._emit(job.info)
        job.finished.set()
        if not self._closed:
            self._start_queued()

    def _emit(self, info: JobInfo) -> None:
        self._send({"type": "job", "job": info.summary()})

    def _send(self, message: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener(message)
            except Exception:  # a broken listener must not stop the jobs
                continue


def _now() -> datetime:
    return datetime.now(UTC)
