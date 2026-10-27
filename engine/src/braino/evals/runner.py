"""Running a suite of tasks with an agent, and the report it produces.

Each task gets a fresh copy of its dataset in a working folder, a scripted user, and its own
run (so its provenance is separate). Datasets are built once and cached under
``$BRAINO_DATA/evals/``.

A task **passes** when all its required checks pass; its **score** is the share of all its
checks that pass. Reports record the agent, model and hardware tier, so runs on different
computers can be compared, as the plan asks: the agent must work on the minimum tier.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino import __version__
from braino.context import RunContext
from braino.evals.agents import AgentFactory, EvalSession
from braino.evals.checks import CheckResult, evaluate
from braino.evals.tasks import DatasetSpec, EvalTask


class TaskResult(BaseModel):
    id: str
    category: str
    passed: bool
    score: float
    checks: list[CheckResult]
    n_calls: int
    n_failed_calls: int
    n_decisions: int
    seconds: float
    answer: str | None
    error: str | None = None


class CategorySummary(BaseModel):
    tasks: int
    passed: int
    pass_rate: float
    mean_score: float


class EvalReport(BaseModel):
    agent: str
    model: str | None
    tier: str | None = Field(description="Hardware tier the run represents, e.g. 'minimum'")
    hardware: dict[str, Any]
    braino_version: str
    started_at: datetime
    seconds: float
    tasks: list[TaskResult]
    pass_rate: float
    mean_score: float
    by_category: dict[str, CategorySummary]

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def run_suite(
    tasks: list[EvalTask],
    factory: AgentFactory,
    *,
    workdir: Path | None = None,
    tier: str | None = None,
    model: str | None = None,
    progress: Callable[[TaskResult], None] | None = None,
) -> EvalReport:
    started = datetime.now(UTC)
    clock = time.perf_counter()
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="braino-eval-"))
    results = []
    agent_name, agent_model = "unknown", model
    for task in tasks:
        agent = factory(task)
        agent_name = getattr(agent, "name", type(agent).__name__)
        agent_model = model or getattr(agent, "model", None)
        result = run_task(task, agent, base / task.id)
        results.append(result)
        if progress:
            progress(result)
    return _report(results, agent_name, agent_model, tier, started, time.perf_counter() - clock)


def run_task(task: EvalTask, agent: Any, folder: Path) -> TaskResult:
    if folder.exists():
        shutil.rmtree(folder)
    root = _copy_dataset(prepare_dataset(task.dataset), folder / "data")
    session = EvalSession(RunContext(dataset_root=root), task.user)
    clock = time.perf_counter()
    answer, error = None, None
    try:
        answer = agent.solve(task.request, session)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    seconds = time.perf_counter() - clock
    checks = evaluate(task, session, answer)
    _write_transcript(folder, task, session, answer)
    required = [c for c in checks if c.required]
    return TaskResult(
        id=task.id,
        category=task.category,
        passed=error is None and all(c.passed for c in required),
        score=round(sum(c.passed for c in checks) / len(checks), 4),
        checks=checks,
        n_calls=len(session.calls),
        n_failed_calls=sum(not c.ok for c in session.calls),
        n_decisions=len(session.decisions),
        seconds=round(seconds, 2),
        answer=None if answer is None else answer.text,
        error=error,
    )


def prepare_dataset(spec: DatasetSpec) -> Path:
    """The dataset of ``spec``, built (or downloaded) once and cached."""
    from braino.datasets.erpcore import data_dir

    if spec.erpcore is not None:
        from braino.datasets.erpcore import fetch

        return fetch(spec.erpcore.task, spec.erpcore.subjects)
    assert spec.synthetic is not None
    recipe = spec.synthetic.model_dump(mode="json")
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    root = data_dir().parent / "evals" / f"synthetic-{key}"
    done = root / ".complete"
    if not done.is_file():
        from braino.datasets.synthetic import write_dataset

        if root.exists():
            shutil.rmtree(root)
        settings = {k: v for k, v in recipe.items() if k not in ("subjects", "task")}
        write_dataset(root, recipe["subjects"], task=recipe["task"], **settings)
        done.write_text(__version__, encoding="utf-8")
    return root


def _copy_dataset(source: Path, target: Path) -> Path:
    def ignore(folder: str, names: list[str]) -> list[str]:
        return (
            [n for n in names if n in ("derivatives", ".complete")]
            if Path(folder) == source
            else []
        )

    shutil.copytree(source, target, ignore=ignore)
    return target


def _write_transcript(folder: Path, task: EvalTask, session: Any, answer: Any) -> None:
    transcript = {
        "task": task.id,
        "request": task.request,
        "calls": [c.model_dump(mode="json") for c in session.calls],
        "decisions": [d.model_dump(mode="json") for d in session.decisions],
        "questions": [q.model_dump(mode="json") for q in session.questions],
        "answer": None if answer is None else answer.text,
    }
    (folder / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    memory = _total_memory_gb()
    if memory is not None:
        info["memory_gb"] = memory
    return info


def _total_memory_gb() -> float | None:
    try:
        if sys.platform != "win32":
            pages = float(os.sysconf("SC_PHYS_PAGES")) * float(os.sysconf("SC_PAGE_SIZE"))
            return round(pages / 2**30, 1)
        else:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            kernel32 = ctypes.windll.kernel32
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(float(status.total) / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return None
    return None


def _report(
    results: list[TaskResult],
    agent: str,
    model: str | None,
    tier: str | None,
    started: datetime,
    seconds: float,
) -> EvalReport:
    categories: dict[str, list[TaskResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    return EvalReport(
        agent=agent,
        model=model,
        tier=tier,
        hardware=hardware(),
        braino_version=__version__,
        started_at=started,
        seconds=round(seconds, 1),
        tasks=results,
        pass_rate=_rate(results),
        mean_score=_mean(results),
        by_category={
            name: CategorySummary(
                tasks=len(rs),
                passed=sum(r.passed for r in rs),
                pass_rate=_rate(rs),
                mean_score=_mean(rs),
            )
            for name, rs in sorted(categories.items())
        },
    )


def _rate(results: list[TaskResult]) -> float:
    return round(sum(r.passed for r in results) / len(results), 4) if results else 0.0


def _mean(results: list[TaskResult]) -> float:
    return round(sum(r.score for r in results) / len(results), 4) if results else 0.0


class Comparison(BaseModel):
    pass_rate: tuple[float, float]
    mean_score: tuple[float, float]
    fixed: list[str] = Field(description="Tasks failing before and passing now")
    broken: list[str] = Field(description="Tasks passing before and failing now")
    only_before: list[str]
    only_now: list[str]


def compare(before: EvalReport, now: EvalReport) -> Comparison:
    old = {t.id: t for t in before.tasks}
    new = {t.id: t for t in now.tasks}
    both = sorted(set(old) & set(new))
    return Comparison(
        pass_rate=(before.pass_rate, now.pass_rate),
        mean_score=(before.mean_score, now.mean_score),
        fixed=[i for i in both if not old[i].passed and new[i].passed],
        broken=[i for i in both if old[i].passed and not new[i].passed],
        only_before=sorted(set(old) - set(new)),
        only_now=sorted(set(new) - set(old)),
    )
