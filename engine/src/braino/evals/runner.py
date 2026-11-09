"""Running a suite of tasks with an agent, and the report it produces.

Each task gets a fresh copy of its dataset in a working folder, a scripted user, and its own
run (so its provenance is separate). Datasets are built once and cached under
``$BRAINO_DATA/evals/``.

A task **passes** when all its required checks pass; its **score** is the share of all its
checks that pass. Reports record the agent, model and hardware tier, so runs on different
computers can be compared, as the plan asks: the agent must work on the minimum tier.

One pass or fail says little about a model: the same model passed a task on the CPU and failed
it on a GPU, whose arithmetic differs slightly. With ``repeat``, each task runs several times,
going through its request and its other wordings (``variants``) in turn, and the report gives
each task's pass rate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
from braino.paths import data_root


class TaskResult(BaseModel):
    id: str
    category: str
    run: int = Field(default=1, description="Which run of the task (from 1)")
    request: str = Field(default="", description="The wording of the request in this run")
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
    tasks: int = Field(description="Task runs (tasks times repeats)")
    passed: int
    pass_rate: float
    mean_score: float


class TaskSummary(BaseModel):
    category: str
    runs: int
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
    repeat: int = Field(default=1, description="Runs of each task")
    tasks: list[TaskResult] = Field(description="Every run of every task")
    pass_rate: float = Field(description="Share of task runs that passed")
    mean_score: float
    by_category: dict[str, CategorySummary]
    by_task: dict[str, TaskSummary] = Field(default_factory=dict)

    def task_rates(self) -> dict[str, float]:
        """Each task's pass rate (also for reports written before ``by_task``)."""
        return {name: s.pass_rate for name, s in _by_task(self.tasks).items()}

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
    repeat: int = 1,
) -> EvalReport:
    """Run each task ``repeat`` times (each run with the next wording of its request)."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    started = datetime.now(UTC)
    clock = time.perf_counter()
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="braino-eval-"))
    results = []
    agent_name, agent_model = "unknown", model
    for task in tasks:
        for run in range(repeat):
            agent = factory(task)
            agent_name = getattr(agent, "name", type(agent).__name__)
            agent_model = model or getattr(agent, "model", None)
            folder = base / task.id if repeat == 1 else base / f"{task.id}-{run + 1}"
            result = run_task(task, agent, folder, run=run)
            results.append(result)
            if progress:
                progress(result)
    report = _report(results, agent_name, agent_model, tier, started, time.perf_counter() - clock)
    report.repeat = repeat
    return report


def run_task(task: EvalTask, agent: Any, folder: Path, *, run: int = 0) -> TaskResult:
    """Run ``task`` once, with the wording of run ``run`` (from 0)."""
    if folder.exists():
        shutil.rmtree(folder)
    root = _copy_dataset(prepare_dataset(task.dataset), folder / "data")
    session = EvalSession(RunContext(dataset_root=root), task.user)
    request = task.wording(run)
    clock = time.perf_counter()
    answer, error = None, None
    try:
        answer = agent.solve(request, session)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    seconds = time.perf_counter() - clock
    checks = evaluate(task, session, answer, request)
    _write_transcript(folder, task, request, session, answer, getattr(agent, "log", None))
    required = [c for c in checks if c.required]
    return TaskResult(
        id=task.id,
        category=task.category,
        run=run + 1,
        request=request,
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
    if spec.erpcore is not None:
        from braino.datasets.erpcore import fetch

        return fetch(spec.erpcore.task, spec.erpcore.subjects)
    assert spec.synthetic is not None
    recipe = spec.synthetic.model_dump(mode="json")
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    root = data_root() / "evals" / f"synthetic-{key}"
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


def _write_transcript(
    folder: Path, task: EvalTask, request: str, session: Any, answer: Any, model_log: Any = None
) -> None:
    transcript = {
        "task": task.id,
        "request": request,
        "calls": [c.model_dump(mode="json") for c in session.calls],
        "decisions": [d.model_dump(mode="json") for d in session.decisions],
        "questions": [q.model_dump(mode="json") for q in session.questions],
        "answer": None if answer is None else answer.text,
        "unverified": None if answer is None else answer.unverified,
        "stopped": None if answer is None else answer.stopped,
    }
    if model_log is not None:
        transcript["model"] = model_log
    (folder / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def hardware() -> dict[str, Any]:
    """The computer the run is on (CPU, RAM, GPUs), for the report."""
    from braino.runtime.hardware import detect

    return detect().model_dump(mode="json")


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
        by_task=_by_task(results),
    )


def _by_task(results: list[TaskResult]) -> dict[str, TaskSummary]:
    runs: dict[str, list[TaskResult]] = {}
    for r in results:
        runs.setdefault(r.id, []).append(r)
    return {
        name: TaskSummary(
            category=rs[0].category,
            runs=len(rs),
            passed=sum(r.passed for r in rs),
            pass_rate=_rate(rs),
            mean_score=_mean(rs),
        )
        for name, rs in sorted(runs.items())
    }


def _rate(results: list[TaskResult]) -> float:
    return round(sum(r.passed for r in results) / len(results), 4) if results else 0.0


def _mean(results: list[TaskResult]) -> float:
    return round(sum(r.score for r in results) / len(results), 4) if results else 0.0


class Comparison(BaseModel):
    pass_rate: tuple[float, float]
    mean_score: tuple[float, float]
    fixed: list[str] = Field(description="Tasks that pass more often now")
    broken: list[str] = Field(description="Tasks that pass less often now")
    rates: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Pass rate before and now of each task in both"
    )
    only_before: list[str]
    only_now: list[str]


def compare(before: EvalReport, now: EvalReport) -> Comparison:
    """Tasks whose pass rate rose or fell (with one run each: fixed or broken)."""
    old = before.task_rates()
    new = now.task_rates()
    both = sorted(set(old) & set(new))
    return Comparison(
        pass_rate=(before.pass_rate, now.pass_rate),
        mean_score=(before.mean_score, now.mean_score),
        fixed=[i for i in both if new[i] > old[i]],
        broken=[i for i in both if new[i] < old[i]],
        rates={i: (old[i], new[i]) for i in both},
        only_before=sorted(set(old) - set(new)),
        only_now=sorted(set(new) - set(old)),
    )
