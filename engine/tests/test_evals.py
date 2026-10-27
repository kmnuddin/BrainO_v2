"""The agent evaluation harness: task files, checks, grounding, and runs with simple agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from braino.agent import AgentAnswer, Session
from braino.evals import (
    EvalReport,
    EvalTask,
    TaskError,
    builtin_tasks,
    compare,
    select_tasks,
)
from braino.evals.checks import match, numbers_in, render, resolve
from braino.evals.runner import TaskResult
from braino.tools import registry

# --- task files (no science stack needed) -------------------------------------------------


def test_builtin_tasks_load_and_cover_the_categories() -> None:
    tasks = builtin_tasks()
    assert len(tasks) >= 12
    categories = {t.category for t in tasks}
    assert {"qc", "preprocessing", "decisions", "erp", "stats", "grounding"} <= categories
    assert all(any(c.kind == "grounded" for c in t.all_checks()) for t in tasks)
    assert [t.id for t in select_tasks()] == [t.id for t in tasks if not t.large]
    with pytest.raises(TaskError, match="no task"):
        select_tasks(["nope"])


@pytest.mark.parametrize("task", builtin_tasks(), ids=lambda t: t.id)
def test_solutions_use_real_tools_with_valid_arguments(task: EvalTask) -> None:
    from braino.evals.agents import _substitute

    ids = set()
    for step in task.solution:
        if step.tool is None:
            assert step.decide is None or step.decide in ids, f"decides on unknown {step.decide}"
            continue
        tool = registry.get(step.tool)
        subjects: list[str | None] = list(step.foreach) if step.foreach else [None]
        for subject in subjects:
            tool.input_model.model_validate(_substitute(step.args, subject))
        if step.id:
            ids.add(step.id)
    for check in task.all_checks():
        for name in check.tools + [n for n in (check.before, check.after) if n]:
            assert name in registry, f"check names unknown tool {name}"


def test_task_validation() -> None:
    base: dict[str, Any] = {
        "id": "t",
        "category": "x",
        "request": "r",
        "dataset": {"synthetic": {"subjects": ["01"]}},
        "checks": [{"kind": "tool_called", "tool": "dataset.inspect"}],
    }
    EvalTask.model_validate(base)
    with pytest.raises(ValueError, match="exactly one"):
        EvalTask.model_validate({**base, "dataset": {}})
    with pytest.raises(ValueError, match="needs tool"):
        EvalTask.model_validate({**base, "checks": [{"kind": "tool_called"}]})
    with pytest.raises(ValueError, match="min and/or max"):
        EvalTask.model_validate({**base, "checks": [{"kind": "answer_number"}]})


# --- checks -------------------------------------------------------------------------------


def test_argument_matchers() -> None:
    assert match(1.0, 1)
    assert match(["P8", "P7"], ["P7", "P8"])
    assert not match(["P7"], ["P7", "P8"])
    assert match("fdr", {"not": "none"})
    assert match(None, {"not": "none"})
    assert match(0.026, {"range": [0.025, 0.027]})
    assert match("a", {"any": ["a", "b"]})
    assert match(["x", "y", "z"], {"contains": ["z", "x"]})
    conditions = {"targets": ["stimulus/rare"], "others": ["stimulus/frequent", "stimulus/x"]}
    assert match(conditions, {"covers": ["stimulus/rare", "stimulus/frequent"]})
    assert not match(conditions, {"covers": ["stimulus/odd"]})


def test_numbers_in_text_leave_names_alone() -> None:
    text = "T8 and Fp1 of sub-01 are bad; P3 is 6.38 µV at 300 ms, p = 0.0001, t = -2.5, 10Hz"
    assert numbers_in(text) == [(6.38, 2), (300.0, 0), (0.0001, 4), (-2.5, 1), (10.0, 0)]


def test_templates() -> None:
    results = {"erp": {"measures": [{"waveform": "rare", "v": 1.0}, {"waveform": "d", "v": 6.384}]}}
    assert resolve(results, "erp.measures[waveform=d].v") == 6.384
    assert resolve(results, "erp.measures[0].waveform") == "rare"
    assert render("diff {erp.measures[waveform=d].v:.2f} µV", results) == "diff 6.38 µV"


# --- runs (need the science stack) --------------------------------------------------------


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    monkeypatch.setenv("BRAINO_DATA", str(tmp_path / "cache"))
    return tmp_path


def task(task_id: str) -> EvalTask:
    return next(t for t in builtin_tasks() if t.id == task_id)


def run(agent: Any, task_id: str, folder: Path) -> TaskResult:
    from braino.evals import run_task

    return run_task(task(task_id), agent, folder / task_id)


class Inventor:
    """Calls the right tool but states a number no tool produced."""

    name, model = "inventor", None

    def solve(self, request: str, session: Session) -> AgentAnswer:
        session.call("eeg.preprocess", {"subject": "01"})
        session.call("eeg.spectral", {"subject": "01", "source": "continuous"})
        return AgentAnswer(text="The peak alpha frequency is 10.12 Hz.")


class Impatient:
    """Proposes bad channels and reports them as marked, without asking the user."""

    name, model = "impatient", None

    def solve(self, request: str, session: Session) -> AgentAnswer:
        session.call("eeg.preprocess", {"subject": "02"})
        session.call("eeg.detect_bad_channels", {"subject": "02", "method": "qc"})
        return AgentAnswer(text="T8 is marked bad.")


class Crasher:
    name, model = "crasher", None

    def solve(self, request: str, session: Session) -> AgentAnswer:
        session.call("eeg.erp", {"subject": "01"})  # fails: nothing epoched; logged, not raised
        raise RuntimeError("model server went away")


def test_oracle_passes_every_task(isolated_data: Path) -> None:
    pytest.importorskip("picard")
    from braino.evals import agent_factory, run_suite

    report = run_suite(
        select_tasks(), agent_factory("oracle"), workdir=isolated_data / "work", tier="ci"
    )
    failed = {t.id: [c for c in t.checks if not c.passed] or t.error for t in report.tasks
              if not t.passed}  # fmt: skip
    assert not failed, failed
    assert report.pass_rate == 1.0
    assert report.tier == "ci"
    assert report.hardware["cpu_count"]
    transcript = json.loads((isolated_data / "work/qc-noisy-channel/transcript.json").read_text())
    assert transcript["calls"][0]["tool"] == "eeg.qc_report"


def test_null_agent_fails_tasks_that_need_work(isolated_data: Path) -> None:
    from braino.evals import NullAgent

    result = run(NullAgent(), "erp-p3-one-subject", isolated_data)
    assert not result.passed
    assert 0 < result.score < 1  # passes only the checks doing nothing passes
    assert result.n_calls == 0


def test_invented_numbers_fail_grounding(isolated_data: Path) -> None:
    result = run(Inventor(), "spectral-alpha", isolated_data)
    checks = {c.kind: c for c in result.checks}
    assert checks["tool_called"].passed
    assert checks["answer_number"].passed  # the number is plausible...
    assert not checks["grounded"].passed  # ...but no tool produced it
    assert "10.12" in checks["grounded"].detail
    assert not result.passed


def test_skipping_the_decision_fails(isolated_data: Path) -> None:
    result = run(Impatient(), "bad-channels-ask", isolated_data)
    checks = {c.kind: c for c in result.checks}
    assert not checks["decision_requested"].passed
    assert checks["order"].passed
    assert not result.passed


def test_agent_errors_are_recorded(isolated_data: Path) -> None:
    result = run(Crasher(), "erp-p3-one-subject", isolated_data)
    assert result.error == "RuntimeError: model server went away"
    assert result.n_failed_calls == 1
    assert (isolated_data / "erp-p3-one-subject" / "error.txt").is_file()


def test_reports_save_load_and_compare(tmp_path: Path) -> None:
    from braino.evals.runner import _report

    def result(task_id: str, passed: bool) -> TaskResult:
        return TaskResult(
            id=task_id, category="x", passed=passed, score=1.0 if passed else 0.5, checks=[],
            n_calls=1, n_failed_calls=0, n_decisions=0, seconds=0.1, answer="", error=None,
        )  # fmt: skip

    from datetime import UTC, datetime

    before = _report([result("a", True), result("b", False)], "x", None, "minimum",
                     datetime.now(UTC), 1.0)  # fmt: skip
    now = _report([result("a", False), result("b", True), result("c", True)], "x", "qwen",
                  "minimum", datetime.now(UTC), 1.0)  # fmt: skip
    path = before.save(tmp_path / "r.json")
    assert EvalReport.load(path).pass_rate == 0.5
    diff = compare(before, now)
    assert diff.broken == ["a"] and diff.fixed == ["b"] and diff.only_now == ["c"]
    assert now.by_category["x"].passed == 2


def test_cli(isolated_data: Path) -> None:
    from typer.testing import CliRunner

    from braino.cli import app

    runner = CliRunner()
    listed = runner.invoke(app, ["eval", "list"])
    assert "qc-noisy-channel" in listed.output and "erpcore" not in listed.output
    out = isolated_data / "report.json"
    result = runner.invoke(
        app, ["eval", "run", "--tasks", "inspect-dataset,qc-noisy-channel", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "2 of 2 tasks passed" in result.output
    null = isolated_data / "null.json"
    runner.invoke(app, ["eval", "run", "--agent", "null", "--tasks", "inspect-dataset",
                        "-o", str(null)])  # fmt: skip
    compared = runner.invoke(app, ["eval", "compare", str(out), str(null)])
    assert compared.exit_code == 1 and "broken: inspect-dataset" in compared.output
    bad = runner.invoke(app, ["eval", "run", "--agent", "nonsense"])
    assert bad.exit_code == 1
