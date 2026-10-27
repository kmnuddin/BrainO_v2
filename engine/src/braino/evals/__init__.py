"""Evaluating the analysis agent: tasks with known answers, scored checks, and reports.

Every change to the agent is scored on the same tasks, per hardware tier (plan item 15). See
:mod:`braino.evals.tasks` for the task format and :mod:`braino.evals.runner` for scoring.
"""

from braino.evals.agents import EvalSession, NullAgent, OracleAgent, agent_factory
from braino.evals.checks import CheckResult
from braino.evals.runner import Comparison, EvalReport, TaskResult, compare, run_suite, run_task
from braino.evals.tasks import Check, EvalTask, TaskError, builtin_tasks, load_task, select_tasks

__all__ = [
    "Check",
    "CheckResult",
    "Comparison",
    "EvalReport",
    "EvalSession",
    "EvalTask",
    "NullAgent",
    "OracleAgent",
    "TaskError",
    "TaskResult",
    "agent_factory",
    "builtin_tasks",
    "compare",
    "load_task",
    "run_suite",
    "run_task",
    "select_tasks",
]
