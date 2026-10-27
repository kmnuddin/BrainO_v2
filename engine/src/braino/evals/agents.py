"""The scripted session that plays the user, and the built-in agents that check the harness.

- :class:`OracleAgent` follows each task's reference solution. It should pass every task; if it
  does not, the task (not an agent) is wrong.
- :class:`NullAgent` calls no tool and answers that it cannot help. It should fail the tasks
  that need work, and shows which checks pass by accident.

Real agents are plugged in by import path (``module:factory``); the factory is called once per
task with no arguments and returns an :class:`~braino.agent.Agent`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from braino.agent.session import AgentAnswer, Session
from braino.context import RunContext
from braino.evals.checks import render
from braino.evals.tasks import EvalTask, ScriptedUser


class EvalSession(Session):
    """A session whose user is the task's scripted user."""

    def __init__(self, context: RunContext, user: ScriptedUser) -> None:
        super().__init__(context)
        self.user = user
        self._replies = list(user.replies)

    @property
    def user_name(self) -> str:
        return "eval:scripted-user"

    def _decide(
        self, tool: str, proposal: dict[str, Any], message: str
    ) -> tuple[bool, dict[str, Any], str | None]:
        approve = self.user.decisions == "approve"
        return approve, dict(self.user.edits.get(tool, {})) if approve else {}, "scripted"

    def _reply(self, question: str) -> str:
        return self._replies.pop(0) if self._replies else self.user.default_reply


class OracleAgent:
    """Follows the task's reference solution."""

    name = "oracle"
    model: str | None = None

    def __init__(self, task: EvalTask) -> None:
        self.task = task

    def solve(self, request: str, session: Session) -> AgentAnswer:
        results: dict[str, Any] = {}
        proposals: dict[str, str] = {}
        for step in self.task.solution:
            if step.tool is not None:
                subjects: list[str | None] = list(step.foreach) if step.foreach else [None]
                for subject in subjects:
                    args = _substitute(step.args, subject)
                    outcome = session.call(step.tool, args)
                    if not outcome.ok:
                        raise RuntimeError(f"solution step {step.id or step.tool}: {outcome.error}")
                    if step.id:
                        results[step.id] = outcome.result
                        if outcome.proposal_id:
                            proposals[step.id] = outcome.proposal_id
            elif step.decide is not None:
                if step.decide not in proposals:
                    raise RuntimeError(f"solution: step {step.decide!r} made no proposal")
                decision = session.request_decision(
                    proposals[step.decide], f"Please review the proposal of {step.decide}."
                )
                results[f"{step.decide}_decision"] = decision.model_dump(mode="json")
            elif step.ask is not None:
                results.setdefault("replies", []).append(session.ask_user(step.ask))
        return AgentAnswer(text=render(self.task.answer, results))


class NullAgent:
    """Does nothing."""

    name = "null"
    model: str | None = None

    def solve(self, request: str, session: Session) -> AgentAnswer:
        return AgentAnswer(text="I cannot help with that.")


AgentFactory = Callable[[EvalTask], Any]


def agent_factory(spec: str) -> AgentFactory:
    """``oracle``, ``null``, or ``module:factory`` for an agent of your own."""
    if spec == "oracle":
        return OracleAgent
    if spec == "null":
        return lambda task: NullAgent()
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"agent {spec!r}: use oracle, null or module:factory")
    factory = getattr(importlib.import_module(module_name), attr)
    return lambda task: factory()


def _substitute(value: Any, subject: str | None) -> Any:
    if subject is None:
        return value
    if isinstance(value, str):
        return value.replace("{subject}", subject)
    if isinstance(value, dict):
        return {k: _substitute(v, subject) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, subject) for v in value]
    return value
