"""What an agent can do, and the record of what it did.

An agent works through a :class:`Session`:

- :meth:`Session.call` runs a tool. Tool errors come back as a result the agent can read and
  react to, not as exceptions.
- :meth:`Session.request_decision` asks the user to approve, edit or reject what a decision tool
  proposed. The agent never decides itself: there is no tool for it.
- :meth:`Session.ask_user` asks the user a question.

How the user answers is up to the subclass: a person at the command line (the CLI agent) or a
scripted user (the evaluation harness). Every call, decision and question is logged, so the
session can be scored, and every tool call is also in the run's provenance record.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from braino.context import RunContext
from braino.tools.base import PendingProposal
from braino.tools.registry import ToolNotFoundError, ToolRegistry, registry


class ToolOutcome(BaseModel):
    """What the agent sees after a tool call."""

    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    proposal_id: str | None = Field(
        default=None, description="Set when a decision tool made a proposal; ask the user"
    )


class ToolCallLog(BaseModel):
    index: int
    tool: str
    arguments: dict[str, Any] = Field(description="As validated (defaults filled in) if valid")
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    proposal_id: str | None = None
    seconds: float


class DecisionLog(BaseModel):
    proposal_id: str
    tool: str
    message: str = Field(description="What the agent told the user about the proposal")
    approved: bool
    edits: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None
    applied: dict[str, Any] | None = Field(
        default=None, description="What applying the approved proposal returned"
    )
    error: str | None = None


class QuestionLog(BaseModel):
    question: str
    reply: str


class AgentAnswer(BaseModel):
    """The agent's final reply to the request."""

    text: str


class Session(ABC):
    """One request worked on by one agent, in one run context."""

    def __init__(self, context: RunContext, *, tools: ToolRegistry | None = None) -> None:
        self.context = context
        self.tools = registry if tools is None else tools
        self.calls: list[ToolCallLog] = []
        self.decisions: list[DecisionLog] = []
        self.questions: list[QuestionLog] = []

    def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """LLM-facing schemas of all tools, or of the ``names`` given."""
        if names is None:
            return self.tools.schemas()
        return [self.tools.get(n).llm_schema() for n in names]

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> ToolOutcome:
        """Run a tool; errors are returned, not raised."""
        started = time.perf_counter()
        args = dict(arguments or {})
        logged_args = args
        name = tool
        try:
            spec = self.tools.get(tool)
            name = spec.name
            logged_args = spec.input_model.model_validate(args).model_dump(mode="json")
            result = spec.run(args, self.context)
        except ToolNotFoundError:
            outcome = ToolOutcome(ok=False, error=f"no tool {tool!r}")
        except ValidationError as exc:
            outcome = ToolOutcome(ok=False, error=f"invalid arguments: {_errors(exc)}")
        except Exception as exc:
            outcome = ToolOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
        else:
            data = result.model_dump(mode="json")
            proposal = result.proposal_id if isinstance(result, PendingProposal) else None
            outcome = ToolOutcome(ok=True, result=data, proposal_id=proposal)
        self.calls.append(
            ToolCallLog(
                index=len(self.calls) + 1,
                tool=name,
                arguments=logged_args,
                ok=outcome.ok,
                result=outcome.result,
                error=outcome.error,
                proposal_id=outcome.proposal_id,
                seconds=round(time.perf_counter() - started, 3),
            )
        )
        return outcome

    def request_decision(self, proposal_id: str, message: str = "") -> DecisionLog:
        """Ask the user about a proposal; returns their decision (and its application)."""
        record = self.context.proposals.get(proposal_id)
        decision = self._decide(record.tool, record.proposal, message)
        approved, edits, note = decision
        log = DecisionLog(
            proposal_id=proposal_id,
            tool=record.tool,
            message=message,
            approved=approved,
            edits=edits,
            note=note,
        )
        from braino.tools.decisions import decide

        try:
            done = decide(
                proposal_id,
                self.context,
                approve=approved,
                decided_by=self.user_name,
                edits=edits,
                note=note,
                tools=self.tools,
            )
        except Exception as exc:
            log.error = f"{type(exc).__name__}: {exc}"
        else:
            log.applied = done.decision.apply_result if done.decision else None
        self.decisions.append(log)
        return log

    def ask_user(self, question: str) -> str:
        reply = self._reply(question)
        self.questions.append(QuestionLog(question=question, reply=reply))
        return reply

    @property
    @abstractmethod
    def user_name(self) -> str:
        """Recorded as who decided, e.g. ``user:alice``."""

    @abstractmethod
    def _decide(
        self, tool: str, proposal: dict[str, Any], message: str
    ) -> tuple[bool, dict[str, Any], str | None]:
        """The user's decision: (approve, edits, note)."""

    @abstractmethod
    def _reply(self, question: str) -> str:
        """The user's answer to a question."""


class Agent(Protocol):
    """Anything that can work on a request through a session."""

    name: str
    model: str | None

    def solve(self, request: str, session: Session) -> AgentAnswer: ...


def _errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, e['loc'])) or '(input)'}: {e['msg']}" for e in exc.errors()
    )
