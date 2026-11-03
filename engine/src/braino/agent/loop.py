"""The agent loop: the local model works on a request through a :class:`Session`.

Each step the model chooses one action (:mod:`braino.llm`); the loop carries it out and
reports what came of it. On top of the constrained steps, the loop guards against what a
small model does when it loses track:

- **Repeated calls.** A call identical to an earlier one is not run again when nothing has
  changed since (the earlier result still holds), nor when it repeats a failed call with
  nothing run in between. A decision tool is not run again once its proposal was decided.
  The model is told why, and after two such calls in a row it must answer.
- **Decisions.** The user's decision comes back in plain words: what was applied, or that
  nothing was changed, and that the decision is final.
- **Grounding.** Before an answer reaches the user, every number in it is checked against
  the tool results (:mod:`braino.agent.grounding`). Numbers no tool gave are sent back once
  for correction; any that remain are listed with the answer.
- **Budgets.** The model has ``max_steps`` steps, and must answer at the last one, or when
  the conversation nearly fills the model's context.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from braino.agent.grounding import ungrounded_in_answer
from braino.agent.session import AgentAnswer, DecisionLog, Session, ToolOutcome
from braino.llm import (
    Answer,
    AskUser,
    CallTool,
    ChatClient,
    Conversation,
    LLMError,
    RequestDecision,
    StepError,
)
from braino.runtime.settings import DEFAULT_CTX
from braino.tools.base import Risk
from braino.tools.registry import ToolNotFoundError

MAX_STEPS = 60
"""Steps for one request: enough for a group analysis of about 15 subjects one tool call at a
time (the 10-subject group test takes about 35)."""
MAX_REPEATS = 2
"""Repeated calls in a row after which the model must answer."""
CONTEXT_RESERVE = 3
"""The model must answer once fewer than this many replies' worth of context is left."""

EventKind = Literal["thinking", "model", "call", "result", "skipped", "decision", "answer"]


@dataclass(frozen=True)
class AgentEvent:
    """Something the agent did, for a progress display."""

    kind: EventKind
    text: str = ""
    seconds: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Past:
    """The last time a call with these exact arguments was made."""

    ok: bool
    error: str | None
    proposal_id: str | None
    changes: int
    runs: int


class LLMAgent:
    """An agent driven by a language model behind an OpenAI-compatible server.

    :meth:`solve` works on a request; :meth:`follow_up` continues the same conversation with
    a new request (the model keeps what it learned). ``on_event`` receives every step, for a
    progress display; :attr:`log` keeps every model reply with its timing.
    """

    name = "llm"

    def __init__(
        self,
        client: ChatClient,
        *,
        model: str | None = None,
        max_steps: int = MAX_STEPS,
        ctx_size: int = DEFAULT_CTX,
        max_tokens: int = 768,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model or client.model
        self.max_steps = max_steps
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.on_event = on_event
        self.conversation: Conversation | None = None
        self.log: list[dict[str, Any]] = []
        self._requests: list[str] = []
        self._past: dict[str, _Past] = {}
        self._changes = 0  # successful calls that may change data, and applied decisions
        self._runs = 0  # successful calls
        self._noted = 0  # model replies logged

    def solve(self, request: str, session: Session) -> AgentAnswer:
        """Work on a new request, in a new conversation."""
        self.conversation = Conversation(
            self.client,
            request,
            tools=session.tools,
            ctx_size=self.ctx_size,
            max_tokens=self.max_tokens,
        )
        self._requests = [request]
        self._past = {}
        self._noted = 0
        return self._run(session)

    def follow_up(self, request: str, session: Session) -> AgentAnswer:
        """Work on a further request in the same conversation."""
        if self.conversation is None:
            return self.solve(request, session)
        self.conversation.follow_up(request)
        self._requests.append(request)
        return self._run(session)

    # --- the loop -----------------------------------------------------------------------

    def _run(self, session: Session) -> AgentAnswer:
        conv = self.conversation
        assert conv is not None
        repeats = 0
        answer_only = False
        checked = False
        for n in range(self.max_steps):
            final = (
                answer_only
                or repeats >= MAX_REPEATS
                or n == self.max_steps - 1
                or conv.tokens + CONTEXT_RESERVE * self.max_tokens > self.ctx_size
            )
            self._emit("thinking")
            try:
                step = conv.next_step(final=final)
            except (StepError, LLMError) as error:
                self._note_replies(conv)
                return self._stopped(f"the model failed: {error}")
            self._note_replies(conv)

            if isinstance(step, CallTool):
                skip = self._repeat(session, step)
                if skip is not None:
                    repeats += 1
                    self._emit("skipped", skip, data={"tool": step.tool})
                    conv.add_result(f"Not run: {skip}")
                    continue
                repeats = 0
                self._emit("call", step.tool, data={"arguments": step.arguments})
                started = time.perf_counter()
                outcome = session.call(step.tool, step.arguments)
                self._record(session, step, outcome)
                self._emit(
                    "result",
                    "ok" if outcome.ok else str(outcome.error),
                    time.perf_counter() - started,
                    {"ok": outcome.ok, "proposal_id": outcome.proposal_id},
                )
                conv.add_result(outcome, proposal_id=outcome.proposal_id)
            elif isinstance(step, RequestDecision):
                repeats = 0
                decision = session.request_decision(step.proposal_id, step.message)
                if decision.approved and decision.error is None:
                    self._changes += 1
                self._emit("decision", _decision_text(decision), data=decision.model_dump())
                conv.add_result(_decision_text(decision))
            elif isinstance(step, AskUser):
                conv.add_user_reply(session.ask_user(step.question))
            elif isinstance(step, Answer):
                missing = ungrounded_in_answer(step.text, session, "\n".join(self._requests))
                if missing and not checked and not final:
                    checked = answer_only = True
                    conv.add_result(_ungrounded_text(missing))
                    continue
                self._emit("answer", step.text)
                return AgentAnswer(text=step.text, unverified=missing)
        return self._stopped(f"it used all {self.max_steps} steps")  # pragma: no cover

    def _repeat(self, session: Session, step: CallTool) -> str | None:
        """Why this call is not run again, or None to run it."""
        past = self._past.get(_key(session, step))
        if past is None:
            return None
        if past.proposal_id is not None:
            decided = next(
                (d for d in session.decisions if d.proposal_id == past.proposal_id), None
            )
            if decided is None:
                return (
                    f"this exact call already made proposal {past.proposal_id}, which waits "
                    "for the user's decision. Use the decide action."
                )
            verdict = "approved" if decided.approved else "rejected"
            return (
                f"the user already {verdict} the proposal this exact call made. The decision "
                "is final: do not run it again, report it."
            )
        if past.ok and past.changes == self._changes:
            return (
                "you already made this exact call and nothing has changed since; its result "
                "is above. Use it, make a different call, or answer."
            )
        if not past.ok and past.runs == self._runs:
            return (
                f"this exact call already failed ({past.error}) and nothing has run since. "
                "Fix the arguments, run what it needs first, or answer."
            )
        return None

    def _record(self, session: Session, step: CallTool, outcome: ToolOutcome) -> None:
        if outcome.ok:
            self._runs += 1
            risk = _risk(session, step.tool)
            if risk is not None and risk != Risk.READ:
                self._changes += 1
        self._past[_key(session, step)] = _Past(
            ok=outcome.ok,
            error=outcome.error,
            proposal_id=outcome.proposal_id,
            changes=self._changes,
            runs=self._runs,
        )

    def _note_replies(self, conv: Conversation) -> None:
        """Log the replies of the last step (with tool loads and repairs, more than one)."""
        for reply in conv.replies[self._noted :]:
            usage = reply.usage
            entry = {
                "content": reply.content,
                "finish_reason": reply.finish_reason,
                "prompt_tokens": usage.prompt_tokens,
                "prompt_read": usage.prompt_read,
                "completion_tokens": usage.completion_tokens,
                "seconds": usage.seconds,
            }
            self.log.append(entry)
            self._emit("model", reply.content, usage.seconds, entry)
        self._noted = len(conv.replies)

    def _stopped(self, reason: str) -> AgentAnswer:
        text = f"I stopped before finishing: {reason}."
        self._emit("answer", text)
        return AgentAnswer(text=text, stopped=reason)

    def _emit(
        self,
        kind: EventKind,
        text: str = "",
        seconds: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self.on_event is not None:
            self.on_event(AgentEvent(kind, text, seconds, data or {}))


def _key(session: Session, step: CallTool) -> str:
    """The call with its arguments as the tool reads them (defaults filled in)."""
    arguments: Any = step.arguments
    try:
        arguments = session.tools.get(step.tool).input_model.model_validate(arguments)
        arguments = arguments.model_dump(mode="json")
    except (ToolNotFoundError, ValidationError):
        pass
    return json.dumps([step.tool, arguments], sort_keys=True, default=str)


def _risk(session: Session, tool: str) -> Risk | None:
    try:
        return session.tools.get(tool).risk
    except ToolNotFoundError:
        return None


def _decision_text(d: DecisionLog) -> str:
    if not d.approved:
        note = f" (their note: {d.note})" if d.note else ""
        return (
            f"The user rejected the proposal of {d.tool}{note}, so nothing was changed. "
            "This is final: do not ask again; tell the user nothing was changed."
        )
    edited = f" with changes {_json(d.edits)}" if d.edits else ""
    if d.error is not None:
        return (
            f"The user approved the proposal of {d.tool}{edited}, but applying it failed: "
            f"{d.error}. Nothing was changed."
        )
    return (
        f"The user approved the proposal of {d.tool}{edited}. Applied: {_json(d.applied)}. "
        "This is final: do not ask again; report what was applied, not what was proposed."
    )


def _ungrounded_text(numbers: list[float]) -> str:
    listed = ", ".join(f"{n:g}" for n in numbers)
    return (
        f"Error: these numbers in your answer are not in any tool result: {listed}. Copy "
        "every number exactly as a tool gave it, or leave it out. Reply with the corrected "
        "answer."
    )


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))
