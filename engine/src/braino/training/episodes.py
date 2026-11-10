"""Episodes: a request, a dataset and a scripted way to handle it, played through the real agent.

A teacher stands in for the model behind :class:`~braino.agent.LLMAgent`: each step it writes
the next action of the episode's script as the JSON the model would write (loading a tool
first when it is not loaded yet). The tools run for real, the scripted user answers questions
and decides on proposals, and the answer is written from the actual results. So the
conversation that comes out is exactly what the model sees and should write at run time:
the same system prompt, tool cards, tool results and step format.

An episode becomes a training example only if it went cleanly (no error sent back to the
"model", no call skipped by the loop's guards, the answer's numbers grounded) and its checks,
the same kinds as the evaluation's, pass.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.agent.loop import LLMAgent
from braino.agent.session import Session
from braino.context import RunContext
from braino.evals.agents import EvalSession
from braino.evals.checks import CheckResult, evaluate
from braino.evals.runner import _copy_dataset, prepare_dataset
from braino.evals.tasks import Check, DatasetSpec, EvalTask, ScriptedUser
from braino.llm.client import ChatReply, LLMError, Message, Usage
from braino.llm.conversation import CHARS_PER_TOKEN
from braino.llm.steps import Answer, AskUser, CallTool, LoadTools, RequestDecision

Results = dict[str, Any]
"""What the script has seen so far: each step id's result, ``<id>_error`` for an expected
failure, ``<id>_decision`` for the user's decision, and ``replies`` to questions."""

Text = str | Callable[[Results], str]
Args = dict[str, Any] | Callable[[Results], dict[str, Any]]


@dataclass(frozen=True)
class Call:
    """Call a tool. ``expect_error`` marks a call that should fail (e.g. a reference to
    channels the recording lacks): its error becomes ``<id>_error``."""

    tool: str
    args: Args = field(default_factory=dict)
    id: str | None = None
    expect_error: bool = False


@dataclass(frozen=True)
class Decide:
    """Ask the user to decide on the proposal of step ``id``."""

    id: str
    message: Text


@dataclass(frozen=True)
class Ask:
    question: Text


@dataclass(frozen=True)
class Say:
    """The final answer."""

    text: Text


Action = Call | Decide | Ask | Say


@dataclass
class Episode:
    family: str
    request: str
    dataset: DatasetSpec
    script: list[Action]
    checks: list[Check]
    user: ScriptedUser = field(default_factory=ScriptedUser)
    grounded: bool = True

    def task(self) -> EvalTask:
        """The episode as an evaluation task, for its checks."""
        return EvalTask(
            id=self.family,
            category=self.family,
            request=self.request,
            dataset=self.dataset,
            user=self.user,
            checks=self.checks,
            grounded=self.grounded,
        )


class TrainingMessage(BaseModel):
    role: str
    content: str
    weight: int | None = Field(
        default=None, description="0 for an assistant message the model did not write"
    )


class Example(BaseModel):
    family: str
    request: str
    messages: list[TrainingMessage]
    seconds: float


class Rejected(BaseModel):
    family: str
    request: str
    reason: str
    checks: list[CheckResult] = Field(default_factory=list)


class TeacherError(LLMError):
    """The script cannot go on (a call failed that should not have, nothing left to say)."""


class Teacher:
    """A stand-in for the chat client: replies with the episode's script, step by step."""

    model = "teacher"

    def __init__(self, episode: Episode, session: Session) -> None:
        self.queue: list[Action] = list(episode.script)
        self.session = session
        self.results: Results = {"replies": []}
        self._waiting: Action | None = None
        self._calls = 0
        self._decisions = 0
        self._questions = 0

    def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        **_: Any,
    ) -> ChatReply:
        self._collect()
        if not self.queue:
            # The loop sends an answer back (e.g. numbers no tool gave): say why.
            last = str(messages[-1].get("content", "")) if messages else ""
            why = last.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
            raise TeacherError(f"the answer was sent back: {why[:300]}")
        loaded, pending = _options(json_schema)
        action = self.queue[0]
        if isinstance(action, Call):
            if action.tool not in loaded:
                return self._reply(messages, LoadTools(tools=[action.tool]))
            args = action.args(self.results) if callable(action.args) else action.args
            step: BaseModel = CallTool(tool=action.tool, arguments=args)
        elif isinstance(action, Decide):
            proposal = self.results.get(f"{action.id}_proposal")
            if proposal is None or proposal not in pending:
                raise TeacherError(f"step {action.id!r} has no pending proposal")
            step = RequestDecision(
                proposal_id=proposal, message=_text(action.message, self.results)
            )
        elif isinstance(action, Ask):
            step = AskUser(question=_text(action.question, self.results))
        else:
            step = Answer(text=_text(action.text, self.results))
        self.queue.pop(0)
        self._waiting = action
        self._calls = len(self.session.calls)
        self._decisions = len(self.session.decisions)
        self._questions = len(self.session.questions)
        return self._reply(messages, step)

    def _collect(self) -> None:
        """Note what came of the last action."""
        action, self._waiting = self._waiting, None
        if isinstance(action, Call):
            own = [c for c in self.session.calls[self._calls :] if c.within is None]
            if not own:
                raise TeacherError(f"{action.tool} was not run (skipped by the agent loop)")
            call = own[-1]
            if call.ok and action.expect_error:
                raise TeacherError(f"{action.tool} succeeded but was expected to fail")
            if not call.ok and not action.expect_error:
                raise TeacherError(f"{action.tool} failed: {call.error}")
            if action.id is not None:
                if call.ok:
                    self.results[action.id] = call.result
                    if call.proposal_id is not None:
                        self.results[f"{action.id}_proposal"] = call.proposal_id
                else:
                    self.results[f"{action.id}_error"] = call.error
        elif isinstance(action, Decide):
            decision = self.session.decisions[self._decisions]
            self.results[f"{action.id}_decision"] = decision.model_dump(mode="json")
        elif isinstance(action, Ask):
            self.results["replies"].append(self.session.questions[self._questions].reply)

    def _reply(self, messages: list[Message], step: BaseModel) -> ChatReply:
        content = step.model_dump_json()
        prompt = int(sum(len(m.get("content") or "") for m in messages) / CHARS_PER_TOKEN)
        return ChatReply(
            content=content,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=int(len(content) / CHARS_PER_TOKEN),
                seconds=0.0,
            ),
        )


def _options(schema: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    """The loaded tools and pending proposals a step's schema allows."""
    loaded: set[str] = set()
    pending: set[str] = set()
    for option in (schema or {}).get("anyOf", []):
        props = option.get("properties", {})
        action = props.get("action", {}).get("const")
        if action == "call" and "const" in props.get("tool", {}):
            loaded.add(props["tool"]["const"])
        elif action == "decide":
            pending.update(props["proposal_id"]["enum"])
    return loaded, pending


def _text(text: Text, results: Results) -> str:
    return text(results) if callable(text) else text


# --- running an episode ---------------------------------------------------------------------

#: Messages that show something went wrong, which a training example must not teach.
_REPAIRS = ("<tool_response>\nError:", "<tool_response>\nNot run:")


def run_episode(episode: Episode, folder: Path) -> Example | Rejected:
    """Play ``episode`` through the agent loop with the teacher; the example, or why not."""
    if folder.exists():
        shutil.rmtree(folder)
    root = _copy_dataset(prepare_dataset(episode.dataset), folder / "data")
    session = EvalSession(RunContext(dataset_root=root), episode.user)
    teacher = Teacher(episode, session)
    agent = LLMAgent(teacher)  # type: ignore[arg-type]
    clock = time.perf_counter()

    def rejected(reason: str, checks: list[CheckResult] | None = None) -> Rejected:
        return Rejected(
            family=episode.family, request=episode.request, reason=reason, checks=checks or []
        )

    try:
        answer = agent.solve(episode.request, session)
    except Exception as exc:  # a tool or the script broke: not an example
        return rejected(f"{type(exc).__name__}: {exc}")
    if answer.stopped:
        return rejected(answer.stopped)
    if answer.unverified:
        return rejected(f"ungrounded numbers {answer.unverified}")
    conversation = agent.conversation
    assert conversation is not None
    repaired = [m["content"][:120] for m in conversation.messages if _is_repair(m)]
    if repaired:
        return rejected(f"the conversation has a repair: {repaired[0]!r}")
    checks = evaluate(episode.task(), session, answer, episode.request)
    failed = [c for c in checks if c.required and not c.passed]
    if failed:
        return rejected(
            f"checks failed: {', '.join(c.description or c.kind for c in failed)}", checks
        )
    return Example(
        family=episode.family,
        request=episode.request,
        messages=_training_messages(
            conversation.messages, [r.content for r in conversation.replies]
        ),
        seconds=round(time.perf_counter() - clock, 2),
    )


def _is_repair(message: Message) -> bool:
    return message["role"] == "user" and str(message.get("content", "")).startswith(_REPAIRS)


def _training_messages(messages: list[Message], replies: list[str]) -> list[TrainingMessage]:
    """The conversation, with the assistant messages the model did not write (the tool loads
    the conversation adds from the request's words) weighted 0."""
    out = []
    written = iter(replies)
    expected = next(written, None)
    for m in messages:
        weight = None
        if m["role"] == "assistant":
            if m["content"] == expected:
                weight, expected = 1, next(written, None)
            else:
                weight = 0
        out.append(TrainingMessage(role=m["role"], content=str(m["content"]), weight=weight))
    return out


def write_jsonl(examples: list[Example], path: Path) -> Path:
    """One example per line: ``{"messages": [...], "family": ..., "request": ...}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            line = {
                "family": ex.family,
                "request": ex.request,
                "messages": [m.model_dump(exclude_none=True) for m in ex.messages],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
