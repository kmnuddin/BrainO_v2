"""The agent loop: tools, decisions, questions and answers driven by a (scripted) model, its
guards against repeats and invented numbers, the console session and ``braino agent``.

The model is a scripted OpenAI-compatible server (``scripted_llm.py``); the tools are small
fakes, except in the CLI tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from scripted_llm import ScriptedServer
from typer.testing import CliRunner

from braino import __version__
from braino.agent import (
    AgentEvent,
    ConsoleSession,
    LLMAgent,
    Session,
    numbers_in,
    ungrounded_in_answer,
)
from braino.cli import app
from braino.context import RunContext
from braino.tools import Risk, ToolRegistry, make_tool

# --- fake tools ---------------------------------------------------------------------------


class SubjectIn(BaseModel):
    subject: str
    level: int = 0


class Prepared(BaseModel):
    subject: str
    sfreq_hz: float


class Amplitude(BaseModel):
    subject: str
    amplitude_uv: float


class BadProposal(BaseModel):
    channels: list[str]


class BadApplied(BaseModel):
    bad_channels: list[str]


def prepare(params: SubjectIn, ctx: RunContext) -> Prepared:
    """Prepare a subject's recording."""
    assert ctx.dataset_root is not None
    (ctx.dataset_root / f"prepared-{params.subject}").touch()
    return Prepared(subject=params.subject, sfreq_hz=250.0)


def measure(params: SubjectIn, ctx: RunContext) -> Amplitude:
    """Measure the amplitude."""
    assert ctx.dataset_root is not None
    if not (ctx.dataset_root / f"prepared-{params.subject}").exists():
        raise RuntimeError("run demo.prepare first")
    return Amplitude(subject=params.subject, amplitude_uv=6.384)


def find_bad(params: SubjectIn) -> BadProposal:
    """Propose bad channels."""
    return BadProposal(channels=["T8", "Pz"])


def apply_bad(proposal: BadProposal, ctx: RunContext) -> BadApplied:
    return BadApplied(bad_channels=proposal.channels)


TOOLS = ToolRegistry()
TOOLS.register(make_tool(prepare, name="demo.prepare", risk=Risk.COMPUTE))
TOOLS.register(make_tool(measure, name="demo.measure", risk=Risk.READ))
TOOLS.register(make_tool(find_bad, name="demo.find_bad", risk=Risk.DECISION, apply=apply_bad))
LOAD_ALL = {"action": "load_tools", "tools": ["demo.prepare", "demo.measure", "demo.find_bad"]}


class ScriptedSession(Session):
    def __init__(
        self,
        root: Path,
        *,
        approve: bool = True,
        edits: dict[str, Any] | None = None,
        replies: list[str] | None = None,
    ) -> None:
        super().__init__(RunContext(dataset_root=root, run_id="r1"), tools=TOOLS)
        self.approve = approve
        self.edits = edits or {}
        self.replies = list(replies or [])

    @property
    def user_name(self) -> str:
        return "test:user"

    def _decide(
        self, tool: str, proposal: dict[str, Any], message: str
    ) -> tuple[bool, dict[str, Any], str | None]:
        return self.approve, self.edits if self.approve else {}, None

    def _reply(self, question: str) -> str:
        return self.replies.pop(0)


@pytest.fixture
def server() -> Iterator[ScriptedServer]:
    s = ScriptedServer()
    yield s
    s.close()


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"action": "call", "tool": tool, "arguments": arguments}


def _answer(text: str) -> dict[str, Any]:
    return {"action": "answer", "text": text}


def _actions(request: dict[str, Any]) -> set[str]:
    """The actions the step's schema allowed."""
    options = request["response_format"]["json_schema"]["schema"]["anyOf"]
    return {o["properties"]["action"]["const"] for o in options}


def _responses(request: dict[str, Any]) -> list[str]:
    """The tool responses the model had seen at this request."""
    return [
        m["content"]
        for m in request["messages"]
        if m["role"] == "user" and m["content"].startswith("<tool_response>")
    ]


# --- grounding ----------------------------------------------------------------------------


def test_numbers_in_ranges_minus_signs_and_thousands() -> None:
    minus, dash = chr(0x2212), chr(0x2013)
    text = (
        f"band-pass 0.1-40 Hz, t = {minus}14.45, 5,020 tests, 0.316{dash}0.48 s, sub-01, T8, "
        f"t = {dash}9.189"  # a model wrote negative t values with an en dash
    )
    assert numbers_in(text) == [
        (0.1, 1),
        (40.0, 0),
        (-14.45, 2),
        (5020.0, 0),
        (0.316, 3),
        (0.48, 2),
        (-9.189, 3),
    ]


def test_ungrounded_numbers(tmp_path: Path) -> None:
    session = ScriptedSession(tmp_path)
    session.call("demo.prepare", {"subject": "01"})
    assert ungrounded_in_answer("Sampled at 250 Hz, 25000% of 1 Hz", session) == []
    assert ungrounded_in_answer("Sampled at 251 Hz", session) == [251.0]
    assert ungrounded_in_answer("Use 30 epochs", session, "Use 30 epochs") == []


# --- the loop -----------------------------------------------------------------------------


def test_the_agent_runs_tools_and_answers(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _call("demo.measure", subject="01"),
        _answer("Subject 01: 6.38 µV (sampled at 250 Hz)."),
    ]
    events: list[AgentEvent] = []
    session = ScriptedSession(tmp_path)
    agent = LLMAgent(server.client(), on_event=events.append)
    answer = agent.solve("Measure subject 01", session)

    assert answer.text.startswith("Subject 01: 6.38")
    assert answer.unverified == [] and answer.stopped is None
    assert [c.tool for c in session.calls] == ["demo.prepare", "demo.measure"]
    assert [e.kind for e in events if e.kind not in ("thinking", "model")] == [
        "call",
        "result",
        "call",
        "result",
        "answer",
    ]
    assert len(agent.log) == 4 and agent.log[0]["prompt_tokens"] > 0
    assert '"amplitude_uv":6.384' in _responses(server.requests[-1])[-1]


def test_a_repeated_call_is_not_run_and_then_the_agent_must_answer(
    server: ScriptedServer, tmp_path: Path
) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _call("demo.prepare", subject="01"),
        _call("demo.prepare", subject="01"),
        _answer("Prepared."),
    ]
    session = ScriptedSession(tmp_path)
    answer = LLMAgent(server.client()).solve("Prepare subject 01", session)

    assert answer.text == "Prepared."
    assert len(session.calls) == 1
    seen = _responses(server.requests[-1])
    assert sum("Not run: you already made this exact call" in r for r in seen) == 2
    assert _actions(server.requests[-1]) == {"answer"}
    assert "call" in _actions(server.requests[-2])


def test_a_call_repeated_after_the_same_tool_ran_on_others_is_not_run(
    server: ScriptedServer, tmp_path: Path
) -> None:
    """A model cycled a quality check over three subjects ten times; each call writes a table,
    but the tool's own calls do not change what a repeat would find."""
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _call("demo.prepare", subject="02"),
        _call("demo.prepare", subject="01"),  # nothing else ran: not run again
        _call("demo.find_bad", subject="01"),  # a decision tool...
        {"action": "decide", "proposal_id": "r1-4", "message": "Mark T8, Pz?"},
        _call("demo.prepare", subject="01"),  # ...whose applied decision is a change: runs
        _answer("Done."),
    ]
    session = ScriptedSession(tmp_path)
    LLMAgent(server.client()).solve("Prepare the subjects", session)

    prepared = [c.arguments["subject"] for c in session.calls if c.tool == "demo.prepare"]
    assert prepared == ["01", "02", "01"]
    seen = _responses(server.requests[-1])
    assert sum("Not run: you already made this exact call" in r for r in seen) == 1


def test_a_streak_of_one_tool_on_one_subject_is_stopped(
    server: ScriptedServer, tmp_path: Path
) -> None:
    """Calls that differ a little each time (a model added a band per call) are not repeats,
    but after MAX_STREAK in a row on one subject the next is not run."""
    from braino.agent.loop import MAX_STREAK

    calls = [_call("demo.prepare", subject="01", level=i) for i in range(MAX_STREAK + 1)]
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="02"),  # another subject: not part of the streak
        *calls,
        _call("demo.measure", subject="01"),  # another tool: runs
        _answer("Done."),
    ]
    session = ScriptedSession(tmp_path)
    LLMAgent(server.client()).solve("Prepare subject 01", session)

    assert [c.tool for c in session.calls] == [
        "demo.prepare",
        *["demo.prepare"] * MAX_STREAK,
        "demo.measure",
    ]
    seen = _responses(server.requests[-1])
    assert sum(f"called demo.prepare for this subject {MAX_STREAK} times" in r for r in seen) == 1


def test_a_failed_call_is_retried_only_after_something_ran(
    server: ScriptedServer, tmp_path: Path
) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.measure", subject="01"),
        _call("demo.measure", subject="01"),
        _call("demo.prepare", subject="01"),
        _call("demo.measure", subject="01"),
        _answer("6.38 µV"),
    ]
    session = ScriptedSession(tmp_path)
    LLMAgent(server.client()).solve("Measure subject 01", session)

    assert [(c.tool, c.ok) for c in session.calls] == [
        ("demo.measure", False),
        ("demo.prepare", True),
        ("demo.measure", True),
    ]
    assert any("already failed (RuntimeError: run demo.prepare first)" in r for r in
               _responses(server.requests[-1]))  # fmt: skip


def test_an_approved_decision_is_reported_and_final(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.find_bad", subject="02"),
        {"action": "decide", "proposal_id": "r1-1", "message": "T8 and Pz look bad."},
        _call("demo.find_bad", subject="02"),
        _answer("T8 is marked bad."),
    ]
    session = ScriptedSession(tmp_path, edits={"channels": ["T8"]})
    answer = LLMAgent(server.client()).solve("Find and mark bad channels in 02", session)

    assert answer.text == "T8 is marked bad."
    assert len(session.calls) == 1 and len(session.decisions) == 1
    assert session.decisions[0].message == "T8 and Pz look bad."
    after_decision = server.requests[3]
    assert "ask" not in _actions(after_decision) and "decide" not in _actions(after_decision)
    seen = _responses(server.requests[-1])
    assert (
        'The user approved the proposal of demo.find_bad with changes {"channels": ["T8"]}. '
        'Applied: {"bad_channels": ["T8"]}.' in seen[-2]
    )
    assert "Not run: the user already approved the proposal" in seen[-1]


def test_a_rejected_decision_says_nothing_was_changed(
    server: ScriptedServer, tmp_path: Path
) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.find_bad", subject="02"),
        {"action": "decide", "proposal_id": "r1-1", "message": "Mark T8 and Pz?"},
        _answer("You rejected it; nothing was changed."),
    ]
    session = ScriptedSession(tmp_path, approve=False)
    LLMAgent(server.client()).solve("Mark bad channels in 02", session)
    assert (
        "The user rejected the proposal of demo.find_bad, so nothing was changed"
        in (_responses(server.requests[-1])[-1])
    )


def test_a_waiting_proposal_is_not_proposed_again(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.find_bad", subject="02"),
        _call("demo.find_bad", subject="02"),
        _answer("Waiting for you."),
    ]
    session = ScriptedSession(tmp_path)
    LLMAgent(server.client()).solve("Mark bad channels in 02", session)
    assert len(session.calls) == 1
    assert "already made proposal r1-1, which waits" in _responses(server.requests[-1])[-1]


def test_questions_go_to_the_user(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        {"action": "ask", "question": "Which subject?"},
        _call("demo.prepare", subject="03"),
        _answer("Prepared subject 03."),
    ]
    session = ScriptedSession(tmp_path, replies=["03"])
    LLMAgent(server.client()).solve("Prepare a subject", session)
    assert session.questions[0].reply == "03"
    assert "ask" not in _actions(server.requests[2])  # no second question right away


def test_invented_numbers_are_sent_back_once(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _answer("Sampled at 251 Hz."),
        _answer("Sampled at 250 Hz."),
    ]
    answer = LLMAgent(server.client()).solve("Prepare 01", ScriptedSession(tmp_path))
    assert answer.text == "Sampled at 250 Hz." and answer.unverified == []
    assert "not in any tool result: 251" in _responses(server.requests[-1])[-1]
    assert _actions(server.requests[-1]) == {"answer"}


def test_numbers_still_invented_after_the_correction_are_listed(
    server: ScriptedServer, tmp_path: Path
) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _answer("Sampled at 251 Hz."),
        _answer("Sampled at 252 Hz."),
    ]
    answer = LLMAgent(server.client()).solve("Prepare 01", ScriptedSession(tmp_path))
    assert answer.unverified == [252.0]


def test_the_last_step_must_be_an_answer(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _call("demo.prepare", subject="02"),
        _answer("Prepared 01 and 02."),
    ]
    session = ScriptedSession(tmp_path)
    LLMAgent(server.client(), max_steps=3).solve("Prepare 01, 02 and 03", session)
    assert len(session.calls) == 2
    assert _actions(server.requests[-1]) == {"answer"}


def test_the_agent_stops_when_the_model_fails(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += ["not json", "still not json"]
    answer = LLMAgent(server.client()).solve("Prepare 01", ScriptedSession(tmp_path))
    assert answer.stopped is not None and answer.stopped.startswith("the model failed")
    assert answer.text.startswith("I stopped before finishing")


def test_a_follow_up_continues_the_conversation(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        LOAD_ALL,
        _call("demo.prepare", subject="01"),
        _answer("Prepared 01."),
        _call("demo.prepare", subject="02"),
        _answer("Prepared 02."),
    ]
    session = ScriptedSession(tmp_path)
    agent = LLMAgent(server.client())
    agent.solve("Prepare 01", session)
    assert agent.follow_up("Now 02", session).text == "Prepared 02."
    first, last = server.requests[2]["messages"], server.requests[-1]["messages"]
    assert last[: len(first)] == first
    assert {"role": "user", "content": "Now 02"} in last
    assert len(session.calls) == 2


# --- the console session ------------------------------------------------------------------


class Terminal:
    def __init__(self, *inputs: str) -> None:
        self.inputs = list(inputs)
        self.lines: list[str] = []

    def prompt(self, text: str, **kwargs: Any) -> str:
        self.lines.append(f"? {text}")
        return self.inputs.pop(0)

    def echo(self, text: str) -> None:
        self.lines.append(text)


def _console(tmp_path: Path, terminal: Terminal) -> ConsoleSession:
    context = RunContext(dataset_root=tmp_path, run_id="r1")
    return ConsoleSession(context, tools=TOOLS, prompt=terminal.prompt, echo=terminal.echo)


def test_console_approval_with_edits(tmp_path: Path) -> None:
    terminal = Terminal(
        "maybe", "e", "{not json", '{"nope": 1}', '{"channels": "T8"}', '{"channels": ["T8"]}',
        "Pz carries the P3",
    )  # fmt: skip
    session = _console(tmp_path, terminal)
    outcome = session.call("demo.find_bad", {"subject": "02"})
    assert outcome.proposal_id is not None
    log = session.request_decision(outcome.proposal_id, "T8 and Pz look noisy.")

    assert log.approved and log.edits == {"channels": ["T8"]}
    assert log.note == "Pz carries the P3"
    assert log.applied == {"bad_channels": ["T8"]}
    text = "\n".join(terminal.lines)
    assert "Decision needed: demo.find_bad proposes" in text
    assert "    channels: T8, Pz" in text and "    T8 and Pz look noisy." in text
    assert "Type a to approve" in text
    assert "not valid JSON" in text and "Unknown fields: nope" in text
    assert "Those changes are not valid" in text


def test_console_rejection_and_questions(tmp_path: Path) -> None:
    terminal = Terminal("r", "", "  subject 03 ")
    session = _console(tmp_path, terminal)
    outcome = session.call("demo.find_bad", {"subject": "02"})
    assert outcome.proposal_id is not None
    log = session.request_decision(outcome.proposal_id, "")
    assert not log.approved and log.note is None
    assert session.ask_user("Which subject?") == "subject 03"
    assert "The agent asks: Which subject?" in terminal.lines


# --- the CLI ------------------------------------------------------------------------------

runner = CliRunner()


def test_cli_agent_answers_a_request(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [
        _call("system.info"),  # not loaded yet: this loads it
        _call("system.info"),
        _answer(f"This is BrainO {__version__}."),
    ]
    result = runner.invoke(
        app, ["agent", "--dataset", str(tmp_path), "--server", server.url, "Which version?"]
    )
    assert result.exit_code == 0, result.output
    assert "> system.info()" in result.stdout
    assert "  done in" in result.stdout
    assert f"This is BrainO {__version__}." in result.stdout
    assert "braino runs show" in result.stderr


def test_cli_agent_takes_requests_one_by_one(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [_answer("Hello."), _answer("Bye.")]
    result = runner.invoke(
        app,
        ["agent", "--dataset", str(tmp_path), "--server", server.url, "-v"],
        input="Hi\nThanks\n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Hello." in result.stdout and "Bye." in result.stdout
    assert "[model" in result.stderr
    assert {"role": "user", "content": "Thanks"} in server.requests[-1]["messages"]


def test_cli_agent_lists_unverified_numbers(server: ScriptedServer, tmp_path: Path) -> None:
    server.replies += [_answer("It is 42.5."), _answer("It is 42.5.")]
    result = runner.invoke(
        app, ["agent", "--dataset", str(tmp_path), "--server", server.url, "How much?"]
    )
    assert result.exit_code == 0, result.output
    assert "Check these numbers: they are not in any tool result (42.5)" in result.stderr


def test_cli_eval_with_the_llm_agent(server: ScriptedServer, tmp_path: Path) -> None:
    pytest.importorskip("mne")  # makes the task's synthetic dataset when it is not cached
    server.replies += [_call("dataset.inspect"), _answer("The dataset has 3 subjects.")]
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["eval", "run", "--agent", "llm", "--server", server.url, "--tasks", "inspect-dataset",
         "--workdir", str(tmp_path / "work"), "--output", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "llm: " in result.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["agent"] == "llm"
    transcript = json.loads(
        (tmp_path / "work" / "inspect-dataset" / "transcript.json").read_text(encoding="utf-8")
    )
    assert transcript["calls"][0]["tool"] == "dataset.inspect"
    assert len(transcript["model"]) == 2


# --- the real model -------------------------------------------------------------------------


@pytest.mark.llama
def test_real_model_bad_channels_decision(tmp_path: Path) -> None:
    """The real model on the eval task where the 4B model used to loop: it must ask for the
    decision once and report what the user applied."""
    from braino.evals import builtin_tasks
    from braino.evals.runner import run_task
    from braino.llm import ChatClient
    from braino.runtime import start_server

    task = next(t for t in builtin_tasks() if t.id == "bad-channels-ask")
    with start_server() as llama:
        agent = LLMAgent(ChatClient.for_server(llama), ctx_size=llama.ctx_size)
        result = run_task(task, agent, tmp_path / task.id)
    assert result.n_decisions == 1  # asked once, no loop
    assert result.error is None and result.answer is not None and "T8" in result.answer
    for check in result.checks:
        if check.kind in ("order", "decision_requested", "answer_mentions", "grounded"):
            assert check.passed, check
