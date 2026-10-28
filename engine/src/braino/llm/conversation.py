"""The conversation with the model about one request, laid out for llama.cpp's prompt cache.

On a CPU, reading the prompt is the slow part, and llama.cpp skips whatever starts the same
as the previous request's prompt. So the conversation only ever grows at the end:

- The **system prompt** is the same for every request (the rules and a one-line catalogue of
  every tool), so the server reads it once and keeps it cached across requests.
- **Tool schemas** are added where they are loaded, as the result of a ``load_tools`` step,
  not in the system prompt (where Qwen's chat template puts native tools, so that loading one
  more would mean reading the whole conversation again). The schemas the request's words
  point to are loaded before the first step (:func:`~braino.llm.select.select_tools`), and
  a tool named in a result (``run eeg.preprocess first``) is loaded right after it.
- **Tool results and the user's replies** come back as tool responses, which keeps the chat
  template's view of "the last user message" fixed, so earlier turns render identically.

Each step costs only what is new: the last tool result, the schemas just loaded, and the reply.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from braino.llm.client import ChatClient, ChatReply, LLMError, Message
from braino.llm.schemas import catalogue, render_cards
from braino.llm.select import select_tools
from braino.llm.steps import (
    AskUser,
    LoadTools,
    RequestDecision,
    Step,
    StepError,
    parse_step,
    step_schema,
)
from braino.runtime.settings import DEFAULT_CTX
from braino.tools.base import Tool
from braino.tools.registry import ToolRegistry, registry

SYSTEM_PROMPT = """\
You are BrainO's EEG analysis agent. You analyse the user's open dataset only through tools, \
one step at a time. Every reply is one JSON object with one action:
- "call": run a loaded tool with arguments. Its result comes back as a tool response.
- "load_tools": load tools you need that are not loaded yet; their descriptions come back.
- "decide": a decision tool returned a proposal_id. Ask the user to approve it, explaining \
the proposal in "message". Never treat a proposal as done before the user decides.
- "ask": ask the user when the request is unclear. Do not guess subjects, conditions, events \
or channels: find them with tools, or ask.
- "answer": your final reply to the request.
Rules: every number in an answer must come from a tool result. Report what the tools found, \
including warnings. If a tool fails, read its error: if another tool must run first, run it, \
then retry; otherwise fix the call or explain the problem. Never repeat a call that failed \
unchanged. If the information asked for is not in the dataset, answer that it is not \
available.

All tools (load a tool before calling it):
{catalogue}"""

#: Characters of a tool result shown to the model (about 1,500 tokens); the rest is cut.
MAX_RESULT_CHARS = 6000
#: Rough characters per token, to estimate the prompt before sending it.
CHARS_PER_TOKEN = 3.5


class ContextFullError(LLMError):
    """The conversation no longer fits in the model's context."""


def system_prompt(tools: ToolRegistry) -> str:
    return SYSTEM_PROMPT.format(catalogue=catalogue(list(tools)))


class Conversation:
    """One request, step by step. :meth:`next_step` returns the model's next action
    (``call``, ``decide``, ``ask`` or ``answer``; tool loading is handled here), and the
    caller reports what came of it with :meth:`add_result`.
    """

    def __init__(
        self,
        client: ChatClient,
        request: str,
        *,
        tools: ToolRegistry | None = None,
        preload: Sequence[str] | None = None,
        ctx_size: int = DEFAULT_CTX,
        max_tokens: int = 768,
        max_loads: int = 3,
        can_ask: bool = True,
    ) -> None:
        self.client = client
        self.tools = registry if tools is None else tools
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.max_loads = max_loads
        self.can_ask = can_ask
        self.messages: list[Message] = [
            {"role": "system", "content": system_prompt(self.tools)},
            {"role": "user", "content": request},
        ]
        self.loaded: list[Tool] = []
        self.pending: list[str] = []
        self.steps: list[Step] = []
        self.replies: list[ChatReply] = []
        names = select_tools(request, self.tools) if preload is None else list(preload)
        if names:
            self._append({"role": "assistant", "content": _dump(LoadTools(tools=names))})
            self._load(names)

    @property
    def loadable(self) -> list[str]:
        loaded = {t.name for t in self.loaded}
        return [t.name for t in self.tools if t.name not in loaded]

    @property
    def tokens(self) -> int:
        """Tokens in the conversation so far (measured at the last reply, estimated after)."""
        if not self.replies:
            return self._estimate(self.messages)
        last = self.replies[-1].usage
        return last.prompt_tokens + last.completion_tokens + self._estimate(self._unsent)

    def next_step(self) -> Step:
        """Ask the model for its next action."""
        loads = 0
        repaired = False
        while True:
            # No two questions in a row: a small model may otherwise ask again and again.
            just_asked = bool(self.steps) and isinstance(self.steps[-1], AskUser)
            can_ask = self.can_ask and not just_asked
            schema = step_schema(self.loaded, self.loadable, self.pending, can_ask=can_ask)
            if self.tokens + self.max_tokens > self.ctx_size:
                raise ContextFullError(
                    f"the conversation ({self.tokens} tokens) no longer fits in the "
                    f"model's context ({self.ctx_size})"
                )
            reply = self.client.chat(self.messages, json_schema=schema, max_tokens=self.max_tokens)
            self.replies.append(reply)
            self._sent = len(self.messages)
            self._append({"role": "assistant", "content": reply.content})
            try:
                if reply.finish_reason == "length":
                    raise StepError(f"the reply was cut off at {self.max_tokens} tokens")
                step = parse_step(reply.content, self.loaded, self.loadable, self.pending)
            except StepError as error:
                if repaired:
                    raise
                repaired = True
                self._append(_tool(f"Error: {error}. Reply again with one valid action."))
                continue
            self.steps.append(step)
            if isinstance(step, LoadTools):
                loads += 1
                if loads > self.max_loads:
                    raise StepError(f"the model loaded tools {loads} times in a row")
                self._load(step.tools)
                continue
            if isinstance(step, RequestDecision):
                self.pending.remove(step.proposal_id)
            return step

    def add_result(
        self, result: BaseModel | dict[str, Any] | str, *, proposal_id: str | None = None
    ) -> None:
        """Report what the last action led to: a tool's result (or error), the user's
        decision, or the user's reply to a question. A decision tool's ``proposal_id``
        becomes something the model can ask the user to decide."""
        if isinstance(result, BaseModel):
            text = result.model_dump_json(exclude_none=True)
        elif isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        if len(text) > MAX_RESULT_CHARS:
            cut = len(text) - MAX_RESULT_CHARS
            text = f"{text[:MAX_RESULT_CHARS]} ... ({cut} more characters not shown)"
        self._append(_tool(text))
        if proposal_id is not None and proposal_id not in self.pending:
            self.pending.append(proposal_id)
        # "run eeg.preprocess first": load the tools a result names, so the next step can
        # call them (only loaded tools are in the step's schema).
        named = [n for n in self.loadable if _mentions(text, n)]
        if named:
            self._load(named)

    def add_user_reply(self, reply: str) -> None:
        """The user's answer to an ``ask`` step."""
        self.add_result(f"The user replied: {reply}")

    def _load(self, names: Sequence[str]) -> None:
        tools = [self.tools.get(n) for n in names]
        self.loaded += [t for t in tools if t not in self.loaded]
        self._append(_tool(f"Loaded. You can now call these tools:\n{render_cards(tools)}"))

    _sent = 0

    @property
    def _unsent(self) -> list[Message]:
        # Everything after the last reply (which the usage already counts).
        return self.messages[self._sent + 1 :]

    def _append(self, message: Message) -> None:
        self.messages.append(message)

    def _estimate(self, messages: list[Message]) -> int:
        return int(sum(len(m.get("content") or "") for m in messages) / CHARS_PER_TOKEN)


def _tool(content: str) -> Message:
    # A user message in Qwen's tool-response wrapper: rendered exactly like a "tool" message
    # by Qwen's template, but accepted by every OpenAI-compatible server (a "tool" message
    # must otherwise answer a native tool call).
    return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}


def _mentions(text: str, tool_name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(tool_name)}(?![\w])", text) is not None


def _dump(step: BaseModel) -> str:
    return step.model_dump_json()
