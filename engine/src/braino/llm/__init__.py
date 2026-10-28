"""The agent's connection to the language model.

- :class:`ChatClient` talks to an OpenAI-compatible server: BrainO's own llama.cpp
  (:meth:`ChatClient.for_server`), or Ollama, vLLM and others.
- :class:`Conversation` runs one request step by step. Every reply is one action (call a tool,
  load tools, ask for a decision, ask the user, answer), constrained by a JSON schema built
  from the loaded tools' own parameter schemas (:func:`step_schema`), so the model cannot make
  an invalid tool call.
- Only the tool schemas a step needs are sent (:func:`select_tools`, ``load_tools``), in a
  compact form (:func:`compact_schema`), and the prompt only grows at the end, so llama.cpp's
  prompt cache is reused: reading the prompt is the slow part on a CPU.
"""

from braino.llm.client import ChatClient, ChatReply, LLMError, Message, Usage
from braino.llm.conversation import SYSTEM_PROMPT, ContextFullError, Conversation, system_prompt
from braino.llm.schemas import catalogue, compact_schema, tool_card
from braino.llm.select import select_tools
from braino.llm.steps import (
    Answer,
    AskUser,
    CallTool,
    LoadTools,
    RequestDecision,
    Step,
    StepError,
    parse_step,
    step_schema,
)

__all__ = [
    "SYSTEM_PROMPT",
    "Answer",
    "AskUser",
    "CallTool",
    "ChatClient",
    "ChatReply",
    "ContextFullError",
    "Conversation",
    "LLMError",
    "LoadTools",
    "Message",
    "RequestDecision",
    "Step",
    "StepError",
    "Usage",
    "catalogue",
    "compact_schema",
    "parse_step",
    "select_tools",
    "step_schema",
    "system_prompt",
    "tool_card",
]
