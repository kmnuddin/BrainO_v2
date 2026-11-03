"""The AI analysis agent: what it can do (a :class:`Session`), what it returns, and the loop
that drives it with the local model (:class:`LLMAgent`)."""

from braino.agent.console import ConsoleSession
from braino.agent.grounding import numbers_in, ungrounded_in_answer
from braino.agent.loop import AgentEvent, LLMAgent
from braino.agent.session import (
    Agent,
    AgentAnswer,
    DecisionLog,
    QuestionLog,
    Session,
    ToolCallLog,
    ToolOutcome,
)

__all__ = [
    "Agent",
    "AgentAnswer",
    "AgentEvent",
    "ConsoleSession",
    "DecisionLog",
    "LLMAgent",
    "QuestionLog",
    "Session",
    "ToolCallLog",
    "ToolOutcome",
    "numbers_in",
    "ungrounded_in_answer",
]
