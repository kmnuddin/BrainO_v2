"""Typed, registered analysis tools."""

from braino.context import NoDatasetError, RunContext

# Importing these modules registers the built-in tools.
from braino.tools import dataset as _dataset  # noqa: F401
from braino.tools import eeg as _eeg  # noqa: F401
from braino.tools import system as _system  # noqa: F401
from braino.tools.base import (
    MissingContextError,
    PendingProposal,
    Risk,
    Tool,
    ToolDefinitionError,
    make_tool,
)
from braino.tools.decisions import ProposalAlreadyDecidedError, current_user, decide
from braino.tools.registry import ToolNotFoundError, ToolRegistry, registry, tool

__all__ = [
    "MissingContextError",
    "NoDatasetError",
    "PendingProposal",
    "ProposalAlreadyDecidedError",
    "Risk",
    "RunContext",
    "Tool",
    "ToolDefinitionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "current_user",
    "decide",
    "make_tool",
    "registry",
    "tool",
]
