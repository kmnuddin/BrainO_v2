"""Registry of the tools available to the agent, the CLI and the UI."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from braino.tools.base import Risk, Tool, ToolFuncT, make_tool


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, *, name: str, risk: Risk) -> Callable[[ToolFuncT], ToolFuncT]:
        """Decorator that registers a function as a tool and returns it unchanged."""

        def decorator(func: ToolFuncT) -> ToolFuncT:
            self.register(make_tool(func, name=name, risk=risk))
            return func

        return decorator

    def get(self, name: str) -> Tool:
        """Look a tool up by its name or by the name the LLM uses for it."""
        tool = self._tools.get(name) or self._by_llm_name().get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.llm_schema() for tool in self]

    def _by_llm_name(self) -> dict[str, Tool]:
        return {tool.llm_name: tool for tool in self._tools.values()}

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(sorted(self._tools.values(), key=lambda t: t.name))

    def __len__(self) -> int:
        return len(self._tools)


registry = ToolRegistry()
"""The default registry. Built-in tools register themselves here on import."""

tool = registry.tool
