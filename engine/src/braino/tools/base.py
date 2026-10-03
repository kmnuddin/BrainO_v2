"""The tool contract shared by the UI, the CLI, the Python API and the AI agent.

A tool is a plain Python function that takes one Pydantic model and returns another. The
input model's JSON schema is what the language model sees, so field descriptions matter.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel

# Dotted lower-case names, e.g. "eeg.preprocess" or "stats.cluster_permutation".
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")

# OpenAI-style function names may not contain dots, so the LLM sees "eeg__preprocess".
LLM_NAME_SEPARATOR = "__"

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class Risk(str, Enum):
    """How much a tool can change an analysis.

    ``DECISION`` tools change which data or which statistical model the results rest on, so
    the agent must get the user's approval before running them.
    """

    READ = "read"
    COMPUTE = "compute"
    DECISION = "decision"


class ToolDefinitionError(TypeError):
    """Raised when a function cannot be turned into a tool."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: Callable[[Any], BaseModel]

    @property
    def llm_name(self) -> str:
        return self.name.replace(".", LLM_NAME_SEPARATOR)

    @property
    def requires_approval(self) -> bool:
        return self.risk is Risk.DECISION

    def run(self, arguments: Mapping[str, Any]) -> BaseModel:
        """Validate ``arguments``, call the tool and validate its result.

        Raises ``pydantic.ValidationError`` if the arguments or the result do not match the
        declared models.
        """
        params = self.input_model.model_validate(dict(arguments))
        result = self.func(params)
        if not isinstance(result, self.output_model):
            result = self.output_model.model_validate(result)
        return result

    def llm_schema(self) -> dict[str, Any]:
        """Function definition in the OpenAI-compatible format served by Ollama, llama.cpp
        and vLLM."""
        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


def make_tool(func: Callable[[InT], OutT], *, name: str, risk: Risk) -> Tool:
    """Build a :class:`Tool` from a function annotated ``(params: InModel) -> OutModel``.

    The function's docstring becomes the description shown to the language model.
    """
    if not _NAME_PATTERN.fullmatch(name):
        raise ToolDefinitionError(
            f"invalid tool name {name!r}: use dotted lower-case names like 'eeg.preprocess'"
        )

    params = list(inspect.signature(func).parameters.values())
    if len(params) != 1:
        raise ToolDefinitionError(f"tool {name!r} must take exactly one parameter")

    hints = get_type_hints(func)
    input_model = hints.get(params[0].name)
    output_model = hints.get("return")
    for role, model in (("parameter", input_model), ("return", output_model)):
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ToolDefinitionError(
                f"tool {name!r}: {role} must be annotated with a Pydantic model"
            )
    assert input_model is not None and output_model is not None

    description = inspect.getdoc(func)
    if not description:
        raise ToolDefinitionError(
            f"tool {name!r} needs a docstring; it is the LLM-facing description"
        )

    return Tool(
        name=name,
        description=description,
        risk=risk,
        input_model=input_model,
        output_model=output_model,
        func=func,
    )
