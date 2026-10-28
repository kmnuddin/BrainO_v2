"""The agent's replies: one action per step, constrained to a JSON schema.

Each step the model writes exactly one JSON object, one of:

- ``call``: run a loaded tool, ``{"action": "call", "tool": "eeg.qc_report", "arguments": {...}}``
  (a call of a tool that is not loaded yet loads it, and the model calls it again)
- ``load_tools``: load the full description of tools it needs (only loaded tools can be called)
- ``decide``: show a decision tool's proposal to the user for approval
- ``ask``: ask the user a question
- ``answer``: the final reply

:func:`step_schema` builds the JSON schema of the actions possible at this step: a ``call``
for each loaded tool with that tool's own parameters, ``load_tools`` (or a ``call`` with any
arguments) naming only tools that exist, ``decide`` naming only pending proposals. The
server turns it into a grammar, so the model cannot call a tool that does not exist, misspell
a parameter or give a value of the wrong type. This is why the agent does not use the API's
native tool calls: llama.cpp does not enforce the parameter schemas of Qwen's tool-call
format (it accepts values outside an enum).

Checks that a JSON schema cannot express (e.g. ``tmin < tmax``) are made when the tool runs,
and their errors go back to the model as the tool's result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from braino.llm.schemas import parameters
from braino.tools.base import Tool


class CallTool(BaseModel):
    action: Literal["call"] = "call"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LoadTools(BaseModel):
    action: Literal["load_tools"] = "load_tools"
    tools: list[str]


class RequestDecision(BaseModel):
    action: Literal["decide"] = "decide"
    proposal_id: str
    message: str = Field(description="What the agent tells the user about the proposal")


class AskUser(BaseModel):
    action: Literal["ask"] = "ask"
    question: str


class Answer(BaseModel):
    action: Literal["answer"] = "answer"
    text: str


Step = Annotated[
    CallTool | LoadTools | RequestDecision | AskUser | Answer, Field(discriminator="action")
]
_STEP: TypeAdapter[Step] = TypeAdapter(Step)


class StepError(ValueError):
    """The reply is not a valid step (only possible when the server does not enforce the
    schema, or the reply was cut off)."""


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _text() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def step_schema(
    loaded: Sequence[Tool],
    loadable: Sequence[str] = (),
    proposals: Sequence[str] = (),
    *,
    can_ask: bool = True,
) -> dict[str, Any]:
    """The JSON schema of every action possible at this step."""
    options: list[dict[str, Any]] = [
        _object(
            {"action": {"const": "call"}, "tool": {"const": t.name}, "arguments": parameters(t)}
        )
        for t in loaded
    ]
    if loadable:
        # A small model often just calls the tool it wants: a call of a tool that is not
        # loaded yet is taken as loading it (see parse_step), and the model calls it again.
        options.append(
            _object(
                {
                    "action": {"const": "call"},
                    "tool": {"enum": list(loadable)},
                    "arguments": {"type": "object"},
                }
            )
        )
        options.append(
            _object(
                {
                    "action": {"const": "load_tools"},
                    "tools": {"type": "array", "items": {"enum": list(loadable)}, "minItems": 1},
                }
            )
        )
    if proposals:
        options.append(
            _object(
                {
                    "action": {"const": "decide"},
                    "proposal_id": {"enum": list(proposals)},
                    "message": _text(),
                }
            )
        )
    if can_ask:
        options.append(_object({"action": {"const": "ask"}, "question": _text()}))
    options.append(_object({"action": {"const": "answer"}, "text": _text()}))
    return {"anyOf": options}


def parse_step(
    content: str,
    loaded: Sequence[Tool],
    loadable: Sequence[str] = (),
    proposals: Sequence[str] = (),
) -> Step:
    """Parse and check a reply against what :func:`step_schema` allowed.

    The server already enforces the schema; this catches servers that do not, and replies cut
    off by the token limit. Arguments are checked against the tool's input model.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise StepError(f"the reply is not JSON ({error.msg}): {content[:200]!r}") from error
    try:
        step = _STEP.validate_python(data)
    except ValidationError as error:
        raise StepError(f"the reply is not a valid action: {_errors(error)}") from error

    if isinstance(step, CallTool):
        tool = next((t for t in loaded if t.name == step.tool), None)
        if tool is None:
            if step.tool in loadable:
                return LoadTools(tools=[step.tool])
            raise StepError(f"there is no tool {step.tool!r}")
        try:
            tool.input_model.model_validate(step.arguments)
        except ValidationError as error:
            raise StepError(f"invalid arguments for {tool.name}: {_errors(error)}") from error
    elif isinstance(step, LoadTools):
        unknown = [name for name in step.tools if name not in loadable]
        if unknown:
            raise StepError(f"cannot load {', '.join(unknown)}")
    elif isinstance(step, RequestDecision) and step.proposal_id not in proposals:
        raise StepError(f"no pending proposal {step.proposal_id!r}")
    return step


def _errors(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, e['loc'])) or '(input)'}: {e['msg']}" for e in error.errors()
    )
