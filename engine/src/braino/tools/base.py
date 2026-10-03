"""The tool contract shared by the UI, the CLI, the Python API and the AI agent.

A tool is a plain Python function that takes one Pydantic model and returns another. The
input model's JSON schema is what the language model sees, so field descriptions matter.

A tool that reads a dataset or writes outputs takes a second parameter annotated
:class:`~braino.context.RunContext`. The caller supplies the context; it is not part of the
input schema, so the language model never sees or chooses it.

A ``decision`` tool works in two steps. Running it only *proposes* (e.g. channels to drop) and
returns a :class:`PendingProposal`. A person then approves, edits or rejects the proposal with
:func:`braino.tools.decide`, and only an approved proposal is passed to the tool's ``apply``
function, which commits it.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.provenance.models import ProposalRecord

# Dotted lower-case names, e.g. "eeg.preprocess" or "stats.cluster_permutation".
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")

# OpenAI-style function names may not contain dots, so the LLM sees "eeg__preprocess".
LLM_NAME_SEPARATOR = "__"

ToolFuncT = TypeVar("ToolFuncT", bound=Callable[..., BaseModel])
"""A tool function: ``(params: In) -> Out`` or ``(params: In, ctx: RunContext) -> Out``."""

DECISION_NOTE = (
    "This is a decision tool: it returns a proposal, which takes effect only after the user "
    "approves it."
)


class Risk(StrEnum):
    """How much a tool can change an analysis.

    ``DECISION`` tools change which data or which statistical model the results rest on, so
    what they propose takes effect only after the user approves it.
    """

    READ = "read"
    COMPUTE = "compute"
    DECISION = "decision"


class ToolDefinitionError(TypeError):
    """Raised when a function cannot be turned into a tool."""


class MissingContextError(RuntimeError):
    """Raised when a tool that needs a :class:`RunContext` is run without one."""


class PendingProposal(BaseModel):
    """What running a decision tool returns: a proposal awaiting the user's decision."""

    proposal_id: str
    tool: str
    proposal: dict[str, Any] = Field(description="What the tool proposes")
    status: str = Field(default="pending", description="Always 'pending' when returned")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: Callable[..., BaseModel]
    needs_context: bool = False
    apply_func: Callable[..., BaseModel] | None = None
    """Decision tools only: commits an approved proposal, ``(proposal, ctx) -> result``."""
    apply_output_model: type[BaseModel] | None = None

    @property
    def llm_name(self) -> str:
        return self.name.replace(".", LLM_NAME_SEPARATOR)

    @property
    def requires_approval(self) -> bool:
        return self.risk is Risk.DECISION

    def run(self, arguments: Mapping[str, Any], context: RunContext | None = None) -> BaseModel:
        """Validate ``arguments``, call the tool and validate its result.

        Raises ``pydantic.ValidationError`` if the arguments or the result do not match the
        declared models, and :class:`MissingContextError` if the tool needs a context and
        none was given.

        With a context, the call (including a failed one) is recorded in its provenance. A
        decision tool always needs a context: it stores its result as a proposal and returns
        a :class:`PendingProposal`.
        """
        params = self.input_model.model_validate(dict(arguments))
        if context is None:
            if self.needs_context or self.requires_approval:
                raise MissingContextError(f"tool {self.name!r} needs a RunContext")
            return self._call(params, None)

        with context.provenance.tool_call(self.name, self.risk.value, params) as call:
            if self.requires_approval:
                call.proposal_id = f"{context.run_id}-{call.call_index}"
            result = self._call(params, context)
            call.result = result
        if call.proposal_id is None:
            return result

        record = ProposalRecord(
            proposal_id=call.proposal_id,
            tool=self.name,
            run_id=context.run_id,
            created_at=datetime.now(UTC),
            arguments=params.model_dump(mode="json"),
            proposal=result.model_dump(mode="json"),
        )
        context.proposals.save(record)
        return PendingProposal(
            proposal_id=record.proposal_id, tool=self.name, proposal=record.proposal
        )

    def apply_approved(
        self, proposal: BaseModel, context: RunContext, *, proposal_id: str
    ) -> BaseModel:
        """Commit a proposal the user approved. Use :func:`braino.tools.decide`, which checks
        and records the approval, rather than calling this directly."""
        if self.apply_func is None or self.apply_output_model is None:
            raise TypeError(f"tool {self.name!r} is not a decision tool")
        with context.provenance.tool_call(
            f"{self.name}.apply", self.risk.value, proposal, proposal_id=proposal_id
        ) as call:
            result = self.apply_func(proposal, context)
            if not isinstance(result, self.apply_output_model):
                result = self.apply_output_model.model_validate(result)
            call.result = result
        return result

    def _call(self, params: BaseModel, context: RunContext | None) -> BaseModel:
        result = self.func(params, context) if self.needs_context else self.func(params)
        if not isinstance(result, self.output_model):
            result = self.output_model.model_validate(result)
        return result

    def llm_schema(self) -> dict[str, Any]:
        """Function definition in the OpenAI-compatible format served by Ollama, llama.cpp
        and vLLM."""
        description = self.description
        if self.requires_approval:
            description = f"{description}\n\n{DECISION_NOTE}"
        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


def make_tool(
    func: Callable[..., BaseModel],
    *,
    name: str,
    risk: Risk,
    apply: Callable[..., BaseModel] | None = None,
) -> Tool:
    """Build a :class:`Tool` from a function annotated ``(params: InModel) -> OutModel`` or
    ``(params: InModel, ctx: RunContext) -> OutModel``.

    The function's docstring becomes the description shown to the language model. Decision
    tools must also pass ``apply``, annotated ``(proposal: OutModel, ctx: RunContext) ->
    ResultModel``; other tools must not.
    """
    if not _NAME_PATTERN.fullmatch(name):
        raise ToolDefinitionError(
            f"invalid tool name {name!r}: use dotted lower-case names like 'eeg.preprocess'"
        )

    params = list(inspect.signature(func).parameters.values())
    if len(params) not in (1, 2):
        raise ToolDefinitionError(f"tool {name!r} must take (params) or (params, ctx: RunContext)")

    hints = get_type_hints(func)
    needs_context = len(params) == 2
    if needs_context and hints.get(params[1].name) is not RunContext:
        raise ToolDefinitionError(
            f"tool {name!r}: second parameter must be annotated with RunContext"
        )

    input_model = hints.get(params[0].name)
    output_model = hints.get("return")
    for role, model in (("parameter", input_model), ("return", output_model)):
        if not _is_model(model):
            raise ToolDefinitionError(
                f"tool {name!r}: {role} must be annotated with a Pydantic model"
            )
    assert input_model is not None and output_model is not None

    description = inspect.getdoc(func)
    if not description:
        raise ToolDefinitionError(
            f"tool {name!r} needs a docstring; it is the LLM-facing description"
        )

    apply_output_model = _check_apply(apply, name=name, risk=risk, output_model=output_model)

    return Tool(
        name=name,
        description=description,
        risk=risk,
        input_model=input_model,
        output_model=output_model,
        func=func,
        needs_context=needs_context,
        apply_func=apply,
        apply_output_model=apply_output_model,
    )


def _check_apply(
    apply: Callable[..., BaseModel] | None,
    *,
    name: str,
    risk: Risk,
    output_model: type[BaseModel],
) -> type[BaseModel] | None:
    if risk is not Risk.DECISION:
        if apply is not None:
            raise ToolDefinitionError(f"tool {name!r}: only decision tools take an apply function")
        return None
    if apply is None:
        raise ToolDefinitionError(f"decision tool {name!r} needs an apply function")

    params = list(inspect.signature(apply).parameters.values())
    hints = get_type_hints(apply)
    result_model: type[BaseModel] | None = hints.get("return")
    if (
        len(params) != 2
        or hints.get(params[0].name) is not output_model
        or hints.get(params[1].name) is not RunContext
        or not _is_model(result_model)
    ):
        raise ToolDefinitionError(
            f"decision tool {name!r}: apply must be annotated "
            f"(proposal: {output_model.__name__}, ctx: RunContext) -> <Pydantic model>"
        )
    assert result_model is not None
    return result_model


def _is_model(obj: object) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)
