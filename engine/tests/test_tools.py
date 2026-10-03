from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from braino import __version__
from braino.tools import (
    MissingContextError,
    Risk,
    RunContext,
    ToolDefinitionError,
    ToolNotFoundError,
    ToolRegistry,
    make_tool,
    registry,
)


class AddInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class AddOutput(BaseModel):
    total: int


def add(params: AddInput) -> AddOutput:
    """Add two numbers."""
    return AddOutput(total=params.a + params.b)


@pytest.fixture
def local_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(name="math.add", risk=Risk.COMPUTE)(add)
    return reg


def test_run_validates_arguments_and_returns_output(local_registry: ToolRegistry) -> None:
    result = local_registry.get("math.add").run({"a": 2, "b": 3})
    assert result == AddOutput(total=5)


def test_run_rejects_invalid_arguments(local_registry: ToolRegistry) -> None:
    with pytest.raises(ValidationError):
        local_registry.get("math.add").run({"a": "not a number", "b": 3})


def test_lookup_by_llm_name(local_registry: ToolRegistry) -> None:
    tool = local_registry.get("math__add")
    assert tool.name == "math.add"
    assert tool.llm_name == "math__add"


def test_unknown_tool_raises(local_registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError):
        local_registry.get("math.subtract")


def test_duplicate_registration_rejected(local_registry: ToolRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        local_registry.tool(name="math.add", risk=Risk.COMPUTE)(add)


def test_llm_schema_shape(local_registry: ToolRegistry) -> None:
    schema = local_registry.get("math.add").llm_schema()
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "math__add"
    assert function["description"] == "Add two numbers."
    assert function["parameters"]["properties"]["a"]["description"] == "First number"
    assert set(function["parameters"]["required"]) == {"a", "b"}


def test_only_decision_tools_require_approval() -> None:
    assert make_tool(add, name="math.add", risk=Risk.DECISION).requires_approval
    assert not make_tool(add, name="math.add", risk=Risk.COMPUTE).requires_approval


@pytest.mark.parametrize("bad_name", ["Math.add", "math add", "math-add", ".add", "math."])
def test_invalid_names_rejected(bad_name: str) -> None:
    with pytest.raises(ToolDefinitionError, match="invalid tool name"):
        make_tool(add, name=bad_name, risk=Risk.READ)


def test_missing_docstring_rejected() -> None:
    def undocumented(params: AddInput) -> AddOutput:
        return AddOutput(total=0)

    with pytest.raises(ToolDefinitionError, match="docstring"):
        make_tool(undocumented, name="math.add", risk=Risk.READ)


def test_non_model_annotations_rejected() -> None:
    def plain(params: int) -> AddOutput:
        """Not a model input."""
        return AddOutput(total=params)

    with pytest.raises(ToolDefinitionError, match="Pydantic model"):
        make_tool(plain, name="math.plain", risk=Risk.READ)


def test_builtin_system_info_registered() -> None:
    result = registry.get("system.info").run({})
    assert result.model_dump()["braino_version"] == __version__


class NoteInput(BaseModel):
    text: str


class NoteOutput(BaseModel):
    path: str


def write_note(params: NoteInput, ctx: RunContext) -> NoteOutput:
    """Write a note into the derivatives folder."""
    path = ctx.output_dir("notes") / f"{ctx.run_id}.txt"
    path.write_text(params.text, encoding="utf-8")
    return NoteOutput(path=str(path))


def test_context_tool_receives_context(tmp_path: Path) -> None:
    tool = make_tool(write_note, name="notes.write", risk=Risk.COMPUTE)
    assert tool.needs_context
    ctx = RunContext(dataset_root=tmp_path, run_id="run-1")
    result = tool.run({"text": "hello"}, ctx)
    written = ctx.derivatives_root / "notes" / "run-1.txt"
    assert result == NoteOutput(path=str(written))
    assert written.read_text(encoding="utf-8") == "hello"


def test_context_tool_without_context_fails() -> None:
    tool = make_tool(write_note, name="notes.write", risk=Risk.COMPUTE)
    with pytest.raises(MissingContextError):
        tool.run({"text": "hello"})


def test_context_is_not_in_llm_schema() -> None:
    schema = make_tool(write_note, name="notes.write", risk=Risk.COMPUTE).llm_schema()
    assert set(schema["function"]["parameters"]["properties"]) == {"text"}


def test_context_free_tool_ignores_context() -> None:
    tool = make_tool(add, name="math.add", risk=Risk.COMPUTE)
    assert not tool.needs_context
    assert tool.run({"a": 1, "b": 1}, RunContext()) == AddOutput(total=2)


def test_second_parameter_must_be_run_context() -> None:
    def wrong(params: AddInput, extra: int) -> AddOutput:
        """Second parameter is not a context."""
        return AddOutput(total=extra)

    with pytest.raises(ToolDefinitionError, match="RunContext"):
        make_tool(wrong, name="math.wrong", risk=Risk.READ)


def test_too_many_parameters_rejected() -> None:
    def three(params: AddInput, ctx: RunContext, extra: int) -> AddOutput:
        """Too many parameters."""
        return AddOutput(total=extra)

    with pytest.raises(ToolDefinitionError, match="must take"):
        make_tool(three, name="math.three", risk=Risk.READ)
