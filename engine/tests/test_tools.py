from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from braino import __version__
from braino.tools import (
    Risk,
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
        make_tool(plain, name="math.plain", risk=Risk.READ)  # type: ignore[type-var]


def test_builtin_system_info_registered() -> None:
    result = registry.get("system.info").run({})
    assert result.model_dump()["braino_version"] == __version__
