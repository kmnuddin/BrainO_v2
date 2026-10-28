"""How tools are shown to the language model: compact schemas and a one-line catalogue.

Reading the prompt is the slow part on a CPU (about 65 tokens/s for the 4B model on a desktop
processor), so every token in a schema costs time. :func:`compact_schema` removes what the
model does not need from Pydantic's JSON schemas (titles, and the ``anyOf`` that Pydantic
writes for every optional field) without changing which values they allow: about 16% fewer
tokens for BrainO's tools.
"""

from __future__ import annotations

import json
from typing import Any

from braino.tools.base import DECISION_NOTE, Tool

_NULL = {"type": "null"}


def compact_schema(schema: Any) -> Any:
    """``schema`` without titles, with ``anyOf: [{type: X}, {type: null}]`` written as
    ``type: [X, null]``. Accepts exactly the same values."""
    if isinstance(schema, list):
        return [compact_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        # A property called "title" is a name, not an annotation: keep it.
        out[key] = (
            {k: compact_schema(v) for k, v in value.items()}
            if key in ("properties", "$defs", "patternProperties") and isinstance(value, dict)
            else compact_schema(value)
        )
    options = out.get("anyOf")
    if isinstance(options, list) and len(options) == 2 and _NULL in options:
        other = options[0] if options[1] == _NULL else options[1]
        if set(other) == {"type"} and isinstance(other["type"], str) and "type" not in out:
            del out["anyOf"]
            out["type"] = [other["type"], "null"]
    return out


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with its ``$ref``s to ``#/$defs/...`` replaced by the definitions, so it can
    be nested inside another schema (a ``$ref`` always points into the outermost one)."""
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in seen:
                raise ValueError(f"recursive schema definition {name!r} cannot be inlined")
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolve(defs[name], (*seen, name)), **resolve(siblings, seen)}
        return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}

    result: dict[str, Any] = resolve(schema, ())
    return result


def parameters(tool: Tool) -> dict[str, Any]:
    """The tool's input schema, compact and self-contained (no ``$ref``)."""
    result: dict[str, Any] = compact_schema(inline_refs(tool.input_model.model_json_schema()))
    return result


def tool_card(tool: Tool) -> dict[str, Any]:
    """What the model reads about a tool it may call: name, description and parameters."""
    description = tool.description
    if tool.requires_approval:
        description = f"{description}\n\n{DECISION_NOTE}"
    return {"name": tool.name, "description": description, "parameters": parameters(tool)}


def render_cards(tools: list[Tool]) -> str:
    """Tool cards in the layout Qwen's chat template uses for tools: one JSON per line."""
    lines = "\n".join(json.dumps(tool_card(t), ensure_ascii=False) for t in tools)
    return f"<tools>\n{lines}\n</tools>"


def summary(tool: Tool) -> str:
    """The first sentence of the tool's description."""
    text = " ".join(tool.description.split())
    end = text.find(". ")
    return text if end < 0 else text[: end + 1]


def catalogue(tools: list[Tool]) -> str:
    """One line per tool, so the model knows every tool exists without reading its schema."""
    lines = []
    for tool in tools:
        tag = " [decision: needs the user's approval]" if tool.requires_approval else ""
        lines.append(f"- {tool.name}: {summary(tool)}{tag}")
    return "\n".join(lines)
