"""MkDocs hook: fills in the reference pages from the code, so they cannot go out of date.

``<!-- braino:tools -->`` becomes one section per registered tool (description, risk and a
table of its parameters, from the same JSON schema the AI agent sees), and
``<!-- braino:cli -->`` the help of every command.
"""

from __future__ import annotations

from typing import Any

TOOLS_MARK = "<!-- braino:tools -->"
CLI_MARK = "<!-- braino:cli -->"


def on_page_markdown(markdown: str, **_: Any) -> str:
    if TOOLS_MARK in markdown:
        markdown = markdown.replace(TOOLS_MARK, tools_reference())
    if CLI_MARK in markdown:
        markdown = markdown.replace(CLI_MARK, cli_reference())
    return markdown


def tools_reference() -> str:
    from braino.tools import registry

    out = []
    for tool in registry:
        schema = tool.input_model.model_json_schema()
        out.append(f"## `{tool.name}`\n")
        out.append(f"**Risk:** {tool.risk.value}")
        if tool.requires_approval:
            out.append(" (its proposal takes effect only once a person approves it)")
        out.append("\n\n" + tool.description + "\n")
        properties = schema.get("properties", {})
        if properties:
            required = set(schema.get("required", []))
            out.append("| Parameter | Type | Default | Description |")
            out.append("|---|---|---|---|")
            for name, prop in properties.items():
                default = "required" if name in required else _code(prop.get("default"))
                description = str(prop.get("description", "")).replace("|", "\\|")
                out.append(f"| `{name}` | {_type(prop, schema)} | {default} | {description} |")
        out.append("")
    return "\n".join(out)


def _type(prop: dict[str, Any], schema: dict[str, Any]) -> str:
    if "enum" in prop:
        return " \\| ".join(f"`{v}`" for v in prop["enum"])
    if "const" in prop:
        return f"`{prop['const']}`"
    if "anyOf" in prop:
        return " or ".join(_type(p, schema) for p in prop["anyOf"])
    if "$ref" in prop:
        return prop["$ref"].rsplit("/", 1)[-1]
    kind = prop.get("type", "any")
    if kind == "array":
        items = prop.get("items") or prop.get("prefixItems")
        if isinstance(items, list):
            return "[" + ", ".join(_type(i, schema) for i in items) + "]"
        return f"list of {_type(items or {}, schema)}"
    if kind == "object":
        values = prop.get("additionalProperties")
        return f"map to {_type(values, schema)}" if isinstance(values, dict) else "object"
    return str(kind)


def _code(value: Any) -> str:
    import json

    return f"`{json.dumps(value, ensure_ascii=False)}`"


def cli_reference() -> str:
    import typer
    from typer.testing import CliRunner

    from braino.cli import app

    root = typer.main.get_command(app)
    runner = CliRunner()
    env = {"COLUMNS": "92", "NO_COLOR": "1", "TERM": "dumb"}
    out: list[str] = []

    def walk(command: Any, path: list[str]) -> None:
        subcommands = getattr(command, "commands", None)
        if subcommands:
            for name in sorted(subcommands):
                walk(subcommands[name], [*path, name])
            return
        result = runner.invoke(app, [*path, "--help"], env=env, prog_name="braino")
        out.append(f"## `braino {' '.join(path)}`\n")
        out.append("```text")
        out.append("\n".join(line.rstrip() for line in result.output.splitlines()).strip("\n"))
        out.append("```\n")

    walk(root, [])
    return "\n".join(out)
