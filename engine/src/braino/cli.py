"""Command-line interface: ``braino --help``."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from pydantic import ValidationError

from braino import __version__
from braino.tools import ToolNotFoundError, registry

app = typer.Typer(help="BrainO: EEG and fMRI analysis engine.", no_args_is_help=True)
tools_app = typer.Typer(help="List, inspect and run analysis tools.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")


@app.command()
def version() -> None:
    """Print the BrainO version."""
    typer.echo(__version__)


@tools_app.command("list")
def list_tools() -> None:
    """List the registered tools."""
    for t in registry:
        summary = t.description.splitlines()[0]
        typer.echo(f"{t.name:<32} {t.risk.value:<9} {summary}")


@tools_app.command("schema")
def tool_schema(
    name: Annotated[str | None, typer.Argument(help="Tool name; omit for all tools")] = None,
) -> None:
    """Print the LLM-facing JSON schema of one tool, or of all tools."""
    try:
        schema = registry.get(name).llm_schema() if name else registry.schemas()
    except ToolNotFoundError:
        typer.echo(f"Unknown tool: {name}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(schema, indent=2))


@tools_app.command("run")
def run_tool(
    name: Annotated[str, typer.Argument(help="Tool name, e.g. system.info")],
    arguments: Annotated[str, typer.Option("--args", help="Tool arguments as JSON")] = "{}",
) -> None:
    """Run a tool and print its result as JSON."""
    try:
        t = registry.get(name)
    except ToolNotFoundError:
        typer.echo(f"Unknown tool: {name}", err=True)
        raise typer.Exit(code=1) from None

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        typer.echo(f"--args is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if not isinstance(parsed, dict):
        typer.echo("--args must be a JSON object", err=True)
        raise typer.Exit(code=1)

    try:
        result = t.run(parsed)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.model_dump_json(indent=2))
