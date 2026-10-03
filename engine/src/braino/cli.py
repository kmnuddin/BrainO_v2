"""Command-line interface: ``braino --help``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from braino import __version__
from braino.context import NoDatasetError, RunContext
from braino.provenance import ProposalNotFoundError, RunNotFoundError, list_runs, load_run
from braino.tools import (
    PendingProposal,
    ProposalAlreadyDecidedError,
    ToolNotFoundError,
    current_user,
    decide,
    registry,
)

app = typer.Typer(help="BrainO: EEG and fMRI analysis engine.", no_args_is_help=True)
tools_app = typer.Typer(help="List, inspect and run analysis tools.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")
runs_app = typer.Typer(help="Inspect the provenance of past runs.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")
proposals_app = typer.Typer(
    help="Review, approve or reject what decision tools propose.", no_args_is_help=True
)
app.add_typer(proposals_app, name="proposals")

_DATASET_HELP = "Root of the BIDS dataset to work on"


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
    dataset: Annotated[Path | None, typer.Option("--dataset", help=_DATASET_HELP)] = None,
) -> None:
    """Run a tool and print its result as JSON."""
    try:
        t = registry.get(name)
    except ToolNotFoundError:
        typer.echo(f"Unknown tool: {name}", err=True)
        raise typer.Exit(code=1) from None

    parsed = _parse_json_object(arguments, "--args")
    if t.requires_approval and dataset is None:
        typer.echo(f"{name} is a decision tool; pass --dataset so its proposal is saved", err=True)
        raise typer.Exit(code=1)

    try:
        context = RunContext(dataset_root=dataset)
    except NotADirectoryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    if dataset is not None:
        typer.echo(f"run {context.run_id}", err=True)

    try:
        result = t.run(parsed, context)
    except (ValidationError, NoDatasetError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.model_dump_json(indent=2))
    if isinstance(result, PendingProposal):
        typer.echo(
            f"proposal {result.proposal_id} awaits your decision: "
            f"braino proposals approve|reject {result.proposal_id} --dataset {dataset}",
            err=True,
        )


def _parse_json_object(text: str, option: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        typer.echo(f"{option} is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=1) from None
    if not isinstance(parsed, dict):
        typer.echo(f"{option} must be a JSON object", err=True)
        raise typer.Exit(code=1)
    return parsed


def _context(dataset: Path) -> RunContext:
    try:
        return RunContext(dataset_root=dataset)
    except NotADirectoryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


def _derivatives_root(dataset: Path) -> Path:
    return _context(dataset).derivatives_root


@runs_app.command("list")
def runs_list(
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
) -> None:
    """List recorded runs, oldest first, with the tools each one called."""
    root = _derivatives_root(dataset)
    for run_id in list_runs(root):
        calls = load_run(root, run_id).calls
        failed = sum(call.status == "error" for call in calls)
        tools = ", ".join(dict.fromkeys(call.tool for call in calls))
        status = f"{len(calls)} calls" + (f", {failed} failed" if failed else "")
        typer.echo(f"{run_id}  {status:<20} {tools}")


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run ID, as printed by 'braino runs list'")],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
) -> None:
    """Print the full provenance record of a run as JSON."""
    try:
        log = load_run(_derivatives_root(dataset), run_id)
    except RunNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(log.model_dump_json(indent=2))


@proposals_app.command("list")
def proposals_list(
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
    pending: Annotated[bool, typer.Option("--pending", help="Only undecided proposals")] = False,
) -> None:
    """List proposals made by decision tools, oldest first."""
    for record in _context(dataset).proposals.list():
        if pending and record.status != "pending":
            continue
        typer.echo(f"{record.proposal_id}  {record.status:<9} {record.tool}")


@proposals_app.command("show")
def proposals_show(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID")],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
) -> None:
    """Print a proposal and its decision, if any, as JSON."""
    try:
        record = _context(dataset).proposals.get(proposal_id)
    except ProposalNotFoundError:
        typer.echo(f"Unknown proposal: {proposal_id}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(record.model_dump_json(indent=2))


@proposals_app.command("approve")
def proposals_approve(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID")],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
    edit: Annotated[
        str | None,
        typer.Option("--edit", help="JSON object of proposal fields to change before applying"),
    ] = None,
    note: Annotated[str | None, typer.Option("--note", help="Reason for the decision")] = None,
) -> None:
    """Approve a proposal (optionally edited) and apply it."""
    edits = _parse_json_object(edit, "--edit") if edit is not None else None
    _decide(proposal_id, dataset, approve=True, edits=edits, note=note)


@proposals_app.command("reject")
def proposals_reject(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID")],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
    note: Annotated[str | None, typer.Option("--note", help="Reason for the decision")] = None,
) -> None:
    """Reject a proposal; nothing is applied."""
    _decide(proposal_id, dataset, approve=False, edits=None, note=note)


def _decide(
    proposal_id: str,
    dataset: Path,
    *,
    approve: bool,
    edits: dict[str, Any] | None,
    note: str | None,
) -> None:
    context = _context(dataset)
    try:
        record = decide(
            proposal_id,
            context,
            approve=approve,
            decided_by=current_user(),
            edits=edits,
            note=note,
        )
    except ProposalNotFoundError:
        typer.echo(f"Unknown proposal: {proposal_id}", err=True)
        raise typer.Exit(code=1) from None
    except (ProposalAlreadyDecidedError, ToolNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(record.model_dump_json(indent=2))
