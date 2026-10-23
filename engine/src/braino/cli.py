"""Command-line interface: ``braino --help``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

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


@app.command("import")
def import_command(
    source: Annotated[
        Path | None, typer.Argument(help="Recording to import (or use --mapping)")
    ] = None,
    dataset: Annotated[Path, typer.Option("--dataset", help="BIDS dataset to import into")] = Path(
        "."
    ),
    key: Annotated[
        Path | None,
        typer.Option("--key", help="Subject key file (private; must be outside the dataset)"),
    ] = None,
    subject_id: Annotated[
        str | None, typer.Option("--subject-id", help="The person's original ID (kept in the key)")
    ] = None,
    task: Annotated[str | None, typer.Option("--task", help="Task label, e.g. rest")] = None,
    session: Annotated[str | None, typer.Option("--session", help="Session label")] = None,
    run: Annotated[int | None, typer.Option("--run", help="Run number")] = None,
    line_freq: Annotated[
        float | None, typer.Option("--line-freq", help="Power-line frequency in Hz, if known")
    ] = None,
    mapping: Annotated[
        Path | None,
        typer.Option("--mapping", help="CSV with source,subject_id,task[,session,run,line_freq]"),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace recordings")] = False,
) -> None:
    """Import recordings into a BIDS dataset, removing personal details on the way.

    Give one recording with --subject-id and --task, or many with --mapping.
    """
    from braino.io.batch import ImportFailedError, ImportJob, MappingError, import_job, read_mapping
    from braino.io.deidentify import SubjectKey, SubjectKeyError

    if key is None:
        _fail("--key is required: the private file that links subject labels to real IDs")
    dataset.mkdir(parents=True, exist_ok=True)
    try:
        subject_key = SubjectKey.open(key, dataset_root=dataset)
        if mapping is not None:
            if source is not None:
                _fail("give either a recording or --mapping, not both")
            jobs = read_mapping(mapping, dataset_root=dataset)
        else:
            if source is None or subject_id is None or task is None:
                _fail("give a recording with --subject-id and --task, or use --mapping")
            jobs = [
                ImportJob(
                    source=source,
                    subject_id=subject_id,
                    task=task,
                    session=session,
                    run=run,
                    line_freq=line_freq,
                )
            ]
    except (SubjectKeyError, MappingError, ValidationError) as exc:
        _fail(str(exc))

    context = _context(dataset)
    typer.echo(f"run {context.run_id}", err=True)
    failed = 0
    for job in jobs:
        try:
            report = import_job(context, subject_key, job, overwrite=overwrite)
        except ImportFailedError as exc:
            failed += 1
            typer.echo(f"FAILED  {job.source}: {exc}", err=True)
            continue
        typer.echo(_import_line(report))
        for text in report.deidentification.free_text_events:
            typer.echo(f"    check this event text by hand: {text!r}")
    typer.echo(f"{len(jobs) - failed} of {len(jobs)} recordings imported into {dataset}")
    if failed:
        raise typer.Exit(code=1)


def _import_line(report: Any) -> str:
    c = report.conversion
    n = report.normalisation
    line = f"line {c.line_freq:g} Hz ({c.line_freq_source})" if c.line_freq else "line n/a"
    events = sum(n.events.values())
    data_file = next((f for f in c.files if f.endswith((".vhdr", ".fif"))), c.files[0])
    return (
        f"OK      {data_file}  [{report.format}] {line}; positions: {n.positions}; {events} events"
    )


@app.command("inspect")
def inspect_command(
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
    as_json: Annotated[bool, typer.Option("--json", help="Print the full summary as JSON")] = False,
) -> None:
    """Summarise a BIDS dataset: subjects, tasks, and each recording's channels and events."""
    summary: Any = registry.get("dataset.inspect").run({"max_recordings": 500}, _context(dataset))
    if as_json:
        typer.echo(summary.model_dump_json(indent=2))
        return
    typer.echo(f"{summary.name or 'Unnamed dataset'} (BIDS {summary.bids_version})")
    typer.echo(
        f"{summary.n_subjects} subjects, {summary.n_recordings} recordings, "
        f"{summary.total_duration_s / 60:.1f} min; tasks: {', '.join(summary.tasks) or '-'}; "
        f"sessions: {', '.join(summary.sessions) or '-'}"
    )
    for r in summary.recordings:
        channels = ", ".join(f"{n} {t}" for t, n in sorted(r.channels.items()))
        sfreq = f"{r.sampling_rate_hz:g} Hz" if r.sampling_rate_hz else "? Hz"
        duration = f"{r.duration_s:.1f} s" if r.duration_s is not None else "? s"
        bads = f"; {len(r.bad_channels)} bad" if r.bad_channels else ""
        typer.echo(
            f"  {r.path}  {sfreq}, {duration}; {channels}{bads}; {sum(r.events.values())} events"
        )


@app.command("qc")
def qc_command(
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
    subject: Annotated[
        str | None, typer.Option("--subject", help="Subject label; omit to check every recording")
    ] = None,
    task: Annotated[str | None, typer.Option("--task", help="Task label")] = None,
    session: Annotated[str | None, typer.Option("--session", help="Session label")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Run number")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summaries as JSON")] = False,
) -> None:
    """Check data quality: flat and noisy channels, line noise, and time with large artifacts.

    Changes nothing; per-channel tables are written to derivatives/braino/.
    """
    from braino.io.dataset import scan_dataset
    from braino.tools.eeg import RecordingNotFoundError

    context = _context(dataset)
    if subject is not None:
        selectors = [{"subject": subject, "task": task, "session": session, "run": run}]
    else:
        selectors = [
            {"subject": r.subject, "task": r.task, "session": r.session, "run": r.run}
            for r in scan_dataset(context.dataset).recordings
            if r.datatype in ("eeg", "ieeg", "meg")
        ]
    qc = registry.get("eeg.qc_report")
    results = []
    for selector in selectors:
        try:
            result: Any = qc.run(selector, context)
        except RecordingNotFoundError as exc:
            _fail(str(exc))
        results.append(result)
        if not as_json:
            typer.echo(_qc_line(result))
    if as_json:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


def _qc_line(r: Any) -> str:
    problems = []
    if r.n_flat:
        problems.append(f"{r.n_flat} flat ({', '.join(r.flat[:5])})")
    if r.n_noisy:
        problems.append(f"{r.n_noisy} noisy ({', '.join(n['channel'] for n in r.noisy[:5])})")
    if r.n_strong_line_noise:
        problems.append(f"strong line noise on {r.n_strong_line_noise}")
    problems.append(f"{r.artifact_fraction:.0%} of time with large artifacts")
    return f"{r.recording}: {r.n_channels_checked} channels; " + "; ".join(problems)


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


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
