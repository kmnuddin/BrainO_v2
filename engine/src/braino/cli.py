"""Command-line interface: ``braino --help``."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
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


_OUTPUT_HELP = "Write the script here instead of printing it"


@runs_app.command("export")
def runs_export(
    run_id: Annotated[str, typer.Argument(help="Run ID, as printed by 'braino runs list'")],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
    output: Annotated[Path | None, typer.Option("--output", "-o", help=_OUTPUT_HELP)] = None,
) -> None:
    """Export a run as a standalone Python script that re-runs it and checks its results."""
    from braino.provenance.export import (
        ExportError,
        original_inputs,
        render_script,
        steps_from_run,
    )

    context = _context(dataset)
    try:
        steps, calls, environment = steps_from_run(
            context.derivatives_root, run_id, context.proposals
        )
    except (RunNotFoundError, ExportError) as exc:
        _fail(str(exc))
    if not steps:
        _fail(f"run {run_id} has no successful analysis steps to export")
    script = render_script(
        steps,
        title=f"BrainO run {run_id}",
        environment=environment,
        inputs=original_inputs(calls),
    )
    _write_script(script, output)


def _write_script(script: str, output: Path | None) -> None:
    if output is None:
        typer.echo(script, nl=False)
        return
    output.write_text(script, encoding="utf-8")
    typer.echo(f"wrote {output}", err=True)


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


# --- pipelines ----------------------------------------------------------------------------

pipeline_app = typer.Typer(
    help="Run analyses declared as pipeline specs (YAML/JSON), with cached results.",
    no_args_is_help=True,
)
app.add_typer(pipeline_app, name="pipeline")

_SPEC_HELP = "Spec file (.yaml/.json) or the name of a built-in spec"
WAITING_EXIT_CODE = 3
"""Exit code of ``braino pipeline run`` when it stopped at decisions waiting for a person."""


def _load_spec(source: str) -> Any:
    from braino.pipelines import SpecError, load_spec

    try:
        return load_spec(source)
    except (SpecError, OSError) as exc:
        _fail(str(exc))


@pipeline_app.command("list")
def pipeline_list() -> None:
    """List the pipeline specs that come with BrainO."""
    from braino.pipelines import builtin_specs, load_spec

    for name, path in sorted(builtin_specs().items()):
        typer.echo(f"{name:<24} {load_spec(path).description}")


@pipeline_app.command("show")
def pipeline_show(spec: Annotated[str, typer.Argument(help=_SPEC_HELP)]) -> None:
    """Print a spec (e.g. a built-in one, to copy and adapt)."""
    from braino.pipelines import builtin_specs

    path = builtin_specs().get(spec) or Path(spec)
    if not path.is_file():
        _fail(f"no spec {spec}")
    _load_spec(str(path))
    typer.echo(path.read_text(encoding="utf-8"), nl=False)


@pipeline_app.command("validate")
def pipeline_validate(
    spec: Annotated[str, typer.Argument(help=_SPEC_HELP)],
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Also check the plan against this dataset")
    ] = None,
) -> None:
    """Check a spec: its tools exist and accept the arguments given, and the plan holds
    together (step order, conditions, windows, multiple-comparison correction; with
    --dataset, also subjects, events, channels and filters)."""
    loaded = _load_spec(spec)
    check = _check_plan(loaded, dataset)
    summary = (
        f"{loaded.name}: {len(loaded.steps)} steps per recording, {len(loaded.group)} group steps"
    )
    if check.errors:
        _fail(f"{summary}; {len(check.errors)} problem(s)")
    typer.echo(f"{summary}; OK" + (f", {len(check.warnings)} warning(s)" if check.warnings else ""))


def _check_plan(spec: Any, dataset: Path | None) -> Any:
    """Run the plan checks and print what they find."""
    from braino.pipelines.checks import check_plan

    check = check_plan(spec, dataset=dataset.resolve() if dataset is not None else None)
    for finding in check.findings:
        typer.echo(str(finding), err=True)
    return check


@pipeline_app.command("templates")
def pipeline_templates() -> None:
    """List the analysis templates: tested pipelines filled in with a few settings."""
    from braino.pipelines.templates import TEMPLATES

    for template in TEMPLATES.values():
        typer.echo(f"{template.name:<22} {template.description}")
        for name, field in template.settings.model_fields.items():
            required = " (required)" if field.is_required() else ""
            typer.echo(f"    {name}{required}: {field.description or ''}")


@pipeline_app.command("template")
def pipeline_template(
    name: Annotated[str, typer.Argument(help="Template name (see 'braino pipeline templates')")],
    settings: Annotated[
        str,
        typer.Option(
            "--settings",
            help="The template's settings as JSON, e.g. "
            '\'{"condition_a": "rare", "events_a": ["stimulus/rare"], ...}\'',
        ),
    ],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the spec here")
    ] = None,
) -> None:
    """Make a pipeline spec from a template, checked against the dataset, to read, change and
    run with 'braino pipeline run'."""
    import yaml
    from pydantic import ValidationError

    from braino.pipelines.templates import TemplateError, build

    try:
        values = json.loads(settings)
    except json.JSONDecodeError as exc:
        _fail(f"--settings is not valid JSON: {exc.msg}")
    try:
        spec = build(name, values, dataset.resolve())
    except (TemplateError, ValidationError) as exc:
        _fail(str(exc))
    check = _check_plan(spec, dataset)
    text = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    text = f"# {spec.description}\n# Made from the {name} template.\n{text}"
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {output}", err=True)
    if check.errors:
        raise typer.Exit(code=1)


@pipeline_app.command("run")
def pipeline_run(
    spec: Annotated[str, typer.Argument(help=_SPEC_HELP)],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
    subjects: Annotated[
        str | None, typer.Option("--subjects", help="Comma-separated subset of the subjects")
    ] = None,
    from_step: Annotated[
        str | None, typer.Option("--from", help="Run this step and the ones after it again")
    ] = None,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--auto-approve",
            help="Approve every proposal as made, in your name (e.g. for validation runs)",
        ),
    ] = False,
) -> None:
    """Run a pipeline over a dataset. Steps whose inputs and settings are unchanged are
    skipped. Decision steps wait for your approval unless --auto-approve is given; approve
    with 'braino proposals approve' and run again to continue."""
    from braino.pipelines import SpecError, run_pipeline

    loaded = _load_spec(spec)
    if _check_plan(loaded, dataset).errors:
        _fail("the plan check found problems; nothing was run")
    context = _context(dataset)
    typer.echo(f"run {context.run_id}", err=True)
    approver = f"{current_user()} (auto-approved in a pipeline run)" if auto_approve else None
    try:
        report = run_pipeline(
            loaded,
            context,
            subjects=subjects.split(",") if subjects else None,
            from_step=from_step,
            auto_approve=approver,
            progress=lambda o: typer.echo(_outcome_line(o)),
        )
    except SpecError as exc:
        _fail(str(exc))
    counts = ", ".join(
        f"{report.count(s)} {s}"
        for s in ("ran", "cached", "pending", "rejected", "failed", "blocked")
        if report.count(s)
    )
    typer.echo(f"{loaded.name}: {len(report.subjects)} subjects; steps {counts}")
    if report.failed:
        raise typer.Exit(code=1)
    if report.pending:
        typer.echo(
            f"{len(report.pending)} decisions wait for you: braino proposals list --pending "
            f"--dataset {dataset}",
            err=True,
        )
        raise typer.Exit(code=WAITING_EXIT_CODE)


@pipeline_app.command("status")
def pipeline_status(
    spec: Annotated[str, typer.Argument(help=_SPEC_HELP)],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
) -> None:
    """Show, for each subject, which steps have a current result."""
    from braino.pipelines import load_state

    loaded = _load_spec(spec)
    state = load_state(_context(dataset), loaded)
    if not state.records:
        typer.echo(f"{loaded.name} has not run on this dataset")
        return
    by_subject: dict[str, list[str]] = {}
    for record in state.records.values():
        mark = record.step
        if record.proposal_id and record.apply is None:
            mark += " (decision pending or rejected)"
        by_subject.setdefault(record.subject, []).append(mark)
    for subject, steps in sorted(by_subject.items()):
        typer.echo(f"{subject:<8} {', '.join(steps)}")


@pipeline_app.command("export")
def pipeline_export(
    spec: Annotated[str, typer.Argument(help=_SPEC_HELP)],
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)] = Path("."),
    output: Annotated[Path | None, typer.Option("--output", "-o", help=_OUTPUT_HELP)] = None,
) -> None:
    """Export a pipeline's current results as a standalone Python script that re-runs every
    step (including steps reused from earlier runs) and checks the results."""
    from braino.pipelines import load_state
    from braino.provenance.export import (
        ExportError,
        original_inputs,
        render_script,
        steps_from_pipeline,
    )

    loaded = _load_spec(spec)
    context = _context(dataset)
    state = load_state(context, loaded)
    try:
        steps, calls, environment = steps_from_pipeline(
            context.derivatives_root, state, loaded, context.proposals
        )
    except (RunNotFoundError, ExportError) as exc:
        _fail(str(exc))
    script = render_script(
        steps,
        title=f"BrainO pipeline {loaded.name}: {loaded.description}".rstrip(": "),
        environment=environment,
        inputs=original_inputs(calls),
    )
    _write_script(script, output)


def _outcome_line(outcome: Any) -> str:
    where = outcome.subject if outcome.subject == "group" else f"sub-{outcome.subject}"
    text = f"{where:<10} {outcome.step:<16} {outcome.status:<8}"
    if outcome.message:
        text += f" {outcome.message}"
    return text


validate_app = typer.Typer(
    help="Check BrainO's analyses against published results.", no_args_is_help=True
)
app.add_typer(validate_app, name="validate")


@validate_app.command("erpcore")
def validate_erpcore(
    component: Annotated[str, typer.Argument(help="P3 or N170")],
    subjects: Annotated[
        str | None,
        typer.Option("--subjects", help="Comma-separated participants, e.g. 001,002; default all"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON")] = False,
) -> None:
    """Run the built-in ERP CORE pipeline and compare with the published scores.

    Downloads the data on first use (P3 2.2 GB, N170 3.4 GB for all 40 participants) into
    $BRAINO_DATA (default ~/.cache/braino). Decisions are auto-approved in your name.
    """
    from braino.datasets.erpcore import DownloadError, references
    from braino.validation.erpcore import run as run_validation

    if component not in references():
        _fail(f"unknown component {component!r}; choose from {', '.join(references())}")
    try:
        result = run_validation(
            component,
            subjects.split(",") if subjects else None,
            progress=None if as_json else lambda o: typer.echo(_outcome_line(o), err=True),
        )
    except (DownloadError, OSError) as exc:
        _fail(f"download failed ({exc}); run the command again to resume")
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    for subject, reason in result.failed.items():
        typer.echo(f"sub-{subject} failed at {reason}")
    report = result.report
    if report is None:
        typer.echo("fewer than 3 participants scored; nothing to compare")
        raise typer.Exit(code=1)
    lo, hi = report.braino_ci95_uv
    typer.echo(
        f"{component}: BrainO {report.braino_mean_uv:+.2f} µV [95% CI {lo:+.2f}, {hi:+.2f}], "
        f"n = {report.n_braino}; published {report.published_mean_uv:+.2f} µV"
    )
    if report.correlation is not None:
        typer.echo(
            f"participant level: r = {report.correlation:.3f} over {report.n_compared}, mean "
            f"absolute difference {report.mean_abs_difference_uv:.2f} µV"
        )
    typer.echo(f"excluded (> 25% rejected): {', '.join(result.excluded) or 'none'}")
    for check, ok in report.checks.items():
        typer.echo(f"  {'PASS' if ok else 'FAIL'}  {check}")
    typer.echo("REPRODUCED" if report.passed else "NOT REPRODUCED")
    if not report.passed:
        raise typer.Exit(code=1)


# --- the agent ----------------------------------------------------------------------------

_SERVER_HELP = (
    "Use this OpenAI-compatible server (its .../v1 address) instead of starting the local model"
)


@app.command("agent")
def agent_command(
    dataset: Annotated[Path, typer.Option("--dataset", help=_DATASET_HELP)],
    request: Annotated[
        str | None,
        typer.Argument(help="What to do, in your words; omit it to type requests one by one"),
    ] = None,
    server: Annotated[str | None, typer.Option("--server", help=_SERVER_HELP)] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="BRAINO_LLM_API_KEY", help="API key for --server"),
    ] = None,
    model_name: Annotated[
        str, typer.Option("--model-name", help="The model's name on --server")
    ] = "local",
    llama_cpp: Annotated[
        bool,
        typer.Option(
            "--llama-cpp/--other-server",
            help="Whether --server is llama.cpp (prompt cache and reasoning switch)",
        ),
    ] = True,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Steps the agent may take per request")
    ] = 60,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show the model's replies and timings")
    ] = False,
) -> None:
    """Ask the AI agent to analyse a dataset, in your own words.

    The agent runs on the local model (see 'braino model setup'). It works through BrainO's
    tools step by step; when a tool proposes a decision (bad channels, ICA components) you
    approve, edit or reject it here, and the agent never decides for you. To compare
    conditions it plans a tested pipeline from a template, which you approve before it runs.
    Every step is in the run's provenance record.
    """
    from braino.agent import ConsoleSession, LLMAgent
    from braino.tools.analysis import step_progress

    _safe_console()
    step_progress.set(lambda o: typer.echo(f"    {_outcome_line(o)}"))
    context = _context(dataset)
    session = ConsoleSession(context)
    with _model_client(server, api_key, model_name, llama_cpp) as (client, ctx_size):
        agent = LLMAgent(
            client, max_steps=max_steps, ctx_size=ctx_size, on_event=_AgentPrinter(verbose)
        )
        typer.echo(f"run {context.run_id}", err=True)
        try:
            if request is not None:
                _show_answer(agent.solve(request, session))
            else:
                typer.echo("Type a request, or press Enter on an empty line to finish.")
                while True:
                    text = typer.prompt("\nYou", default="", show_default=False).strip()
                    if not text:
                        break
                    _show_answer(agent.follow_up(text, session))
        except (KeyboardInterrupt, typer.Abort):
            typer.echo("\nStopped.", err=True)
            raise typer.Exit(code=130) from None
    typer.echo(
        f"Steps recorded in run {context.run_id}: braino runs show {context.run_id} "
        f"--dataset {dataset}",
        err=True,
    )


def _safe_console() -> None:
    """Print characters the console cannot show (such as µ or a minus sign) as '?' instead of
    failing, e.g. on a Windows console with a legacy code page."""
    import io
    import sys

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(errors="replace")


@contextlib.contextmanager
def _model_client(
    server: str | None, api_key: str | None, model_name: str, llama_cpp: bool
) -> Iterator[tuple[Any, int]]:
    """A chat client for ``server``, or for the local model, started here and stopped after."""
    from braino.llm import ChatClient
    from braino.runtime import ServerError, SetupError, load_settings, start_server

    if server is not None:
        client = ChatClient(server, api_key=api_key, model=model_name, llama_cpp=llama_cpp)
        yield client, load_settings().ctx_size
        return
    settings = load_settings()
    typer.echo(f"Starting the local model ({settings.model or 'not set up'})...", err=True)
    try:
        running = start_server()
    except (SetupError, ServerError) as exc:
        _fail(str(exc))
    with running:
        typer.echo(f"Model loaded in {running.load_seconds} s.", err=True)
        yield ChatClient.for_server(running), running.ctx_size


class _AgentPrinter:
    """Shows the agent's steps as it works: tool calls, their outcome, and (verbose) the
    model's replies with timings."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def __call__(self, event: Any) -> None:
        if event.kind == "model":
            if self.verbose:
                d = event.data
                typer.echo(
                    f"  [model {d['seconds']:.1f} s, read {d['prompt_read']} of "
                    f"{d['prompt_tokens']} tokens, wrote {d['completion_tokens']}] "
                    f"{_clip(event.text, 300)}",
                    err=True,
                )
        elif event.kind == "call":
            arguments = ", ".join(
                f"{k}={json.dumps(v, ensure_ascii=False)}"
                for k, v in event.data["arguments"].items()
            )
            typer.echo(f"> {event.text}({_clip(arguments, 160)})")
        elif event.kind == "result":
            if event.data["ok"]:
                proposal = event.data.get("proposal_id")
                extra = f", proposal {proposal}" if proposal else ""
                typer.echo(f"  done in {event.seconds:.1f} s{extra}")
            else:
                typer.echo(f"  failed: {_clip(event.text, 300)}")
        elif event.kind == "skipped":
            typer.echo(f"> {event.data['tool']}: not run again ({_clip(event.text, 120)})")
        elif event.kind == "decision" and self.verbose:
            typer.echo(f"  {event.text}", err=True)


def _show_answer(answer: Any) -> None:
    typer.echo("")
    typer.echo(answer.text)
    if answer.unverified:
        numbers = ", ".join(f"{n:g}" for n in answer.unverified)
        typer.echo(f"\nCheck these numbers: they are not in any tool result ({numbers}).", err=True)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- agent evaluation ---------------------------------------------------------------------

eval_app = typer.Typer(
    help="Score an analysis agent on tasks with known answers.", no_args_is_help=True
)
app.add_typer(eval_app, name="eval")


@eval_app.command("list")
def eval_list(
    include_large: Annotated[
        bool, typer.Option("--include-large", help="Also tasks that need large downloads")
    ] = False,
) -> None:
    """List the evaluation tasks."""
    from braino.evals import select_tasks

    for t in select_tasks(include_large=include_large):
        large = " (large download)" if t.large else ""
        typer.echo(f"{t.id:<32} {t.category:<14} {t.description}{large}")


@eval_app.command("run")
def eval_run(
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="llm (the AI agent on the local model, or on --server), oracle, null, or "
            "module:factory for your own agent",
        ),
    ] = "oracle",
    server: Annotated[str | None, typer.Option("--server", help=_SERVER_HELP)] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="BRAINO_LLM_API_KEY", help="API key for --server"),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Steps the llm agent may take per task")
    ] = 60,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show the llm agent's steps")
    ] = False,
    tasks: Annotated[
        str | None, typer.Option("--tasks", help="Comma-separated task ids; default all")
    ] = None,
    include_large: Annotated[
        bool, typer.Option("--include-large", help="Also tasks that need large downloads")
    ] = False,
    tier: Annotated[
        str | None,
        typer.Option("--tier", help="Hardware tier this run stands for, e.g. minimum, gpu"),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model name, for the report")] = None,
    workdir: Annotated[
        Path | None, typer.Option("--workdir", help="Keep each task's data and transcript here")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the report (JSON) here")
    ] = None,
) -> None:
    """Run the evaluation tasks with an agent and score them.

    Each task gets a fresh copy of its dataset and a scripted user who answers questions and
    decides on proposals. A task passes when all its required checks pass, including that every
    number in the answer comes from a tool result.
    """
    from braino.evals import TaskError, agent_factory, run_suite, select_tasks

    _safe_console()
    try:
        selected = select_tasks(tasks.split(",") if tasks else None, include_large=include_large)
        factory = None if agent == "llm" else agent_factory(agent)
    except (TaskError, ValueError, ImportError, AttributeError) as exc:
        _fail(str(exc))

    def line(result: Any) -> None:
        mark = "PASS" if result.passed else "FAIL"
        typer.echo(f"{mark}  {result.id:<32} score {result.score:.2f}  {result.seconds:6.1f} s")
        for c in result.checks:
            if not c.passed:
                flag = "required" if c.required else "optional"
                typer.echo(f"        - {c.description} ({flag}): {c.detail}")
        if result.error:
            typer.echo(f"        ! {result.error}")

    with contextlib.ExitStack() as stack:
        if factory is None:
            from braino.agent import LLMAgent
            from braino.runtime import load_settings

            client, ctx_size = stack.enter_context(_model_client(server, api_key, "local", True))
            printer = _AgentPrinter(verbose=True) if verbose else None
            if model is None and server is None:
                model = load_settings().model
            if tier is None and server is None:
                tier = load_settings().tier

            def factory(task: Any) -> Any:
                return LLMAgent(client, max_steps=max_steps, ctx_size=ctx_size, on_event=printer)

        report = run_suite(
            selected, factory, workdir=workdir, tier=tier, model=model, progress=line
        )
    passed = sum(t.passed for t in report.tasks)
    typer.echo(
        f"{report.agent}: {passed} of {len(report.tasks)} tasks passed "
        f"({report.pass_rate:.0%}), mean score {report.mean_score:.2f}, {report.seconds:.0f} s"
    )
    for name, c in report.by_category.items():
        typer.echo(f"  {name:<14} {c.passed}/{c.tasks}")
    if output is not None:
        report.save(output)
        typer.echo(f"report written to {output}", err=True)


@eval_app.command("compare")
def eval_compare(
    before: Annotated[Path, typer.Argument(help="Earlier report (JSON)")],
    now: Annotated[Path, typer.Argument(help="Later report (JSON)")],
) -> None:
    """Compare two evaluation reports: pass rate, and tasks fixed or broken."""
    from braino.evals import EvalReport, compare

    try:
        old, new = EvalReport.load(before), EvalReport.load(now)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    result = compare(old, new)
    a, b = result.pass_rate
    typer.echo(f"pass rate {a:.0%} -> {b:.0%}; mean score {result.mean_score[0]:.2f} -> "
               f"{result.mean_score[1]:.2f}")  # fmt: skip
    for label, ids in (("fixed", result.fixed), ("broken", result.broken),
                       ("only in the earlier report", result.only_before),
                       ("only in the later report", result.only_now)):  # fmt: skip
        if ids:
            typer.echo(f"{label}: {', '.join(ids)}")
    if result.broken:
        raise typer.Exit(code=1)


# --- local model runtime ------------------------------------------------------------------

model_app = typer.Typer(
    help="Set up and run the local AI model (llama.cpp with a Qwen model).",
    no_args_is_help=True,
)
app.add_typer(model_app, name="model")


def _progress_bar(label: str) -> Any:
    """A download progress callback that prints one updating line to stderr."""
    import sys
    import time

    state = {"shown": -1.0, "start": time.monotonic()}

    def update(done: int, total: int) -> None:
        share = done / total if total else 1.0
        if share < 1.0 and share - state["shown"] < 0.005:
            return
        state["shown"] = share
        rate = done / max(time.monotonic() - state["start"], 1e-6) / 2**20
        sys.stderr.write(
            f"\r  {label}: {share:6.1%} of {total / 2**30:.2f} GB ({rate:.1f} MiB/s)   "
        )
        if share >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return update


@model_app.command("hardware")
def model_hardware(
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON")] = False,
) -> None:
    """Show this computer's CPU, RAM and GPUs, and the model tier BrainO proposes."""
    from braino.runtime import catalog, detect, recommend

    hw = detect()
    rec = recommend(hw)
    if as_json:
        data = {"hardware": hw.model_dump(), "recommendation": rec.model_dump()}
        typer.echo(json.dumps(data, indent=2))
        return
    typer.echo(f"Computer:  {hw.platform}, {hw.cpu or 'unknown CPU'}, {hw.cpu_count} threads")
    typer.echo(f"RAM:       {hw.memory_gb} GB")
    for g in hw.gpus:
        kind = "shared memory" if g.unified else "dedicated"
        typer.echo(f"GPU:       {g.name}, {g.memory_gb:g} GB {kind} (from {g.source})")
    if not hw.gpus:
        typer.echo("GPU:       none with its own memory")
    model = catalog().model(rec.model)
    typer.echo(f"\nProposed:  tier '{rec.tier}': {model.id} on {rec.placement} ({rec.backend})")
    for reason in rec.reasons:
        typer.echo(f"  {reason}")
    for warning in rec.warnings:
        typer.echo(f"  Warning: {warning}")
    others = [o for o in rec.options if o.model != rec.model]
    if others:
        typer.echo("Also fits: " + "; ".join(f"{o.model} ({o.placement})" for o in others))


@model_app.command("list")
def model_list() -> None:
    """List the models BrainO can run, and which are downloaded or fit here."""
    from braino.runtime import catalog, detect, installed_models, options

    fits = {o.model: o.placement for o in options(detect())}
    have = {m.id for m in installed_models()}
    for m in catalog().models:
        where = fits.get(m.id, "does not fit")
        mark = "downloaded" if m.id in have else ""
        typer.echo(f"{m.id:<18} {m.size_gb:5.1f} GB  {m.licence:<11} {where:<13} {mark}")


@model_app.command("setup")
def model_setup(
    model: Annotated[
        str | None, typer.Option("--model", help="Model id (default: the proposed one)")
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="cpu, vulkan, cuda or metal (default: auto)")
    ] = None,
    bench: Annotated[bool, typer.Option("--bench/--no-bench", help="Measure the speed")] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before downloading")] = False,
) -> None:
    """Install llama.cpp and download the model for this computer, once.

    Downloads are checked against pinned SHA-256 hashes and resume if interrupted. Afterwards a
    short benchmark tells you how fast the agent will be.
    """
    from braino.runtime import (
        DownloadError,
        ServerError,
        SetupError,
        catalog,
        choose,
        describe,
        detect,
        setup,
    )
    from braino.runtime.catalog import BACKENDS
    from braino.runtime.install import find_server, is_model_installed

    if backend is not None and backend not in BACKENDS:
        _fail(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    hw = detect()
    cat = catalog()
    try:
        option, chosen = choose(hw, model, backend)
        spec = cat.model(option.model)
        build = cat.build(hw.os, hw.arch, chosen)
    except (SetupError, KeyError) as exc:
        _fail(str(exc).strip("'\""))
    need = 0 if find_server(chosen) else build.size
    need += 0 if is_model_installed(spec) else spec.size
    typer.echo(f"Model:    {spec.name} ({spec.licence}), runs on {option.placement}")
    typer.echo(f"Runtime:  llama.cpp {cat.llama_cpp.build} ({chosen})")
    typer.echo(f"Download: {need / 2**30:.2f} GB")
    if need and not yes and not typer.confirm("Download now?", default=True):
        raise typer.Exit(code=1)
    try:
        settings = setup(option.model, chosen, benchmark=bench, progress=_progress_bar, hw=hw)
    except (SetupError, DownloadError, ServerError) as exc:
        _fail(str(exc))
    typer.echo(
        f"Ready: {settings.model} on {settings.placement} ({settings.backend}), "
        f"tier '{settings.tier}'."
    )
    if settings.benchmark is not None:
        typer.echo(describe(settings.benchmark))


@model_app.command("status")
def model_status() -> None:
    """Show the chosen model, what is installed, and the last benchmark."""
    from braino.runtime import describe, installed_backends, installed_models, load_settings
    from braino.runtime.install import find_server

    s = load_settings()
    if s.model is None:
        typer.echo("No model set up yet: run 'braino model setup'.")
    else:
        typer.echo(
            f"Model:     {s.model} on {s.placement} ({s.backend}), tier '{s.tier}', "
            f"context {s.ctx_size} tokens"
        )
    typer.echo("Runtimes:  " + (", ".join(installed_backends()) or "none installed"))
    server = find_server(s.backend)
    if server is not None:
        typer.echo(f"Server:    {server}")
    typer.echo("Models:    " + (", ".join(m.id for m in installed_models()) or "none downloaded"))
    if s.benchmark is not None:
        typer.echo(f"Speed:     {describe(s.benchmark)}")


@model_app.command("bench")
def model_bench() -> None:
    """Measure how fast the agent will be with the chosen model."""
    from braino.runtime import ServerError, SetupError, describe, run_saved_benchmark

    try:
        settings = run_saved_benchmark()
    except (SetupError, ServerError) as exc:
        _fail(str(exc))
    assert settings.benchmark is not None
    typer.echo(describe(settings.benchmark))


@model_app.command("serve")
def model_serve(
    port: Annotated[int | None, typer.Option("--port", help="Port (default: a free one)")] = None,
) -> None:
    """Run the model server in the foreground until Ctrl+C (for development).

    Prints the OpenAI-compatible base URL and this session's API key.
    """
    import time

    from braino.runtime import ServerError, SetupError, start_server

    try:
        server = start_server(port=port)
    except (SetupError, ServerError) as exc:
        _fail(str(exc))
    typer.echo(
        f"Serving {server.alias} at {server.base_url}/v1 (loaded in {server.load_seconds} s). "
        f"Press Ctrl+C to stop."
    )
    typer.echo(f"API key: {server.api_key}")
    try:
        while server.running:
            time.sleep(0.5)
        typer.echo("The server stopped:\n" + server.log_tail(), err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


@model_app.command("remove")
def model_remove(
    model: Annotated[str | None, typer.Argument(help="Model id to delete")] = None,
    runtime: Annotated[
        str | None, typer.Option("--runtime", help="Delete an installed llama.cpp backend")
    ] = None,
) -> None:
    """Delete a downloaded model or an installed llama.cpp build, to free disk space."""
    from braino.runtime import catalog
    from braino.runtime.install import remove_model, remove_runtime

    if model is None and runtime is None:
        _fail("name a model, or --runtime BACKEND")
    if model is not None:
        try:
            spec = catalog().model(model)
        except KeyError as exc:
            _fail(str(exc).strip("'\""))
        typer.echo(f"{model}: " + ("deleted" if remove_model(spec) else "not downloaded"))
    if runtime is not None:
        from braino.runtime.catalog import BACKENDS

        if runtime not in BACKENDS:
            _fail(f"unknown backend {runtime!r}; choose from {', '.join(BACKENDS)}")
        done = remove_runtime(runtime)
        typer.echo(f"llama.cpp {runtime}: " + ("deleted" if done else "not installed"))
