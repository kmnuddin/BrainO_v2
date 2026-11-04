"""Analysis tools: a whole analysis from a template, planned, approved, then run.

``analysis.erp_comparison`` and ``analysis.spectral_comparison`` turn a few settings into a
pipeline (:mod:`braino.pipelines.templates`) and check the plan
(:mod:`braino.pipelines.checks`). They are decision tools: the plan, with its steps and any
warnings, is shown to the user, and only an approved plan is saved, under
``derivatives/braino/pipelines/<name>/``. ``analysis.run`` runs a saved plan over every
subject, reusing steps already run, and reports each subject's values, their mean and the
group test. Decision steps in the plan (ICA, bad channels) wait for the user; running again
continues from there.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.pipelines.checks import check_plan
from braino.pipelines.runner import GROUP, PIPELINES_DIR, StepOutcome, run_pipeline
from braino.pipelines.spec import NAME_PATTERN, PipelineSpec, SpecError, Step, load_spec
from braino.pipelines.templates import (
    ErpComparisonSettings,
    SpectralComparisonSettings,
    TemplateError,
    get_template,
    resolve,
)
from braino.tools.base import Risk
from braino.tools.registry import tool

PLAN_FILE = "plan.json"
SPEC_FILE = "spec.yaml"
MAX_NOTES = 8

step_progress: ContextVar[Callable[[StepOutcome], None] | None] = ContextVar(
    "step_progress", default=None
)
"""Called with every step ``analysis.run`` finishes, for a progress display."""


class PlanProposal(BaseModel):
    template: str
    name: str = Field(description="Name of the plan; analysis.run runs it once approved")
    task: str
    subjects: list[str]
    steps: list[str] = Field(description="Run on each subject, in order")
    group_steps: list[str] = Field(description="Run once over the subjects")
    warnings: list[str] = Field(description="What the plan check found worth knowing")
    settings: dict[str, Any] = Field(description="The template's settings")


class PlanSaved(BaseModel):
    name: str
    spec: str = Field(description="The pipeline spec, relative to the dataset")
    n_subjects: int
    next: str


class RunInput(BaseModel):
    name: str = Field(pattern=NAME_PATTERN, description="Name of an approved plan")


class WaitingStep(BaseModel):
    subject: str
    step: str
    tool: str
    proposal_id: str


class FailedStep(BaseModel):
    subject: str
    step: str
    message: str


class AnalysisResult(BaseModel):
    name: str
    template: str
    status: Literal["complete", "waiting", "failed"]
    spec: str = Field(description="The pipeline spec, relative to the dataset")
    n_subjects: int
    unit: str
    per_subject: list[dict[str, Any]] = Field(
        description="Each subject's values per condition and difference, with epochs kept"
    )
    mean: dict[str, float] = Field(description="Mean over the subjects with values")
    group_test: dict[str, Any] | None = Field(description="The group test's result, if run")
    waiting: list[WaitingStep] = Field(description="Decisions the user still has to make")
    failed: list[FailedStep]
    notes: list[str]
    next: str | None = None


# --- planning -----------------------------------------------------------------------------


def _plan(template_name: str, settings: BaseModel, ctx: RunContext) -> PlanProposal:
    template = get_template(template_name)
    task, subjects = resolve(settings, ctx.dataset)  # type: ignore[arg-type]
    spec = template.build(settings, task, subjects)
    check = check_plan(spec, dataset=ctx.dataset, subjects=subjects)
    if check.errors:
        raise TemplateError(
            "the plan does not fit the data: " + "; ".join(str(f) for f in check.errors)
        )
    return PlanProposal(
        template=template.name,
        name=spec.name,
        task=task,
        subjects=subjects,
        steps=[_describe(s) for s in spec.steps],
        group_steps=[_describe(s) for s in spec.group],
        warnings=[
            f.message if f.step is None else f"{f.step}: {f.message}" for f in check.warnings
        ],
        settings=settings.model_dump(mode="json", exclude={"task", "subjects"}),
    )


def _describe(step: Step) -> str:
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in step.args.items())
    return f"{step.tool}({args})"


def _save(proposal: PlanProposal, ctx: RunContext) -> PlanSaved:
    """Rebuild the plan from the approved settings (the user may have edited them), check it
    again and save it."""
    template = get_template(proposal.template)
    settings = template.settings.model_validate(
        {**proposal.settings, "task": proposal.task, "subjects": proposal.subjects}
    )
    task, subjects = resolve(settings, ctx.dataset)
    spec = template.build(settings, task, subjects)
    check = check_plan(spec, dataset=ctx.dataset, subjects=subjects)
    if check.errors:
        raise TemplateError(
            "the edited plan does not fit the data: " + "; ".join(str(f) for f in check.errors)
        )
    folder = ctx.output_dir(PIPELINES_DIR, spec.name)
    spec_path = folder / SPEC_FILE
    header = f"# {spec.description}\n# Made from the {template.name} template and approved.\n"
    spec_path.write_text(
        header + yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    plan_path = folder / PLAN_FILE
    plan = {"template": template.name, "settings": settings.model_dump(mode="json")}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ctx.provenance.record_output(spec_path)
    ctx.provenance.record_output(plan_path)
    return PlanSaved(
        name=spec.name,
        spec=spec_path.relative_to(ctx.dataset).as_posix(),
        n_subjects=len(subjects),
        next=f"Call analysis.run with name {spec.name!r} to run it.",
    )


def save_erp_plan(proposal: PlanProposal, ctx: RunContext) -> PlanSaved:
    return _save(proposal, ctx)


def save_spectral_plan(proposal: PlanProposal, ctx: RunContext) -> PlanSaved:
    return _save(proposal, ctx)


@tool(name="analysis.erp_comparison", risk=Risk.DECISION, apply=save_erp_plan)
def erp_comparison(params: ErpComparisonSettings, ctx: RunContext) -> PlanProposal:
    """Plan a tested ERP comparison of two conditions, for one subject or a group: preprocess,
    epoch, average, measure the difference wave in a window at some channels, and (with three
    subjects or more) test it across subjects, corrected for multiple comparisons. Use it to
    compare conditions instead of calling each step yourself. Find the events with
    dataset.inspect first. Once the user approves the plan, run it with analysis.run."""
    return _plan("erp-comparison", params, ctx)


@tool(name="analysis.spectral_comparison", risk=Risk.DECISION, apply=save_spectral_plan)
def spectral_comparison(params: SpectralComparisonSettings, ctx: RunContext) -> PlanProposal:
    """Plan a tested comparison of band power (alpha by default) between two conditions'
    epochs, for one subject or a group, with a corrected group test of the spectra in the
    band. Find the events with dataset.inspect first. Once the user approves the plan, run it
    with analysis.run."""
    return _plan("spectral-comparison", params, ctx)


# --- running ------------------------------------------------------------------------------


@tool(name="analysis.run", risk=Risk.COMPUTE, runs_steps=True)
def run_analysis(params: RunInput, ctx: RunContext) -> AnalysisResult:
    """Run an approved plan over its subjects (steps already run are reused) and report each
    subject's values, their mean and the group test. If decisions wait for the user, ask for
    them, then call analysis.run again to continue."""
    folder = ctx.derivatives_root / PIPELINES_DIR / params.name
    plan_path, spec_path = folder / PLAN_FILE, folder / SPEC_FILE
    if not plan_path.is_file() or not spec_path.is_file():
        raise SpecError(
            f"no approved plan {params.name!r}: plan it with analysis.erp_comparison or "
            "analysis.spectral_comparison and have the user approve it"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    template = get_template(plan["template"])
    settings = template.settings.model_validate(plan["settings"])
    spec: PipelineSpec = load_spec(spec_path)
    report = run_pipeline(spec, ctx, progress=step_progress.get())
    summary = template.summarize(settings, report)
    waiting = [
        WaitingStep(subject=o.subject, step=o.step, tool=o.tool, proposal_id=o.proposal_id)
        for o in report.pending
        if o.proposal_id is not None
    ]
    failed = [
        FailedStep(subject=o.subject, step=o.step, message=o.message or "failed")
        for o in report.failed
    ]
    status: Literal["complete", "waiting", "failed"] = (
        "failed" if failed else "waiting" if waiting else "complete"
    )
    next_step = None
    if waiting:
        next_step = (
            f"{len(waiting)} decisions wait for the user: ask for each (decide), then call "
            f"analysis.run with name {params.name!r} again to continue."
        )
    return AnalysisResult(
        name=params.name,
        template=template.name,
        status=status,
        spec=spec_path.relative_to(ctx.dataset).as_posix(),
        n_subjects=len(report.subjects),
        notes=_notes(report.outcomes),
        waiting=waiting,
        failed=failed,
        next=next_step,
        **summary,
    )


def _notes(outcomes: list[StepOutcome]) -> list[str]:
    """The tools' notes, each once, with the subjects it concerns."""
    by_note: dict[str, list[str]] = {}
    for o in outcomes:
        for note in (o.result or {}).get("notes", []) if isinstance(o.result, dict) else []:
            if isinstance(note, str):
                by_note.setdefault(note, []).append(o.subject)
    notes = []
    for note, subjects in by_note.items():
        who = "group" if subjects == [GROUP] else f"subject {', '.join(subjects)}"
        notes.append(f"{who}: {note}")
    if len(notes) > MAX_NOTES:
        notes = [*notes[:MAX_NOTES], f"... and {len(notes) - MAX_NOTES} more notes"]
    return notes
