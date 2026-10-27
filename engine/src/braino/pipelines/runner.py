"""Running a pipeline spec over a dataset, with cached results.

For each subject the steps run in order; then the group steps run once. A step is **skipped**
(its earlier result reused) when its tool, arguments and the BrainO version are unchanged and
every file it read and wrote still has the hash it had then. Re-running an earlier step writes
new files, so the steps after it run again by themselves. ``from_step`` forces a step, and the
ones after it, to run again.

Decision steps stop a subject until the proposal is decided (``braino proposals approve``);
running the pipeline again then continues from there. Only the person running the pipeline can
choose ``auto_approve``, which approves every proposal as made (e.g. for validation runs); it
is recorded as such in every decision.

The state lives in ``derivatives/braino/pipelines/<name>/state.json``: for every subject and
step, which tool call produced the current result, with its arguments and file hashes. Script
export reads it to rebuild the analysis.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from braino import __version__
from braino.context import RunContext
from braino.io.dataset import scan_dataset
from braino.pipelines.spec import PipelineSpec, SpecError, Step
from braino.provenance.models import FileRecord, ToolCallRecord
from braino.provenance.store import RunNotFoundError, load_run
from braino.tools.base import PendingProposal, Tool
from braino.tools.decisions import decide
from braino.tools.registry import ToolRegistry, registry

PIPELINES_DIR = "pipelines"
STATE_FILE = "state.json"
GROUP = "group"
"""Subject key of the group steps in the state."""

Status = Literal["ran", "cached", "pending", "rejected", "failed", "blocked"]


class CallRef(BaseModel):
    run_id: str
    call_index: int


class StepRecord(BaseModel):
    """The tool call behind a step's current result."""

    step: str
    subject: str
    tool: str
    args: dict[str, Any] = Field(description="The validated arguments")
    key: str = Field(description="Hash of tool, arguments and BrainO version")
    call: CallRef
    inputs: list[FileRecord]
    outputs: list[FileRecord]
    result: dict[str, Any] | None
    proposal_id: str | None = None
    apply: CallRef | None = None
    apply_outputs: list[FileRecord] = Field(default_factory=list)
    finished_at: datetime


class PipelineState(BaseModel):
    name: str
    spec_sha256: str
    records: dict[str, StepRecord] = Field(
        default_factory=dict, description="'<subject>/<step id>' -> record"
    )
    file_stats: dict[str, tuple[int, int, str]] = Field(
        default_factory=dict, description="path -> (size, mtime_ns, sha256), to skip rehashing"
    )


class StepOutcome(BaseModel):
    subject: str
    step: str
    tool: str
    status: Status
    proposal_id: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None
    changed: bool = Field(
        default=False, description="Whether the step's result is new since the last run"
    )


class PipelineReport(BaseModel):
    name: str
    run_id: str
    subjects: list[str]
    outcomes: list[StepOutcome]

    def count(self, status: Status) -> int:
        return sum(o.status == status for o in self.outcomes)

    @property
    def pending(self) -> list[StepOutcome]:
        return [o for o in self.outcomes if o.status == "pending"]

    @property
    def failed(self) -> list[StepOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def complete(self) -> bool:
        return all(o.status in ("ran", "cached", "rejected") for o in self.outcomes)


def state_path(ctx: RunContext, name: str) -> Path:
    return ctx.derivatives_root / PIPELINES_DIR / name / STATE_FILE


def load_state(ctx: RunContext, spec: PipelineSpec) -> PipelineState:
    path = state_path(ctx, spec.name)
    if path.is_file():
        return PipelineState.model_validate_json(path.read_text(encoding="utf-8"))
    return PipelineState(name=spec.name, spec_sha256=spec.sha256())


def save_state(ctx: RunContext, state: PipelineState) -> None:
    folder = ctx.output_dir(PIPELINES_DIR, state.name)
    path = folder / STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    # On Windows, replacing fails while another process (a virus scanner, the search indexer)
    # briefly has the file open; such locks last milliseconds, so try again a few times.
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


_REPLACE_ATTEMPTS = 10


def pipeline_subjects(ctx: RunContext, spec: PipelineSpec) -> list[str]:
    available = sorted(
        {
            r.subject
            for r in scan_dataset(ctx.dataset).recordings
            if r.task == spec.task and (spec.session is None or r.session == spec.session)
        }
    )
    wanted = available if spec.subjects is None else list(spec.subjects)
    missing = [s for s in wanted if s not in available]
    if missing:
        raise SpecError(f"no {spec.task} recording for subject(s) {', '.join(missing)}")
    return [s for s in wanted if s not in spec.exclude_subjects]


def run_pipeline(
    spec: PipelineSpec,
    ctx: RunContext,
    *,
    subjects: list[str] | None = None,
    from_step: str | None = None,
    auto_approve: str | None = None,
    tools: ToolRegistry | None = None,
    progress: Callable[[StepOutcome], None] | None = None,
) -> PipelineReport:
    """Run ``spec`` over the dataset of ``ctx``.

    ``subjects`` limits the run to some of the spec's subjects (the group steps then use only
    those). ``auto_approve`` is the identity recorded as approving every proposal, e.g.
    ``"user:alice (auto-approved in a pipeline run)"``; without it, decision steps wait.
    """
    reg = registry if tools is None else tools
    all_subjects = pipeline_subjects(ctx, spec)
    if subjects is not None:
        unknown = [s for s in subjects if s not in all_subjects]
        if unknown:
            raise SpecError(f"subject(s) {', '.join(unknown)} are not in this pipeline")
        all_subjects = [s for s in all_subjects if s in subjects]
    if from_step is not None:
        spec.step(from_step)  # check it exists
    state = load_state(ctx, spec)
    state.spec_sha256 = spec.sha256()
    hasher = _Hasher(ctx.dataset, state)
    runner = _Runner(spec, ctx, reg, state, hasher, auto_approve, progress)

    outcomes: list[StepOutcome] = []
    finished = []
    for subject in all_subjects:
        forced = from_step is not None and from_step in [s.id for s in spec.steps]
        subject_outcomes = runner.run_steps(
            subject,
            spec.steps,
            functools.partial(_recording_args, spec, subject),
            first_forced=from_step if forced else None,
        )
        outcomes.extend(subject_outcomes)
        if all(o.status in ("ran", "cached", "rejected") for o in subject_outcomes):
            finished.append(subject)

    if spec.group:
        unfinished = [s for s in all_subjects if s not in finished]
        if unfinished:
            for step in spec.group:
                outcome = StepOutcome(
                    subject=GROUP, step=step.id, tool=step.tool, status="blocked",
                    message=f"waiting for subject(s) {', '.join(unfinished)}",
                )  # fmt: skip
                outcomes.append(outcome)
                if progress:
                    progress(outcome)
        else:
            # Any per-subject step forced or re-run means the group steps re-run too (their
            # inputs changed); from_step naming a group step forces from there.
            group_forced = from_step if from_step in [s.id for s in spec.group] else None
            if group_forced is None and (from_step is not None or any(o.changed for o in outcomes)):
                group_forced = spec.group[0].id
            outcomes.extend(
                runner.run_steps(
                    GROUP,
                    spec.group,
                    lambda step: spec.group_args(step, finished),
                    first_forced=group_forced,
                )
            )
    save_state(ctx, state)
    return PipelineReport(
        name=spec.name, run_id=ctx.run_id, subjects=all_subjects, outcomes=outcomes
    )


class _Runner:
    def __init__(
        self,
        spec: PipelineSpec,
        ctx: RunContext,
        tools: ToolRegistry,
        state: PipelineState,
        hasher: _Hasher,
        auto_approve: str | None,
        progress: Callable[[StepOutcome], None] | None,
    ) -> None:
        self.spec = spec
        self.ctx = ctx
        self.tools = tools
        self.state = state
        self.hasher = hasher
        self.auto_approve = auto_approve
        self.progress = progress

    def run_steps(
        self,
        subject: str,
        steps: list[Step],
        make_args: Callable[[Step], dict[str, Any]],
        *,
        first_forced: str | None,
    ) -> list[StepOutcome]:
        outcomes = []
        forcing = False
        stopped: str | None = None
        for step in steps:
            if step.id == first_forced:
                forcing = True
            if stopped is not None:
                outcome = StepOutcome(
                    subject=subject, step=step.id, tool=step.tool, status="blocked",
                    message=stopped,
                )  # fmt: skip
            else:
                outcome = self._run_step(subject, step, make_args(step), force=forcing)
                if outcome.changed:
                    forcing = True  # everything after a changed step runs again
                if outcome.status == "pending":
                    stopped = f"waiting for the decision on {step.id}"
                elif outcome.status == "failed":
                    stopped = f"{step.id} failed"
            outcomes.append(outcome)
            if self.progress:
                self.progress(outcome)
            save_state(self.ctx, self.state)
        return outcomes

    def _run_step(
        self, subject: str, step: Step, args: dict[str, Any], *, force: bool
    ) -> StepOutcome:
        tool = self.tools.get(step.tool)
        base = {"subject": subject, "step": step.id, "tool": tool.name}
        try:
            validated = tool.input_model.model_validate(args).model_dump(mode="json")
        except ValueError as exc:
            return StepOutcome(**base, status="failed", message=str(exc))
        key = _key(tool, validated)
        record_key = f"{subject}/{step.id}"
        record = self.state.records.get(record_key)

        if not force and record is not None and record.key == key:
            reused = self._reuse(record, base)
            if reused is not None:
                return reused

        try:
            result = tool.run(validated, self.ctx)
        except Exception as exc:  # a failed step stops this subject, not the pipeline
            return StepOutcome(**base, status="failed", message=f"{type(exc).__name__}: {exc}")
        call = self.ctx.provenance.calls[-1]
        record = StepRecord(
            step=step.id,
            subject=subject,
            tool=tool.name,
            args=validated,
            key=key,
            call=CallRef(run_id=call.run_id, call_index=call.call_index),
            inputs=call.inputs,
            outputs=call.outputs,
            result=call.result,
            proposal_id=call.proposal_id,
            finished_at=call.finished_at,
        )
        self.hasher.remember(record.inputs + record.outputs)
        self.state.records[record_key] = record
        if not isinstance(result, PendingProposal):
            return StepOutcome(**base, status="ran", result=record.result, changed=True)
        if self.auto_approve is None:
            return StepOutcome(
                **base, status="pending", proposal_id=result.proposal_id,
                message=f"approve or reject proposal {result.proposal_id}, then run again",
            )  # fmt: skip
        try:
            decision = decide(
                result.proposal_id, self.ctx, approve=True, decided_by=self.auto_approve,
                note="auto-approved in a pipeline run", tools=self.tools,
            )  # fmt: skip
        except Exception as exc:
            return StepOutcome(**base, status="failed", message=f"{type(exc).__name__}: {exc}")
        apply_call = self.ctx.provenance.calls[-1]
        record.apply = CallRef(run_id=apply_call.run_id, call_index=apply_call.call_index)
        record.apply_outputs = apply_call.outputs
        self.hasher.remember(record.apply_outputs)
        applied = decision.decision.apply_result if decision.decision else None
        return StepOutcome(
            **base, status="ran", proposal_id=result.proposal_id, result=applied, changed=True
        )

    def _reuse(self, record: StepRecord, base: dict[str, Any]) -> StepOutcome | None:
        """The outcome of reusing ``record``, or ``None`` if it must run again."""
        if not self.hasher.unchanged(record.inputs + record.outputs):
            return None
        if record.proposal_id is None:
            return StepOutcome(**base, status="cached", result=record.result)
        proposal = self.ctx.proposals.get(record.proposal_id)
        if proposal.status == "pending":
            if self.auto_approve is None:
                return StepOutcome(
                    **base, status="pending", proposal_id=record.proposal_id,
                    message=f"approve or reject proposal {record.proposal_id}, then run again",
                )  # fmt: skip
            return None  # make and approve a new proposal
        if proposal.status == "rejected":
            return StepOutcome(
                **base, status="rejected", proposal_id=record.proposal_id,
                message="proposal rejected; nothing applied",
            )  # fmt: skip
        decision = proposal.decision
        if decision is None or decision.applied_in is None:
            return None
        newly_applied = record.apply is None
        if newly_applied:  # approved outside the pipeline since the last run
            apply_call = _find_call(self.ctx, decision.applied_in)
            if apply_call is None:
                return None
            record.apply = CallRef(run_id=apply_call.run_id, call_index=apply_call.call_index)
            record.apply_outputs = apply_call.outputs
            self.hasher.remember(record.apply_outputs)
        if not self.hasher.unchanged(record.apply_outputs):
            return None
        return StepOutcome(
            **base,
            status="cached",
            proposal_id=record.proposal_id,
            result=decision.apply_result,
            message="decision applied since the last run" if newly_applied else None,
            changed=newly_applied,
        )


def _recording_args(spec: PipelineSpec, subject: str, step: Step) -> dict[str, Any]:
    return spec.recording_args(step, subject)


def _key(tool: Tool, args: dict[str, Any]) -> str:
    text = json.dumps({"tool": tool.name, "args": args, "braino": __version__}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _find_call(ctx: RunContext, ref: str) -> ToolCallRecord | None:
    """The call ``'<run_id>-<call_index>'``."""
    run_id, _, index = ref.rpartition("-")
    if not index.isdigit():
        return None
    calls = ctx.provenance.calls if run_id == ctx.run_id else None
    if calls is None:
        try:
            calls = load_run(ctx.derivatives_root, run_id).calls
        except RunNotFoundError:
            return None
    return next((c for c in calls if c.call_index == int(index)), None)


class _Hasher:
    """Checks recorded files against the dataset, rehashing only files whose size or
    modification time changed."""

    def __init__(self, root: Path, state: PipelineState) -> None:
        self.root = root
        self.state = state

    def remember(self, files: list[FileRecord]) -> None:
        for f in files:
            path = self._path(f.path)
            if path is not None and path.is_file():
                stat = path.stat()
                self.state.file_stats[f.path] = (stat.st_size, stat.st_mtime_ns, f.sha256)

    def unchanged(self, files: list[FileRecord]) -> bool:
        return all(self._sha256(f.path) == f.sha256 for f in files)

    def _sha256(self, relative: str) -> str | None:
        path = self._path(relative)
        if path is None or not path.is_file():
            return None
        stat = path.stat()
        known = self.state.file_stats.get(relative)
        if known is not None and known[0] == stat.st_size and known[1] == stat.st_mtime_ns:
            return known[2]
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        sha = digest.hexdigest()
        self.state.file_stats[relative] = (stat.st_size, stat.st_mtime_ns, sha)
        return sha

    def _path(self, relative: str) -> Path | None:
        if relative.startswith("<") or Path(relative).is_absolute():
            return None  # outside the dataset: not tracked
        return self.root / relative
