"""Pipeline specifications: the steps of an analysis, declared in YAML or JSON.

A spec names the task (and session) it analyses, the subjects (all by default), the steps run
on each subject's recording, and the group steps run once over all subjects::

    name: p3-oddball
    description: P3 to rare vs frequent letters
    task: P3
    steps:
      - id: preprocess
        tool: eeg.preprocess
        args: {l_freq: 0.1, h_freq: 30, reference: [P9, P10]}
      - id: ica
        tool: eeg.ica_clean
      - id: epochs
        tool: eeg.epoch
        args: {conditions: {rare: [stimulus/11], frequent: [stimulus/12]}}
    group:
      - id: clusters
        tool: stats.cluster_permutation
        args: {conditions: [rare, frequent]}

The runner fills in ``subject``, ``task`` and ``session`` (and the group steps' ``subjects``);
specs must not set them. Every step's arguments are checked against its tool's input schema
when the spec is loaded, so mistakes show before anything runs.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from braino.tools.registry import ToolNotFoundError, ToolRegistry, registry

STEP_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
RUNNER_FIELDS = ("subject", "task", "session", "run")
"""Arguments of per-recording steps that the runner sets."""
GROUP_RUNNER_FIELDS = ("task", "session", "subjects")


class SpecError(ValueError):
    """Raised when a spec is malformed or does not fit the tools it names."""


class Step(BaseModel):
    id: str = Field(pattern=STEP_ID_PATTERN, description="Short name of the step, unique")
    tool: str = Field(description="Registered tool name, e.g. 'eeg.preprocess'")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class PipelineSpec(BaseModel):
    name: str = Field(pattern=NAME_PATTERN, description="Name; its state is kept under it")
    description: str = ""
    task: str
    session: str | None = None
    run: str | None = None
    subjects: list[str] | None = Field(
        default=None, description="Subjects to analyse; null for every subject with the task"
    )
    exclude_subjects: list[str] = Field(
        default_factory=list, description="Subjects left out, with the reason in 'description'"
    )
    steps: list[Step] = Field(min_length=1, description="Run on each subject's recording")
    group: list[Step] = Field(default_factory=list, description="Run once over the subjects")

    @model_validator(mode="after")
    def _unique_ids(self) -> PipelineSpec:
        ids = [s.id for s in [*self.steps, *self.group]]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"step ids must be unique; repeated: {', '.join(duplicates)}")
        return self

    def step(self, step_id: str) -> Step:
        for s in [*self.steps, *self.group]:
            if s.id == step_id:
                return s
        raise SpecError(f"no step {step_id!r}; the steps are {', '.join(self.step_ids)}")

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in [*self.steps, *self.group]]

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()

    def recording_args(self, step: Step, subject: str) -> dict[str, Any]:
        args = {**step.args, "subject": subject, "task": self.task}
        if self.session is not None:
            args["session"] = self.session
        if self.run is not None:
            args["run"] = self.run
        return args

    def group_args(self, step: Step, subjects: list[str]) -> dict[str, Any]:
        args = {**step.args, "task": self.task, "subjects": subjects}
        if self.session is not None:
            args["session"] = self.session
        if self.run is not None and "run" not in args:
            args["run"] = self.run
        return args


def load_spec(source: str | Path, *, tools: ToolRegistry | None = None) -> PipelineSpec:
    """Read and check a spec from a ``.yaml``/``.yml``/``.json`` file, or the name of a built-in
    spec (see :func:`builtin_specs`)."""
    path = Path(source)
    if not path.is_file():
        specs = builtin_specs()
        if str(source) in specs:
            path = specs[str(source)]
        else:
            raise SpecError(
                f"{source} is not a file or a built-in spec ({', '.join(sorted(specs))})"
            )
    text = path.read_text(encoding="utf-8")
    return parse_spec(text, json_format=path.suffix.lower() == ".json", tools=tools)


def parse_spec(
    text: str, *, json_format: bool = False, tools: ToolRegistry | None = None
) -> PipelineSpec:
    if json_format:
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SpecError("a spec must be a mapping with name, task and steps")
    try:
        spec = PipelineSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(str(exc)) from None
    check_spec(spec, tools=tools)
    return spec


def check_spec(spec: PipelineSpec, *, tools: ToolRegistry | None = None) -> None:
    """Check that every tool exists and accepts its arguments; raises :class:`SpecError`."""
    reg = registry if tools is None else tools
    problems = []
    for kind, steps, fields in (
        ("step", spec.steps, RUNNER_FIELDS),
        ("group step", spec.group, GROUP_RUNNER_FIELDS),
    ):
        for step in steps:
            try:
                tool = reg.get(step.tool)
            except ToolNotFoundError:
                problems.append(f"{kind} {step.id!r}: no tool {step.tool!r}")
                continue
            schema = tool.input_model.model_fields
            per_recording = "subject" in schema
            if kind == "step" and not per_recording:
                problems.append(
                    f"step {step.id!r}: {step.tool} works on the group, not one "
                    "recording; put it under 'group'"
                )
                continue
            if kind == "group step" and per_recording:
                problems.append(
                    f"group step {step.id!r}: {step.tool} works on one recording; "
                    "put it under 'steps'"
                )
                continue
            reserved = [f for f in fields if f in step.args]
            if reserved:
                problems.append(f"{kind} {step.id!r}: the runner sets {', '.join(reserved)}")
                continue
            sample = (
                spec.recording_args(step, "x") if kind == "step"
                else spec.group_args(step, ["x", "y", "z"])
            )  # fmt: skip
            try:
                tool.input_model.model_validate(sample)
            except ValidationError as exc:
                errors = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                )
                problems.append(f"{kind} {step.id!r} ({step.tool}): {errors}")
    if problems:
        raise SpecError("\n".join(problems))


@cache
def builtin_specs() -> dict[str, Path]:
    """Specs shipped with BrainO, by name (e.g. ``erpcore-p3``)."""
    folder = resources.files("braino.pipelines") / "specs"
    out = {}
    for entry in folder.iterdir():
        if entry.name.endswith((".yaml", ".yml")):
            out[entry.name.rsplit(".", 1)[0]] = Path(str(entry))
    return out
