"""Evaluation tasks: a request, a dataset with a known answer, and checks on what the agent did.

A task is a YAML file::

    id: erp-p3-one-subject
    category: erp
    request: >
      Compare the rare and frequent trials of subject 01 at Pz between 300 and 500 ms.
    dataset:
      synthetic: {subjects: ["01"], seconds: 60}
    checks:
      - {kind: tool_called, tool: eeg.erp}
      - {kind: answer_number, min: 3, max: 10, description: the P3 difference in µV}
    solution:            # how the oracle agent solves it (and a worked example)
      - {id: pre, tool: eeg.preprocess, args: {subject: "01"}}
      - ...
    answer: "The difference at Pz is {erp.measures[waveform=difference].value_uv:.2f} µV."

Every task also checks, unless it says ``grounded: false``, that each number in the answer comes
from a tool result: the agent must never invent a number.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CheckKind = Literal[
    "tool_called",
    "tool_not_called",
    "order",
    "decision_requested",
    "no_decision_requested",
    "answer_number",
    "answer_mentions",
    "answer_not_mentions",
    "max_calls",
    "grounded",
]


class SyntheticData(BaseModel):
    """A dataset made by :mod:`braino.datasets.synthetic`, one recording per subject."""

    subjects: list[str] = Field(min_length=1)
    task: str = "oddball"
    seconds: float = 60.0
    p3_uv: float = 8.0
    blinks: int = 30
    noisy: list[str] = Field(default_factory=list)
    flat: list[str] = Field(default_factory=list)
    line_noise_uv: float = 0.0


class ErpCoreData(BaseModel):
    task: Literal["P3", "N170"]
    subjects: list[str] = Field(min_length=1)


class DatasetSpec(BaseModel):
    synthetic: SyntheticData | None = None
    erpcore: ErpCoreData | None = None

    @model_validator(mode="after")
    def _one(self) -> DatasetSpec:
        if (self.synthetic is None) == (self.erpcore is None):
            raise ValueError("give exactly one of synthetic or erpcore")
        return self


class ScriptedUser(BaseModel):
    """How the simulated user answers the agent."""

    decisions: Literal["approve", "reject"] = "approve"
    edits: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Tool name -> proposal fields the user changes"
    )
    replies: list[str] = Field(default_factory=list, description="Answers to questions, in order")
    default_reply: str = "Please use your judgement; I have no preference."


class Check(BaseModel):
    kind: CheckKind
    description: str = ""
    required: bool = True
    tool: str | list[str] | None = Field(default=None, description="A tool, or any of several")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Argument matchers: a value (equal), or {not|range|any|covers|contains: ...}",
    )
    before: str | None = None
    after: str | None = None
    min: float | None = None
    max: float | None = None
    terms: list[str] = Field(default_factory=list)
    any_term: bool = Field(default=False, description="answer_mentions: any term instead of all")
    count: int | None = None

    @model_validator(mode="after")
    def _fields(self) -> Check:
        needs = {
            "tool_called": ("tool",),
            "tool_not_called": ("tool",),
            "decision_requested": ("tool",),
            "order": ("before", "after"),
            "answer_mentions": ("terms",),
            "answer_not_mentions": ("terms",),
            "max_calls": ("count",),
        }.get(self.kind, ())
        missing = [f for f in needs if not getattr(self, f)]
        if missing:
            raise ValueError(f"{self.kind} check needs {', '.join(missing)}")
        if self.kind == "answer_number" and self.min is None and self.max is None:
            raise ValueError("answer_number check needs min and/or max")
        return self

    @property
    def tools(self) -> list[str]:
        if self.tool is None:
            return []
        return [self.tool] if isinstance(self.tool, str) else list(self.tool)


class SolutionStep(BaseModel):
    id: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    foreach: list[str] | None = Field(
        default=None, description="Run once per subject; '{subject}' in args is replaced"
    )
    decide: str | None = Field(
        default=None, description="Ask the user about the proposal of this step (by id)"
    )
    ask: str | None = Field(default=None, description="Ask the user this question")

    @model_validator(mode="after")
    def _one(self) -> SolutionStep:
        if sum(x is not None for x in (self.tool, self.decide, self.ask)) != 1:
            raise ValueError("a solution step has one of tool, decide or ask")
        return self


class EvalTask(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str
    description: str = ""
    request: str = Field(description="What the user asks the agent, in their words")
    dataset: DatasetSpec
    user: ScriptedUser = Field(default_factory=ScriptedUser)
    checks: list[Check] = Field(min_length=1)
    grounded: bool = Field(default=True, description="Add the check that numbers are grounded")
    solution: list[SolutionStep] = Field(default_factory=list)
    answer: str = Field(default="", description="The oracle's answer, a template over results")
    large: bool = Field(default=False, description="Needs a large download (ERP CORE)")

    def all_checks(self) -> list[Check]:
        checks = list(self.checks)
        if self.grounded and not any(c.kind == "grounded" for c in checks):
            checks.append(
                Check(kind="grounded", description="every number in the answer is from a tool")
            )
        return checks


class TaskError(ValueError):
    pass


def load_task(path: str | Path) -> EvalTask:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        return EvalTask.model_validate(data)
    except ValueError as exc:
        raise TaskError(f"{Path(path).name}: {exc}") from None


@cache
def builtin_tasks() -> tuple[EvalTask, ...]:
    """The tasks shipped with BrainO, sorted by id."""
    folder = resources.files("braino.evals") / "tasks"
    tasks = [
        load_task(Path(str(entry)))
        for entry in folder.iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    ]
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise TaskError(f"duplicate task ids: {', '.join(duplicates)}")
    return tuple(sorted(tasks, key=lambda t: t.id))


def select_tasks(ids: list[str] | None = None, *, include_large: bool = False) -> list[EvalTask]:
    tasks = list(builtin_tasks())
    if ids:
        known = {t.id: t for t in tasks}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise TaskError(f"no task {', '.join(unknown)}")
        return [known[i] for i in ids]
    return [t for t in tasks if include_large or not t.large]
