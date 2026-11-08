"""Templates: tested pipelines for common analyses, filled in with a few settings.

A template turns a handful of typed settings (the two conditions, their events, where and when
to measure) into a full pipeline spec, so the agent fills in a form instead of planning a
dozen steps for every subject. Each template is built from the steps BrainO validated on ERP
CORE, and its settings are the tool arguments that differ between studies; everything else
keeps the tools' tested defaults.

- ``erp-comparison``: two conditions' ERPs, their difference wave measured in a window at
  some channels, and a group test of the difference (with three subjects or more).
- ``spectral-comparison``: band power in two conditions' epochs, and a group test of the
  spectra in the band.

Specs from templates are ordinary specs: ``braino pipeline template`` writes one to a file to
read, change and run with ``braino pipeline run``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from braino.pipelines.runner import GROUP, PipelineReport
from braino.pipelines.spec import PipelineSpec, SpecError, Step, check_spec
from braino.stats.group import MIN_SUBJECTS

CONDITION_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"


class TemplateError(ValueError):
    """Raised when a template's settings do not fit the dataset."""


# --- settings shared by the templates -----------------------------------------------------


class _Common(BaseModel):
    task: str | None = Field(
        default=None, description="Task label of the recordings; null if the dataset has one"
    )
    subjects: list[str] | None = Field(
        default=None, description="Subjects to analyse; null for every subject with the task"
    )
    condition_a: str = Field(
        pattern=CONDITION_PATTERN, description="Name of the first condition, e.g. 'rare'"
    )
    events_a: list[str] = Field(
        min_length=1,
        description="Event descriptions of the first condition, as dataset.inspect lists "
        "them, e.g. ['stimulus/rare']",
    )
    condition_b: str = Field(
        pattern=CONDITION_PATTERN, description="Name of the second condition, e.g. 'frequent'"
    )
    events_b: list[str] = Field(min_length=1, description="Event descriptions of the second")
    reference: Literal["average", "average_with_eog"] | list[str] = Field(
        default="average",
        description="'average', 'average_with_eog' or the channels to reference to, e.g. "
        "['P9', 'P10']",
    )
    ica: bool = Field(
        default=False,
        description="Remove blinks and other artifacts with ICA (the user reviews each "
        "subject's components)",
    )
    bad_channels: bool = Field(
        default=False,
        description="Detect bad channels and interpolate them (the user reviews each subject's)",
    )
    reject: Literal["autoreject", "peak_to_peak", "none"] = Field(
        default="autoreject",
        description="Bad epochs: 'autoreject' (learned thresholds), 'peak_to_peak' "
        "(fixed threshold) or 'none'",
    )
    peak_to_peak_uv: float = Field(default=150.0, gt=0, description="For 'peak_to_peak', µV")

    @model_validator(mode="after")
    def _distinct(self) -> _Common:
        if self.condition_a == self.condition_b:
            raise ValueError("the two conditions need different names")
        both = set(self.events_a) & set(self.events_b)
        if both:
            raise ValueError(f"events {', '.join(sorted(both))} are in both conditions")
        return self


class ErpComparisonSettings(_Common):
    # Required, so the model has to choose: a small model leaves optional fields as they are.
    test_scope: Literal["measured", "all"] = Field(
        description="Where the group test looks: 'measured' (the channels and window below) "
        "or 'all' (every channel and time point of the epochs, when asked to test every "
        "channel or time point)"
    )
    channels: list[str] = Field(
        min_length=1,
        description="Channel names to measure at (averaged), e.g. ['Pz']; with test_scope "
        "'all' the group test still covers every channel",
    )
    tmin: float = Field(description="Start of the measurement window, s (e.g. 0.3)")
    tmax: float = Field(description="End of the measurement window, s (e.g. 0.6)")
    measure: Literal["mean", "peak_positive", "peak_negative"] = Field(
        default="mean", description="'mean' amplitude, or the positive or negative peak"
    )
    epoch_tmin: float = Field(default=-0.2, description="Epoch start, s")
    epoch_tmax: float = Field(default=0.8, description="Epoch end, s")
    l_freq: float = Field(default=0.1, ge=0, description="High-pass, Hz")
    h_freq: float = Field(default=30.0, gt=0, description="Low-pass, Hz")
    test: Literal["mass_univariate", "cluster_permutation"] = Field(
        default="mass_univariate",
        description="Group test: 'mass_univariate' (t-test at each point, corrected) or "
        "'cluster_permutation' (clusters of neighbouring points)",
    )
    correction: Literal["fdr", "bonferroni", "max_stat"] = Field(
        default="fdr", description="Multiple-comparison correction of 'mass_univariate'"
    )

    @model_validator(mode="after")
    def _windows(self) -> ErpComparisonSettings:
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be before tmax")
        if self.epoch_tmin >= self.epoch_tmax:
            raise ValueError("epoch_tmin must be before epoch_tmax")
        if any(ch.strip().lower() in ALL_CHANNELS for ch in self.channels):
            raise ValueError(
                "channels names the channels to measure at, e.g. ['Pz'], not 'all': "
                "test_scope 'all' already tests every channel and time point"
            )
        return self


#: What a model writes for "every channel" in a list that needs channel names.
ALL_CHANNELS = frozenset({"all", "*", "every", "all channels"})


class SpectralComparisonSettings(_Common):
    band: str = Field(default="alpha", pattern=r"^[a-z][a-z0-9_]*$", description="Band name")
    fmin: float = Field(default=8.0, gt=0, description="Lower edge of the band, Hz")
    fmax: float = Field(default=13.0, gt=0, description="Upper edge of the band, Hz")
    channels: list[str] | None = Field(default=None, description="Channels to test; null for all")
    epoch_tmin: float = Field(default=0.0, description="Epoch start, s")
    epoch_tmax: float = Field(default=1.0, description="Epoch end, s (1 s gives 1 Hz bins)")
    l_freq: float = Field(default=1.0, ge=0, description="High-pass, Hz")
    h_freq: float = Field(default=45.0, gt=0, description="Low-pass, Hz")
    correction: Literal["fdr", "bonferroni", "max_stat"] = Field(
        default="fdr", description="Multiple-comparison correction"
    )

    @model_validator(mode="after")
    def _band(self) -> SpectralComparisonSettings:
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be below fmax")
        if self.fmax > self.h_freq:
            raise ValueError(f"the band ends above the low-pass ({self.h_freq:g} Hz)")
        if self.epoch_tmin >= self.epoch_tmax:
            raise ValueError("epoch_tmin must be before epoch_tmax")
        return self


# --- the templates -------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    settings: type[_Common]
    build: Callable[[Any, str, list[str]], PipelineSpec]
    """``(settings, task, subjects) -> spec``."""
    summarize: Callable[[Any, PipelineReport], dict[str, Any]]
    """``(settings, report) -> per-subject values, their mean and the group result``."""


def _cleaning(s: _Common) -> list[Step]:
    steps = []
    if s.ica:
        steps.append(Step(id="ica", tool="eeg.ica_clean"))
    if s.bad_channels:
        steps.append(Step(id="bad_channels", tool="eeg.detect_bad_channels"))
    return steps


def _epochs(s: _Common, baseline: list[float] | None) -> Step:
    args: dict[str, Any] = {
        "conditions": {s.condition_a: list(s.events_a), s.condition_b: list(s.events_b)},
        "tmin": s.epoch_tmin,  # type: ignore[attr-defined]
        "tmax": s.epoch_tmax,  # type: ignore[attr-defined]
        "baseline": baseline,
        "reject": s.reject,
    }
    if s.reject == "peak_to_peak":
        args["peak_to_peak_uv"] = s.peak_to_peak_uv
    return Step(id="epochs", tool="eeg.epoch", args=args)


def contrast_name(s: _Common) -> str:
    return f"{s.condition_a}_minus_{s.condition_b}"


def _build_erp(s: ErpComparisonSettings, task: str, subjects: list[str]) -> PipelineSpec:
    contrast = contrast_name(s)
    steps = [
        Step(
            id="preprocess",
            tool="eeg.preprocess",
            args={"l_freq": s.l_freq, "h_freq": s.h_freq, "reference": s.reference},
        ),
        *_cleaning(s),
        _epochs(s, [s.epoch_tmin, 0.0]),
        Step(
            id="erp",
            tool="eeg.erp",
            args={
                "contrasts": [{"name": contrast, "plus": s.condition_a, "minus": s.condition_b}],
                "measures": [
                    {
                        "name": "amplitude",
                        "channels": list(s.channels),
                        "tmin": s.tmin,
                        "tmax": s.tmax,
                        "kind": s.measure,
                    }
                ],
            },
        ),
    ]
    group = []
    if len(subjects) >= MIN_SUBJECTS:
        args: dict[str, Any] = {"conditions": [contrast]}
        if s.test_scope == "measured":
            args.update(channels=list(s.channels), tmin=s.tmin, tmax=s.tmax)
        else:  # every channel; the whole epoch
            args.update(tmin=s.epoch_tmin, tmax=s.epoch_tmax)
        if s.test == "mass_univariate":
            args["correction"] = s.correction
        group.append(Step(id="group_test", tool=f"stats.{s.test}", args=args))
    return _spec(
        f"erp-{s.condition_a}-vs-{s.condition_b}",
        f"ERP comparison: {s.condition_a} minus {s.condition_b}, {s.measure} amplitude at "
        f"{', '.join(s.channels)}, {s.tmin:g}-{s.tmax:g} s",
        task,
        subjects,
        steps,
        group,
    )


def _build_spectral(s: SpectralComparisonSettings, task: str, subjects: list[str]) -> PipelineSpec:
    steps = [
        Step(
            id="preprocess",
            tool="eeg.preprocess",
            args={"l_freq": s.l_freq, "h_freq": s.h_freq, "reference": s.reference},
        ),
        *_cleaning(s),
        _epochs(s, None),
        Step(
            id="spectra",
            tool="eeg.spectral",
            args={
                "source": "epochs",
                "fmin": max(s.l_freq, 1.0),
                "fmax": s.h_freq,
                "bands": {s.band: [s.fmin, s.fmax]},
            },
        ),
    ]
    group = []
    if len(subjects) >= MIN_SUBJECTS:
        args: dict[str, Any] = {
            "data": "psd",
            "conditions": [s.condition_a, s.condition_b],
            "fmin": s.fmin,
            "fmax": s.fmax,
            "correction": s.correction,
        }
        if s.channels:
            args["channels"] = list(s.channels)
        group.append(Step(id="group_test", tool="stats.mass_univariate", args=args))
    return _spec(
        f"spectral-{s.condition_a}-vs-{s.condition_b}",
        f"Spectral comparison: {s.band} power ({s.fmin:g}-{s.fmax:g} Hz), "
        f"{s.condition_a} vs {s.condition_b}",
        task,
        subjects,
        steps,
        group,
    )


def _spec(
    name: str, description: str, task: str, subjects: list[str], steps: list[Step],
    group: list[Step],
) -> PipelineSpec:  # fmt: skip
    spec = PipelineSpec(
        name=re.sub(r"[^a-z0-9_-]", "-", name.lower()),
        description=description,
        task=task,
        subjects=subjects,
        steps=steps,
        group=group,
    )
    try:
        check_spec(spec)
    except SpecError as exc:
        raise TemplateError(str(exc)) from None
    return spec


# --- summaries -----------------------------------------------------------------------------


def _subject_results(report: PipelineReport, step: str) -> dict[str, dict[str, Any]]:
    return {
        o.subject: o.result
        for o in report.outcomes
        if o.step == step and o.subject != GROUP and o.result is not None
    }


def _group_result(report: PipelineReport) -> dict[str, Any] | None:
    for o in report.outcomes:
        if o.subject == GROUP and o.step == "group_test" and o.result is not None:
            result = o.result.get("result")
            return result if isinstance(result, dict) else None
    return None


def _kept(report: PipelineReport) -> dict[str, dict[str, int]]:
    return {
        subject: {c: v["kept"] for c, v in result.get("conditions", {}).items()}
        for subject, result in _subject_results(report, "epochs").items()
    }


def _summary(
    values: dict[str, dict[str, float]], unit: str, report: PipelineReport
) -> dict[str, Any]:
    names = sorted({k for v in values.values() for k in v})
    mean = {
        n: round(fmean(v[n] for v in values.values() if n in v), 3)
        for n in names
        if any(n in v for v in values.values())
    }
    kept = _kept(report)
    return {
        "unit": unit,
        "per_subject": [
            {"subject": s, **{k: round(x, 3) for k, x in v.items()}, "epochs_kept": kept.get(s)}
            for s, v in sorted(values.items())
        ],
        "mean": mean,
        "group_test": _group_result(report),
    }


def _summarize_erp(s: ErpComparisonSettings, report: PipelineReport) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = {}
    for subject, result in _subject_results(report, "erp").items():
        values[subject] = {m["waveform"]: m["value_uv"] for m in result.get("measures", [])}
    return _summary(values, "µV", report)


def _summarize_spectral(s: SpectralComparisonSettings, report: PipelineReport) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = {}
    for subject, result in _subject_results(report, "spectra").items():
        row = {}
        for condition, spectrum in result.get("conditions", {}).items():
            for band in spectrum.get("bands", []):
                if band["band"] == s.band:
                    row[condition] = band["mean_absolute_uv2"]
        values[subject] = row
    return _summary(values, "µV²", report)


TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in (
        Template(
            name="erp-comparison",
            description="Compare two conditions' ERPs: the difference wave in a window at "
            "some channels, per subject, and a group test of the difference",
            settings=ErpComparisonSettings,
            build=_build_erp,
            summarize=_summarize_erp,
        ),
        Template(
            name="spectral-comparison",
            description="Compare band power (e.g. alpha) between two conditions' epochs, per "
            "subject, and a group test of the spectra in the band",
            settings=SpectralComparisonSettings,
            build=_build_spectral,
            summarize=_summarize_spectral,
        ),
    )
}


def get_template(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise TemplateError(
            f"no template {name!r}; the templates are {', '.join(TEMPLATES)}"
        ) from None


def resolve(settings: _Common, dataset: Path) -> tuple[str, list[str]]:
    """The task and the subjects the settings select in ``dataset``."""
    from braino.io.dataset import scan_dataset

    summary = scan_dataset(dataset)
    task = settings.task
    if task is None:
        if len(summary.tasks) != 1:
            raise TemplateError(
                "give the task: the dataset has "
                + (f"tasks {', '.join(summary.tasks)}" if summary.tasks else "no task labels")
            )
        task = summary.tasks[0]
    available = sorted({r.subject for r in summary.recordings if r.task == task})
    if not available:
        raise TemplateError(
            f"no recording of task {task!r}; the tasks are {', '.join(summary.tasks)}"
        )
    subjects = list(settings.subjects) if settings.subjects else available
    missing = [s for s in subjects if s not in available]
    if missing:
        raise TemplateError(
            f"no {task} recording for subject(s) {', '.join(missing)}; there are "
            f"{', '.join(available)}"
        )
    return task, subjects


def build(name: str, settings: dict[str, Any] | BaseModel, dataset: Path) -> PipelineSpec:
    """The spec of template ``name`` with ``settings``, for the subjects of ``dataset``."""
    template = get_template(name)
    model = template.settings.model_validate(
        settings.model_dump() if isinstance(settings, BaseModel) else settings
    )
    task, subjects = resolve(model, dataset)
    return template.build(model, task, subjects)
