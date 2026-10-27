"""Group-level statistics tools: tests across subjects on the per-subject results that
``eeg.erp``, ``eeg.spectral`` and ``eeg.tfr`` write.

Outputs go to ``derivatives/braino/group/task-<task>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from braino.context import RunContext
from braino.io.dataset import scan_dataset
from braino.stats.group import ClusterResult, Correction, GroupData, UnivariateResult
from braino.tools.base import Risk
from braino.tools.recordings import (
    LABEL_PATTERN,
    RecordingSelector,
    derivative,
    find_recording,
    relative,
    write_tsv,
)
from braino.tools.registry import tool

DataKind = Literal["erp", "psd", "tfr"]

_SUFFIX = {"erp": "ave.fif", "psd": "psd.npz", "tfr": "tfr.npz"}
_MAKER = {"erp": "eeg.erp", "psd": "eeg.spectral", "tfr": "eeg.tfr"}


class NotEnoughSubjectsError(ValueError):
    pass


class GroupInput(BaseModel):
    task: str = Field(description="Task label of the recordings")
    session: str | None = Field(default=None, description="Session label, if any")
    run: str | None = Field(default=None, description="Run number, if subjects have several")
    subjects: list[str] | None = Field(
        default=None, description="Subjects to include; null for every subject with results"
    )
    data: DataKind = Field(
        default="erp",
        description="Which per-subject results: 'erp' (eeg.erp averages, µV), 'psd' "
        "(eeg.spectral spectra, dB), 'tfr' (eeg.tfr power)",
    )
    label: str = Field(
        default="epochs",
        pattern=LABEL_PATTERN,
        description="Label of the results (the epochs label; 'continuous' for spectra of "
        "continuous data)",
    )
    conditions: list[str] = Field(
        min_length=1,
        description="One condition or difference wave (tested against 0), two conditions "
        "(paired test of the first minus the second), or more (repeated-measures ANOVA, "
        "stats.mass_univariate only)",
    )
    channels: list[str] | None = Field(default=None, description="Channels; null for all")
    tmin: float | None = Field(default=None, description="Start of the time window, s")
    tmax: float | None = Field(default=None, description="End of the time window, s")
    fmin: float | None = Field(default=None, description="Lowest frequency (psd, tfr), Hz")
    fmax: float | None = Field(default=None, description="Highest frequency (psd, tfr), Hz")
    output_label: str = Field(
        default="group", pattern=LABEL_PATTERN, description="Name for the output files"
    )

    @model_validator(mode="after")
    def _windows(self) -> GroupInput:
        if self.tmin is not None and self.tmax is not None and self.tmin >= self.tmax:
            raise ValueError("tmin must be before tmax")
        if self.fmin is not None and self.fmax is not None and self.fmin >= self.fmax:
            raise ValueError("fmin must be below fmax")
        if self.data == "psd" and (self.tmin is not None or self.tmax is not None):
            raise ValueError("spectra have no time axis; leave tmin and tmax out")
        if self.data == "erp" and (self.fmin is not None or self.fmax is not None):
            raise ValueError("ERPs have no frequency axis; leave fmin and fmax out")
        return self


class MassUnivariateInput(GroupInput):
    correction: Correction = Field(
        default="fdr",
        description="Multiple-comparison correction: 'fdr' (false discovery rate), "
        "'bonferroni', 'max_stat' (permutation, family-wise) or 'none' (exploration only)",
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)
    n_permutations: int = Field(default=1000, ge=100, le=100_000, description="For max_stat")
    seed: int = 42


class ClusterInput(GroupInput):
    p_threshold: float = Field(
        default=0.05, gt=0, lt=1, description="Two-sided p of the t that forms clusters"
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)
    n_permutations: int = Field(default=1024, ge=100, le=100_000)
    seed: int = 42


class GroupSummary(BaseModel):
    subjects: list[str]
    missing_subjects: list[str] = Field(description="Subjects without results, left out")
    n_channels: int
    unit: str
    window: dict[str, tuple[float, float]] = Field(description="Extent of each axis tested")


class MassUnivariateOutput(GroupSummary):
    result: UnivariateResult
    table: str = Field(description="Every tested point, relative to the dataset")
    notes: list[str]


class ClusterOutput(GroupSummary):
    result: ClusterResult
    table: str = Field(description="Clusters, relative to the dataset")
    notes: list[str]


@tool(name="stats.mass_univariate", risk=Risk.COMPUTE)
def mass_univariate(params: MassUnivariateInput, ctx: RunContext) -> MassUnivariateOutput:
    """Test across subjects at every channel and time (or frequency) point: t-test of one
    condition against 0, paired t-test of two conditions, or repeated-measures ANOVA of more,
    corrected for multiple comparisons (FDR by default). Reads the per-subject results of
    eeg.erp, eeg.spectral or eeg.tfr. Restrict channels and windows to planned ones when you
    can: fewer tests, more power."""
    from braino.stats.group import univariate

    group, missing = load_group(ctx, params)
    result, maps = univariate(
        group,
        correction=params.correction,
        alpha=params.alpha,
        n_permutations=params.n_permutations,
        seed=params.seed,
    )
    table = _group_dir(ctx, params) / f"task-{params.task}_desc-{params.output_label}_stats.tsv"
    _write_maps(ctx, table, group, maps)
    notes = list(group.notes)
    if params.correction == "none":
        notes.append("p-values are not corrected for multiple comparisons.")
    return MassUnivariateOutput(
        **_summary(group, missing),
        result=result,
        table=relative(ctx, table),
        notes=notes,
    )


@tool(name="stats.cluster_permutation", risk=Risk.COMPUTE)
def cluster_permutation(params: ClusterInput, ctx: RunContext) -> ClusterOutput:
    """Cluster-based permutation test across subjects (one condition against 0, or a paired
    test of two): finds groups of neighbouring channels and time (or frequency) points with
    an effect, and gives each cluster a p-value that controls the family-wise error rate.
    Reads the per-subject results of eeg.erp, eeg.spectral or eeg.tfr."""
    from braino.stats.group import cluster_test

    if len(params.conditions) > 2:
        raise ValueError("give one condition (tested against 0) or two (paired)")
    group, missing = load_group(ctx, params)
    result, _ = cluster_test(
        group,
        p_threshold=params.p_threshold,
        alpha=params.alpha,
        n_permutations=params.n_permutations,
        seed=params.seed,
    )
    table = _group_dir(ctx, params) / f"task-{params.task}_desc-{params.output_label}_clusters.tsv"
    axes = list(group.axes)
    rows = [
        [
            i,
            c.p,
            c.sign,
            c.mass,
            c.n_points,
            ",".join(c.channels),
            *[v for a in axes for v in c.ranges[a]],
        ]
        for i, c in enumerate(result.clusters, start=1)
    ]
    header = [
        "cluster",
        "p",
        "sign",
        "mass",
        "n_points",
        "channels",
        *[f"{a}_{end}" for a in axes for end in ("start", "end")],
    ]
    write_tsv(ctx, table, header, rows)
    return ClusterOutput(
        **_summary(group, missing),
        result=result,
        table=relative(ctx, table),
        notes=list(group.notes),
    )


def load_group(ctx: RunContext, params: GroupInput) -> tuple[GroupData, list[str]]:
    """Every subject's ``params.data`` results for ``params.conditions``, aligned on common
    channels and the requested windows. Returns the data and the subjects left out."""
    import numpy as np

    wanted = params.subjects
    if wanted is None:
        wanted = sorted(
            {r.subject for r in scan_dataset(ctx.dataset).recordings if r.task == params.task}
        )
    loaded: dict[str, tuple[list[Any], list[str], dict[str, Any], str, Any]] = {}
    missing = []
    for subject in wanted:
        selector = RecordingSelector(
            subject=subject, task=params.task, session=params.session, run=params.run
        )
        recording = find_recording(ctx.dataset, selector)
        path = derivative(ctx, recording, f"desc-{params.label}_{_SUFFIX[params.data]}")
        if not path.is_file():
            if params.subjects is not None:
                raise FileNotFoundError(
                    f"sub-{subject}: no {path.name}; run {_MAKER[params.data]} first"
                )
            missing.append(subject)
            continue
        ctx.record_input(path)
        loaded[subject] = _load_subject(ctx, path, params)

    subjects = sorted(loaded)
    if len(subjects) < 3:
        raise NotEnoughSubjectsError(
            f"{len(subjects)} subjects have {params.data} results labelled {params.label!r}; "
            "group tests need at least 3"
        )
    first = loaded[subjects[0]]
    common = [ch for ch in first[1] if all(ch in loaded[s][1] for s in subjects)]
    if params.channels is not None:
        absent = [ch for ch in params.channels if ch not in common]
        if absent:
            raise ValueError(f"channels not in every subject's results: {', '.join(absent)}")
        common = list(params.channels)
    notes = []
    dropped = [ch for ch in first[1] if ch not in common and params.channels is None]
    if dropped:
        notes.append(f"Channels missing in some subjects were left out: {', '.join(dropped)}")

    axes = first[2]
    for s in subjects[1:]:
        for name, coords in axes.items():
            other = loaded[s][2][name]
            if other.shape != coords.shape or not np.allclose(other, coords, atol=1e-6):
                raise ValueError(
                    f"sub-{s} has different {name} points from sub-{subjects[0]}; process every "
                    "subject with the same settings"
                )
    masks = {name: _window(name, coords, params) for name, coords in axes.items()}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"the {name} window contains no data points")

    data = []
    for k in range(len(params.conditions)):
        stacked = []
        for s in subjects:
            arrays, names, _, _, _ = loaded[s]
            idx = [names.index(ch) for ch in common]
            x = arrays[k][idx]
            for a, mask in enumerate(masks.values()):
                x = np.compress(mask, x, axis=a + 1)
            stacked.append(x)
        data.append(np.stack(stacked))
    group = GroupData(
        conditions=list(params.conditions),
        data=data,
        subjects=subjects,
        ch_names=common,
        axes={name: coords[masks[name]] for name, coords in axes.items()},
        unit=first[3],
        info=first[4],
        notes=notes,
    )
    return group, missing


def _load_subject(
    ctx: RunContext, path: Path, params: GroupInput
) -> tuple[list[Any], list[str], dict[str, Any], str, Any]:
    """One subject's arrays per condition (channels x axes), channel names, axes, unit, info."""
    import mne
    import numpy as np

    if params.data == "erp":
        evokeds = {ev.comment: ev for ev in mne.read_evokeds(path, verbose="error")}
        _check_conditions(path, list(evokeds), params.conditions)
        picked = [evokeds[c].copy().pick("eeg") for c in params.conditions]
        names = list(picked[0].ch_names)
        return (
            [ev.data * 1e6 for ev in picked],
            names,
            {"time": picked[0].times},
            "µV",
            picked[0].info,
        )
    with np.load(path, allow_pickle=False) as npz:
        conditions = [str(c) for c in npz["conditions"]]
        _check_conditions(path, conditions, params.conditions)
        names = [str(c) for c in npz["ch_names"]]
        arrays = [npz[f"data_{conditions.index(c)}"] for c in params.conditions]
        freqs = npz["freqs"]
        if params.data == "psd":
            axes = {"freq": freqs}
            arrays = [10 * np.log10(np.maximum(a * 1e12, 1e-30)) for a in arrays]
            unit = "dB (10·log10 µV²/Hz)"
        else:
            axes = {"freq": freqs, "time": npz["times"]}
            unit = str(npz["unit"])
    info = _epochs_info(path, params.label)
    return arrays, names, axes, unit, info


def _epochs_info(path: Path, label: str) -> Any:
    """Channel positions from the subject's epochs file, if there is one."""
    import mne

    epochs = path.with_name(path.name.rsplit(f"desc-{label}_", 1)[0] + f"desc-{label}_epo.fif")
    if not epochs.is_file():
        return None
    return mne.io.read_info(epochs, verbose="error")


def _check_conditions(path: Path, available: list[str], wanted: list[str]) -> None:
    absent = [c for c in wanted if c not in available]
    if absent:
        raise ValueError(
            f"{path.name} has no {', '.join(map(repr, absent))}; it has {', '.join(available)}"
        )


def _window(name: str, coords: Any, params: GroupInput) -> Any:
    import numpy as np

    lo, hi = (params.tmin, params.tmax) if name == "time" else (params.fmin, params.fmax)
    mask = np.ones(coords.shape, dtype=bool)
    if lo is not None:
        mask &= coords >= lo - 1e-9
    if hi is not None:
        mask &= coords <= hi + 1e-9
    return mask


def _group_dir(ctx: RunContext, params: GroupInput) -> Path:
    parts = ["group", f"task-{params.task}"]
    if params.session:
        parts.append(f"ses-{params.session}")
    return ctx.output_dir(*parts)


def _summary(group: GroupData, missing: list[str]) -> dict[str, Any]:
    return {
        "subjects": group.subjects,
        "missing_subjects": missing,
        "n_channels": len(group.ch_names),
        "unit": group.unit,
        "window": {
            name: (round(float(c[0]), 4), round(float(c[-1]), 4)) for name, c in group.axes.items()
        },
    }


def _write_maps(ctx: RunContext, path: Path, group: GroupData, maps: dict[str, Any]) -> None:
    import numpy as np

    axes = list(group.axes)
    grids = np.meshgrid(*[np.arange(len(group.axes[a])) for a in axes], indexing="ij")
    rows = []
    for c, ch in enumerate(group.ch_names):
        for point in zip(*[g.ravel() for g in grids], strict=True):
            index = (c, *point)
            rows.append(
                [
                    ch,
                    *[round(float(group.axes[a][i]), 4) for a, i in zip(axes, point, strict=True)],
                    round(float(maps["stat"][index]), 4),
                    float(maps["p"][index]),
                    float(maps["p_corrected"][index]),
                    int(bool(maps["significant"][index])),
                ]
            )
    write_tsv(ctx, path, ["channel", *axes, "stat", "p", "p_corrected", "significant"], rows)
