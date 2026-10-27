"""Per-subject data arranged for group statistics, and the tests run on it.

Data are arrays of shape ``(n_subjects, n_channels, *axes)``: one axis (time for ERPs, frequency
for spectra) or two (frequency and time for time-frequency power).

With one condition, the test is a one-sample t-test against zero (e.g. of a difference wave
already computed per subject); with two, a paired t-test (the same as a one-sample t-test of
their difference); with three or more, a one-way repeated-measures ANOVA (mass-univariate only).

Corrections for testing many points (channels x times x ...):

- ``fdr``: Benjamini-Hochberg false discovery rate.
- ``bonferroni``: family-wise error rate, conservative.
- ``max_stat``: family-wise error rate by permutation of the maximum |t| (Nichols & Holmes,
  2002); exact for any correlation between points, t-tests only.
- ``none``: uncorrected p-values (only for exploration; results must say so).

Cluster-based permutation tests (Maris & Oostenveld, 2007) group neighbouring points (in time,
frequency and between adjacent channels) that pass a threshold, and test each cluster's summed t
against the largest clusters found when the sign of each subject's effect is flipped at random.
They control the family-wise error rate for the claim "there is an effect", not for where it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Correction = Literal["fdr", "bonferroni", "max_stat", "none"]


@dataclass
class GroupData:
    conditions: list[str]
    data: list[Any]
    """One array per condition: subjects x channels x axes."""
    subjects: list[str]
    ch_names: list[str]
    axes: dict[str, Any]
    """Name -> coordinates of each axis after the channel axis, e.g. {"time": times}."""
    unit: str
    info: Any = None
    """MNE measurement info with channel positions, for channel adjacency."""
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data[0].shape)


class Segment(BaseModel):
    """A run of significant points on one channel along the first axis."""

    channel: str
    start: float
    end: float
    peak_stat: float
    min_p: float


class UnivariateResult(BaseModel):
    test: str
    correction: Correction
    alpha: float
    n_subjects: int
    n_tests: int
    n_significant: int
    min_p: float = Field(description="Smallest corrected p-value")
    peak_stat: float = Field(description="Largest statistic by absolute value")
    peak_channel: str
    peak_at: dict[str, float] = Field(description="Axis coordinates of the peak")
    segments: list[Segment] = Field(description="Significant runs, longest first")
    n_segments: int


class Cluster(BaseModel):
    p: float
    mass: float = Field(description="Sum of t over the cluster's points")
    sign: Literal["positive", "negative"]
    channels: list[str]
    ranges: dict[str, tuple[float, float]] = Field(description="Extent along each axis")
    n_points: int


class ClusterResult(BaseModel):
    test: str
    n_subjects: int
    n_permutations: int
    threshold_t: float
    alpha: float
    adjacency: str = Field(description="How channels are neighbours")
    clusters: list[Cluster] = Field(description="Clusters, most significant first")
    n_clusters: int
    n_significant: int


def univariate(
    group: GroupData,
    *,
    correction: Correction = "fdr",
    alpha: float = 0.05,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[UnivariateResult, dict[str, Any]]:
    """Test every point; returns the summary and full maps ``{"stat", "p", "p_corrected",
    "significant"}`` (each channels x axes)."""
    import mne
    import numpy as np
    from scipy import stats

    shape = group.shape[1:]
    n = group.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 subjects, have {n}")
    k = len(group.data)
    if k <= 2:
        x = group.data[0] if k == 1 else group.data[0] - group.data[1]
        test = (
            "one-sample t-test against 0"
            if k == 1
            else (f"paired t-test: {group.conditions[0]} vs {group.conditions[1]}")
        )
        flat = x.reshape(n, -1)
        stat, p = stats.ttest_1samp(flat, 0.0, axis=0)
        stat = np.nan_to_num(stat)
        p = np.nan_to_num(p, nan=1.0)
    else:
        if correction == "max_stat":
            raise ValueError("max_stat correction is for t-tests (one or two conditions)")
        test = f"repeated-measures ANOVA over {k} conditions"
        stacked = np.stack([d.reshape(n, -1) for d in group.data], axis=1)  # n x k x tests
        stat, p = mne.stats.f_mway_rm(stacked, factor_levels=[k], effects="A", correction=True)
        stat, p = np.nan_to_num(stat), np.nan_to_num(p, nan=1.0)

    if correction == "fdr":
        significant, p_corr = mne.stats.fdr_correction(p, alpha=alpha)
    elif correction == "bonferroni":
        significant, p_corr = mne.stats.bonferroni_correction(p, alpha=alpha)
    elif correction == "max_stat":
        _, p_corr, _ = mne.stats.permutation_t_test(
            flat, n_permutations=n_permutations, tail=0, rng=seed, verbose="error"
        )
        significant = p_corr < alpha
    else:
        p_corr, significant = p, p < alpha

    maps = {
        "stat": stat.reshape(shape),
        "p": p.reshape(shape),
        "p_corrected": np.asarray(p_corr).reshape(shape),
        "significant": np.asarray(significant).reshape(shape),
    }
    peak = np.unravel_index(int(np.argmax(np.abs(maps["stat"]))), shape)
    axis_names = list(group.axes)
    segments = _segments(group, maps)
    return (
        UnivariateResult(
            test=test,
            correction=correction,
            alpha=alpha,
            n_subjects=n,
            n_tests=int(np.prod(shape)),
            n_significant=int(np.asarray(significant).sum()),
            min_p=float(np.min(p_corr)),
            peak_stat=round(float(maps["stat"][peak]), 3),
            peak_channel=group.ch_names[peak[0]],
            peak_at={
                name: round(float(group.axes[name][peak[i + 1]]), 4)
                for i, name in enumerate(axis_names)
            },
            segments=segments[:10],
            n_segments=len(segments),
        ),
        maps,
    )


def cluster_test(
    group: GroupData,
    *,
    p_threshold: float = 0.05,
    alpha: float = 0.05,
    n_permutations: int = 1024,
    seed: int = 42,
) -> tuple[ClusterResult, Any]:
    """Cluster-based permutation t-test; returns the summary and a cluster label map (channels
    x axes; 0 = no cluster, i = the i-th cluster of the summary)."""
    import mne
    import numpy as np
    from scipy import stats

    n = group.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 subjects, have {n}")
    if len(group.data) > 2:
        raise ValueError("cluster tests here compare one condition with 0, or two conditions")
    x = group.data[0] if len(group.data) == 1 else group.data[0] - group.data[1]
    test = (
        "one-sample t-test against 0"
        if len(group.data) == 1
        else (f"paired t-test: {group.conditions[0]} vs {group.conditions[1]}")
    )
    # MNE wants channels last: subjects x axes x channels.
    moved = np.moveaxis(x, 1, -1)
    axis_sizes = moved.shape[1:-1]
    ch_adjacency, adjacency_text = _channel_adjacency(group)
    adjacency = mne.stats.combine_adjacency(*axis_sizes, ch_adjacency)
    threshold = float(stats.t.ppf(1 - p_threshold / 2, n - 1))
    t_obs, clusters, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
        moved,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,
        adjacency=adjacency,
        out_type="mask",
        rng=seed,
        verbose="error",
    )
    order = np.argsort(cluster_p, kind="stable")
    labels = np.zeros(moved.shape[1:], dtype=int)
    summary = []
    axis_names = list(group.axes)
    for rank, i in enumerate(order, start=1):
        mask = clusters[i]
        labels[mask] = rank
        idx = np.nonzero(mask)
        mass = float(t_obs[mask].sum())
        ranges = {
            name: (
                round(float(group.axes[name][idx[a].min()]), 4),
                round(float(group.axes[name][idx[a].max()]), 4),
            )
            for a, name in enumerate(axis_names)
        }
        summary.append(
            Cluster(
                p=float(cluster_p[i]),
                mass=round(mass, 2),
                sign="positive" if mass > 0 else "negative",
                channels=[group.ch_names[c] for c in sorted(set(idx[-1].tolist()))],
                ranges=ranges,
                n_points=int(mask.sum()),
            )
        )
    return (
        ClusterResult(
            test=test,
            n_subjects=n,
            n_permutations=n_permutations,
            threshold_t=round(threshold, 3),
            alpha=alpha,
            adjacency=adjacency_text,
            clusters=summary[:10],
            n_clusters=len(summary),
            n_significant=sum(c.p < alpha for c in summary),
        ),
        np.moveaxis(labels, -1, 0),
    )


def _channel_adjacency(group: GroupData) -> tuple[Any, str]:
    import mne
    from scipy import sparse

    n = len(group.ch_names)
    if group.info is not None:
        info = group.info.copy()
        picked = mne.pick_info(info, [info.ch_names.index(c) for c in group.ch_names])
        try:
            with mne.utils.use_log_level("error"):
                adjacency, names = mne.channels.find_ch_adjacency(picked, ch_type="eeg")
        except (ValueError, RuntimeError):
            pass
        else:
            if list(names) == group.ch_names:
                return adjacency, "neighbouring electrodes (Delaunay triangulation of positions)"
    group.notes.append("No electrode positions: channels are not neighbours of each other.")
    return sparse.csr_matrix((n, n)), "none (each channel on its own)"


def _segments(group: GroupData, maps: dict[str, Any]) -> list[Segment]:
    """Runs of significant points along the first axis, per channel (and per second-axis
    value for two-axis data, which are reported at their first-axis extent)."""
    import numpy as np

    significant = maps["significant"]
    first = next(iter(group.axes))
    coords = group.axes[first]
    out = []
    for c, channel in enumerate(group.ch_names):
        sig = significant[c]
        if sig.ndim > 1:
            sig = sig.any(axis=tuple(range(1, sig.ndim)))
        stat = maps["stat"][c]
        p = maps["p_corrected"][c]
        if stat.ndim > 1:
            stat = np.take_along_axis(
                stat.reshape(stat.shape[0], -1),
                np.abs(stat.reshape(stat.shape[0], -1)).argmax(axis=1)[:, None],
                axis=1,
            )[:, 0]
            p = p.reshape(p.shape[0], -1).min(axis=1)
        start = None
        for i, flag in enumerate([*sig.tolist(), False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                run = slice(start, i)
                k = start + int(np.argmax(np.abs(stat[run])))
                out.append(
                    Segment(
                        channel=channel,
                        start=round(float(coords[start]), 4),
                        end=round(float(coords[i - 1]), 4),
                        peak_stat=round(float(stat[k]), 3),
                        min_p=float(p[run].min()),
                    )
                )
                start = None
    return sorted(out, key=lambda s: (-(s.end - s.start), s.min_p))
