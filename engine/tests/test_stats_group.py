"""Group statistics on simulated subjects: a planted effect must be found where it is, and null
data must not give more false positives than the correction allows."""

from __future__ import annotations

import pytest

pytest.importorskip("mne")
pytest.importorskip("scipy")

import mne
import numpy as np

from braino.stats.group import GroupData, cluster_test, univariate

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4", "O1", "O2", "F3", "F4"]
TIMES = np.linspace(-0.1, 0.6, 71)
EFFECT_CHANNELS = ["Pz", "P3", "P4"]
EFFECT_TIMES = (0.3, 0.45)


def info() -> mne.Info:
    out = mne.create_info(CHANNELS, 100.0, "eeg")
    out.set_montage("colin27_1020")
    return out


def simulate(n: int = 20, effect: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two conditions per subject (subjects x channels x times); B - A has ``effect`` over the
    effect channels and times."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, len(CHANNELS), TIMES.size))
    b = a + rng.standard_normal(a.shape)
    mask = np.zeros((len(CHANNELS), TIMES.size), dtype=bool)
    rows = [CHANNELS.index(c) for c in EFFECT_CHANNELS]
    cols = (EFFECT_TIMES[0] <= TIMES) & (EFFECT_TIMES[1] >= TIMES)
    mask[np.ix_(np.array(rows), cols)] = True
    b[:, mask] += effect
    return a, b


def group(data: list[np.ndarray], names: list[str]) -> GroupData:
    return GroupData(
        conditions=names,
        data=data,
        subjects=[f"{i:02d}" for i in range(data[0].shape[0])],
        ch_names=CHANNELS,
        axes={"time": TIMES},
        unit="µV",
        info=info(),
    )


@pytest.mark.parametrize("correction", ["fdr", "bonferroni", "max_stat"])
def test_paired_test_finds_the_effect_where_it_is(correction: str) -> None:
    a, b = simulate()
    result, maps = univariate(group([b, a], ["b", "a"]), correction=correction)  # type: ignore[arg-type]
    assert result.test.startswith("paired")
    assert result.n_tests == len(CHANNELS) * TIMES.size
    assert result.peak_channel in EFFECT_CHANNELS
    assert EFFECT_TIMES[0] <= result.peak_at["time"] <= EFFECT_TIMES[1]
    sig = maps["significant"]
    rows = [CHANNELS.index(c) for c in EFFECT_CHANNELS]
    others = [i for i in range(len(CHANNELS)) if i not in rows]
    assert sig[rows].sum() > 0.8 * sig.sum()
    assert sig[others].sum() <= 2
    assert {s.channel for s in result.segments} <= set(EFFECT_CHANNELS) | {
        CHANNELS[i] for i in others if sig[i].any()
    }


def test_one_sample_equals_paired_on_the_difference() -> None:
    a, b = simulate()
    paired, _ = univariate(group([b, a], ["b", "a"]))
    single, _ = univariate(group([b - a], ["difference"]))
    assert paired.peak_stat == pytest.approx(single.peak_stat)
    assert single.test == "one-sample t-test against 0"


def test_null_data_have_few_false_positives() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20, len(CHANNELS), TIMES.size))
    result, _ = univariate(group([x], ["null"]), correction="max_stat", n_permutations=500)
    assert result.n_significant <= 2
    uncorrected, _ = univariate(group([x], ["null"]), correction="none")
    assert uncorrected.n_significant > result.n_significant


def test_anova_for_three_conditions() -> None:
    a, b = simulate()
    c = a.copy()
    result, maps = univariate(group([a, b, c], ["a", "b", "c"]))
    assert result.test.startswith("repeated-measures ANOVA")
    assert result.peak_channel in EFFECT_CHANNELS
    with pytest.raises(ValueError, match="max_stat"):
        univariate(group([a, b, c], ["a", "b", "c"]), correction="max_stat")
    assert maps["stat"].shape == (len(CHANNELS), TIMES.size)


def test_cluster_test_finds_one_cluster_at_the_effect() -> None:
    a, b = simulate(effect=1.5)
    result, labels = cluster_test(group([b, a], ["b", "a"]), n_permutations=500)
    assert result.n_significant >= 1
    top = result.clusters[0]
    assert top.p < 0.05
    assert top.sign == "positive"
    assert set(EFFECT_CHANNELS) <= set(top.channels)
    start, end = top.ranges["time"]
    assert start <= EFFECT_TIMES[0] + 0.05 and end >= EFFECT_TIMES[1] - 0.05
    assert "neighbouring electrodes" in result.adjacency
    assert labels.shape == (len(CHANNELS), TIMES.size)
    assert (labels == 1).sum() == top.n_points


def test_cluster_test_on_null_data() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20, len(CHANNELS), TIMES.size))
    result, _ = cluster_test(group([x], ["null"]), n_permutations=500)
    assert result.n_significant == 0


def test_cluster_test_over_frequency_and_time() -> None:
    rng = np.random.default_rng(1)
    freqs = np.arange(4, 30, 2.0)
    x = rng.standard_normal((15, len(CHANNELS), freqs.size, 20))
    rows = [CHANNELS.index(c) for c in EFFECT_CHANNELS]
    x[:, rows, 2:5, 8:14] += 1.5
    data = GroupData(
        conditions=["power"], data=[x], subjects=[str(i) for i in range(15)], ch_names=CHANNELS,
        axes={"freq": freqs, "time": np.linspace(0, 1, 20)}, unit="dB", info=info(),
    )  # fmt: skip
    result, _ = cluster_test(data, n_permutations=300)
    top = result.clusters[0]
    assert top.p < 0.05
    assert set(EFFECT_CHANNELS) <= set(top.channels)
    assert top.ranges["freq"][0] <= 8 and top.ranges["freq"][1] >= 12


def test_without_positions_channels_are_not_neighbours() -> None:
    a, b = simulate(effect=1.5)
    g = group([b, a], ["b", "a"])
    g.info = None
    result, _ = cluster_test(g, n_permutations=200)
    assert result.adjacency.startswith("none")
    assert all(len(c.channels) == 1 for c in result.clusters)


def test_needs_three_subjects() -> None:
    a, b = simulate(n=2)
    with pytest.raises(ValueError, match="at least 3"):
        univariate(group([b, a], ["b", "a"]))
