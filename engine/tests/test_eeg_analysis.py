"""ICA, epoching, ERPs, spectra and time-frequency power, on synthetic data with known effects."""

from __future__ import annotations

import pytest

pytest.importorskip("mne")

import mne
import numpy as np

from braino.datasets.synthetic import EEG, P3_UV, P3_WINDOW, SFREQ, blink_weights, make_raw
from braino.eeg.epochs import EpochParams, make_epochs
from braino.eeg.erp import Contrast, Measure, condition_averages, measure
from braino.eeg.ica import fit_ica, remove_components
from braino.eeg.preprocess import PreprocessParams, apply_reference, preprocess
from braino.eeg.spectral import DEFAULT_BANDS, peak_alpha, spectra, summarise_bands
from braino.eeg.tfr import extremes, frequencies, morlet_power

CONDITIONS = {"rare": ["stimulus/rare"], "frequent": ["stimulus/frequent"]}


@pytest.fixture(scope="module")
def recording() -> mne.io.BaseRaw:
    raw = make_raw(seconds=120.0, blinks=60)
    out, _ = preprocess(raw, PreprocessParams(l_freq=0.1, h_freq=30.0, reference="average"))
    return out


def blink_size(raw: mne.io.BaseRaw) -> float:
    """Amplitude of the blink pattern in the data (projection on its topography)."""
    w = blink_weights()
    w = w - w.mean()
    return float(np.abs(w @ raw.get_data(picks=EEG)).max())


def test_ica_with_eog_labels_removes_blinks(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("picard")
    ica, result = fit_ica(recording, labeler="eog", seed=1)
    assert result.proposed, result
    assert all(result.components[i].label == "eye blink" for i in result.proposed)
    clean = remove_components(recording, ica, result.proposed, "average")
    assert blink_size(clean) < 0.3 * blink_size(recording)
    assert np.abs(clean.get_data(picks=EEG).mean(axis=0)).max() < 1e-12  # still average


def test_ica_with_iclabel(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("picard")
    pytest.importorskip("mne_icalabel")
    pytest.importorskip("onnxruntime")
    _, result = fit_ica(recording, labeler="iclabel", seed=1)
    assert len(result.components) == result.n_components
    assert {c.label for c in result.components} <= {
        "brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise",
        "other",
    }  # fmt: skip
    assert all(0 <= c.probability <= 1 for c in result.components)


def test_components_are_removed_in_the_data_reference(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("picard")
    mastoid = recording.copy()
    apply_reference(mastoid, ["P7", "P8"])
    ica, result = fit_ica(mastoid, labeler="eog", seed=1)
    clean = remove_components(mastoid, ica, result.proposed, ["P7", "P8"])
    assert np.abs(clean.get_data(picks=["P7", "P8"]).sum(axis=0)).max() < 1e-12
    with pytest.raises(ValueError, match="no components"):
        remove_components(mastoid, ica, [999], "average")


def test_ica_leaves_bad_channels_out(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("picard")
    raw = recording.copy()
    raw.info["bads"] = ["T7"]
    ica, _ = fit_ica(raw, labeler="eog", n_components=5)
    assert "T7" not in ica.ch_names
    assert ica.n_components_ == 5


def test_epochs_and_the_known_p3(recording: mne.io.BaseRaw) -> None:
    params = EpochParams(conditions=CONDITIONS, reject="none", baseline=(-0.2, 0.0))
    epochs, report = make_epochs(recording, params, reference="average")
    assert report.n_events == report.n_kept
    assert set(report.conditions) == {"rare", "frequent"}
    evokeds, notes = condition_averages(
        epochs, contrasts=[Contrast(name="difference", plus="rare", minus="frequent")]
    )
    assert notes == []
    spec = Measure(name="P3", channels=["Pz"], tmin=P3_WINDOW[0], tmax=P3_WINDOW[1])
    value = measure(evokeds["difference"], spec, "difference")
    # The average reference takes the mean of all channels' P3 off Pz.
    from braino.datasets.synthetic import p3_weights

    expected = P3_UV * (1 - p3_weights().mean())
    assert value.value_uv == pytest.approx(expected, rel=0.15)
    assert measure(evokeds["frequent"], spec, "frequent").value_uv == pytest.approx(0, abs=1.0)
    peak = measure(
        evokeds["difference"],
        Measure(name="peak", channels=["Pz", "P3", "P4"], tmin=0.2, tmax=0.7, kind="peak_positive"),
        "difference",
    )
    assert 350 <= (peak.latency_ms or 0) <= 450


def test_peak_to_peak_rejection_drops_artifact_epochs(recording: mne.io.BaseRaw) -> None:
    params = EpochParams(conditions=CONDITIONS, reject="peak_to_peak", peak_to_peak_uv=80)
    _, report = make_epochs(recording, params, reference="average")
    assert 0 < report.n_kept < report.n_events  # the blinks exceed 80 µV at Fp1/Fp2
    assert report.threshold_uv == 80


def test_autoreject(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("autoreject")
    short = recording.copy().crop(0, 50)
    params = EpochParams(conditions=CONDITIONS, reject="autoreject_global")
    _, report = make_epochs(short, params, reference="average")
    assert report.reject == "autoreject_global"
    assert report.threshold_uv is not None and report.threshold_uv > 0
    assert report.n_kept <= report.n_events


def test_too_few_epochs_for_autoreject_fall_back(recording: mne.io.BaseRaw) -> None:
    short = recording.copy().crop(0, 15)
    _, report = make_epochs(short, EpochParams(conditions=CONDITIONS), reference="average")
    assert report.reject == "peak_to_peak"
    assert any("too few" in n for n in report.notes)


def test_bad_channels_are_interpolated_first(recording: mne.io.BaseRaw) -> None:
    raw = recording.copy()
    raw._data[EEG.index("O2")] = 0.0
    raw.info["bads"] = ["O2"]
    epochs, report = make_epochs(
        raw, EpochParams(conditions=CONDITIONS, reject="none"), reference="average"
    )
    assert report.interpolated_channels == ["O2"]
    assert epochs.info["bads"] == []
    assert epochs.get_data(picks="O2").std() > 1e-7


def test_missing_events_and_bad_conditions(recording: mne.io.BaseRaw) -> None:
    params = EpochParams(conditions={"rare": ["stimulus/rare", "stimulus/missing"]}, reject="none")
    _, report = make_epochs(recording, params)
    assert report.missing_events == ["stimulus/missing"]
    with pytest.raises(ValueError, match="none of the requested events"):
        make_epochs(recording, EpochParams(conditions={"x": ["nope"]}, reject="none"))
    with pytest.raises(ValueError, match="both"):
        EpochParams(conditions={"a": ["e"], "b": ["e"]})
    with pytest.raises(ValueError, match="tmin"):
        EpochParams(conditions=CONDITIONS, tmin=0.5, tmax=0.1)


def test_measure_errors(recording: mne.io.BaseRaw) -> None:
    epochs, _ = make_epochs(recording, EpochParams(conditions=CONDITIONS, reject="none"))
    evokeds, _ = condition_averages(epochs)
    with pytest.raises(ValueError, match="no channel"):
        measure(evokeds["rare"], Measure(name="x", channels=["M1"], tmin=0, tmax=0.1), "rare")
    with pytest.raises(ValueError, match="outside the epoch"):
        measure(evokeds["rare"], Measure(name="x", channels=["Pz"], tmin=0, tmax=2), "rare")
    with pytest.raises(ValueError, match="no condition"):
        condition_averages(epochs, ["standard"])


def test_spectra_find_the_alpha_rhythm(recording: mne.io.BaseRaw) -> None:
    result = spectra(recording)
    s = result["continuous"]
    assert peak_alpha(s) == pytest.approx(10.0, abs=0.5)
    bands = {b.band: b for b in summarise_bands(s, DEFAULT_BANDS)}
    assert bands["alpha"].mean_absolute_uv2 > bands["beta"].mean_absolute_uv2 / 3
    assert sum(b.mean_relative for b in bands.values()) == pytest.approx(1.0, abs=0.1)

    epochs, _ = make_epochs(recording, EpochParams(conditions=CONDITIONS, reject="none"))
    by_condition = spectra(epochs, method="multitaper", fmin=2, fmax=30)
    assert set(by_condition) == {"rare", "frequent"}
    assert by_condition["rare"].n_segments == len(epochs["rare"])


def test_tfr_finds_a_beta_burst() -> None:
    raw = make_raw(seconds=80.0, blinks=0, p3_uv=0)
    t = raw.times
    burst = np.zeros_like(t)
    for onset in raw.annotations.onset:
        mask = (t >= onset + 0.3) & (t < onset + 0.6)
        burst[mask] = np.sin(2 * np.pi * 20 * t[mask]) * 6e-6
    raw._data[: len(EEG)] += burst
    epochs, _ = make_epochs(
        raw,
        EpochParams(
            conditions={"all": ["stimulus/rare", "stimulus/frequent"]},
            reject="none",
            tmin=-0.5,
            tmax=1.0,
        ),
    )
    tfrs, _ = morlet_power(epochs, freqs=frequencies(8, 30, 12), baseline=(-0.4, 0.0))
    found = {e.band: e for e in extremes(tfrs, {"beta": (13, 30)})}
    assert found["beta"].value > 3  # dB
    assert 300 <= found["beta"].time_ms <= 600
    assert found["beta"].freq_hz == pytest.approx(20, abs=3)


def test_tfr_refuses_wavelets_longer_than_the_epochs(recording: mne.io.BaseRaw) -> None:
    epochs, _ = make_epochs(recording, EpochParams(conditions=CONDITIONS, reject="none"))
    with pytest.raises(ValueError, match="raise fmin"):
        morlet_power(epochs, freqs=frequencies(2, 30, 5))


def test_sampling_rate_is_kept() -> None:
    assert SFREQ == 250.0


def test_event_delay_moves_the_effect_earlier(recording: mne.io.BaseRaw) -> None:
    peak = Measure(name="p", channels=["Pz"], tmin=0.2, tmax=0.7, kind="peak_positive")
    latencies = []
    for delay in (0.0, 0.1):
        params = EpochParams(conditions=CONDITIONS, reject="none", event_delay_s=delay)
        epochs, report = make_epochs(recording, params, reference="average")
        evokeds, _ = condition_averages(epochs)
        latencies.append(measure(evokeds["rare"], peak, "rare").latency_ms or 0)
    assert latencies[0] - latencies[1] == pytest.approx(100, abs=8)
    assert any("later" in n for n in report.notes)


def test_ica_keeps_the_average_with_eog_reference(recording: mne.io.BaseRaw) -> None:
    pytest.importorskip("picard")
    raw = recording.copy()
    apply_reference(raw, "average_with_eog")
    ica, result = fit_ica(raw, labeler="eog", seed=1)
    clean = remove_components(raw, ica, result.proposed, "average_with_eog")
    assert np.abs(clean.get_data(picks=["eeg", "eog"]).mean(axis=0)).max() < 1e-10
    assert blink_size(clean) < 0.3 * blink_size(raw)
