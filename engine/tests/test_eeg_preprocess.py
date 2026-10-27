"""Preprocessing and bad-channel detection, on synthetic recordings with known content."""

from __future__ import annotations

import pytest

pytest.importorskip("mne")

import mne
import numpy as np

from braino.datasets.synthetic import EEG, SFREQ, make_raw
from braino.eeg.bad_channels import find_bad_channels
from braino.eeg.preprocess import PreprocessParams, apply_reference, preprocess


def power_at(raw: mne.io.BaseRaw, freq: float, ch: str = "Cz") -> float:
    spectrum = raw.compute_psd(picks=[ch], fmin=freq - 1, fmax=freq + 1, verbose="error")
    return float(spectrum.get_data().max())


def sines(freqs: list[float], seconds: float = 20.0) -> mne.io.RawArray:
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    signal = sum(10e-6 * np.sin(2 * np.pi * f * t) for f in freqs)
    data = np.tile(signal, (3, 1)) * np.array([[1.0], [0.5], [0.25]])
    info = mne.create_info(["Cz", "Pz", "STI"], SFREQ, ["eeg", "eeg", "stim"])
    return mne.io.RawArray(data, info, verbose="error")


def test_band_pass_removes_frequencies_outside_the_band() -> None:
    raw = sines([0.05, 10.0, 80.0], seconds=60.0)
    out, report = preprocess(raw, PreprocessParams(l_freq=1.0, h_freq=40.0, reference="none"))
    assert power_at(out, 10.0) == pytest.approx(power_at(raw, 10.0), rel=0.05)
    assert power_at(out, 80.0) < 1e-3 * power_at(raw, 80.0)
    assert out.get_data(picks="Cz").std() < raw.get_data(picks="Cz").std()
    assert report.dropped_channels == ["STI"]
    assert "band-pass 1-40 Hz (fir, zero-phase)" in report.steps


def test_iir_filter_and_notch() -> None:
    raw = sines([10.0, 50.0, 100.0])
    params = PreprocessParams(
        l_freq=None, h_freq=None, filter_method="iir", notch_line_noise=True, reference="none"
    )
    out, report = preprocess(raw, params, line_freq=50.0)
    assert power_at(out, 50.0) < 1e-2 * power_at(raw, 50.0)
    assert power_at(out, 100.0) < 1e-2 * power_at(raw, 100.0)
    assert power_at(out, 10.0) == pytest.approx(power_at(raw, 10.0), rel=0.05)
    assert any(step.startswith("notch 50, 100") for step in report.steps)


def test_unknown_line_frequency_is_noted() -> None:
    _, report = preprocess(sines([10.0]), PreprocessParams(notch_line_noise=True, reference="none"))
    assert any("unknown" in n for n in report.notes)


def test_resample() -> None:
    out, report = preprocess(sines([10.0]), PreprocessParams(resample_hz=100.0, reference="none"))
    assert out.info["sfreq"] == 100.0
    assert report.sfreq_hz == 100.0


def test_average_reference_leaves_out_bad_channels() -> None:
    raw = make_raw(seconds=10.0)
    raw.info["bads"] = ["T8"]
    out, report = preprocess(raw, PreprocessParams(l_freq=None, h_freq=None))
    good = [ch for ch in EEG if ch != "T8"]
    assert np.abs(out.get_data(picks=good).mean(axis=0)).max() < 1e-12
    assert report.reference == "average"
    assert report.bad_channels == ["T8"]


def test_channel_reference() -> None:
    raw = make_raw(seconds=10.0)
    out, _ = preprocess(raw, PreprocessParams(l_freq=None, h_freq=None, reference=["P7", "P8"]))
    data = out.get_data(picks=["P7", "P8"])
    assert np.abs(data.sum(axis=0)).max() < 1e-12  # each is minus the other


def test_reference_can_be_reapplied_whatever_the_previous_one() -> None:
    raw = make_raw(seconds=5.0).load_data()
    a = raw.copy()
    apply_reference(a, ["P7", "P8"])
    b = raw.copy()
    apply_reference(b, "average")
    apply_reference(b, ["P7", "P8"])
    np.testing.assert_allclose(a.get_data(), b.get_data(), atol=1e-15)


def test_reference_errors() -> None:
    raw = make_raw(seconds=5.0).load_data()
    with pytest.raises(ValueError, match="not among"):
        apply_reference(raw, ["M1"])
    raw.info["bads"] = ["P7"]
    with pytest.raises(ValueError, match="marked bad"):
        apply_reference(raw, ["P7", "P8"])
    with pytest.raises(ValueError, match="at least one"):
        PreprocessParams(reference=[])
    with pytest.raises(ValueError, match="below low-pass"):
        preprocess(raw, PreprocessParams(l_freq=30.0, h_freq=20.0))


@pytest.mark.parametrize("method", ["pyprep", "qc"])
def test_flat_and_noisy_channels_are_found(method: str) -> None:
    pytest.importorskip("pyprep")
    raw = make_raw(seconds=60.0, flat=["O1"], noisy=["T7"], blinks=0)
    result = find_bad_channels(raw, method=method)  # type: ignore[arg-type]
    assert "O1" in result.channels
    assert "T7" in result.channels
    assert len(result.channels) <= 4
    assert result.n_checked == len(EEG)


def test_detection_does_not_depend_on_the_reference() -> None:
    pytest.importorskip("pyprep")
    raw = make_raw(seconds=60.0, noisy=["C4"], blinks=0).load_data()
    mastoid = raw.copy()
    apply_reference(mastoid, ["P7", "P8"])
    a = find_bad_channels(raw, ransac=False)
    b = find_bad_channels(mastoid, ransac=False)
    assert a.channels == b.channels
    assert "P7" not in b.channels and "P8" not in b.channels


def test_already_marked_channels_stay_bad() -> None:
    raw = make_raw(seconds=30.0, blinks=0)
    raw.info["bads"] = ["Cz"]
    result = find_bad_channels(raw, method="qc")
    assert "Cz" in result.channels
    assert result.reasons["Cz"][0] == "already marked bad"
    assert result.already_marked == ["Cz"]


def test_blinks_are_left_out_of_detection() -> None:
    raw = make_raw(seconds=60.0, blinks=30)
    result = find_bad_channels(raw, method="qc")
    assert not {"Fp1", "Fp2"} & set(result.channels)
    assert any("blinks" in n and "left out" in n for n in result.notes)


def test_average_with_eog_reference() -> None:
    raw = make_raw(seconds=10.0).load_data()
    once = raw.copy()
    apply_reference(once, "average_with_eog")
    data = once.get_data(picks=["eeg", "eog"])
    assert np.abs(data.mean(axis=0)).max() < 1e-12
    # Applied again (e.g. after interpolating channels) it changes nothing. It must not follow
    # a reference that changed only the EEG channels: EEG and EOG must share their reference.
    twice = once.copy()
    apply_reference(twice, "average_with_eog")
    np.testing.assert_allclose(once.get_data(), twice.get_data(), atol=1e-15)
