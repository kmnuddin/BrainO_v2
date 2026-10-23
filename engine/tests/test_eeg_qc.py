"""Data-quality screening, on synthetic recordings with known problems."""

from __future__ import annotations

import pytest

pytest.importorskip("mne")

import mne
import numpy as np

from braino.eeg.qc import compute_qc

SFREQ = 250.0
NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]


def clean(seconds: float = 30.0, sfreq: float = SFREQ, seed: int = 1) -> np.ndarray:
    """Background EEG: 1/f-like noise of about 10 µV with a 10 Hz rhythm."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    times = np.arange(n) / sfreq
    white = rng.standard_normal((len(NAMES), n))
    pink = np.cumsum(white, axis=1)
    pink -= np.linspace(pink[:, 0], pink[:, -1], n, axis=1)  # no drift across the recording
    pink *= 5e-6 / pink.std(axis=1, keepdims=True)
    return pink + 5e-6 * white + 8e-6 * np.sin(2 * np.pi * 10 * times)


def raw_from(
    data: np.ndarray, sfreq: float = SFREQ, bads: list[str] | None = None
) -> mne.io.RawArray:
    info = mne.create_info(NAMES, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.info["bads"] = bads or []
    return raw


def channel(name: str) -> int:
    return NAMES.index(name)


def test_clean_recording_has_no_findings() -> None:
    result = compute_qc(raw_from(clean()), line_freq=50.0)
    assert result.flat == []
    assert result.noisy == []
    assert result.strong_line_noise == []
    assert result.artifact_fraction == 0.0
    assert result.n_channels_checked == len(NAMES)
    assert len(result.channels) == len(NAMES)


def test_flat_channel() -> None:
    data = clean()
    data[channel("O1")] = 1e-9
    result = compute_qc(raw_from(data), line_freq=50.0)
    assert result.flat == ["O1"]
    assert "flat" in result.channels[channel("O1")].flags


def test_loud_channel_is_noisy() -> None:
    data = clean()
    data[channel("F4")] *= 8
    result = compute_qc(raw_from(data), line_freq=50.0)
    assert [n.channel for n in result.noisy] == ["F4"]
    assert "amplitude" in result.noisy[0].reason


def test_high_frequency_noise_is_noisy() -> None:
    data = clean()
    rng = np.random.default_rng(7)
    muscle = mne.filter.filter_data(
        rng.standard_normal(data.shape[1]) * 20e-6, SFREQ, 60.0, 110.0, verbose="error"
    )
    data[channel("P3")] += muscle
    result = compute_qc(raw_from(data), line_freq=50.0)
    noisy = {n.channel: n for n in result.noisy}
    assert "P3" in noisy
    assert "high-frequency" in noisy["P3"].reason


def test_line_noise() -> None:
    data = clean()
    times = np.arange(data.shape[1]) / SFREQ
    data[channel("C3")] += 20e-6 * np.sin(2 * np.pi * 50 * times)
    result = compute_qc(raw_from(data), line_freq=50.0)
    assert result.strong_line_noise == ["C3"]
    ratio = result.channels[channel("C3")].line_noise_ratio
    assert ratio is not None
    assert ratio > 10


def test_artifacts() -> None:
    data = clean()
    n = int(SFREQ)
    for second in (3, 11, 20):  # 3 of 30 one-second windows, mid-window (filtering smears an
        start = second * n + n // 2  # artifact at a window edge into the neighbouring window)
        data[channel("Fz"), start : start + 20] += 400e-6
    result = compute_qc(raw_from(data), line_freq=50.0)
    assert result.artifact_fraction == pytest.approx(0.1, abs=0.04)


def test_channels_marked_bad_do_not_count_for_artifacts() -> None:
    data = clean()
    data[channel("Fz"), 1000:1020] += 400e-6
    result = compute_qc(raw_from(data, bads=["Fz"]), line_freq=50.0)
    assert result.marked_bad_in_file == ["Fz"]
    assert result.artifact_fraction == 0.0


def test_slow_drift_is_not_noise() -> None:
    data = clean()
    data[channel("Pz")] += np.linspace(0, 2e-3, data.shape[1])  # 2 mV DC drift
    result = compute_qc(raw_from(data), line_freq=50.0)
    assert result.noisy == []
    assert result.artifact_fraction == 0.0


def test_unknown_line_frequency_and_low_sampling_rate() -> None:
    result = compute_qc(raw_from(clean()), line_freq=None)
    assert result.line_noise_median_ratio is None
    assert any("line frequency unknown" in n for n in result.notes)
    low = compute_qc(raw_from(clean(sfreq=64.0), sfreq=64.0), line_freq=50.0)
    assert any("below 100 Hz" in n for n in low.notes)


def test_short_recording_is_noted() -> None:
    result = compute_qc(raw_from(clean(seconds=5.0)), line_freq=50.0)
    assert any("rough" in n for n in result.notes)


def test_no_eeg_channels() -> None:
    info = mne.create_info(["ECG"], SFREQ, ch_types="ecg")
    raw = mne.io.RawArray(np.zeros((1, 2500)), info, verbose="error")
    result = compute_qc(raw)
    assert result.n_channels_checked == 0
    assert result.notes
