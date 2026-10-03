"""Checks that the test-data setup works: every sample recording downloads and MNE reads it."""

from __future__ import annotations

from functools import partial

import pytest

pytest.importorskip("mne")
pytest.importorskip("pooch")

import mne
import sample_data

# Formats MNE cannot detect on its own. The Neuroscan CNT header of this file reports 0 samples,
# so the sample width has to be given; BrainO's own reader dispatch (M1.1) must handle this.
READERS = {"cnt": partial(mne.io.read_raw_cnt, data_format="int16")}


def test_every_recording_has_files() -> None:
    for fmt, recording in sample_data.RECORDINGS.items():
        assert recording.main.startswith(recording.prefix.rstrip("/")), fmt
        assert any(name.startswith(recording.prefix) for name in sample_data.REGISTRY), fmt


@pytest.mark.network
@pytest.mark.parametrize("fmt", sorted(sample_data.RECORDINGS))
def test_sample_recording_downloads_and_reads(fmt: str) -> None:
    path = sample_data.fetch_recording(fmt)
    assert path.exists()
    raw = READERS.get(fmt, mne.io.read_raw)(path, preload=False, verbose="error")
    assert raw.info["sfreq"] > 0
    assert len(raw.ch_names) > 0
    assert raw.n_times > 0


def test_synthetic_raw(synthetic_raw: mne.io.RawArray) -> None:
    assert synthetic_raw.info["sfreq"] == 250.0
    assert len(synthetic_raw.ch_names) == 8
    assert set(synthetic_raw.annotations.description) == {"standard", "target"}
    assert synthetic_raw.get_montage() is not None
