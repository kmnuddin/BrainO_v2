"""Reading every supported format from the public sample recordings."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mne")
pytest.importorskip("pooch")

import mne
import sample_data
from pydantic import BaseModel

from braino.io import RecordingReadError, read_raw
from braino.tools import Risk, RunContext, ToolRegistry


class Empty(BaseModel):
    pass


@pytest.mark.network
@pytest.mark.parametrize("fmt", sorted(sample_data.RECORDINGS))
def test_reads_sample_recording(fmt: str) -> None:
    path = sample_data.fetch_recording(fmt)
    recording = read_raw(path)
    assert recording.format.id == fmt
    assert recording.path == path
    if fmt != "egi_mff":  # an MFF recording lists the files inside its folder instead
        assert recording.files[0] == path
    assert all(p.is_file() for p in recording.files)
    assert isinstance(recording.raw, mne.io.BaseRaw)
    assert recording.raw.n_times > 0


@pytest.mark.network
@pytest.mark.parametrize("fmt", sorted(sample_data.RECORDINGS))
def test_recording_files_can_be_recorded_in_provenance(fmt: str) -> None:
    recording = read_raw(sample_data.fetch_recording(fmt))
    reg = ToolRegistry()

    @reg.tool(name="test.record_inputs", risk=Risk.READ)
    def record_inputs(params: Empty, ctx: RunContext) -> Empty:
        """Record the recording's files as inputs."""
        for path in recording.files:
            ctx.record_input(path)
        return Empty()

    ctx = RunContext()
    reg.get("test.record_inputs").run({}, ctx)
    assert len(ctx.provenance.calls[0].inputs) == len(recording.files)


@pytest.mark.network
def test_reads_from_companion_file() -> None:
    header = sample_data.fetch_recording("brainvision")
    recording = read_raw(header.with_suffix(".eeg"))
    assert recording.path == header
    assert len(recording.files) == 3


@pytest.mark.network
def test_cnt_sample_width_is_inferred_and_reported() -> None:
    path = sample_data.fetch_recording("cnt")
    recording = read_raw(path)
    assert any("read as int16" in note for note in recording.notes)
    explicit = read_raw(path, cnt_data_format="int16")
    assert explicit.raw.n_times == recording.raw.n_times
    assert not any("sample width" in note for note in explicit.notes)


@pytest.mark.network
def test_preload() -> None:
    recording = read_raw(sample_data.fetch_recording("edf"), preload=True)
    assert recording.raw.preload


def test_unreadable_recording(tmp_path: Path) -> None:
    path = tmp_path / "broken.edf"
    path.write_bytes(b"0       " + b"\x00" * 64)
    with pytest.raises(RecordingReadError, match="could not read European Data Format"):
        read_raw(path)


def test_epoched_eeglab_file_is_reported(tmp_path: Path) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    import numpy as np

    path = tmp_path / "epochs.set"
    eeg = {
        "nbchan": 2,
        "trials": 3,
        "pnts": 10,
        "srate": 100.0,
        "xmin": 0.0,
        "data": np.zeros((2, 10, 3)),
        "chanlocs": np.array([]),
        "event": np.array([]),
        "epoch": np.array([]),
    }
    scipy_io.savemat(path, {"EEG": eeg}, appendmat=False)
    with pytest.raises(RecordingReadError, match="trials"):
        read_raw(path)
