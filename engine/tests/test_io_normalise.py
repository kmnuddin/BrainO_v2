"""Normalising channel names, types, positions and events."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("mne")

import mne
import numpy as np
import sample_data

from braino.io import FORMATS, Recording, normalise, read_raw


def recording(
    names: list[str],
    types: str | list[str] = "eeg",
    *,
    data: np.ndarray | None = None,
    first_samp: int = 0,
    meas_date: datetime | None = None,
) -> Recording:
    sfreq = 100.0
    info = mne.create_info(names, sfreq, ch_types=types)
    raw = mne.io.RawArray(
        np.zeros((len(names), 1000)) if data is None else data,
        info,
        first_samp=first_samp,
        verbose="error",
    )
    raw.set_meas_date(meas_date)
    return Recording(raw, FORMATS["fif"], Path("rec.fif"), (Path("rec.fif"),))


def test_standard_names_get_standard_spelling() -> None:
    rec, report = normalise(recording(["FP1", "EEG Cz-REF", "fz", "EEG 001", "Weird"]))
    assert rec.raw.ch_names == ["Fp1", "Cz", "Fz", "EEG 001", "Weird"]
    assert report.renamed == {"FP1": "Fp1", "EEG Cz-REF": "Cz", "fz": "Fz"}


def test_rename_that_would_clash_is_skipped() -> None:
    rec, report = normalise(recording(["FZ", "Fz", "CZ"]))
    assert rec.raw.ch_names == ["FZ", "Fz", "Cz"]
    assert any("FZ" in note for note in report.notes)


def test_bipolar_names_are_kept() -> None:
    rec, _ = normalise(recording(["Fp1-F7", "F7-T7", "Cz"]))
    assert rec.raw.ch_names[:2] == ["Fp1-F7", "F7-T7"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("VEOG", "eog"),
        ("HEOGR", "eog"),
        ("LOC", "eog"),
        ("ECG", "ecg"),
        ("EKG2", "ecg"),
        ("POL EKG1", "ecg"),
        ("EMG chin", "emg"),
        ("Chin1", "emg"),
        ("Resp", "resp"),
        ("POL X1", "misc"),
        ("Cz", "eeg"),
    ],
)
def test_channel_types_from_names(name: str, expected: str) -> None:
    rec, report = normalise(recording(["Fz", name]))
    assert rec.raw.get_channel_types(picks=[name])[0] == expected
    assert (name in report.retyped) == (expected != "eeg")


def test_only_eeg_channels_are_retyped() -> None:
    rec, report = normalise(recording(["Fz", "ECG"], ["eeg", "misc"]))
    assert rec.raw.get_channel_types() == ["eeg", "misc"]
    assert report.retyped == {}


def test_positions_from_the_file_are_kept(synthetic_raw: mne.io.RawArray) -> None:
    before = synthetic_raw.get_montage().get_positions()["ch_pos"]["Cz"]
    rec = Recording(synthetic_raw, FORMATS["fif"], Path("x.fif"), (Path("x.fif"),))
    normalised, report = normalise(rec)
    assert report.positions == "file"
    assert report.template is None
    after = normalised.raw.get_montage().get_positions()["ch_pos"]["Cz"]
    assert np.allclose(before, after)


def test_10_05_names_get_template_positions() -> None:
    rec, report = normalise(recording(["Fp1", "Cz", "O2", "X1"]))
    assert report.positions == "template"
    assert report.template == "colin27_1005"
    assert report.eeg_without_position == ["X1"]
    positions = rec.raw.get_montage().get_positions()["ch_pos"]
    assert np.isfinite(positions["Cz"]).all()


def test_complete_biosemi_cap_gets_its_layout() -> None:
    names = [f"{block}{i}" for block in "ABCD" for i in range(1, 33)]
    _, report = normalise(recording(names))
    assert report.template == "biosemi128"


def test_partial_cap_names_get_no_positions() -> None:
    # BioSemi-style names, but not any BioSemi cap: a template would put them in wrong places.
    names = [f"{block}{i}" for block in "ABCDEFGH" for i in range(1, 17)]
    _, report = normalise(recording(names))
    assert report.positions == "none"
    assert len(report.eeg_without_position) == len(names)


def test_numbered_channels_get_no_positions_unless_asked() -> None:
    names = [str(i) for i in range(1, 65)]
    _, report = normalise(recording(names))
    assert report.positions == "none"
    _, report = normalise(recording(names), montage="easycap-M43")
    assert report.positions == "template"
    assert report.template == "easycap-M43"


def test_positions_can_be_skipped() -> None:
    _, report = normalise(recording(["Fp1", "Cz"]), montage=None)
    assert report.positions == "none"
    assert report.eeg_without_position == ["Fp1", "Cz"]


def trigger_data(n_channels: int, samples: dict[int, int], channel: int) -> np.ndarray:
    data = np.zeros((n_channels, 1000))
    for sample, code in samples.items():
        data[channel, sample : sample + 3] = code
    return data


@pytest.mark.parametrize("meas_date", [None, datetime(2024, 1, 2, tzinfo=UTC)])
@pytest.mark.parametrize("first_samp", [0, 500])
def test_trigger_events_become_annotations(meas_date: datetime | None, first_samp: int) -> None:
    triggers = {100: 1, 400: 2, 700: 1}
    rec = recording(
        ["Cz", "STI 014"],
        ["eeg", "stim"],
        data=trigger_data(2, triggers, 1),
        first_samp=first_samp,
        meas_date=meas_date,
    )
    normalised, report = normalise(rec)
    assert report.events_from_stim == {"STI 014": 3}
    assert report.events == {"1": 2, "2": 1}
    events, ids = mne.events_from_annotations(normalised.raw, verbose="error")
    found = {int(s) - first_samp: int(c) for s, c in zip(events[:, 0], events[:, 2], strict=True)}
    by_code = {ids[str(code)]: code for code in (1, 2)}
    assert {s: by_code[c] for s, c in found.items()} == triggers


def test_composite_trigger_channel_is_preferred() -> None:
    data = trigger_data(3, {100: 5}, 1)
    data[2] = data[1]  # STI 001 carries the same event as bit 1 of STI 014
    rec = recording(["Cz", "STI 014", "STI 001"], ["eeg", "stim", "stim"], data=data)
    _, report = normalise(rec)
    assert report.events_from_stim == {"STI 014": 1}


def test_triggers_already_annotated_are_not_read_again() -> None:
    rec = recording(["Cz", "DIN1"], ["eeg", "stim"], data=trigger_data(2, {100: 1}, 1))
    rec.raw.set_annotations(mne.Annotations([1.0], [0.0], ["DIN1"]))
    _, report = normalise(rec)
    assert report.events_from_stim == {}
    assert report.events == {"DIN1": 1}


def test_requested_stim_channels() -> None:
    rec = recording(["Cz", "TRIG"], ["eeg", "misc"], data=trigger_data(2, {100: 3}, 1))
    _, report = normalise(rec, stim_channels=["TRIG"])
    assert report.events_from_stim == {"TRIG": 1}


def test_overrides_win() -> None:
    rec, report = normalise(
        recording(["FP1", "X7"]),
        rename={"X7": "VEOG_upper"},
        channel_types={"VEOG_upper": "eog"},
    )
    assert rec.raw.ch_names == ["Fp1", "VEOG_upper"]
    assert report.renamed["X7"] == "VEOG_upper"
    assert report.retyped["VEOG_upper"].reason == "requested"


def test_original_is_unchanged() -> None:
    original = recording(["FP1", "ECG"])
    normalised, _ = normalise(original)
    assert original.raw.ch_names == ["FP1", "ECG"]
    assert original.raw.get_channel_types() == ["eeg", "eeg"]
    assert normalised.raw is not original.raw


@pytest.mark.network
@pytest.mark.parametrize("fmt", sorted(sample_data.RECORDINGS))
def test_sample_recordings(fmt: str) -> None:
    original = read_raw(sample_data.fetch_recording(fmt))
    normalised, report = normalise(original)
    assert sum(report.channel_types.values()) == len(normalised.raw.ch_names)
    assert set(report.notes) <= set(normalised.notes)
    if report.positions == "template":
        assert report.template is not None


@pytest.mark.network
def test_sample_fif_trigger_events() -> None:
    original = read_raw(sample_data.fetch_recording("fif"))
    expected = mne.find_events(original.raw, "STI 014", shortest_event=1, verbose="error")
    normalised, report = normalise(original)
    assert report.events_from_stim == {"STI 014": len(expected)}
    events, _ = mne.events_from_annotations(normalised.raw, verbose="error")
    assert np.array_equal(events[:, 0], expected[:, 0])


@pytest.mark.network
def test_sample_biosemi_like_edf_gets_no_template() -> None:
    _, report = normalise(read_raw(sample_data.fetch_recording("edf")))
    assert report.positions == "none"
