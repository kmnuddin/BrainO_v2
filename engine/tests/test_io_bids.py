"""BIDS conversion and the whole import pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_bids")
pytest.importorskip("pybv")

import mne
import mne_bids
import numpy as np
import sample_data
from test_io_deidentify import PLANTED, planted_recording

from braino.io import (
    FORMATS,
    NotDeidentifiedError,
    Recording,
    SubjectKey,
    estimate_line_freq,
    import_recording,
    write_bids,
)

# The official validator, run with Deno when available (CI installs it). BRAINO_DENO may point
# to a deno executable that is not on PATH.
VALIDATOR = "jsr:@bids/validator@3.0.2"
DENO = os.environ.get("BRAINO_DENO") or shutil.which("deno")


def validate(root: Path) -> list[dict[str, str]]:
    """Errors the official BIDS validator reports for ``root``."""
    if DENO is None:
        pytest.skip("deno is not installed; the full BIDS validator cannot run")
    result = subprocess.run(
        [DENO, "run", "-A", VALIDATOR, "--json", str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    report = json.loads(result.stdout)
    return [i for i in report["issues"]["issues"] if i["severity"] == "error"]


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, SubjectKey]:
    source, bids = tmp_path / "Jane Doe records", tmp_path / "bids"
    source.mkdir()
    key = SubjectKey.open(tmp_path / "private" / "key.json", dataset_root=bids)
    return source, bids, key


def planted_source(folder: Path) -> Path:
    path = folder / "Jane_Doe_2024-03-12_raw.fif"
    planted_recording().raw.save(path, verbose="error")
    return path


def test_planted_details_are_gone_from_the_whole_dataset(
    dirs: tuple[Path, Path, SubjectKey],
) -> None:
    source, bids, key = dirs
    report = import_recording(
        planted_source(source), bids, key=key, original_id="Jane Doe", task="rest"
    )
    assert report.conversion.files
    for path in bids.rglob("*"):
        if path.is_file():
            content = path.read_bytes().lower()
            for value in PLANTED:
                assert value.lower().encode() not in content, (value, path.name)
    dumped = report.model_dump_json().lower()
    for value in PLANTED:
        assert value.lower() not in dumped, value


def test_data_are_written_not_copied(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    original = planted_source(source)
    report = import_recording(original, bids, key=key, original_id="Jane Doe", task="rest")
    assert report.conversion.format == "BrainVision"
    assert not any(f.endswith(".fif") for f in report.conversion.files)


def test_round_trip_keeps_data_and_events(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    original = planted_recording().raw
    import_recording(planted_source(source), bids, key=key, original_id="Jane Doe", task="rest")
    back = mne_bids.read_raw_bids(
        mne_bids.BIDSPath(subject="001", task="rest", datatype="eeg", root=bids), verbose="error"
    )
    assert np.allclose(back.get_data(), original.get_data(), atol=1e-12)
    assert len(back.annotations) == len(original.annotations)
    assert back.info["meas_date"].year < 1925


def test_template_positions_are_described(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    report = import_recording(
        planted_source(source), bids, key=key, original_id="Jane Doe", task="rest"
    )
    assert report.normalisation.positions == "template"
    [coordsystem] = bids.rglob("*_coordsystem.json")
    description = json.loads(coordsystem.read_text(encoding="utf-8"))
    assert "not measured" in description["EEGCoordinateSystemDescription"]


def test_existing_recording_is_not_overwritten_by_default(
    dirs: tuple[Path, Path, SubjectKey],
) -> None:
    source, bids, key = dirs
    path = planted_source(source)
    import_recording(path, bids, key=key, original_id="Jane Doe", task="rest")
    with pytest.raises(FileExistsError):
        import_recording(path, bids, key=key, original_id="Jane Doe", task="rest")
    again = import_recording(
        path, bids, key=key, original_id="Jane Doe", task="rest", overwrite=True
    )
    assert again.conversion.files


def test_sessions_and_runs(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    path = planted_source(source)
    report = import_recording(
        path, bids, key=key, original_id="Jane Doe", task="rest", session="pre", run=2
    )
    assert "sub-001/ses-pre/eeg/sub-001_ses-pre_task-rest_run-2_eeg.vhdr" in report.conversion.files


def test_second_run_of_a_subject_with_template_positions(
    dirs: tuple[Path, Path, SubjectKey],
) -> None:
    source, bids, key = dirs
    path = planted_source(source)
    for run in (1, 2):
        import_recording(path, bids, key=key, original_id="Jane Doe", task="rest", run=run)
    [coordsystem] = bids.rglob("*_coordsystem.json")
    description = json.loads(coordsystem.read_text(encoding="utf-8"))
    assert "not measured" in description["EEGCoordinateSystemDescription"]
    assert len(list(bids.rglob("*_eeg.vhdr"))) == 2


def test_recording_without_eeg_takes_no_label(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    info = mne.create_info(["ECG"], 100.0, ch_types="ecg")
    path = source / "heart_raw.fif"
    mne.io.RawArray(np.zeros((1, 500)), info, verbose="error").save(path, verbose="error")
    with pytest.raises(ValueError, match="no EEG, MEG or intracranial"):
        import_recording(path, bids, key=key, original_id="Jane Doe", task="rest")
    assert key.labels() == {}


def recording(raw: mne.io.BaseRaw) -> Recording:
    return Recording(raw, FORMATS["fif"], Path("rec.fif"), (Path("rec.fif"),))


def test_identifiable_recording_is_refused(tmp_path: Path) -> None:
    raw = planted_recording().raw  # real date and names, not de-identified
    with pytest.raises(NotDeidentifiedError):
        write_bids(recording(raw), tmp_path, subject="001", task="rest")


@pytest.mark.parametrize("label", ["sub_01", "pre-op", "01 a"])
def test_labels_must_be_alphanumeric(tmp_path: Path, label: str) -> None:
    raw = planted_recording().raw
    with pytest.raises(ValueError, match="letters and digits"):
        write_bids(recording(raw), tmp_path, subject=label, task="rest")


def sine_raw(freq: float | None, sfreq: float = 500.0, seconds: float = 20.0) -> mne.io.RawArray:
    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(0)
    data = 1e-6 * rng.standard_normal((4, times.size))
    if freq is not None:
        data += 5e-6 * np.sin(2 * np.pi * freq * times)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="error")


@pytest.mark.parametrize("freq", [50.0, 60.0, None])
def test_line_frequency_estimate(freq: float | None) -> None:
    assert estimate_line_freq(sine_raw(freq)) == freq


def test_line_frequency_needs_a_high_enough_sampling_rate() -> None:
    assert estimate_line_freq(sine_raw(50.0, sfreq=100.0)) is None


def test_line_frequency_given_wins(tmp_path: Path) -> None:
    raw = sine_raw(50.0)
    raw.set_meas_date(datetime(1910, 1, 1, tzinfo=UTC))
    report = write_bids(recording(raw), tmp_path, subject="001", task="rest", line_freq=60.0)
    assert (report.line_freq, report.line_freq_source) == (60.0, "given")
    sidecar = next(tmp_path.rglob("*_eeg.json"))
    assert json.loads(sidecar.read_text(encoding="utf-8"))["PowerLineFrequency"] == 60.0


def test_planted_dataset_passes_the_validator(dirs: tuple[Path, Path, SubjectKey]) -> None:
    source, bids, key = dirs
    import_recording(planted_source(source), bids, key=key, original_id="Jane Doe", task="rest")
    assert validate(bids) == []


@pytest.mark.network
def test_sample_recordings_make_a_valid_dataset(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    key = SubjectKey.open(tmp_path / "key.json", dataset_root=bids)
    imported = 0
    for fmt in sorted(sample_data.RECORDINGS):
        path = sample_data.fetch_recording(fmt)
        if fmt == "gdf":  # its only channel is an ECG: nothing to convert
            with pytest.raises(ValueError, match="no EEG"):
                import_recording(path, bids, key=key, original_id=fmt, task="test")
            continue
        report = import_recording(path, bids, key=key, original_id=fmt, task="test")
        assert report.conversion.files
        imported += 1
    assert imported == len(sample_data.RECORDINGS) - 1
    if DENO is not None:
        assert validate(bids) == []
