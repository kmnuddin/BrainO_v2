"""De-identification: planted personal details must be gone from everything written out."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("mne")

import mne
import numpy as np
import sample_data

from braino.io import (
    FORMATS,
    Recording,
    SubjectKey,
    SubjectKeyError,
    deidentify,
    read_raw,
)

MEAS_DATE = datetime(2024, 3, 12, 9, 30, tzinfo=UTC)
BIRTHDAY = date(1980, 5, 17)

# Every value planted below, as it could appear in written bytes.
PLANTED = [
    "Jane",
    "Doe",
    "Quinn",
    "MRN12345678",
    "Dr Watson",
    "St Mary",
    "SN-998877",
    "Ward 7",
    "555-123-4567",
    "jane.doe@example.org",
    "2024-03-12",
    "1980-05-17",
]


def planted_recording() -> Recording:
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((3, 1000)), info, verbose="error")
    raw.set_meas_date(MEAS_DATE)
    raw.info["subject_info"] = {
        "id": 42,
        "his_id": "MRN12345678",
        "first_name": "Jane",
        "middle_name": "Quinn",
        "last_name": "Doe",
        "birthday": BIRTHDAY,
        "sex": 2,
        "hand": 1,
        "weight": 61.0,
        "height": 1.68,
    }
    raw.info["experimenter"] = "Dr Watson"
    raw.info["description"] = "Resting EEG of Jane Doe"
    raw.info["proj_name"] = "Sleep study at St Mary"
    raw.info["device_info"] = {"type": "TestAmp", "serial": "SN-998877", "site": "Ward 7"}
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 4.0, 5.0],
            duration=0.0,
            description=[
                "target",
                "Jane said she felt tired",
                "call 555-123-4567 if late",
                "note 2024-03-12 from jane.doe@example.org",
                "Stimulus/S  1",
            ],
            orig_time=MEAS_DATE,
        )
    )
    return Recording(raw, FORMATS["fif"], Path("Jane_Doe.fif"), (Path("Jane_Doe.fif"),))


@pytest.fixture
def key(tmp_path: Path) -> SubjectKey:
    (tmp_path / "dataset").mkdir()
    return SubjectKey.open(
        tmp_path / "private" / "subjects.json", dataset_root=tmp_path / "dataset"
    )


def test_nothing_planted_survives_in_the_written_file(key: SubjectKey, tmp_path: Path) -> None:
    clean, report = deidentify(planted_recording(), key, "Jane Doe")
    out = tmp_path / "dataset" / "sub-001_eeg.fif"
    clean.raw.save(out, verbose="error")
    written = out.read_bytes()
    for value in PLANTED:
        assert value.encode() not in written, value
        assert value.lower().encode() not in written.lower(), value
    assert report.subject == "001"


def test_report_lists_removed_fields_but_not_their_values(key: SubjectKey) -> None:
    _, report = deidentify(planted_recording(), key, "Jane Doe")
    assert {
        "subject_info.his_id",
        "subject_info.first_name",
        "subject_info.last_name",
        "experimenter",
        "description",
        "proj_name",
        "device_info.serial",
        "device_info.site",
    } <= set(report.removed_fields)
    dumped = report.model_dump_json()
    for value in ("MRN12345678", "Dr Watson", "SN-998877", "St Mary"):
        assert value not in dumped


def test_research_fields_are_kept(key: SubjectKey) -> None:
    clean, _ = deidentify(planted_recording(), key, "Jane Doe")
    subject = clean.raw.info["subject_info"]
    assert subject["his_id"] == "001"
    assert subject["sex"] == 2
    assert subject["hand"] == 1
    assert clean.raw.info["device_info"] == {"type": "TestAmp"}


def test_dates_are_shifted_keeping_age_and_time_of_day(key: SubjectKey) -> None:
    clean, report = deidentify(planted_recording(), key, "Jane Doe")
    shifted = clean.raw.info["meas_date"]
    assert report.dates_shifted
    assert datetime(1902, 1, 1, tzinfo=UTC) <= shifted <= datetime(1924, 12, 31, tzinfo=UTC)
    assert (shifted.hour, shifted.minute) == (9, 30)
    birthday = clean.raw.info["subject_info"]["birthday"]
    assert (MEAS_DATE.date() - BIRTHDAY) == (shifted.date() - birthday)


def test_event_timing_is_unchanged(key: SubjectKey) -> None:
    original = planted_recording()
    clean, _ = deidentify(original, key, "Jane Doe")
    assert np.allclose(
        clean.raw.annotations.onset - clean.raw.first_time,
        original.raw.annotations.onset - original.raw.first_time,
    )


def test_event_text_is_scrubbed_and_free_text_listed(key: SubjectKey) -> None:
    clean, report = deidentify(planted_recording(), key, "Jane Doe")
    descriptions = list(clean.raw.annotations.description)
    assert descriptions[0] == "target"
    assert descriptions[1] == "[removed] said she felt tired"
    assert descriptions[2] == "call [removed] if late"
    assert descriptions[3] == "note [removed] from [removed]"
    assert descriptions[4] == "Stimulus/S  1"
    assert report.redacted_events == 3
    assert "[removed] said she felt tired" in report.free_text_events
    assert "target" not in report.free_text_events
    assert "Stimulus/S  1" not in report.free_text_events


def test_extra_identifiers_are_removed(key: SubjectKey) -> None:
    rec = planted_recording()
    rec.raw.set_annotations(mne.Annotations([1.0], [0.0], ["seen by Nurse Ratched"], MEAS_DATE))
    clean, _ = deidentify(rec, key, "Jane Doe", identifiers=["Ratched"])
    assert clean.raw.annotations.description[0] == "seen by Nurse [removed]"


def test_original_is_unchanged(key: SubjectKey) -> None:
    original = planted_recording()
    deidentify(original, key, "Jane Doe")
    assert original.raw.info["subject_info"]["first_name"] == "Jane"
    assert original.raw.info["meas_date"] == MEAS_DATE


def test_recording_without_date_stays_without_date(key: SubjectKey) -> None:
    rec = planted_recording()
    rec.raw.set_meas_date(None)
    clean, report = deidentify(rec, key, "Jane Doe")
    assert clean.raw.info["meas_date"] is None
    assert not report.dates_shifted
    assert "birthday" not in clean.raw.info["subject_info"]  # cannot be shifted, so removed


# The subject key


def test_key_assigns_stable_labels_and_shifts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    key_path = tmp_path / "private" / "subjects.json"
    key = SubjectKey.open(key_path, dataset_root=dataset)
    first = key.entry("Jane Doe", MEAS_DATE)
    second = key.entry("John Roe", MEAS_DATE)
    assert (first.label, second.label) == ("001", "002")

    reopened = SubjectKey.open(key_path, dataset_root=dataset)
    later = reopened.entry("Jane Doe", MEAS_DATE + timedelta(days=200))
    assert later.label == "001"
    assert later.daysback == first.daysback  # same shift for every session of a subject
    assert reopened.labels() == {"Jane Doe": "001", "John Roe": "002"}


def test_key_file_is_written_and_labelled(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    key = SubjectKey.open(tmp_path / "subjects.json", dataset_root=dataset)
    key.entry("Jane Doe", MEAS_DATE)
    data = json.loads((tmp_path / "subjects.json").read_text(encoding="utf-8"))
    assert data["braino_subject_key"] == 1
    assert "private" in data["warning"]
    assert data["subjects"]["Jane Doe"]["label"] == "001"


def test_key_inside_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SubjectKeyError, match="outside the dataset"):
        SubjectKey.open(tmp_path / "code" / "subjects.json", dataset_root=tmp_path)


def test_session_that_cannot_be_shifted_is_refused(key: SubjectKey) -> None:
    key.entry("Jane Doe", MEAS_DATE)
    with pytest.raises(SubjectKeyError, match="cannot be shifted"):
        key.entry("Jane Doe", MEAS_DATE + timedelta(days=365 * 40))


def test_shifts_differ_between_subjects(key: SubjectKey) -> None:
    shifts = {key.entry(f"person {i}", MEAS_DATE).daysback for i in range(20)}
    assert len(shifts) > 1


@pytest.mark.network
@pytest.mark.parametrize("fmt", sorted(sample_data.RECORDINGS))
def test_sample_recordings(fmt: str, key: SubjectKey) -> None:
    original = read_raw(sample_data.fetch_recording(fmt))
    clean, report = deidentify(original, key, f"sample {fmt}")
    meas_date = clean.raw.info["meas_date"]
    if original.raw.info["meas_date"] is None:
        assert meas_date is None
    else:
        assert meas_date.year < 1925
    assert clean.raw.info["subject_info"]["his_id"] == report.subject
    assert clean.raw.info["experimenter"] is None
