"""Summarising a BIDS dataset from its text files (no MNE needed)."""

from __future__ import annotations

import json
from pathlib import Path

from braino.io.dataset import scan_dataset
from braino.tools import RunContext, registry


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def recording(root: Path, entities: str, *, sfreq: float, duration: float, folder: str) -> None:
    stem = root / folder / entities
    write(Path(f"{stem}_eeg.vhdr"))
    write(
        Path(f"{stem}_eeg.json"),
        json.dumps(
            {"SamplingFrequency": sfreq, "RecordingDuration": duration, "PowerLineFrequency": 50}
        ),
    )
    write(
        Path(f"{stem}_channels.tsv"),
        "name\ttype\tunits\tstatus\nFz\tEEG\tuV\tgood\nCz\tEEG\tuV\tbad\nVEOG\tEOG\tuV\tgood\n",
    )
    write(
        Path(f"{stem}_events.tsv"),
        "onset\tduration\ttrial_type\n1.0\t0\ttarget\n2.0\t0\tstandard\n3.0\t0\tstandard\n",
    )


def dataset(root: Path) -> Path:
    write(root / "dataset_description.json", json.dumps({"Name": "Demo", "BIDSVersion": "1.9.0"}))
    recording(root, "sub-001_task-rest", sfreq=500.0, duration=60.0, folder="sub-001/eeg")
    recording(
        root,
        "sub-002_ses-pre_task-oddball_run-1",
        sfreq=250.0,
        duration=30.0,
        folder="sub-002/ses-pre/eeg",
    )
    write(root / "sub-002" / "ses-pre" / "eeg" / "notes.txt", "not a recording")
    return root


def test_summary(tmp_path: Path) -> None:
    summary = scan_dataset(dataset(tmp_path))
    assert (summary.name, summary.bids_version) == ("Demo", "1.9.0")
    assert summary.subjects == ["001", "002"]
    assert summary.sessions == ["pre"]
    assert summary.tasks == ["oddball", "rest"]
    assert summary.datatypes == ["eeg"]
    assert summary.n_recordings == 2
    assert summary.total_duration_s == 90.0

    first, second = summary.recordings
    assert first.path == "sub-001/eeg/sub-001_task-rest_eeg.vhdr"
    assert (first.sampling_rate_hz, first.duration_s, first.line_freq_hz) == (500.0, 60.0, 50.0)
    assert first.channels == {"eeg": 2, "eog": 1}
    assert first.bad_channels == ["Cz"]
    assert first.events == {"target": 1, "standard": 2}
    assert (second.session, second.task, second.run) == ("pre", "oddball", "1")


def test_listing_can_be_limited(tmp_path: Path) -> None:
    summary = scan_dataset(dataset(tmp_path), max_recordings=1)
    assert summary.n_recordings == 2
    assert len(summary.recordings) == 1
    assert summary.truncated


def test_empty_folder(tmp_path: Path) -> None:
    summary = scan_dataset(tmp_path)
    assert summary.n_recordings == 0
    assert summary.name is None


def test_inspect_tool(tmp_path: Path) -> None:
    ctx = RunContext(dataset_root=dataset(tmp_path))
    result = registry.get("dataset.inspect").run({"max_recordings": 5}, ctx)
    assert result.model_dump()["n_subjects"] == 2
    assert ctx.provenance.calls[0].tool == "dataset.inspect"


def test_inspect_tool_schema_is_small() -> None:
    schema = registry.get("dataset.inspect").llm_schema()
    assert set(schema["function"]["parameters"]["properties"]) == {"max_recordings"}
