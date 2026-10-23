"""Summarise a BIDS dataset from its text files (sidecars, channel and event tables).

No recording data are read, so this is fast and works in the core install.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Data files of the datatypes BrainO writes: <entities>_<datatype>.<extension>
_DATA_FILE = re.compile(
    r"^(?P<entities>sub-[^_]+(?:_[a-z]+-[^_]+)*)_(?P<suffix>eeg|meg|ieeg)\.(vhdr|fif|edf|bdf|set)$"
)
_ENTITY = re.compile(r"(?P<key>[a-z]+)-(?P<value>[^_]+)")


class RecordingSummary(BaseModel):
    subject: str
    session: str | None = None
    task: str | None = None
    run: str | None = None
    datatype: str
    path: str = Field(description="Data file, relative to the dataset root")
    sampling_rate_hz: float | None = None
    duration_s: float | None = None
    line_freq_hz: float | None = None
    channels: dict[str, int] = Field(default_factory=dict, description="Channels per type")
    bad_channels: list[str] = Field(default_factory=list)
    events: dict[str, int] = Field(default_factory=dict, description="Events per type")


class DatasetSummary(BaseModel):
    name: str | None
    bids_version: str | None
    n_subjects: int
    subjects: list[str]
    sessions: list[str]
    tasks: list[str]
    datatypes: list[str]
    n_recordings: int
    total_duration_s: float
    recordings: list[RecordingSummary] = Field(
        description="Per-recording details (possibly only the first ones; see truncated)"
    )
    truncated: bool = False


def scan_dataset(root: str | Path, *, max_recordings: int | None = None) -> DatasetSummary:
    """Summarise the BIDS dataset at ``root``. ``max_recordings`` limits how many recordings
    are listed in detail (the totals always cover all of them)."""
    root = Path(root)
    description = _read_json(root / "dataset_description.json")
    recordings = sorted(
        (_summarise(root, path) for path in _data_files(root)),
        key=lambda r: (r.subject, r.session or "", r.task or "", r.run or "", r.datatype),
    )
    shown = recordings if max_recordings is None else recordings[:max_recordings]
    return DatasetSummary(
        name=description.get("Name"),
        bids_version=description.get("BIDSVersion"),
        n_subjects=len({r.subject for r in recordings}),
        subjects=sorted({r.subject for r in recordings}),
        sessions=sorted({r.session for r in recordings if r.session}),
        tasks=sorted({r.task for r in recordings if r.task}),
        datatypes=sorted({r.datatype for r in recordings}),
        n_recordings=len(recordings),
        total_duration_s=round(sum(r.duration_s or 0.0 for r in recordings), 3),
        recordings=shown,
        truncated=len(shown) < len(recordings),
    )


def _data_files(root: Path) -> list[Path]:
    files = []
    for subject_dir in sorted(root.glob("sub-*")):
        for path in subject_dir.rglob("*"):
            if path.is_file() and _DATA_FILE.match(path.name):
                files.append(path)
    return files


def _summarise(root: Path, path: Path) -> RecordingSummary:
    match = _DATA_FILE.match(path.name)
    assert match is not None
    entities = {m["key"]: m["value"] for m in _ENTITY.finditer(match["entities"])}
    stem = path.name[: -len(path.name.split("_")[-1])]  # "<entities>_"
    sidecar = _read_json(path.parent / f"{stem}{match['suffix']}.json")
    channels = _read_tsv(path.parent / f"{stem}channels.tsv")
    events = _read_tsv(path.parent / f"{stem}events.tsv")
    return RecordingSummary(
        subject=entities["sub"],
        session=entities.get("ses"),
        task=entities.get("task"),
        run=entities.get("run"),
        datatype=match["suffix"],
        path=path.relative_to(root).as_posix(),
        sampling_rate_hz=_number(sidecar.get("SamplingFrequency")),
        duration_s=_number(sidecar.get("RecordingDuration")),
        line_freq_hz=_number(sidecar.get("PowerLineFrequency")),
        channels=dict(Counter(row.get("type", "n/a").lower() for row in channels)),
        bad_channels=[row["name"] for row in channels if row.get("status") == "bad"],
        events=dict(Counter(row.get("trial_type", "n/a") for row in events)),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
