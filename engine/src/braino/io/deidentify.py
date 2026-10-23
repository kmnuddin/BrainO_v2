"""Remove information that identifies the person recorded.

Each person gets a neutral subject label (``001``, ``002``, ...) and a random date shift. The table
linking labels back to the original identifiers is a :class:`SubjectKey`: a separate file that
must live outside the dataset folder and be kept private. It is the only place the shift is kept.

What :func:`deidentify` changes:

- **Header fields:** names, IDs, experimenter, free-text description, project name, device
  serial number and site, machine IDs and time zone are removed.
- **Dates:** the recording date and the birth date move back by the subject's shift, so age at
  recording, times of day and the intervals between sessions are kept. Shifted dates fall
  between 1902 and 1925, as BIDS requires.
- **Event text:** the person's own identifiers (name, ID, birth date) and dates, e-mail addresses
  and phone numbers are replaced with ``[removed]``. Other free text cannot be checked by machine,
  so it is listed in the report for a person to review.

File names are not carried over: converted files are named after the subject label.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.io.readers import Recording, _import_mne

REDACTED = "[removed]"
KEY_FORMAT_VERSION = 1

# BIDS requires shifted dates before 1925; MNE cannot store dates before 1902. A subject's shift
# puts its first recording in the middle of that window, leaving room for earlier and later
# sessions.
_LATEST_SHIFTED = datetime(1924, 12, 31, tzinfo=UTC)
_EARLIEST_SHIFTED = datetime(1902, 1, 1, tzinfo=UTC)
_TARGET_WINDOW = (datetime(1908, 1, 1, tzinfo=UTC), datetime(1918, 12, 31, tzinfo=UTC))

_TEXT_PATTERNS = (
    re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"),  # 2024-03-12
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),  # 12.03.2024, 3/12/24
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # e-mail
    re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)"),  # phone numbers, long ID numbers
)
_MIN_IDENTIFIER_LENGTH = 3


class SubjectKeyError(ValueError):
    """Raised for a subject key that is unsafe (inside the dataset) or cannot be used."""


class SubjectEntry(BaseModel):
    label: str
    daysback: int | None = Field(description="Days every date of this subject is moved back")
    assigned_at: datetime


class SubjectKey:
    """The private table linking subject labels to original identifiers, kept in a JSON file
    outside the dataset folder."""

    def __init__(self, path: Path, subjects: dict[str, SubjectEntry]) -> None:
        self.path = path
        self._subjects = subjects

    @classmethod
    def open(cls, path: str | Path, *, dataset_root: str | Path) -> SubjectKey:
        """Open the key at ``path``, or start a new one there.

        Raises :class:`SubjectKeyError` if ``path`` is inside ``dataset_root``.
        """
        key_path = Path(path).expanduser().resolve()
        root = Path(dataset_root).expanduser().resolve()
        if key_path.is_relative_to(root):
            raise SubjectKeyError(
                f"the subject key {key_path} must be kept outside the dataset folder {root}"
            )
        if not key_path.exists():
            return cls(key_path, {})
        data = json.loads(key_path.read_text(encoding="utf-8"))
        if data.get("braino_subject_key") != KEY_FORMAT_VERSION:
            raise SubjectKeyError(f"{key_path} is not a BrainO subject key")
        subjects = {k: SubjectEntry.model_validate(v) for k, v in data["subjects"].items()}
        return cls(key_path, subjects)

    def entry(self, original_id: str, meas_date: datetime | None) -> SubjectEntry:
        """The subject for ``original_id``, assigning a label and date shift the first time.

        Saves the key when it changes. Raises :class:`SubjectKeyError` if ``meas_date`` cannot
        be shifted with the subject's existing shift into the allowed window.
        """
        existing = self._subjects.get(original_id)
        if existing is not None:
            if existing.daysback is None and meas_date is not None:
                existing = existing.model_copy(update={"daysback": _new_daysback(meas_date)})
                self._subjects[original_id] = existing
                self.save()
            _check_shift(meas_date, existing)
            return existing
        width = max(3, len(str(len(self._subjects) + 1)))
        entry = SubjectEntry(
            label=f"{len(self._subjects) + 1:0{width}d}",
            daysback=_new_daysback(meas_date) if meas_date is not None else None,
            assigned_at=datetime.now(UTC),
        )
        self._subjects[original_id] = entry
        self.save()
        return entry

    def labels(self) -> dict[str, str]:
        return {original: entry.label for original, entry in self._subjects.items()}

    def save(self) -> None:
        data = {
            "braino_subject_key": KEY_FORMAT_VERSION,
            "warning": (
                "This file links BrainO subject labels to real identities. Keep it private and "
                "never put it inside the dataset folder or share it."
            ),
            "subjects": {k: v.model_dump(mode="json") for k, v in self._subjects.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def _new_daysback(meas_date: datetime) -> int:
    start, end = _TARGET_WINDOW
    target = start + timedelta(days=random.SystemRandom().randint(0, (end - start).days))
    return (meas_date - target).days


def _check_shift(meas_date: datetime | None, entry: SubjectEntry) -> None:
    if meas_date is None or entry.daysback is None:
        return
    shifted = meas_date - timedelta(days=entry.daysback)
    if not _EARLIEST_SHIFTED <= shifted <= _LATEST_SHIFTED:
        raise SubjectKeyError(
            f"subject {entry.label}: a recording from this date cannot be shifted into "
            "1902-1924 with the subject's existing date shift"
        )


class DeidentificationReport(BaseModel):
    """What de-identification did. Holds no identifying values except ``free_text_events``,
    which a person must review and which are already in the dataset's events."""

    subject: str = Field(description="The subject label the recording now carries")
    removed_fields: list[str] = Field(description="Header fields that had values and were removed")
    dates_shifted: bool
    redacted_events: int = Field(description="Event descriptions in which text was replaced")
    free_text_events: list[str] = Field(
        default_factory=list,
        description="Distinct event descriptions that are free text, to check by hand",
    )


def deidentify(
    recording: Recording,
    key: SubjectKey,
    original_id: str,
    *,
    identifiers: Iterable[str] = (),
) -> tuple[Recording, DeidentificationReport]:
    """De-identify a recording as the subject known as ``original_id`` in ``key``.

    ``identifiers`` are further strings to remove from event text (e.g. names from a referral
    letter). Returns a new :class:`Recording` (the original is unchanged) and the report.
    """
    mne = _import_mne()
    raw = recording.raw.copy()
    meas_date = raw.info["meas_date"]
    entry = key.entry(original_id, meas_date)
    known = _identifiers(raw.info, original_id, identifiers)
    removed = _filled_fields(raw.info)

    # Keep only what research needs: sex, hand, and the birth date if it can be shifted along
    # with the recording date (which keeps the age at recording).
    kept = ("sex", "hand", "birthday") if entry.daysback is not None else ("sex", "hand")
    subject_info = raw.info.get("subject_info") or {}
    raw.info["subject_info"] = {k: v for k, v in subject_info.items() if k in kept}
    research = {k: v for k, v in subject_info.items() if k in ("sex", "hand")}
    raw.anonymize(daysback=entry.daysback, keep_his=False, verbose="error")
    if meas_date is None:
        raw.set_meas_date(None)  # keep "no date" rather than a made-up one

    # anonymize() resets sex and hand; they describe the group studied, so they are restored.
    raw.info["subject_info"] = {
        **(raw.info.get("subject_info") or {}),
        **research,
        "his_id": entry.label,
    }
    raw.info["experimenter"] = None
    raw.info["description"] = None
    raw.info["proj_name"] = None
    raw.info["proj_id"] = None
    device = {
        k: v for k, v in (raw.info.get("device_info") or {}).items() if k not in ("serial", "site")
    }
    raw.info["device_info"] = device or None

    redacted, free_text = _scrub_events(mne, raw, known)
    report = DeidentificationReport(
        subject=entry.label,
        removed_fields=removed,
        dates_shifted=meas_date is not None,
        redacted_events=redacted,
        free_text_events=free_text,
    )
    return dataclasses.replace(recording, raw=raw), report


def _identifiers(info: Any, original_id: str, extra: Iterable[str]) -> list[str]:
    """Strings that identify the person, to remove from event text."""
    subject = info.get("subject_info") or {}
    values: list[str] = [original_id, *extra]
    for field in ("first_name", "middle_name", "last_name", "his_id"):
        if subject.get(field):
            values.append(str(subject[field]))
    birthday = subject.get("birthday")
    if isinstance(birthday, date):
        values += [
            birthday.isoformat(),
            birthday.strftime("%d.%m.%Y"),
            birthday.strftime("%m/%d/%Y"),
        ]
    if info.get("experimenter"):
        values.append(str(info["experimenter"]))
    words = {w for v in values for w in [v, *v.split()] if len(w) >= _MIN_IDENTIFIER_LENGTH}
    return sorted(words, key=len, reverse=True)  # longest first, so whole names go before parts


def _filled_fields(info: Any) -> list[str]:
    removed = []
    subject = info.get("subject_info") or {}
    for field in ("id", "his_id", "first_name", "middle_name", "last_name", "weight", "height"):
        if subject.get(field) not in (None, "", 0):
            removed.append(f"subject_info.{field}")
    for field in ("experimenter", "description", "proj_name", "proj_id", "utc_offset"):
        if info.get(field) not in (None, "", 0):
            removed.append(field)
    device = info.get("device_info") or {}
    removed += [f"device_info.{f}" for f in ("serial", "site") if device.get(f)]
    return removed


def _scrub_events(mne: Any, raw: Any, identifiers: list[str]) -> tuple[int, list[str]]:
    # Patterns first: removing a name inside an e-mail address would stop the e-mail matching.
    patterns = list(_TEXT_PATTERNS)
    patterns += [re.compile(re.escape(s), re.IGNORECASE) for s in identifiers]
    annotations = raw.annotations
    descriptions = []
    redacted = 0
    for description in annotations.description:
        clean = str(description)
        for pattern in patterns:
            clean = pattern.sub(REDACTED, clean)
        redacted += clean != description
        descriptions.append(clean)
    if redacted:
        # A new Annotations object: assigning longer strings into the old array would cut them.
        raw.set_annotations(
            mne.Annotations(
                onset=annotations.onset,
                duration=annotations.duration,
                description=descriptions,
                orig_time=annotations.orig_time,
                ch_names=annotations.ch_names,
                extras=annotations.extras,
            )
        )
    free_text = sorted({d for d in descriptions if _is_free_text(d)})
    return redacted, free_text


def _is_free_text(description: str) -> bool:
    """Whether an event description reads like typed text rather than a code or label."""
    words = re.findall(r"[^\W\d_]{2,}", description.replace(REDACTED, ""))
    return len(words) >= 2
