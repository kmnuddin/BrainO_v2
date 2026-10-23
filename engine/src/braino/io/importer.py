"""Import one recording into a BIDS dataset: detect, read, normalise, de-identify, write.

The original files are only read; nothing is copied from them. The returned report holds no
identifying values (no original file names or header values), so it can be shown to the agent
and kept in provenance.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from braino.io.bids import ConversionReport, Datatype, detect_datatype, write_bids
from braino.io.deidentify import DeidentificationReport, SubjectKey, deidentify
from braino.io.normalise import ChannelType, NormalisationReport, normalise
from braino.io.readers import CntDataFormat, read_raw


class ImportReport(BaseModel):
    format: str
    n_source_files: int
    normalisation: NormalisationReport
    deidentification: DeidentificationReport
    conversion: ConversionReport
    reader_notes: list[str]


def import_recording(
    source: str | Path,
    bids_root: str | Path,
    *,
    key: SubjectKey,
    original_id: str,
    task: str,
    session: str | None = None,
    run: int | None = None,
    datatype: Datatype | None = None,
    line_freq: float | None = None,
    rename: Mapping[str, str] | None = None,
    channel_types: Mapping[str, ChannelType] | None = None,
    montage: str | None = "auto",
    stim_channels: Sequence[str] | None = None,
    cnt_data_format: CntDataFormat = "auto",
    identifiers: Sequence[str] = (),
    overwrite: bool = False,
) -> ImportReport:
    """Import the recording at ``source`` as the person ``original_id`` (looked up in, or added
    to, ``key``) into ``bids_root``. See :func:`~braino.io.normalise`,
    :func:`~braino.io.deidentify` and :func:`~braino.io.bids.write_bids` for the options."""
    # Loaded in full: conversion needs all the data, and reading EDF files with mixed sampling
    # rates in chunks causes edge artifacts.
    recording = read_raw(source, preload=True, cnt_data_format=cnt_data_format)
    recording, normalisation = normalise(
        recording,
        rename=rename,
        channel_types=channel_types,
        montage=montage,
        stim_channels=stim_channels,
    )
    # Checked before de-identifying, so a recording that cannot be converted takes no label.
    datatype = datatype or detect_datatype(recording.raw)
    recording, deidentification = deidentify(recording, key, original_id, identifiers=identifiers)
    conversion = write_bids(
        recording,
        bids_root,
        subject=deidentification.subject,
        task=task,
        session=session,
        run=run,
        datatype=datatype,
        line_freq=line_freq,
        overwrite=overwrite,
        template_positions=normalisation.template
        if normalisation.positions == "template"
        else None,
    )
    # Event counts as written: the ones taken before de-identification hold the original text.
    events = Counter(str(d) for d in recording.raw.annotations.description)
    normalisation = normalisation.model_copy(update={"events": dict(events)})
    return ImportReport(
        format=recording.format.id,
        n_source_files=len(recording.files),
        normalisation=normalisation,
        deidentification=deidentification,
        conversion=conversion,
        reader_notes=_safe_notes(recording.notes, recording.files),
    )


def _safe_notes(notes: Sequence[str], files: Sequence[Path]) -> list[str]:
    """Reader notes with the source file and folder names taken out."""
    names = sorted(
        {p.name for p in files} | {str(p.parent) for p in files} | {str(p) for p in files},
        key=len,
        reverse=True,
    )
    cleaned = []
    for note in notes:
        for name in names:
            note = note.replace(name, "<source>")
        cleaned.append(note)
    return cleaned
