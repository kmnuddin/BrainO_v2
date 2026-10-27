"""Finding recordings and the derivative files that analysis steps write for them.

Each step writes its outputs next to each other under ``derivatives/braino/sub-<label>/
[ses-<label>/]<datatype>/``, named after the recording with a ``desc`` entity:

==========================  ===========================================  =======================
Step                        File                                         Read by
==========================  ===========================================  =======================
``eeg.preprocess``          ``<rec>_desc-preproc_eeg.fif`` (+ ``.json``)   all later steps
``eeg.detect_bad_channels`` ``<rec>_desc-preproc_channels.tsv``            ICA, epoching
``eeg.ica_clean``           ``<rec>_desc-preproc_ica.fif`` (fitted ICA),
                            ``<rec>_desc-clean_eeg.fif`` (applied)         epoching
``eeg.epoch``               ``<rec>_desc-<label>_epo.fif``                 ERP, spectra, TFR
``eeg.erp``                 ``<rec>_desc-<label>_ave.fif``, ``_erp.tsv``    group statistics
``eeg.spectral``            ``<rec>_desc-<label>_psd.npz``, ``_bands.tsv``  group statistics
``eeg.tfr``                 ``<rec>_desc-<label>_tfr.npz``                 group statistics
==========================  ===========================================  =======================

Every file is written new; none is changed in place, so each step's inputs keep the hash their
provenance record holds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.io.dataset import RecordingSummary, scan_dataset

LABEL_PATTERN = r"^[A-Za-z0-9]+$"
_LABEL = re.compile(LABEL_PATTERN)

PREPROC = "preproc"
CLEAN = "clean"


class RecordingSelector(BaseModel):
    subject: str = Field(description="Subject label without 'sub-', e.g. '001'")
    task: str | None = Field(default=None, description="Task label, if the subject has several")
    session: str | None = Field(default=None, description="Session label, if any")
    run: str | None = Field(default=None, description="Run number, if any")


class RecordingNotFoundError(LookupError):
    pass


class MissingStepError(LookupError):
    """Raised when a step's input (the output of an earlier step) does not exist."""


def find_recording(root: Path, selector: RecordingSelector) -> RecordingSummary:
    """The one recording matching ``selector``; raises :class:`RecordingNotFoundError` if none
    or several match, listing the candidates."""
    recordings = scan_dataset(root).recordings
    matches = [
        r
        for r in recordings
        if r.subject == selector.subject
        and (selector.task is None or r.task == selector.task)
        and (selector.session is None or r.session == selector.session)
        and (selector.run is None or (r.run or "").lstrip("0") == selector.run.lstrip("0"))
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RecordingNotFoundError(
            f"no recording matches {selector.model_dump(exclude_none=True)}"
        )
    options = ", ".join(r.path for r in matches)
    raise RecordingNotFoundError(f"several recordings match; say which: {options}")


def recording_files(data_file: Path) -> list[Path]:
    """The data file and its companions (e.g. BrainVision .eeg and .vmrk)."""
    stem = data_file.name.rsplit(".", 1)[0]
    return sorted(p for p in data_file.parent.glob(f"{stem}.*") if p.is_file())


def stem(recording: RecordingSummary) -> str:
    """``sub-001_task-P3_``: the recording's entities, as a file-name prefix."""
    name = Path(recording.path).name
    return name[: -len(name.split("_")[-1])]


def output_dir(ctx: RunContext, recording: RecordingSummary) -> Path:
    parts = [f"sub-{recording.subject}"]
    if recording.session:
        parts.append(f"ses-{recording.session}")
    return ctx.output_dir(*parts, recording.datatype)


def derivative(ctx: RunContext, recording: RecordingSummary, name: str) -> Path:
    """Path of a derivative of ``recording``, e.g. ``derivative(ctx, rec, "desc-preproc_eeg.fif")``
    (the folder is created; the file may not exist)."""
    return output_dir(ctx, recording) / f"{stem(recording)}{name}"


def check_label(value: str, what: str) -> str:
    if not _LABEL.fullmatch(value):
        raise ValueError(f"{what} {value!r} may contain only letters and digits")
    return value


def relative(ctx: RunContext, path: Path) -> str:
    return path.relative_to(ctx.dataset).as_posix()


def require(path: Path, step: str) -> Path:
    if not path.is_file():
        raise MissingStepError(f"{path.name} does not exist; run {step} first")
    return path


def read_original(ctx: RunContext, recording: RecordingSummary) -> Any:
    """The recording as stored in the dataset (inputs recorded for provenance)."""
    import mne_bids

    data_file = ctx.dataset / recording.path
    for path in recording_files(data_file):
        ctx.record_input(path)
    bids_path = mne_bids.get_bids_path_from_fname(data_file)
    return mne_bids.read_raw_bids(bids_path, verbose="error")


def read_raw_derivative(ctx: RunContext, path: Path) -> Any:
    import mne

    ctx.record_input(path)
    return mne.io.read_raw_fif(path, preload=True, verbose="error")


def write_raw(ctx: RunContext, raw: Any, path: Path) -> Path:
    ctx.record_output(path)
    raw.save(path, overwrite=True, verbose="error")
    return path


def write_json(ctx: RunContext, data: dict[str, Any], path: Path) -> Path:
    ctx.record_output(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(ctx: RunContext, path: Path) -> dict[str, Any]:
    ctx.record_input(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_tsv(ctx: RunContext, path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    ctx.record_output(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["n/a" if v is None else v for v in row])
    return path


def read_tsv(ctx: RunContext, path: Path) -> list[dict[str, str]]:
    ctx.record_input(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def preprocessed(
    ctx: RunContext, recording: RecordingSummary, *, with_decision: bool = True
) -> tuple[Any, dict[str, Any]]:
    """The preprocessed recording with the approved bad channels marked (unless
    ``with_decision`` is false: then only those marked in the file), and the sidecar of its
    preprocessing (which holds the reference)."""
    path = require(derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif"), "eeg.preprocess")
    raw = read_raw_derivative(ctx, path)
    sidecar = read_json(ctx, path.with_suffix(".json"))
    if with_decision:
        raw.info["bads"] = approved_bads(ctx, recording, raw.ch_names, raw.info["bads"])
    return raw, sidecar


def approved_bads(
    ctx: RunContext, recording: RecordingSummary, ch_names: list[str], current: list[str]
) -> list[str]:
    """Channels marked bad in the approved channels table, if there is one, else ``current``.

    Raises :class:`StaleStepError` if the table was approved for an earlier preprocessing.
    """
    table = derivative(ctx, recording, f"desc-{PREPROC}_channels.tsv")
    if not table.is_file():
        return list(current)
    if bad_channels_stale(ctx, recording):
        raise StaleStepError(
            f"{table.name} was approved for an earlier preprocessing; run "
            "eeg.detect_bad_channels again"
        )
    rows = read_tsv(ctx, table)
    return [r["name"] for r in rows if r.get("status") == "bad" and r["name"] in ch_names]


def bad_channels_stale(ctx: RunContext, recording: RecordingSummary) -> bool:
    """Whether there is an approved channels table made for an earlier preprocessing."""
    table = derivative(ctx, recording, f"desc-{PREPROC}_channels.tsv")
    if not table.is_file():
        return False
    preproc = derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif")
    return bool(read_json(ctx, table.with_suffix(".json")).get("SourceSha256") != sha256(preproc))


def clean_stale(ctx: RunContext, recording: RecordingSummary) -> bool:
    """Whether there is an ICA-cleaned recording made from an earlier preprocessing."""
    clean = derivative(ctx, recording, f"desc-{CLEAN}_eeg.fif")
    if not clean.is_file():
        return False
    preproc = derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif")
    return bool(read_json(ctx, clean.with_suffix(".json")).get("SourceSha256") != sha256(preproc))


def cleanest(
    ctx: RunContext, recording: RecordingSummary, *, with_decision: bool = True
) -> tuple[Any, dict[str, Any], str]:
    """The ICA-cleaned recording if there is one, else the preprocessed one, with the approved
    bad channels marked (unless ``with_decision`` is false: then only those marked in the
    original file); also the preprocessing sidecar and which of the two was used.

    Raises :class:`StaleStepError` if the cleaned recording was made from an earlier
    preprocessing than the current one.
    """
    import mne

    clean = derivative(ctx, recording, f"desc-{CLEAN}_eeg.fif")
    if not clean.is_file():
        raw, sidecar = preprocessed(ctx, recording, with_decision=with_decision)
        return raw, sidecar, PREPROC
    preproc = require(derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif"), "eeg.preprocess")
    if clean_stale(ctx, recording):
        raise StaleStepError(
            f"{clean.name} was made from an earlier preprocessing; run eeg.ica_clean again"
        )
    raw = read_raw_derivative(ctx, clean)
    sidecar = read_json(ctx, preproc.with_suffix(".json"))
    in_file = list(mne.io.read_info(preproc, verbose="error")["bads"])
    raw.info["bads"] = (
        approved_bads(ctx, recording, raw.ch_names, in_file) if with_decision else in_file
    )
    return raw, sidecar, CLEAN


class StaleStepError(RuntimeError):
    """Raised when a step's input was made from an older version of an earlier step's output."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_epochs(ctx: RunContext, recording: RecordingSummary, label: str) -> tuple[Any, Path]:
    import mne

    path = require(derivative(ctx, recording, f"desc-{label}_epo.fif"), "eeg.epoch")
    ctx.record_input(path)
    return mne.read_epochs(path, preload=True, verbose="error"), path
