"""Write a de-identified recording into a BIDS dataset.

Data are always written as new files (BrainVision for EEG and iEEG, FIF for MEG), never copied:
the headers of original files still hold the personal details that de-identification removed.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from braino.io.readers import Recording, _import_mne

Datatype = Literal["eeg", "meg", "ieeg"]
LineFreqSource = Literal["given", "file", "estimated", "unknown"]

_FORMATS: dict[str, str] = {"eeg": "BrainVision", "ieeg": "BrainVision", "meg": "FIF"}
_EXTENSIONS: dict[str, str] = {"eeg": ".vhdr", "ieeg": ".vhdr", "meg": ".fif"}
_DESCRIPTION_KEY = "EEGCoordinateSystemDescription"
_TEMPLATE_NOTE_START = "Approximate electrode positions from the standard"
_LABEL = re.compile(r"^[A-Za-z0-9]+$")
_LATEST_DEIDENTIFIED = datetime(1925, 1, 1, tzinfo=UTC)

# Line noise must stand this far above the neighbouring frequencies (median over channels),
# and this far above the other candidate frequency, to count as found.
_LINE_NOISE_RATIO = 2.0
_LINE_CONTRAST = 1.5
_LINE_ESTIMATE_SECONDS = 60.0


class NotDeidentifiedError(ValueError):
    """Raised when asked to write a recording that still looks identifiable."""


class ConversionReport(BaseModel):
    bids_root: str
    files: list[str] = Field(description="Files written, relative to the BIDS root")
    datatype: Datatype
    format: str
    line_freq: float | None = Field(description="Power-line frequency in Hz, if known")
    line_freq_source: LineFreqSource
    notes: list[str] = Field(default_factory=list)


def write_bids(
    recording: Recording,
    bids_root: str | Path,
    *,
    subject: str,
    task: str,
    session: str | None = None,
    run: int | None = None,
    datatype: Datatype | None = None,
    line_freq: float | None = None,
    overwrite: bool = False,
    allow_identifiable: bool = False,
    template_positions: str | None = None,
    manufacturer: str | None = None,
) -> ConversionReport:
    """Write ``recording`` (normalised and de-identified) to ``bids_root``.

    ``datatype`` defaults to ``meg`` when there are MEG channels, ``ieeg`` when there are
    intracranial channels, and ``eeg`` otherwise. ``line_freq`` defaults to the value in the
    file, or else one estimated from the data (50 or 60 Hz); if neither works, BIDS gets "n/a".

    Raises :class:`NotDeidentifiedError` if the recording still has a real-looking date or
    names, unless ``allow_identifiable`` is set (for data that never identified anyone).

    ``template_positions`` names the template layout the electrode positions came from, if
    they were not measured; the coordinate-system sidecar then says so.

    ``manufacturer`` is the maker of the recording system. MNE-BIDS fills in the maker of the
    *output* format (BrainVision files say "Brain Products"), which is wrong for data recorded
    on other systems, so it is replaced; ``"n/a"`` when the source format does not say.
    """
    mne = _import_mne()
    import mne_bids

    for name, value in (("subject", subject), ("task", task), ("session", session)):
        if value is not None and not _LABEL.fullmatch(value):
            raise ValueError(f"{name} label {value!r} may contain only letters and digits")
    raw = recording.raw.copy()
    if not allow_identifiable:
        _check_deidentified(raw)

    datatype = datatype or detect_datatype(raw)
    notes: list[str] = []
    # BIDS has no type for MNE's generic "bio" channels; they are miscellaneous channels there,
    # and must be typed so before writing or the sidecar's channel counts are wrong.
    bio = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True) if kind == "bio"
    ]
    if bio:
        raw.set_channel_types(dict.fromkeys(bio, "misc"), on_unit_change="ignore")
    freq, source = _line_freq(mne, raw, line_freq, notes)
    raw.info["line_freq"] = freq

    root = Path(bids_root).expanduser().resolve()
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
        root=root,
    )
    data_file = bids_path.copy().update(suffix=datatype, extension=_EXTENSIONS[datatype])
    if not overwrite and data_file.fpath.exists():
        raise FileExistsError(
            f"{data_file.basename} is already in the dataset; pass overwrite=True to replace it"
        )
    raw.load_data(verbose="error")
    # A subject's coordinate-system file is shared by its recordings, and MNE-BIDS refuses to
    # write one that differs from the existing file; so the template notes BrainO added are
    # taken out for the write and put back afterwards.
    notes_taken_out = _take_out_template_notes(Path(data_file.directory))
    started = time.time_ns()
    try:
        with warnings.catch_warnings():
            # Trigger and other non-voltage channels have no BrainVision unit; expected.
            warnings.filterwarnings("ignore", message="Encountered unsupported non-voltage units")
            mne_bids.write_raw_bids(
                raw,
                bids_path,
                format=_FORMATS[datatype],
                allow_preload=True,
                overwrite=overwrite,
                verbose="error",
            )
    finally:
        for path, note in notes_taken_out.items():
            _set_template_note(path, note)
    written = sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_mtime_ns >= started)
    if template_positions:
        for path in written:
            if path.name.endswith("_coordsystem.json"):
                _set_template_note(path, _template_note(template_positions))
    if manufacturer is not None:
        sidecar = Path(data_file.copy().update(extension=".json").fpath)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["Manufacturer"] = manufacturer
        sidecar.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return ConversionReport(
        bids_root=str(root),
        files=[p.relative_to(root).as_posix() for p in written],
        datatype=datatype,
        format=_FORMATS[datatype],
        line_freq=freq,
        line_freq_source=source,
        notes=notes,
    )


def _template_note(template: str) -> str:
    return (
        f"{_TEMPLATE_NOTE_START} {template} template layout, matched by channel name; not "
        "measured for this recording."
    )


def _split_description(description: object) -> tuple[str | None, str | None]:
    """MNE-BIDS's own description, and BrainO's template note appended to it, if any."""
    if not isinstance(description, str):
        return None, None
    start = description.find(_TEMPLATE_NOTE_START)
    if start < 0:
        return description, None
    return (description[:start].rstrip() or None), description[start:]


def _set_template_note(coordsystem: Path, note: str) -> None:
    """Append ``note`` to the description MNE-BIDS wrote (replacing an earlier note)."""
    if not coordsystem.is_file():
        return
    data = json.loads(coordsystem.read_text(encoding="utf-8"))
    base, _ = _split_description(data.get(_DESCRIPTION_KEY))
    data[_DESCRIPTION_KEY] = f"{base} {note}" if base else note
    coordsystem.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _take_out_template_notes(folder: Path) -> dict[Path, str]:
    """Remove BrainO's template notes from the coordinate-system files in ``folder``, leaving
    exactly what MNE-BIDS wrote."""
    taken_out: dict[Path, str] = {}
    for path in folder.glob("*_coordsystem.json") if folder.is_dir() else ():
        data = json.loads(path.read_text(encoding="utf-8"))
        base, note = _split_description(data.get(_DESCRIPTION_KEY))
        if note is not None:
            data[_DESCRIPTION_KEY] = base
            path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
            taken_out[path] = note
    return taken_out


def _check_deidentified(raw: Any) -> None:
    meas_date = raw.info["meas_date"]
    if meas_date is not None and meas_date >= _LATEST_DEIDENTIFIED:
        raise NotDeidentifiedError(
            "the recording date has not been shifted; de-identify the recording first"
        )
    subject = raw.info.get("subject_info") or {}
    if any(subject.get(f) for f in ("first_name", "middle_name", "last_name")):
        raise NotDeidentifiedError("the recording still holds a name; de-identify it first")


def detect_datatype(raw: Any) -> Datatype:
    """``meg``, ``ieeg`` or ``eeg``, from the channel types. Raises :class:`ValueError` if the
    recording has none of these channels (e.g. only ECG), since BIDS has no place for it."""
    types = set(raw.get_channel_types())
    if types & {"mag", "grad"}:
        return "meg"
    if types & {"seeg", "ecog", "dbs"}:
        return "ieeg"
    if "eeg" in types:
        return "eeg"
    raise ValueError(
        f"the recording has no EEG, MEG or intracranial channels (only: {', '.join(sorted(types))})"
        "; check the channel types"
    )


def _line_freq(
    mne: Any, raw: Any, given: float | None, notes: list[str]
) -> tuple[float | None, LineFreqSource]:
    if given is not None:
        return float(given), "given"
    if raw.info["line_freq"]:
        return float(raw.info["line_freq"]), "file"
    estimate = estimate_line_freq(raw)
    if estimate is not None:
        notes.append(f"Power-line frequency not in the file; estimated as {estimate:g} Hz.")
        return estimate, "estimated"
    notes.append("Power-line frequency unknown (not in the file, no clear 50 or 60 Hz noise).")
    return None, "unknown"


def estimate_line_freq(raw: Any) -> float | None:
    """50 or 60 if the data show clear line noise at exactly one of them, otherwise ``None``."""
    mne = _import_mne()
    if raw.info["sfreq"] < 2 * 65:
        return None
    picks = mne.pick_types(raw.info, eeg=True, meg=True, seeg=True, ecog=True, exclude="bads")
    if not len(picks):
        return None
    import numpy as np

    tmax = min(raw.times[-1], _LINE_ESTIMATE_SECONDS)
    spectrum = raw.compute_psd(
        picks=picks, fmin=40.0, fmax=70.0, tmax=tmax, n_fft=_n_fft(raw), verbose="error"
    )
    power = spectrum.get_data()  # channels x frequencies; channels may have different units
    freqs = spectrum.freqs
    # Judged per channel (the median over channels), so that channel types with large values
    # (e.g. MEG gradiometers) do not drown the others. Flat channels are left out.
    ratios: dict[float, float] = {}
    for line in (50.0, 60.0):
        near = np.abs(freqs - line) <= 0.6
        around = (np.abs(freqs - line) >= 2.0) & (np.abs(freqs - line) <= 5.0)
        if not near.any() or not around.any():
            return None
        baseline = np.median(power[:, around], axis=1)
        live = baseline > 0
        if not live.any():
            return None
        ratios[line] = float(np.median(power[live][:, near].max(axis=1) / baseline[live]))
    best, other = (50.0, 60.0) if ratios[50.0] >= ratios[60.0] else (60.0, 50.0)
    if ratios[best] >= _LINE_NOISE_RATIO and ratios[best] >= _LINE_CONTRAST * ratios[other]:
        return best
    return None


def _n_fft(raw: Any) -> int:
    """At least 0.5 Hz resolution, within the recording's length."""
    n = 1
    while n < 2 * raw.info["sfreq"]:
        n *= 2
    return min(n, int(raw.n_times))
