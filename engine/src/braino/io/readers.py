"""Read any supported EEG recording into an MNE ``Raw`` object.

Needs the ``[eeg]`` extra (``pip install braino[eeg]``).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from braino.io.formats import DetectedRecording, FormatSpec, detect_format

if TYPE_CHECKING:
    import mne

CntDataFormat = Literal["auto", "int16", "int32"]

# A plausible EEG amplitude is far below this; reading CNT data with the wrong sample width
# gives values orders of magnitude larger.
_MAX_PLAUSIBLE_VOLTS = 0.01
_CNT_CHECK_SECONDS = 10.0


class RecordingReadError(RuntimeError):
    """Raised when a recording was recognised but could not be read."""


@dataclass(frozen=True)
class Recording:
    raw: mne.io.BaseRaw
    format: FormatSpec
    path: Path
    """The file (or EGI MFF folder) that was opened."""
    files: tuple[Path, ...]
    """Every file of the recording, ``path`` first."""
    notes: tuple[str, ...] = ()
    """Warnings from the reader and choices made while reading (e.g. an inferred CNT sample
    width), to show the user and keep in the provenance record."""


def read_raw(
    path: str | Path,
    *,
    preload: bool = False,
    cnt_data_format: CntDataFormat = "auto",
) -> Recording:
    """Detect the format of ``path`` and read it.

    ``cnt_data_format`` sets the sample width of Neuroscan CNT files. With ``"auto"``, it is
    taken from the header, or, when the header does not say, inferred from which width gives
    plausible EEG amplitudes; the choice is reported in :attr:`Recording.notes`.

    Raises the errors of :func:`~braino.io.detect_format`, and :class:`RecordingReadError` if
    the recording cannot be read.
    """
    mne = _import_mne()
    detected = detect_format(path)
    notes: list[str] = []
    kwargs: dict[str, Any] = {}
    if detected.format.id == "cnt":
        kwargs["data_format"] = cnt_data_format
        if cnt_data_format == "auto":
            kwargs["data_format"] = _cnt_data_format(detected, notes)

    reader = _reader(mne, detected.format.id)
    raw = _read(reader, detected, notes, preload=preload, **kwargs)
    return Recording(raw, detected.format, detected.path, detected.files, tuple(notes))


def _reader(mne: Any, format_id: str) -> Callable[..., mne.io.BaseRaw]:
    readers: dict[str, Callable[..., mne.io.BaseRaw]] = {
        "edf": mne.io.read_raw_edf,
        "bdf": mne.io.read_raw_bdf,
        "gdf": mne.io.read_raw_gdf,
        "brainvision": mne.io.read_raw_brainvision,
        "eeglab": mne.io.read_raw_eeglab,
        "fif": mne.io.read_raw_fif,
        "cnt": mne.io.read_raw_cnt,
        "egi_mff": mne.io.read_raw_egi,
        "curry": mne.io.read_raw_curry,
        "nihon_kohden": mne.io.read_raw_nihon,
        "persyst": mne.io.read_raw_persyst,
    }
    return readers[format_id]


def _read(
    reader: Callable[..., mne.io.BaseRaw],
    detected: DetectedRecording,
    notes: list[str],
    **kwargs: Any,
) -> mne.io.BaseRaw:
    """Call ``reader``, collecting MNE's warnings into ``notes``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            raw = reader(detected.path, verbose="warning", **kwargs)
        except Exception as exc:
            raise RecordingReadError(
                f"could not read {detected.format.name} recording {detected.path}: {exc}"
            ) from exc
    for warning in caught:
        message = " ".join(str(warning.message).split())
        if message not in notes:
            notes.append(message)
    return raw


def _cnt_data_format(detected: DetectedRecording, notes: list[str]) -> CntDataFormat:
    """The sample width of a Neuroscan CNT file: from the header if it says, otherwise the one
    width that gives plausible EEG amplitudes."""
    mne = _import_mne()
    try:
        _read(mne.io.read_raw_cnt, detected, [], data_format="auto")
        return "auto"
    except RecordingReadError as exc:
        if "bytes per sample" not in str(exc.__cause__):
            raise

    plausible: list[CntDataFormat] = []
    for width in ("int16", "int32"):
        try:
            raw = _read(mne.io.read_raw_cnt, detected, [], data_format=width)
        except RecordingReadError:
            continue
        stop = min(raw.n_times, int(_CNT_CHECK_SECONDS * raw.info["sfreq"]))
        picks = mne.pick_types(raw.info, eeg=True) if "eeg" in raw else None
        peak = float(abs(raw.get_data(picks=picks, stop=stop)).max())
        if 0 < peak < _MAX_PLAUSIBLE_VOLTS:
            plausible.append(width)
    if len(plausible) != 1:
        raise RecordingReadError(
            f"the header of {detected.path} does not give its sample width, and it could not "
            "be inferred; pass cnt_data_format='int16' or 'int32'"
        )
    width = plausible[0]
    notes.append(
        f"The CNT header does not give the sample width; read as {width} because only that "
        "width gives plausible EEG amplitudes."
    )
    return width


def _import_mne() -> Any:
    try:
        import mne
    except ImportError:
        raise ImportError(
            "reading EEG recordings needs the scientific stack: pip install 'braino[eeg]'"
        ) from None
    return mne
