"""Recognise which EEG format a file is in, and which files make up the recording.

Detection uses the file extension and, where extensions are ambiguous or often wrong, the first
bytes of the file. Pointing at any file of a multi-file recording (e.g. a BrainVision ``.eeg``)
resolves to the file a reader opens (the ``.vhdr``).

This module does not import MNE, so it works in the core install. With the ``[eeg]`` extra it
also uses SciPy to find an EEGLAB data file that was renamed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormatSpec:
    id: str
    name: str
    extensions: tuple[str, ...]
    """Extensions of the file a reader opens, lower case."""


FORMATS: dict[str, FormatSpec] = {
    spec.id: spec
    for spec in [
        FormatSpec("edf", "European Data Format (EDF/EDF+)", (".edf",)),
        FormatSpec("bdf", "BioSemi Data Format (BDF/BDF+)", (".bdf",)),
        FormatSpec("gdf", "General Data Format (GDF)", (".gdf",)),
        FormatSpec("brainvision", "BrainVision", (".vhdr",)),
        FormatSpec("eeglab", "EEGLAB", (".set",)),
        FormatSpec("fif", "MNE/Elekta FIF", (".fif", ".fif.gz")),
        FormatSpec("cnt", "Neuroscan CNT", (".cnt",)),
        FormatSpec("egi_mff", "EGI MFF", (".mff",)),
        FormatSpec("curry", "Curry", (".cdt", ".dat")),
        FormatSpec("nihon_kohden", "Nihon Kohden", (".eeg",)),
        FormatSpec("persyst", "Persyst", (".lay",)),
    ]
}
"""Supported formats, keyed by format ID."""

# Leading bytes of each format that has a fixed signature.
_EDF_MAGIC = b"0       "
_BDF_MAGIC = b"\xffBIOSEMI"
_GDF_MAGIC = b"GDF "
_NEUROSCAN_MAGIC = b"Version "
_ANT_MAGIC = (b"RIFF", b"RF64")
_NIHON_KOHDEN_MAGIC = (b"EEG-", b"QI-", b"DAE-")

_CURRY_HEADERS = (".cdt.dpa", ".cdt.dpo", ".dap")
_CURRY_COMPANIONS = {
    ".cdt": (".cdt.dpa", ".cdt.dpo", ".cdt.cef", ".cdt.ceo"),
    ".dat": (".dap", ".rs3", ".cef", ".ceo"),
}
_NIHON_KOHDEN_COMPANIONS = (".pnt", ".log", ".21e")

# Parts of a FIF recording split at 2 GB: "rec_raw.fif", "rec_raw-1.fif", ... (Neuromag naming)
# or "..._split-01_meg.fif", "..._split-02_meg.fif", ... (BIDS naming).
_FIF_NEUROMAG_PART = re.compile(r"^(?P<base>.+)-\d+(?P<ext>\.fif(?:\.gz)?)$", re.IGNORECASE)
_FIF_BIDS_SPLIT = re.compile(r"_split-(?P<n>\d+)_", re.IGNORECASE)

# An EEGLAB .set that keeps its data in a separate file holds only metadata, so it is small;
# larger .set files are not opened to look for a renamed data file.
_EEGLAB_MAX_HEADER_BYTES = 100 * 1024 * 1024


class UnsupportedFormatError(ValueError):
    """Raised when a file is not in a format BrainO can read."""


class MissingFileError(FileNotFoundError):
    """Raised when a file the recording needs (e.g. BrainVision's data file) is missing."""


@dataclass(frozen=True)
class DetectedRecording:
    format: FormatSpec
    path: Path
    """The file (or, for EGI MFF, the folder) a reader opens."""
    files: tuple[Path, ...]
    """Every file of the recording: ``path`` first or, for an EGI MFF folder, the files inside
    it. These are the files to hash, de-identify and convert."""


def detect_format(path: str | Path) -> DetectedRecording:
    """Detect the format of the recording at ``path``.

    Raises :class:`FileNotFoundError` if ``path`` does not exist,
    :class:`UnsupportedFormatError` if it is not a supported format, and
    :class:`MissingFileError` if a file the recording needs is missing.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"no such file or folder: {path}")
    for folder in (path, *path.parents):
        if folder.suffix.lower() == ".mff" and folder.is_dir():
            return _mff(folder)
    if path.is_dir():
        raise UnsupportedFormatError(f"{path} is a folder, not a recording")

    name = path.name.lower()
    for suffix in _CURRY_HEADERS:
        if name.endswith(suffix):
            return _curry(_sibling(path, path.name[: -len(suffix)], _curry_data_ext(suffix)))
    for ext in (".cdt.cef", ".cdt.ceo"):
        if name.endswith(ext):
            return _curry(_sibling(path, path.name[: -len(ext)], ".cdt"))

    ext = ".fif.gz" if name.endswith(".fif.gz") else path.suffix.lower()
    magic = _read_magic(path)

    if ext in (".edf", ".bdf", ".gdf"):
        fmt = _format_from_signature(magic)
        if fmt is None:
            raise UnsupportedFormatError(f"{path} has a {ext} extension but no {ext} header")
        return DetectedRecording(FORMATS[fmt], path, (path,))
    if ext == ".vhdr":
        return _brainvision(path)
    if ext in (".vmrk", ".eeg"):
        header = _find_referencing(path, ".vhdr", ("DataFile", "MarkerFile"))
        if header is not None:
            return _brainvision(header)
        if ext == ".eeg" and magic.startswith(_NIHON_KOHDEN_MAGIC):
            return _nihon_kohden(path)
        raise UnsupportedFormatError(
            f"{path} is not part of a BrainVision recording (no .vhdr refers to it) and is not "
            "a Nihon Kohden file"
        )
    if ext in (".pnt", ".log", ".21e"):
        return _nihon_kohden(_sibling(path, path.stem, ".eeg"))
    if ext == ".set":
        return _eeglab(path)
    if ext == ".fdt":
        return _eeglab(_eeglab_header_for(path))
    if ext in (".fif", ".fif.gz"):
        return _fif(path)
    if ext == ".cnt":
        if magic.startswith(_ANT_MAGIC):
            raise UnsupportedFormatError(f"{path} is an ANT Neuro CNT file, not supported yet")
        if not magic.startswith(_NEUROSCAN_MAGIC):
            raise UnsupportedFormatError(f"{path} is not a Neuroscan CNT file")
        return DetectedRecording(FORMATS["cnt"], path, (path,))
    if ext == ".cdt":
        return _curry(path)
    if ext == ".lay":
        return _persyst(path)
    if ext == ".dat":
        layout = _find_referencing(path, ".lay", ("File",))
        if layout is not None:
            return _persyst(layout)
        if _siblings(path, path.stem, (".dap",)):
            return _curry(path)
        raise UnsupportedFormatError(
            f"{path} is neither Persyst data (no .lay refers to it) nor Curry 7 data (no .dap)"
        )

    fmt = _format_from_signature(magic)
    if fmt is not None:
        return DetectedRecording(FORMATS[fmt], path, (path,))
    supported = ", ".join(sorted({e for spec in FORMATS.values() for e in spec.extensions}))
    raise UnsupportedFormatError(f"{path}: unrecognised format (supported: {supported})")


def _format_from_signature(magic: bytes) -> str | None:
    if magic.startswith(_BDF_MAGIC):
        return "bdf"
    if magic.startswith(_EDF_MAGIC):
        return "edf"
    if magic.startswith(_GDF_MAGIC):
        return "gdf"
    return None


def _brainvision(header: Path) -> DetectedRecording:
    refs = _read_ini_values(header, ("DataFile", "MarkerFile"))
    data_name = refs.get("DataFile")
    if not data_name:
        raise UnsupportedFormatError(f"{header} has no DataFile entry")
    data = header.parent / data_name
    if not data.is_file():
        raise MissingFileError(f"{header} refers to data file {data_name!r}, which is missing")
    files = [header, data]
    marker_name = refs.get("MarkerFile")
    if marker_name and (header.parent / marker_name).is_file():
        files.append(header.parent / marker_name)
    return DetectedRecording(FORMATS["brainvision"], header, tuple(files))


def _mff(folder: Path) -> DetectedRecording:
    files = tuple(sorted(p for p in folder.rglob("*") if p.is_file()))
    if not any(p.name.lower() == "signal1.bin" for p in files):
        raise MissingFileError(f"EGI MFF recording {folder} has no signal1.bin")
    return DetectedRecording(FORMATS["egi_mff"], folder, files)


def _eeglab(path: Path) -> DetectedRecording:
    data = _siblings(path, path.stem, (".fdt",))
    if not data:
        name = _eeglab_data_file(path)
        if name and (path.parent / name).is_file():
            data = (path.parent / name,)
    return DetectedRecording(FORMATS["eeglab"], path, (path, *data))


def _eeglab_header_for(data: Path) -> Path:
    """The .set file whose data is in ``data`` (a .fdt file): same name, or one that names it."""
    same_name = _siblings(data, data.stem, (".set",))
    if same_name:
        return same_name[0]
    for header in sorted(p for p in data.parent.iterdir() if p.suffix.lower() == ".set"):
        if (_eeglab_data_file(header) or "").lower() == data.name.lower():
            return header
    raise MissingFileError(f"no EEGLAB .set file in {data.parent} uses the data file {data.name}")


def _eeglab_data_file(header: Path) -> str | None:
    """The data file an EEGLAB .set names (``EEG.datfile``, or ``EEG.data`` when it is a file
    name), if that can be read: it needs SciPy (the ``[eeg]`` extra) and a MATLAB v5-v7 file."""
    if header.stat().st_size > _EEGLAB_MAX_HEADER_BYTES:
        return None
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        mat = loadmat(
            header,
            variable_names=["EEG", "datfile", "data"],
            squeeze_me=True,
            struct_as_record=False,
        )
    except Exception:  # MATLAB v7.3 (HDF5) or not a MAT file: fall back to the .fdt by name
        return None
    eeg: Any = mat.get("EEG")
    for value in (
        getattr(eeg, "datfile", None),
        getattr(eeg, "data", None),
        mat.get("datfile"),
        mat.get("data"),
    ):
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).name
    return None


def _fif(path: Path) -> DetectedRecording:
    """A FIF recording with all of its split parts, opened from the first part."""
    first = _fif_first_part(path)
    parts = [first]
    while (following := _fif_part(first, len(parts) + 1)) is not None:
        parts.append(following)
    return DetectedRecording(FORMATS["fif"], first, tuple(parts))


def _fif_first_part(path: Path) -> Path:
    if match := _FIF_NEUROMAG_PART.match(path.name):
        first = path.parent / f"{match['base']}{match['ext']}"
        if first.is_file():
            return first
    if _FIF_BIDS_SPLIT.search(path.name):
        bids_first = _fif_part(path, 1)
        if bids_first is not None:
            return bids_first
    return path


def _fif_part(part: Path, number: int) -> Path | None:
    """Part ``number`` (1 is the first) of the split recording that ``part`` belongs to, if it
    exists. Neuromag naming numbers the parts after the first from 1; BIDS numbers all parts."""
    if match := _FIF_BIDS_SPLIT.search(part.name):
        split = f"_split-{number:0{len(match['n'])}d}_"
        name = _FIF_BIDS_SPLIT.sub(split, part.name, count=1)
    else:  # ``part`` is the first part
        ext = part.name[-7:] if part.name.lower().endswith(".fif.gz") else part.name[-4:]
        name = f"{part.name[: -len(ext)]}-{number - 1}{ext}"
    candidate = part.parent / name
    return candidate if candidate.is_file() else None


def _curry(path: Path) -> DetectedRecording:
    data_ext = ".cdt" if path.name.lower().endswith(".cdt") else ".dat"
    companions = _siblings(path, path.stem, _CURRY_COMPANIONS[data_ext])
    if not any(p.name.lower().endswith(_CURRY_HEADERS) for p in companions):
        expected = ".cdt.dpa or .cdt.dpo" if data_ext == ".cdt" else ".dap"
        raise MissingFileError(f"Curry recording {path} is missing its header ({expected})")
    return DetectedRecording(FORMATS["curry"], path, (path, *companions))


def _curry_data_ext(header: str) -> str:
    return ".dat" if header == ".dap" else ".cdt"


def _nihon_kohden(path: Path) -> DetectedRecording:
    files = (path, *_siblings(path, path.stem, _NIHON_KOHDEN_COMPANIONS))
    return DetectedRecording(FORMATS["nihon_kohden"], path, files)


def _persyst(layout: Path) -> DetectedRecording:
    data_name = _read_ini_values(layout, ("File",)).get("File")
    if not data_name:
        raise UnsupportedFormatError(f"{layout} has no File entry")
    data = layout.parent / data_name
    if not data.is_file():
        raise MissingFileError(f"{layout} refers to data file {data_name!r}, which is missing")
    return DetectedRecording(FORMATS["persyst"], layout, (layout, data))


def _read_magic(path: Path, size: int = 16) -> bytes:
    with path.open("rb") as f:
        return f.read(size)


def _read_ini_values(path: Path, keys: Iterable[str]) -> dict[str, str]:
    """Read ``key=value`` lines from a text header (BrainVision ``.vhdr``, Persyst ``.lay``)."""
    wanted = set(keys)
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in wanted and key.strip() not in values:
            values[key.strip()] = value.strip()
    return values


def _find_referencing(path: Path, header_ext: str, keys: tuple[str, ...]) -> Path | None:
    """The header file next to ``path`` whose ``keys`` entries name ``path``, if any."""
    candidates = sorted(
        (p for p in path.parent.iterdir() if p.suffix.lower() == header_ext and p.is_file()),
        key=lambda p: p.stem != path.stem,  # same-stem header first
    )
    for header in candidates:
        values = _read_ini_values(header, keys)
        if any(v.lower() == path.name.lower() for v in values.values()):
            return header
    return None


def _sibling(path: Path, stem: str, ext: str) -> Path:
    """The file ``<stem><ext>`` next to ``path``, matching the extension in any case."""
    found = _siblings(path, stem, (ext,))
    if not found:
        raise MissingFileError(f"{path} belongs to a recording whose {stem}{ext} is missing")
    return found[0]


def _siblings(path: Path, stem: str, exts: Iterable[str]) -> tuple[Path, ...]:
    """Files next to ``path`` named ``<stem><ext>`` for each ``ext``, in any case."""
    found: list[Path] = []
    entries = {p.name.lower(): p for p in path.parent.iterdir() if p.is_file()}
    for ext in exts:
        match = entries.get(f"{stem}{ext}".lower())
        if match is not None and match != path:
            found.append(match)
    return tuple(found)
