"""Recognise which EEG format a file is in, and which files make up the recording.

Detection uses the file extension and, where extensions are ambiguous or often wrong, the first
bytes of the file. Pointing at any file of a multi-file recording (e.g. a BrainVision ``.eeg``)
resolves to the file a reader opens (the ``.vhdr``).

This module does not import MNE, so it works in the core install.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


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
_SIGNATURE_FORMATS = {"edf", "bdf", "gdf"}


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
    """Every file of the recording, ``path`` first. These are the files to hash, de-identify
    and convert."""


def detect_format(path: str | Path) -> DetectedRecording:
    """Detect the format of the recording at ``path``.

    Raises :class:`FileNotFoundError` if ``path`` does not exist,
    :class:`UnsupportedFormatError` if it is not a supported format, and
    :class:`MissingFileError` if a file the recording needs is missing.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"no such file or folder: {path}")
    if path.is_dir():
        if path.suffix.lower() == ".mff":
            return DetectedRecording(FORMATS["egi_mff"], path, (path,))
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
        return _eeglab(_sibling(path, path.stem, ".set"))
    if ext in (".fif", ".fif.gz"):
        return DetectedRecording(FORMATS["fif"], path, (path,))
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


def _eeglab(path: Path) -> DetectedRecording:
    files = (path, *_siblings(path, path.stem, (".fdt",)))
    return DetectedRecording(FORMATS["eeglab"], path, files)


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
