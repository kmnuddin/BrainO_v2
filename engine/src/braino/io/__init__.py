"""Import of EEG recordings: format detection and reading.

Format detection works in the core install; reading needs the ``[eeg]`` extra.
"""

from braino.io.deidentify import (
    DeidentificationReport,
    SubjectEntry,
    SubjectKey,
    SubjectKeyError,
    deidentify,
)
from braino.io.formats import (
    FORMATS,
    DetectedRecording,
    FormatSpec,
    MissingFileError,
    UnsupportedFormatError,
    detect_format,
)
from braino.io.normalise import ChannelRetype, NormalisationReport, normalise
from braino.io.readers import CntDataFormat, Recording, RecordingReadError, read_raw

__all__ = [
    "FORMATS",
    "ChannelRetype",
    "CntDataFormat",
    "DeidentificationReport",
    "DetectedRecording",
    "FormatSpec",
    "MissingFileError",
    "NormalisationReport",
    "Recording",
    "RecordingReadError",
    "SubjectEntry",
    "SubjectKey",
    "SubjectKeyError",
    "UnsupportedFormatError",
    "deidentify",
    "detect_format",
    "normalise",
    "read_raw",
]
