"""Import of EEG recordings: format detection and reading.

Format detection works in the core install; reading needs the ``[eeg]`` extra.
"""

from braino.io.bids import ConversionReport, NotDeidentifiedError, estimate_line_freq, write_bids
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
from braino.io.importer import ImportReport, import_recording
from braino.io.normalise import ChannelRetype, NormalisationReport, normalise
from braino.io.readers import CntDataFormat, Recording, RecordingReadError, read_raw

__all__ = [
    "FORMATS",
    "ChannelRetype",
    "CntDataFormat",
    "ConversionReport",
    "DeidentificationReport",
    "DetectedRecording",
    "FormatSpec",
    "ImportReport",
    "MissingFileError",
    "NormalisationReport",
    "NotDeidentifiedError",
    "Recording",
    "RecordingReadError",
    "SubjectEntry",
    "SubjectKey",
    "SubjectKeyError",
    "UnsupportedFormatError",
    "deidentify",
    "detect_format",
    "estimate_line_freq",
    "import_recording",
    "normalise",
    "read_raw",
    "write_bids",
]
