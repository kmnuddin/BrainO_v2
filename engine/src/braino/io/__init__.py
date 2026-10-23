"""Import of EEG recordings: format detection and reading.

Format detection works in the core install; reading needs the ``[eeg]`` extra.
"""

from braino.io.batch import ImportFailedError, ImportJob, MappingError, import_job, read_mapping
from braino.io.bids import ConversionReport, NotDeidentifiedError, estimate_line_freq, write_bids
from braino.io.dataset import DatasetSummary, RecordingSummary, scan_dataset
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
    "DatasetSummary",
    "DeidentificationReport",
    "DetectedRecording",
    "FormatSpec",
    "ImportFailedError",
    "ImportJob",
    "ImportReport",
    "MappingError",
    "MissingFileError",
    "NormalisationReport",
    "NotDeidentifiedError",
    "Recording",
    "RecordingReadError",
    "RecordingSummary",
    "SubjectEntry",
    "SubjectKey",
    "SubjectKeyError",
    "UnsupportedFormatError",
    "deidentify",
    "detect_format",
    "estimate_line_freq",
    "import_job",
    "import_recording",
    "normalise",
    "read_mapping",
    "read_raw",
    "scan_dataset",
    "write_bids",
]
