"""EEG analysis: quality checks now; preprocessing, ERPs, spectra and statistics to come.

Needs the ``[eeg]`` extra.
"""

from braino.eeg.qc import ChannelQc, NoisyChannel, QcResult, compute_qc

__all__ = ["ChannelQc", "NoisyChannel", "QcResult", "compute_qc"]
