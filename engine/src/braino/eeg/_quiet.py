"""Silencing the libraries BrainO calls: their logs, warnings and progress bars would flood the
output of every tool, while what matters is reported in the tool's result."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def quiet() -> Iterator[None]:
    import mne

    previous = os.environ.get("MNE_TQDM")
    os.environ["MNE_TQDM"] = "off"  # MNE's progress bars, also used by PyPREP
    try:
        with mne.utils.use_log_level("error"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        if previous is None:
            os.environ.pop("MNE_TQDM", None)
        else:
            os.environ["MNE_TQDM"] = previous
