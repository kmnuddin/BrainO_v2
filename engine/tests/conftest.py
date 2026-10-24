from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import mne

SKIP_NETWORK_ENV = "BRAINO_SKIP_NETWORK"
LARGE_DOWNLOADS_ENV = "BRAINO_LARGE_DOWNLOADS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    offline = os.environ.get(SKIP_NETWORK_ENV) == "1"
    large = os.environ.get(LARGE_DOWNLOADS_ENV) == "1"
    skip_network = pytest.mark.skip(reason=f"needs downloaded test data ({SKIP_NETWORK_ENV}=1)")
    skip_large = pytest.mark.skip(reason=f"large download; set {LARGE_DOWNLOADS_ENV}=1 to run")
    for item in items:
        if "large_download" in item.keywords and (offline or not large):
            item.add_marker(skip_large)
        elif "network" in item.keywords and offline:
            item.add_marker(skip_network)


@pytest.fixture
def synthetic_raw() -> mne.io.RawArray:
    """A small, deterministic EEG recording: 8 channels of the 10-20 system at 250 Hz for 10 s,
    with a 10 Hz alpha rhythm, 50 Hz line noise and two event types, for fast offline tests."""
    mne = pytest.importorskip("mne")
    import numpy as np

    sfreq, duration = 250.0, 10.0
    ch_names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "O1", "O2"]
    times = np.arange(int(sfreq * duration)) / sfreq
    rng = np.random.default_rng(seed=42)
    alpha = 10e-6 * np.sin(2 * np.pi * 10 * times)
    line = 2e-6 * np.sin(2 * np.pi * 50 * times)
    data = alpha + line + 5e-6 * rng.standard_normal((len(ch_names), times.size))

    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_montage("colin27_1020")
    onsets = np.arange(1.0, duration - 1.0, 1.0)
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=0.0,
            description=["standard" if i % 4 else "target" for i in range(onsets.size)],
        )
    )
    return raw
