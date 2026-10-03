# ruff: noqa: E501  (registry lines hold full SHA-256 hashes)
"""Small public sample recordings in every supported format, downloaded on demand for tests.

Files come from MNE-Python's test data repository (mne-tools/mne-testing-data) at a pinned tag
and are checked against the SHA-256 hashes below, then cached. The cache lives in the OS cache
folder (e.g. ``~/.cache/braino-test-data``) unless ``BRAINO_TEST_DATA`` points elsewhere.

To add a file: add its path and hash to ``REGISTRY`` (and a ``RECORDINGS`` entry if it is a new
recording), then run the tests. Tests that download data are marked ``network``; set
``BRAINO_SKIP_NETWORK=1`` to skip them when offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pooch

MNE_TESTING_DATA_TAG = "0.176"
BASE_URL = f"https://raw.githubusercontent.com/mne-tools/mne-testing-data/{MNE_TESTING_DATA_TAG}/"
CACHE_ENV = "BRAINO_TEST_DATA"

REGISTRY = {
    "BDF/test_bdf_stim_channel.bdf": "sha256:d555c9550069c0402835092ac3b2136ac238c8b8996069e11594350826d765a3",
    "Brainvision/test_NO.eeg": "sha256:894099d7ea0db262bd2dd84918ff96e5c81e39f7677f0405746faaed1623604b",
    "Brainvision/test_NO.vhdr": "sha256:aa3f2d42a1ad3897e702c27d09abdd261f01ccdeb0dae2e674af25fe7be72261",
    "Brainvision/test_NO.vmrk": "sha256:fc02236e72a90124ea04f171a065ae94784ec707bbd96632a86a44499bf7cf27",
    "CNT/scan41_short.cnt": "sha256:f58b7182f6be670159a79090fb666d3f1ed6645f5b488d0016940fd2b8b7e5b6",
    "EDF/test_reduced.edf": "sha256:644acaf3aa547d85d73ed4a7224008c55be1487227e5eb029274877e2c61a39a",
    "EEGLAB/test_raw.fdt": "sha256:3ec388b9a080c723c00cd380403d64cc754929d828375bbd53bd83b468136759",
    "EEGLAB/test_raw.set": "sha256:95593d49974d3ce8b0bd37e086536b9cedcf0abf38f2453092fd88de0bbde18e",
    "EGI/test_egi.mff/Contents/Info.plist": "sha256:6f27e762e8f58329bece4b4565ab93856edc8ce64a4b5e8994b7843562d80b1e",
    "EGI/test_egi.mff/Contents/PkgInfo": "sha256:ffe6cb5c0493916f026fb8ba01b7b07c7746ca6cb0cb499051ec45575a3bbadb",
    "EGI/test_egi.mff/Events_255 DINs.xml": "sha256:a791b91c341a876b15cac3b9cf52f20e4f1f18d1910f0704af36224ff078af05",
    "EGI/test_egi.mff/coordinates.xml": "sha256:3c49e412457581cf74b6627453a3ab182e1f29263f4d3beab4eb077200d1b9f7",
    "EGI/test_egi.mff/epochs.xml": "sha256:a887f8c70d2aa45a9a8f85b10108454b3dfd33f681d71740bc2fa499104bf262",
    "EGI/test_egi.mff/info.xml": "sha256:0bb8648abe34e06c1adeb6caebce4c71aacc0d94091f424e74476eaa15bcc0b4",
    "EGI/test_egi.mff/info1.xml": "sha256:84a6d45f2848fe0463855f697b0bc6141d998e3589918014b72b517ba58af0e6",
    "EGI/test_egi.mff/log_testfile23022017_20170223_113514.txt": "sha256:cab955b4ffd09896a1e5890e09d4ad6f181fa30e7df0f1b38fbc9190dce14482",
    "EGI/test_egi.mff/recordingSettings.xml": "sha256:ac5aff6be185770496d890f18d8ab0d39a9d25aed3c4ca6486cbeff8d94c1eb2",
    "EGI/test_egi.mff/sensorLayout.xml": "sha256:43e8e4423bc3fc56f85b7cf3ff3106010d1f674505b0e2d555117b07360f9965",
    "EGI/test_egi.mff/signal1.bin": "sha256:3069e20983ca2861bb16a6dc8e5edc32f988380ae9cd06821b7db221d43a0a33",
    "EGI/test_egi.mff/subject.xml": "sha256:09ca61cd8ca308a2c836647ca455acac803050dbb82f071d7472f0d611340dd6",
    "EGI/test_egi.mff/workspace.plist": "sha256:f91a4973940cac0e3bcfca88cf26d3df6e0990241413b5737386b3d7940a70cd",
    "GDF/test_1ch.gdf": "sha256:a01faf7b73742a01deae89e2f4cd18c581ca5c1f719417bf80ac2ef8a2923762",
    "MEG/sample/sample_audvis_trunc_raw.fif": "sha256:d1ecdebf1ae23d9e7a5b56be63f45191778440cce6c8dfb0c270e75484b2aa50",
    "NihonKohden/MB0400FU.11D": "sha256:ea88b3733d755c6ece6e7a2bf710fdc918233e9412600e5bd4922e0b5f383712",
    "NihonKohden/MB0400FU.21E": "sha256:4704c38a17c5e04c6b46ddc3fac2f2ef5cb46ca27f9a7f3df1c5a970ee9eb3a1",
    "NihonKohden/MB0400FU.CMT": "sha256:af966ed45afa4bfcfd8c2ca56a37d0a81d06cb813de64043c24e9572897a41b6",
    "NihonKohden/MB0400FU.CN2": "sha256:ad7131e9101c753ab4c8c7cd68a455fd91136b2f93fa88e2ef24762418c08c13",
    "NihonKohden/MB0400FU.EDF": "sha256:54206fb26f743e2b7306e48557883b2fb953080dd7f8799a3793a650c14061cc",
    "NihonKohden/MB0400FU.EEG": "sha256:a2900ceaa0479182e5845dc3f3f8ee4bd6800bd4ec9555a59008486253c42d0a",
    "NihonKohden/MB0400FU.LOG": "sha256:60262a77a8136c595eae7a4c1fb71853d806a83ef0a40e2c9fc03f0fbc0fa714",
    "NihonKohden/MB0400FU.PNT": "sha256:d1bbf1ff13de54407f3f3afe18ae004c3e23747f82c2b76eea977efd13985684",
    "Persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.dat": "sha256:b21b45af347f2fde710da46af66128be63058db2bcc6b566c2934fa42ce6b0ba",
    "Persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay": "sha256:fa1aae5c47fd6c0e7d282fd27ee8c6a5d3ae06bb74687268a5f64fc00834138b",
    "curry/test_bdf_stim_channel Curry 8.cdt": "sha256:cb6e60301592fa9f797216f9a711445ad670700ecb74547f7394d5ca437f244b",
    "curry/test_bdf_stim_channel Curry 8.cdt.cef": "sha256:b1a4774cc54cf7b4bc0f034ee8fc033725ff1dab35744a57014eb9d058f2905b",
    "curry/test_bdf_stim_channel Curry 8.cdt.dpa": "sha256:7a300fe0c486520cbb6d7e7c61866ae7d216894634ffeda0c898e23c8d5ea240",
}


@dataclass(frozen=True)
class Recording:
    """A recording made of one or more files; ``main`` is the path a reader opens."""

    format: str
    main: str
    prefix: str
    """Every registry file starting with this belongs to the recording (companion files)."""


RECORDINGS = {
    r.format: r
    for r in [
        Recording("edf", "EDF/test_reduced.edf", "EDF/test_reduced.edf"),
        Recording("bdf", "BDF/test_bdf_stim_channel.bdf", "BDF/test_bdf_stim_channel.bdf"),
        Recording("brainvision", "Brainvision/test_NO.vhdr", "Brainvision/test_NO."),
        Recording("eeglab", "EEGLAB/test_raw.set", "EEGLAB/test_raw."),
        Recording(
            "fif",
            "MEG/sample/sample_audvis_trunc_raw.fif",
            "MEG/sample/sample_audvis_trunc_raw.fif",
        ),
        Recording("cnt", "CNT/scan41_short.cnt", "CNT/scan41_short.cnt"),
        Recording("gdf", "GDF/test_1ch.gdf", "GDF/test_1ch.gdf"),
        Recording("egi_mff", "EGI/test_egi.mff", "EGI/test_egi.mff/"),
        Recording(
            "curry",
            "curry/test_bdf_stim_channel Curry 8.cdt",
            "curry/test_bdf_stim_channel Curry 8.cdt",
        ),
        Recording("nihon_kohden", "NihonKohden/MB0400FU.EEG", "NihonKohden/MB0400FU."),
        Recording(
            "persyst",
            "Persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay",
            "Persyst/sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.",
        ),
    ]
}

_fetcher = pooch.create(
    path=os.environ.get(CACHE_ENV) or pooch.os_cache("braino-test-data"),
    base_url=BASE_URL,
    registry=REGISTRY,
    urls={name: BASE_URL + quote(name) for name in REGISTRY},
    retry_if_failed=3,
)


def cache_dir() -> Path:
    return Path(_fetcher.abspath)


def fetch_file(name: str) -> Path:
    """Download (or reuse the cached copy of) one registry file and return its path."""
    return Path(_fetcher.fetch(name))


def fetch_recording(fmt: str) -> Path:
    """Download all files of the sample recording for ``fmt``; return the path to open."""
    recording = RECORDINGS[fmt]
    files = [name for name in REGISTRY if name.startswith(recording.prefix)]
    if not files:
        raise KeyError(f"no registry files for recording {fmt!r}")
    for name in files:
        fetch_file(name)
    return cache_dir() / recording.main
