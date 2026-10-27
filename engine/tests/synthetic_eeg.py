"""Synthetic EEG with known content, for testing analysis steps against a known answer.

Each recording has 1/f-like background activity on 20 scalp channels of the 10-20 system, a
VEOG channel, eye blinks that are strong at the front of the head, and two kinds of event:
``stimulus/rare`` (1 in 4) and ``stimulus/frequent``. Rare events are followed by a positive
deflection over the back of the head (a P3-like effect: ``P3_UV`` µV at Pz from 300 to 500 ms,
weaker with distance from Pz), so a reference to the average does not remove it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

SFREQ = 250.0
EEG = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]  # fmt: skip
P3_UV = 8.0
P3_WINDOW = (0.3, 0.5)
EVENT_SPACING_S = 1.0


def _positions() -> dict[str, Any]:
    montage = mne.channels.make_standard_montage("colin27_1020")
    return montage.get_positions()["ch_pos"]  # type: ignore[no-any-return]


def p3_weights() -> np.ndarray:
    """How strongly each EEG channel shows the P3 (1 at Pz, falling off with distance)."""
    pos = _positions()
    pz = pos["Pz"]
    d = np.array([np.linalg.norm(pos[ch] - pz) for ch in EEG])
    return np.exp(-((d / 0.07) ** 2))


def blink_weights() -> np.ndarray:
    pos = _positions()
    front = (pos["Fp1"] + pos["Fp2"]) / 2
    d = np.array([np.linalg.norm(pos[ch] - front) for ch in EEG])
    return np.exp(-((d / 0.06) ** 2))


def make_raw(
    seconds: float = 90.0,
    *,
    seed: int = 0,
    p3_uv: float = P3_UV,
    blinks: int = 30,
    flat: list[str] | None = None,
    noisy: list[str] | None = None,
    line_noise_uv: float = 0.0,
) -> mne.io.RawArray:
    """One synthetic recording (see the module docstring)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SFREQ)
    times = np.arange(n) / SFREQ
    # Background: 1/f-like sources under the scalp, seen by nearby electrodes (so neighbouring
    # channels are alike, as in real EEG), a little channel noise, and a 10 Hz rhythm at the back.
    pos = _positions()
    xyz = np.array([pos[ch] for ch in EEG])
    centres = xyz[rng.choice(len(EEG), 8, replace=False)] * 0.8
    mixing = np.exp(-((np.linalg.norm(xyz[:, None] - centres[None], axis=2) / 0.08) ** 2))
    sources = np.cumsum(rng.standard_normal((len(centres), n)), axis=1)
    sources -= np.linspace(sources[:, 0], sources[:, -1], n, axis=1)
    data = mixing @ sources
    data *= 6e-6 / data.std(axis=1, keepdims=True)
    data += 1.5e-6 * rng.standard_normal((len(EEG), n))
    alpha_w = np.clip(p3_weights() + 0.3, 0, 1)
    data += 4e-6 * alpha_w[:, None] * np.sin(2 * np.pi * 10 * times + rng.uniform(0, 6.28))

    onsets = np.arange(2.0, seconds - 2.0, EVENT_SPACING_S)
    kinds = np.where(rng.random(onsets.size) < 0.25, "rare", "frequent")
    if kinds.size >= 4:
        kinds[:4] = ["rare", "frequent", "frequent", "frequent"]
    wave = np.zeros(n)
    width = int((P3_WINDOW[1] - P3_WINDOW[0]) * SFREQ)
    bump = np.hanning(width + 20)[10:-10]
    bump /= bump.mean()  # mean over the window is exactly 1
    for onset, kind in zip(onsets, kinds, strict=True):
        if kind == "rare":
            start = round((onset + P3_WINDOW[0]) * SFREQ)
            wave[start : start + width] += bump
    data += p3_uv * 1e-6 * p3_weights()[:, None] * wave[None]

    eye = np.zeros(n)
    for start in rng.integers(0, n - 100, blinks):
        eye[start : start + 75] += np.hanning(75) * 120e-6
    data += blink_weights()[:, None] * eye[None]
    veog = eye + 4e-6 * rng.standard_normal(n)
    if line_noise_uv:
        data += line_noise_uv * 1e-6 * np.sin(2 * np.pi * 50 * times)

    for ch in flat or []:
        data[EEG.index(ch)] = 0.0
    for ch in noisy or []:
        data[EEG.index(ch)] += 60e-6 * rng.standard_normal(n)

    info = mne.create_info([*EEG, "VEOG"], SFREQ, ch_types=["eeg"] * len(EEG) + ["eog"])
    raw = mne.io.RawArray(np.vstack([data, veog[None]]), info, verbose="error")
    raw.set_montage("colin27_1020", on_missing="ignore")
    raw.set_annotations(
        mne.Annotations(onset=onsets, duration=0.0, description=[f"stimulus/{k}" for k in kinds])
    )
    raw.info["line_freq"] = 50.0
    return raw


def write_dataset(root: Path, subjects: list[str], *, task: str = "oddball", **kwargs: Any) -> Path:
    """A BIDS dataset with one synthetic recording per subject (seeded by the label)."""
    from braino.io import FORMATS, Recording, write_bids

    for subject in subjects:
        raw = make_raw(seed=int(subject), **kwargs)
        recording = Recording(raw, FORMATS["fif"], Path("synthetic.fif"), (Path("synthetic.fif"),))
        write_bids(recording, root, subject=subject, task=task, line_freq=50.0)
    return root
