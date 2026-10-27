"""Finding bad EEG channels.

Two methods:

- ``pyprep``: the PREP pipeline's noisy-channel detection (Bigdely-Shamlo et al., 2015, as
  implemented by PyPREP): flat signals, extreme amplitudes, high-frequency noise, low
  correlation with other channels, and channels poorly predicted from the others (RANSAC, which
  needs electrode positions).
- ``qc``: BrainO's data-quality screen (:mod:`braino.eeg.qc`): flat channels and channels whose
  amplitude or high-frequency power is far from the others.

Detection only proposes; dropping or interpolating channels is a separate, approved step.

Needs the ``[eeg]`` extra; ``pyprep`` also needs PyPREP.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from braino.eeg._quiet import quiet

Method = Literal["pyprep", "qc"]

# PyPREP's reasons, in the words shown to the user.
_PYPREP_REASONS = {
    "bad_by_nan": "contains missing values",
    "bad_by_flat": "flat",
    "bad_by_deviation": "amplitude far from the other channels",
    "bad_by_hf_noise": "high-frequency noise",
    "bad_by_correlation": "low correlation with the other channels",
    "bad_by_SNR": "low signal-to-noise ratio",
    "bad_by_psd": "unusual power spectrum",
    "bad_by_dropout": "drops out",
    "bad_by_ransac": "poorly predicted from the other channels (RANSAC)",
}

CORRELATION_MAX_BAD_TIME = 0.05
"""pyprep: a channel is bad by correlation when its highest correlation with any other channel
is below 0.4 in more than this share of 1-s windows. PREP uses 1%, which on low-density caps
marks most channels bad (13 of 30 in ERP CORE participant 001, whose data are clean): with few
channels, neighbours are far apart and brief low correlations are common. 5% keeps the
channels that are bad for longer, which also fail PREP's other criteria."""

MIN_RANSAC_CHANNELS = 16
"""RANSAC predicts each channel from a random subset of the others, so it needs enough of them."""


class BadChannelResult(BaseModel):
    channels: list[str] = Field(description="Channels found bad, in recording order")
    reasons: dict[str, list[str]] = Field(description="Why each channel was found bad")
    already_marked: list[str] = Field(description="Channels already marked bad before detection")
    n_checked: int
    method: Method
    notes: list[str] = Field(default_factory=list)


def find_bad_channels(
    raw: Any,
    *,
    method: Method = "pyprep",
    ransac: bool = True,
    seed: int = 42,
    line_freq: float | None = None,
    correlation_max_bad_time: float = CORRELATION_MAX_BAD_TIME,
    leave_out_blinks: bool = True,
) -> BadChannelResult:
    """Find bad EEG channels in ``raw`` (unchanged). Channels already marked bad stay bad.

    With ``leave_out_blinks``, blinks (found in the EOG channels, or else in Fp1/Fp2) and the
    time around them are left out, so that channels near the eyes are not marked bad for
    showing blinks: ICA removes blinks, and needs those channels to do so.
    """
    eeg = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True) if kind == "eeg"
    ]
    already = [ch for ch in raw.info["bads"] if ch in eeg]
    if not eeg:
        return BadChannelResult(
            channels=[],
            reasons={},
            already_marked=[],
            n_checked=0,
            method=method,
            notes=["No EEG channels to check."],
        )
    # Both methods compare channels with each other, which assumes no channel is special.
    # With a channel reference (e.g. mastoids) the reference channels look flat or extreme, so
    # detection runs on an average-referenced copy. Flat channels are found first, on the data
    # as given: after re-referencing they would carry minus the average and no longer look
    # flat. They and the channels already marked bad are left out of the comparison.
    data = raw.copy().pick(eeg).load_data(verbose="error")
    flat = [ch for ch in flat_channels(data) if ch not in already]
    reasons: dict[str, list[str]] = {ch: ["flat"] for ch in flat}
    usable = [ch for ch in eeg if ch not in flat and ch not in already]
    notes: list[str] = []
    if len(usable) >= 3:
        data.pick(usable)
        data.info["bads"] = []
        data.set_eeg_reference("average", projection=False, verbose="error")
        if leave_out_blinks:
            data = _without_blinks(raw, data, notes)
        if method == "pyprep":
            found, more = _pyprep(
                data,
                usable,
                ransac=ransac,
                seed=seed,
                correlation_max_bad_time=correlation_max_bad_time,
            )
        else:
            found, more = _qc(data, line_freq)
        notes.extend(more)
        for ch, why in found.items():
            reasons.setdefault(ch, []).extend(w for w in why if w not in reasons[ch])
    else:
        notes.append("Fewer than 3 usable EEG channels; only flat channels were looked for.")
    for ch in already:
        reasons.setdefault(ch, []).insert(0, "already marked bad")
    ordered = [ch for ch in eeg if ch in reasons]
    return BadChannelResult(
        channels=ordered,
        reasons={ch: reasons[ch] for ch in ordered},
        already_marked=already,
        n_checked=len(eeg),
        method=method,
        notes=notes,
    )


BLINK_WINDOW_S = (-0.3, 0.5)
"""Time around each blink peak left out of bad-channel detection."""

_FRONTAL = ("Fp1", "Fp2", "Fpz", "FP1", "FP2", "FPz", "FPZ")


def _without_blinks(raw: Any, data: Any, notes: list[str]) -> Any:
    """``data`` without the blinks of ``raw`` (and the time around them), as one recording."""
    import mne
    import numpy as np

    eog = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True) if kind == "eog"
    ]
    channels = eog or [ch for ch in _FRONTAL if ch in raw.ch_names]
    if not channels:
        notes.append("No EOG or Fp channels: blinks were not left out.")
        return data
    try:
        events = mne.preprocessing.find_eog_events(raw, ch_name=channels, verbose="error")
    except (ValueError, RuntimeError):
        events = np.empty((0, 3), dtype=int)
    if not len(events):
        return data
    sfreq = float(data.info["sfreq"])
    keep = np.ones(data.n_times, dtype=bool)
    start, stop = (round(t * sfreq) for t in BLINK_WINDOW_S)
    for sample in events[:, 0] - raw.first_samp:
        keep[max(sample + start, 0) : max(sample + stop, 0)] = False
    if keep.mean() < 0.5:
        notes.append(f"{len(events)} blinks cover most of the recording; they were not left out.")
        return data
    notes.append(f"{len(events)} blinks ({1 - keep.mean():.0%} of the time) were left out.")
    return mne.io.RawArray(data.get_data()[:, keep], data.info, verbose="error")


def flat_channels(raw: Any) -> list[str]:
    """EEG channels whose 1-s standard deviation is below 0.1 µV in most windows (as in
    :mod:`braino.eeg.qc`)."""
    import numpy as np

    from braino.eeg.qc import FLAT_FRACTION, FLAT_STD_V

    picks = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
    window = max(round(float(raw.info["sfreq"])), 1)
    data = raw.get_data(picks=picks)
    usable = (data.shape[1] // window) * window
    if usable == 0:
        return []
    stds = data[:, :usable].reshape(len(picks), -1, window).std(axis=2)
    flat = (stds < FLAT_STD_V).mean(axis=1) > FLAT_FRACTION
    return [raw.ch_names[picks[i]] for i in np.flatnonzero(flat)]


def _pyprep(
    raw: Any, eeg: list[str], *, ransac: bool, seed: int, correlation_max_bad_time: float
) -> tuple[dict[str, list[str]], list[str]]:
    from pyprep import NoisyChannels

    notes: list[str] = []
    data = raw.copy()
    data.info["bads"] = []
    if ransac:
        placed = [ch for ch in eeg if _has_position(data, ch)]
        if len(placed) < len(eeg):
            notes.append("RANSAC skipped: not every EEG channel has a position.")
            ransac = False
        elif len(eeg) < MIN_RANSAC_CHANNELS:
            notes.append(f"RANSAC skipped: fewer than {MIN_RANSAC_CHANNELS} EEG channels.")
            ransac = False
    with quiet():
        noisy = NoisyChannels(data, random_state=seed, ransac=ransac)  # also finds NaN/flat
        noisy.find_bad_by_deviation()
        noisy.find_bad_by_hfnoise()
        noisy.find_bad_by_correlation(frac_bad=correlation_max_bad_time)
        noisy.find_bad_by_SNR()
        noisy.find_bad_by_PSD()
        if ransac:
            noisy.find_bad_by_ransac()
    found: dict[str, list[str]] = {}
    for key, channels in noisy.get_bads(as_dict=True).items():
        if key == "bad_all":
            continue
        for ch in channels:
            found.setdefault(ch, []).append(_PYPREP_REASONS.get(key, key.removeprefix("bad_by_")))
    return found, notes


def _qc(raw: Any, line_freq: float | None) -> tuple[dict[str, list[str]], list[str]]:
    from braino.eeg.qc import compute_qc

    result = compute_qc(raw, line_freq=line_freq)
    found: dict[str, list[str]] = {}
    for ch in result.flat:
        found.setdefault(ch, []).append("flat")
    for n in result.noisy:
        found.setdefault(n.channel, []).append(f"{n.reason} (robust z {n.z:g})")
    return found, list(result.notes)


def _has_position(raw: Any, ch: str) -> bool:
    loc = raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3]
    return bool((loc == loc).all() and (loc != 0).any())
