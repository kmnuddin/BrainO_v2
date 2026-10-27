"""Time-frequency analysis with Morlet wavelets.

Power is computed per epoch and averaged per condition (so it includes activity not phase-locked
to the event), then baseline-corrected:

- ``logratio``: 10·log10(power / baseline mean), in dB.
- ``percent``: (power - baseline) / baseline, as a fraction.
- ``zscore``: (power - baseline mean) / baseline standard deviation.
- ``none``: power in µV².

The number of cycles grows with frequency (``n_cycles = freq / cycles_divisor``, at least 3), so
time resolution is better at high frequencies and frequency resolution at low ones.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

BaselineMode = Literal["logratio", "percent", "zscore", "none"]

UNITS = {"logratio": "dB", "percent": "fraction of baseline", "zscore": "z", "none": "µV²"}

MIN_CYCLES = 3.0


@dataclass(frozen=True)
class Tfr:
    """Baseline-corrected power of one condition."""

    data: Any
    """channels x frequencies x times, in the unit of the baseline mode"""
    freqs: Any
    times: Any
    ch_names: list[str]
    n_epochs: int


class TfrExtreme(BaseModel):
    condition: str
    band: str
    value: float = Field(description="Largest change (by absolute value), channel mean")
    time_ms: float
    freq_hz: float
    strongest_channel: str = Field(description="Channel with the largest change at that point")


def frequencies(fmin: float, fmax: float, n_freqs: int) -> Any:
    """Log-spaced frequencies from ``fmin`` to ``fmax``."""
    import numpy as np

    return np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)


def morlet_power(
    epochs: Any,
    *,
    freqs: Any,
    cycles_divisor: float = 2.0,
    baseline: tuple[float | None, float | None] | None = (None, 0.0),
    mode: BaselineMode = "logratio",
    decim: int = 1,
    conditions: list[str] | None = None,
) -> tuple[dict[str, Tfr], list[str]]:
    """Baseline-corrected Morlet power of the EEG channels, per condition."""
    import mne
    import numpy as np

    notes: list[str] = []
    n_cycles = np.maximum(freqs / cycles_divisor, MIN_CYCLES)
    # MNE's wavelets span 10 standard deviations: 5 * n_cycles / (pi * f) seconds.
    longest = float((5 * n_cycles / (np.pi * freqs)).max())
    duration = float(epochs.times[-1] - epochs.times[0])
    if longest > duration:
        raise ValueError(
            f"the lowest frequency needs {longest:.2f} s of data but epochs are {duration:.2f} "
            "s long; raise fmin or use longer epochs"
        )
    if longest / 2 > (0 - epochs.times[0]) and baseline is not None:
        notes.append(
            "The baseline is within half a wavelet of the epoch start at the lowest "
            "frequencies, so edge effects may affect it; longer epochs would help."
        )
    picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    names = [epochs.ch_names[i] for i in picks]
    wanted = list(epochs.event_id) if conditions is None else conditions
    unknown = [c for c in wanted if c not in epochs.event_id]
    if unknown:
        raise ValueError(f"no condition {', '.join(map(repr, unknown))} in the epochs")
    out = {}
    for name in wanted:
        sub = epochs[name]
        if not len(sub):
            notes.append(f"Condition {name!r} has no epochs left; skipped.")
            continue
        power = sub.compute_tfr(
            "morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            picks=picks,
            decim=decim,
            average=True,
            return_itc=False,
            verbose="error",
        )
        if mode != "none" and baseline is not None:
            power.apply_baseline(baseline, mode=mode, verbose="error")
        data = power.data
        if mode == "logratio" and baseline is not None:
            data = data * 10  # log10 ratio -> dB
        elif mode == "none" or baseline is None:
            data = data * 1e12
        out[name] = Tfr(
            data=data,
            freqs=np.asarray(power.freqs),
            times=np.asarray(power.times),
            ch_names=names,
            n_epochs=len(sub),
        )
    return out, notes


def extremes(
    tfrs: dict[str, Tfr], bands: dict[str, tuple[float, float]], *, tmin: float = 0.0
) -> list[TfrExtreme]:
    """Per condition and band, the largest change after ``tmin`` (s) in the channel mean."""
    import numpy as np

    out = []
    for condition, tfr in tfrs.items():
        late = tfr.times >= tmin
        for band, (lo, hi) in bands.items():
            fmask = (tfr.freqs >= lo) & (tfr.freqs < hi)
            if not fmask.any() or not late.any():
                continue
            block = tfr.data[:, fmask][:, :, late]  # ch x f x t
            mean = block.mean(axis=0)
            f, t = np.unravel_index(int(np.argmax(np.abs(mean))), mean.shape)
            out.append(
                TfrExtreme(
                    condition=condition,
                    band=band,
                    value=round(float(mean[f, t]), 4),
                    time_ms=round(float(tfr.times[late][t]) * 1000, 1),
                    freq_hz=round(float(tfr.freqs[fmask][f]), 2),
                    strongest_channel=tfr.ch_names[int(np.argmax(np.abs(block[:, f, t])))],
                )
            )
    return out
