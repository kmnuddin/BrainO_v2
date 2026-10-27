"""Power spectra and band power.

Spectra are estimated with Welch's method or multitapers, per condition for epoched data (the
average of the epochs' spectra) or for the whole continuous recording. Band power is the
integral of the spectrum over the band (µV²); relative power is that share of the power between
``fmin`` and ``fmax``.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

SpectralMethod = Literal["welch", "multitaper"]

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ALPHA_RANGE = (7.0, 14.0)
"""Where the peak alpha frequency is looked for."""


@dataclass(frozen=True)
class Spectrum:
    """A spectrum per channel, V²/Hz."""

    freqs: Any
    data: Any
    """channels x frequencies"""
    ch_names: list[str]
    n_segments: int
    """Epochs averaged (1 for continuous data)."""


class BandPower(BaseModel):
    band: str
    fmin: float
    fmax: float
    mean_absolute_uv2: float = Field(description="Mean over channels of the band power, µV²")
    mean_relative: float = Field(description="Mean over channels of the relative band power")
    strongest_channel: str


def spectra(
    inst: Any,
    *,
    method: SpectralMethod = "welch",
    fmin: float = 1.0,
    fmax: float = 45.0,
    conditions: list[str] | None = None,
) -> dict[str, Spectrum]:
    """Spectra of the EEG channels of ``inst``: per condition for epochs, or one called
    ``"continuous"`` for raw data."""
    import mne

    picks = mne.pick_types(inst.info, eeg=True, exclude="bads")
    names = [inst.ch_names[i] for i in picks]
    nyquist = float(inst.info["sfreq"]) / 2
    fmax = min(fmax, nyquist)
    if fmin >= fmax:
        raise ValueError(f"fmin {fmin:g} Hz must be below fmax {fmax:g} Hz")
    if isinstance(inst, mne.BaseEpochs):
        wanted = list(inst.event_id) if conditions is None else conditions
        unknown = [c for c in wanted if c not in inst.event_id]
        if unknown:
            raise ValueError(f"no condition {', '.join(map(repr, unknown))} in the epochs")
        out = {}
        for name in wanted:
            sub = inst[name]
            if not len(sub):
                continue
            psd = sub.compute_psd(method=method, fmin=fmin, fmax=fmax, picks=picks, verbose="error")
            out[name] = Spectrum(
                freqs=psd.freqs,
                data=psd.get_data().mean(axis=0),
                ch_names=names,
                n_segments=len(sub),
            )
        return out
    kwargs: dict[str, Any] = {}
    if method == "welch":
        n_fft = 1
        while n_fft < 2 * inst.info["sfreq"]:
            n_fft *= 2
        kwargs["n_fft"] = min(n_fft, inst.n_times)
    psd = inst.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        reject_by_annotation=True,
        verbose="error",
        **kwargs,
    )
    return {
        "continuous": Spectrum(freqs=psd.freqs, data=psd.get_data(), ch_names=names, n_segments=1)
    }


def band_power(
    spectrum: Spectrum, bands: dict[str, tuple[float, float]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Absolute (µV²) and relative band power per channel: ``{band: array}`` twice."""
    import numpy as np

    freqs = spectrum.freqs
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    total = spectrum.data.sum(axis=1) * df
    absolute, relative = {}, {}
    for band, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = (
            spectrum.data[:, mask].sum(axis=1) * df if mask.any() else np.full(len(total), np.nan)
        )
        absolute[band] = power * 1e12
        relative[band] = power / np.where(total > 0, total, np.nan)
    return absolute, relative


def summarise_bands(spectrum: Spectrum, bands: dict[str, tuple[float, float]]) -> list[BandPower]:
    import numpy as np

    absolute, relative = band_power(spectrum, bands)
    out = []
    for band, (lo, hi) in bands.items():
        values = absolute[band]
        if np.isnan(values).all():
            continue
        out.append(
            BandPower(
                band=band,
                fmin=lo,
                fmax=hi,
                mean_absolute_uv2=round(float(np.nanmean(values)), 4),
                mean_relative=round(float(np.nanmean(relative[band])), 4),
                strongest_channel=spectrum.ch_names[int(np.nanargmax(values))],
            )
        )
    return out


def peak_alpha(spectrum: Spectrum) -> float | None:
    """Frequency of the highest point of the channel-mean spectrum in 7-14 Hz, if it is a peak
    (higher than both ends of that range)."""
    import numpy as np

    freqs = spectrum.freqs
    mask = (freqs >= ALPHA_RANGE[0]) & (freqs <= ALPHA_RANGE[1])
    if mask.sum() < 3:
        return None
    mean = spectrum.data.mean(axis=0)[mask]
    k = int(np.argmax(mean))
    if k in (0, mean.size - 1):
        return None
    return round(float(freqs[mask][k]), 2)
