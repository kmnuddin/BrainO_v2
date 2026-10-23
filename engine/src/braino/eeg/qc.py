"""Data-quality checks for EEG recordings.

These are screening measures: they point at channels and stretches worth a look, and change
nothing. Deciding which channels to drop is a separate, approved step.

Amplitude measures use a 1-40 Hz band-passed copy, so slow drifts and DC offsets do not count as
noise. All measures use 1-second windows:

- **Flat:** the signal barely moves (standard deviation below 0.1 µV) in most windows.
- **Noisy:** the channel's typical amplitude, or its share of high-frequency power, is far from
  the other channels (robust z-score above 3.5, using the median and MAD, as PyPREP does).
- **Line noise:** power at the power-line frequency relative to the frequencies around it.
- **Artifacts:** windows where a channel's peak-to-peak amplitude exceeds 150 µV, counted on
  the channels that are not flat, noisy or already marked bad.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

FLAT_STD_V = 1e-7
FLAT_FRACTION = 0.5
NOISY_Z = 3.5
ARTIFACT_PTP_V = 150e-6
STRONG_LINE_NOISE_RATIO = 10.0
_CHUNK_SECONDS = 60
_PSD_SECONDS = 300.0


class ChannelQc(BaseModel):
    channel: str
    median_std_uv: float
    flat_fraction: float = Field(description="Share of 1-s windows that are flat")
    artifact_fraction: float = Field(description="Share of 1-s windows above the threshold")
    high_freq_ratio: float | None = Field(
        description="Power above 40 Hz relative to 1-40 Hz (line frequencies left out)"
    )
    line_noise_ratio: float | None = Field(
        description="Power at the line frequency relative to the neighbouring frequencies"
    )
    flags: list[str] = Field(default_factory=list)


class NoisyChannel(BaseModel):
    channel: str
    reason: str
    z: float = Field(description="Robust z-score that flagged the channel")


class QcResult(BaseModel):
    """The compact result; ``channels`` holds the per-channel table."""

    duration_s: float
    n_channels_checked: int
    marked_bad_in_file: list[str]
    flat: list[str]
    noisy: list[NoisyChannel]
    line_freq_hz: float | None
    line_noise_median_ratio: float | None
    strong_line_noise: list[str] = Field(
        description=f"Channels whose line-noise ratio exceeds {STRONG_LINE_NOISE_RATIO:g}"
    )
    artifact_fraction: float = Field(
        description="Share of 1-s windows in which any checked channel exceeds the threshold"
    )
    artifact_threshold_uv: float = ARTIFACT_PTP_V * 1e6
    notes: list[str] = Field(default_factory=list)
    channels: list[ChannelQc] = Field(default_factory=list, exclude=False)


def compute_qc(raw: Any, *, line_freq: float | None = None) -> QcResult:
    """Screen the EEG (and intracranial) channels of ``raw``."""
    import mne
    import numpy as np

    picks = mne.pick_types(raw.info, eeg=True, seeg=True, ecog=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    sfreq = float(raw.info["sfreq"])
    notes: list[str] = []
    if not len(picks):
        return QcResult(
            duration_s=float(raw.times[-1]),
            n_channels_checked=0,
            marked_bad_in_file=list(raw.info["bads"]),
            flat=[],
            noisy=[],
            line_freq_hz=line_freq,
            line_noise_median_ratio=None,
            strong_line_noise=[],
            artifact_fraction=0.0,
            notes=["No EEG or intracranial channels to check."],
        )

    window = max(round(sfreq), 1)
    band = raw.copy().pick(picks).load_data(verbose="error")
    band.filter(1.0, 40.0 if sfreq / 2 > 45 else None, verbose="error")
    stds, ptps = _window_stats(band, list(range(len(names))), window)  # windows x channels
    n_windows = stds.shape[0]
    flat_fraction = (stds < FLAT_STD_V).mean(axis=0)
    median_std = np.median(stds, axis=0)
    flat = flat_fraction > FLAT_FRACTION

    hf_ratio, line_ratio = _spectral_ratios(raw, picks, line_freq, notes)
    noisy_reasons: dict[int, tuple[str, float]] = {}
    live = ~flat
    amp_z = _robust_z(np.log(np.maximum(median_std, 1e-15)), live)
    for k in np.flatnonzero(live & (amp_z > NOISY_Z)):
        noisy_reasons[int(k)] = ("amplitude far above the other channels", float(amp_z[k]))
    if hf_ratio is not None:
        hf_z = _robust_z(np.log(np.maximum(hf_ratio, 1e-15)), live)
        for k in np.flatnonzero(live & (hf_z > NOISY_Z)):
            noisy_reasons.setdefault(
                int(k), ("high-frequency power far above the other channels", float(hf_z[k]))
            )

    bads = set(raw.info["bads"])
    checked = np.array(
        [not flat[i] and i not in noisy_reasons and names[i] not in bads for i in range(len(names))]
    )
    over = ptps > ARTIFACT_PTP_V
    artifact_fraction = float(over[:, checked].any(axis=1).mean()) if checked.any() else 0.0

    strong = []
    channels = []
    channel_line_ratio: dict[str, float] = {}
    for i, name in enumerate(names):
        flags = []
        if flat[i]:
            flags.append("flat")
        if i in noisy_reasons:
            flags.append("noisy")
        if name in bads:
            flags.append("marked bad in file")
        ratio = None if line_ratio is None else float(line_ratio[i])
        if ratio is not None:
            channel_line_ratio[name] = ratio
        if ratio is not None and ratio > STRONG_LINE_NOISE_RATIO:
            flags.append("strong line noise")
            strong.append(name)
        channels.append(
            ChannelQc(
                channel=name,
                median_std_uv=round(float(median_std[i]) * 1e6, 3),
                flat_fraction=round(float(flat_fraction[i]), 3),
                artifact_fraction=round(float(over[:, i].mean()), 3),
                high_freq_ratio=None if hf_ratio is None else round(float(hf_ratio[i]), 4),
                line_noise_ratio=None if ratio is None else round(ratio, 2),
                flags=flags,
            )
        )
    if n_windows < 10:
        notes.append(f"Only {n_windows} one-second windows; the measures are rough.")
    return QcResult(
        duration_s=round(float(raw.times[-1]), 3),
        n_channels_checked=len(names),
        marked_bad_in_file=sorted(bads),
        flat=[names[i] for i in np.flatnonzero(flat)],
        noisy=sorted(
            (
                NoisyChannel(channel=names[i], reason=reason, z=round(z, 1))
                for i, (reason, z) in noisy_reasons.items()
            ),
            key=lambda n: -n.z,
        ),
        line_freq_hz=line_freq,
        line_noise_median_ratio=None
        if line_ratio is None
        else round(float(np.median(line_ratio[live] if live.any() else line_ratio)), 2),
        strong_line_noise=sorted(strong, key=lambda n: -channel_line_ratio[n]),
        artifact_fraction=round(artifact_fraction, 4),
        notes=notes,
        channels=channels,
    )


def _window_stats(raw: Any, picks: Any, window: int) -> tuple[Any, Any]:
    """Per 1-s window and channel: standard deviation and peak-to-peak amplitude."""
    import numpy as np

    n_windows = max(raw.n_times // window, 1)
    chunk = window * _CHUNK_SECONDS
    stds, ptps = [], []
    for start in range(0, n_windows * window, chunk):
        stop = min(start + chunk, n_windows * window, raw.n_times)
        data = raw.get_data(picks=picks, start=start, stop=stop)
        usable = (data.shape[1] // window) * window
        if usable == 0:
            usable = data.shape[1]
            segments = data[:, None, :usable]
        else:
            segments = data[:, :usable].reshape(data.shape[0], -1, window)
        stds.append(segments.std(axis=2).T)
        ptps.append(np.ptp(segments, axis=2).T)
    return np.concatenate(stds), np.concatenate(ptps)


def _spectral_ratios(
    raw: Any, picks: Any, line_freq: float | None, notes: list[str]
) -> tuple[Any | None, Any | None]:
    """Per channel: high-frequency power ratio, and line-noise ratio (``None`` if not
    measurable at this sampling rate)."""
    import numpy as np

    sfreq = float(raw.info["sfreq"])
    if sfreq < 100:
        notes.append("Sampling rate below 100 Hz: high-frequency and line noise not measured.")
        return None, None
    n_fft = 1
    while n_fft < 2 * sfreq:
        n_fft *= 2
    n_fft = min(n_fft, raw.n_times)
    fmax = min(sfreq / 2 - 1, 100.0)
    tmax = min(float(raw.times[-1]), _PSD_SECONDS)
    # Exactly the checked channels, bad marks cleared, so none is left out of the spectrum.
    checked = raw.copy().pick(picks)
    checked.info["bads"] = []
    spectrum = checked.compute_psd(fmin=1.0, fmax=fmax, tmax=tmax, n_fft=n_fft, verbose="error")
    power, freqs = spectrum.get_data(), spectrum.freqs
    low = (freqs >= 1) & (freqs < 40)
    high = freqs >= 40
    for line in (50.0, 60.0):
        for harmonic in (line, 2 * line):
            high &= np.abs(freqs - harmonic) > 2
    low_power = power[:, low].sum(axis=1)
    hf_ratio = None
    if high.any():
        hf_ratio = power[:, high].sum(axis=1) / np.where(low_power > 0, low_power, np.inf)

    line_ratio = None
    if line_freq is not None and line_freq + 5 < fmax:
        near = np.abs(freqs - line_freq) <= 0.6
        around = (np.abs(freqs - line_freq) >= 2) & (np.abs(freqs - line_freq) <= 5)
        baseline = np.median(power[:, around], axis=1)
        line_ratio = power[:, near].max(axis=1) / np.where(baseline > 0, baseline, np.inf)
    elif line_freq is None:
        notes.append("Power-line frequency unknown: line noise not measured.")
    return hf_ratio, line_ratio


def _robust_z(values: Any, mask: Any) -> Any:
    """Robust z-scores (median and MAD of the ``mask``-ed values); 0 where undefined."""
    import numpy as np

    ref = values[mask]
    if ref.size < 3:
        return np.zeros_like(values)
    median = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - median))
    if mad == 0:
        return np.zeros_like(values)
    return (values - median) / mad
