"""Filtering, notch filtering, resampling and re-referencing of continuous EEG.

The steps run in a fixed order, the one MNE-Python recommends: notch, band-pass, resample,
re-reference. Every step that ran is listed in the report, so the provenance record says
exactly what was done to the data.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

FilterMethod = Literal["fir", "iir"]

DATA_TYPES = ("eeg", "eog", "ecg", "emg")
"""Channel types kept by preprocessing; trigger and miscellaneous channels are dropped."""


class PreprocessParams(BaseModel):
    l_freq: float | None = Field(
        default=0.1, ge=0, description="High-pass cut-off in Hz; null for no high-pass"
    )
    h_freq: float | None = Field(
        default=40.0, gt=0, description="Low-pass cut-off in Hz; null for no low-pass"
    )
    filter_method: FilterMethod = Field(
        default="fir",
        description="'fir' (zero-phase windowed FIR, MNE's default) or 'iir' (zero-phase "
        "4th-order Butterworth)",
    )
    notch_line_noise: bool = Field(
        default=False,
        description="Notch-filter the power-line frequency and its harmonics below the low-pass",
    )
    notch_freqs: list[float] = Field(
        default_factory=list, description="Further frequencies (Hz) to notch-filter"
    )
    resample_hz: float | None = Field(
        default=None, gt=0, description="New sampling rate in Hz; null keeps the original"
    )
    reference: Literal["average", "average_with_eog", "none"] | list[str] = Field(
        default="average",
        description="'average' (of the good EEG channels), 'average_with_eog' (of the good EEG "
        "and EOG channels, subtracted from both, as ERP CORE does), 'none' to keep the "
        "recording's reference, or the channels to reference to, e.g. ['P9', 'P10'] for "
        "linked mastoids",
    )

    @field_validator("reference")
    @classmethod
    def _non_empty(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("give at least one reference channel, or 'average' or 'none'")
        return value


class PreprocessReport(BaseModel):
    steps: list[str] = Field(description="What was done, in order")
    sfreq_hz: float
    duration_s: float
    channels: dict[str, int] = Field(description="Channels kept, per type")
    dropped_channels: list[str] = Field(description="Trigger and other non-data channels")
    bad_channels: list[str] = Field(description="Channels marked bad (left out of the reference)")
    reference: str | list[str]
    notes: list[str] = Field(default_factory=list)


def preprocess(
    raw: Any, params: PreprocessParams, *, line_freq: float | None = None
) -> tuple[Any, PreprocessReport]:
    """Return a preprocessed copy of ``raw`` and a report of what was done."""
    import mne

    notes: list[str] = []
    steps: list[str] = []
    out = raw.copy().load_data(verbose="error")
    keep = [
        ch
        for ch, kind in zip(out.ch_names, out.get_channel_types(), strict=True)
        if kind in DATA_TYPES
    ]
    dropped = [ch for ch in out.ch_names if ch not in keep]
    if not keep:
        raise ValueError("the recording has no EEG, EOG, ECG or EMG channels")
    out.pick(keep)
    if dropped:
        steps.append(f"dropped {len(dropped)} non-data channels")

    sfreq = float(out.info["sfreq"])
    nyquist = sfreq / 2
    h_freq = params.h_freq
    if h_freq is not None and h_freq >= nyquist:
        notes.append(f"Low-pass {h_freq:g} Hz is not below the Nyquist frequency; skipped.")
        h_freq = None
    if params.l_freq is not None and h_freq is not None and params.l_freq >= h_freq:
        raise ValueError(f"high-pass {params.l_freq:g} Hz must be below low-pass {h_freq:g} Hz")

    picks = mne.pick_types(out.info, eeg=True, eog=True, ecg=True, emg=True, exclude=[])
    notch = _notch_freqs(params, line_freq, h_freq, nyquist, notes)
    if notch:
        out.notch_filter(notch, picks=picks, verbose="error")
        steps.append(f"notch {', '.join(f'{f:g}' for f in notch)} Hz")

    l_freq = params.l_freq or None
    if l_freq is not None or h_freq is not None:
        iir = {"order": 4, "ftype": "butter"} if params.filter_method == "iir" else None
        out.filter(
            l_freq,
            h_freq,
            picks=picks,
            method=params.filter_method,
            iir_params=iir,
            phase="zero",
            verbose="error",
        )
        steps.append(f"{_band_text(l_freq, h_freq)} ({params.filter_method}, zero-phase)")

    if params.resample_hz is not None and params.resample_hz != sfreq:
        if h_freq is None or h_freq >= params.resample_hz / 2:
            notes.append(
                "Resampling applies its own anti-alias filter, because the low-pass is not "
                "below the new Nyquist frequency."
            )
        out.resample(params.resample_hz, verbose="error")
        steps.append(f"resample {sfreq:g} -> {params.resample_hz:g} Hz")

    reference = apply_reference(out, params.reference)
    if reference != "none":
        steps.append(f"reference: {_reference_text(reference)}")

    return out, PreprocessReport(
        steps=steps,
        sfreq_hz=float(out.info["sfreq"]),
        duration_s=round(float(out.times[-1]), 3),
        channels=_count_types(out),
        dropped_channels=dropped,
        bad_channels=list(out.info["bads"]),
        reference=reference,
        notes=notes,
    )


def apply_reference(raw: Any, reference: str | list[str]) -> str | list[str]:
    """Re-reference the EEG channels of ``raw`` in place; returns what was applied.

    Re-referencing to channels gives the same result whatever the previous reference was, so it
    can be re-applied after a step that needed another reference (ICA works on average-referenced
    data). The average leaves out channels marked bad.
    """
    if reference == "none":
        return "none"
    eeg = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True) if kind == "eeg"
    ]
    if not eeg:
        return "none"
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="error")
        return "average"
    if reference == "average_with_eog":
        _average_with_eog(raw)
        return "average_with_eog"
    missing = [ch for ch in reference if ch not in eeg]
    if missing:
        raise ValueError(f"reference channels not among the EEG channels: {', '.join(missing)}")
    bad = [ch for ch in reference if ch in raw.info["bads"]]
    if bad:
        raise ValueError(f"reference channels are marked bad: {', '.join(bad)}")
    raw.set_eeg_reference(list(reference), projection=False, verbose="error")
    return list(reference)


def _average_with_eog(raw: Any) -> None:
    """Subtract the mean of the good EEG and EOG channels from both.

    Like any common reference this can be re-applied, but only while EEG and EOG share their
    reference: not after a reference that changed only the EEG channels (MNE's average and
    channel references leave EOG channels as they are)."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude="bads")
    targets = mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])
    if not len(picks):
        return
    data = raw._data  # loaded by every caller
    data[targets] -= data[picks].mean(axis=0, keepdims=True)
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = mne.io.constants.FIFF.FIFFV_MNE_CUSTOM_REF_ON


def _notch_freqs(
    params: PreprocessParams,
    line_freq: float | None,
    h_freq: float | None,
    nyquist: float,
    notes: list[str],
) -> list[float]:
    limit = min(h_freq or nyquist, nyquist - 1)
    freqs = [f for f in params.notch_freqs if f < nyquist]
    if params.notch_line_noise:
        if line_freq is None:
            notes.append("Power-line frequency unknown; line noise not notch-filtered.")
        else:
            k = 1
            while k * line_freq <= limit:
                freqs.append(k * line_freq)
                k += 1
            if k == 1:
                notes.append(
                    f"Line frequency {line_freq:g} Hz is above the low-pass; no notch needed."
                )
    return sorted(set(freqs))


def _band_text(l_freq: float | None, h_freq: float | None) -> str:
    if l_freq is not None and h_freq is not None:
        return f"band-pass {l_freq:g}-{h_freq:g} Hz"
    if l_freq is not None:
        return f"high-pass {l_freq:g} Hz"
    return f"low-pass {h_freq:g} Hz"


def _reference_text(reference: str | list[str]) -> str:
    return reference if isinstance(reference, str) else "mean of " + ", ".join(reference)


def _count_types(raw: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in raw.get_channel_types():
        counts[kind] = counts.get(kind, 0) + 1
    return counts
