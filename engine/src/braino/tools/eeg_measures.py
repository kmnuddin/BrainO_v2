"""EEG tools that measure epoched (or continuous) data: ERPs, power spectra, time-frequency.

Each writes the full result per subject to the derivatives, where the group statistics tools
read it, and returns a compact summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from braino.context import RunContext
from braino.eeg.erp import Contrast, Measure, MeasureValue
from braino.eeg.spectral import DEFAULT_BANDS, BandPower, SpectralMethod
from braino.eeg.tfr import UNITS, BaselineMode, TfrExtreme
from braino.tools.base import Risk
from braino.tools.recordings import (
    LABEL_PATTERN,
    RecordingSelector,
    cleanest,
    derivative,
    find_recording,
    read_epochs,
    relative,
    write_tsv,
)
from braino.tools.registry import tool

_EPOCHS_LABEL = Field(
    default="epochs",
    pattern=LABEL_PATTERN,
    description="Label of the epochs to use (as given to eeg.epoch)",
)


def _save_npz(ctx: RunContext, path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    ctx.record_output(path)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)


# --- eeg.erp ------------------------------------------------------------------------------


class ErpInput(RecordingSelector):
    label: str = _EPOCHS_LABEL
    conditions: list[str] | None = Field(
        default=None, description="Conditions to average; null for all"
    )
    contrasts: list[Contrast] = Field(
        default_factory=list, description="Difference waves to compute, e.g. rare minus frequent"
    )
    measures: list[Measure] = Field(
        default_factory=list,
        description="Amplitude measures to take from every average and difference wave",
    )


class ErpOutput(BaseModel):
    epochs: str = Field(description="The epochs averaged, relative to the dataset")
    output: str = Field(description="Averages and difference waves (MNE .fif), relative path")
    table: str | None = Field(description="Measures table, relative to the dataset")
    n_trials: dict[str, int] = Field(description="Epochs averaged per condition")
    waveforms: list[str] = Field(description="Averages and difference waves written")
    measures: list[MeasureValue]
    notes: list[str]


@tool(name="eeg.erp", risk=Risk.COMPUTE)
def erp(params: ErpInput, ctx: RunContext) -> ErpOutput:
    """Average the epochs of each condition into event-related potentials, compute difference
    waves, and measure amplitudes (mean amplitude or peak and latency in a time window over
    chosen channels). Run eeg.epoch first. The averages are written for group statistics."""
    import mne

    from braino.eeg.erp import condition_averages, measure

    recording = find_recording(ctx.dataset, params)
    epochs, epochs_file = read_epochs(ctx, recording, params.label)
    evokeds, notes = condition_averages(epochs, params.conditions, params.contrasts)
    if not evokeds:
        raise ValueError("no condition has epochs left to average")
    out = derivative(ctx, recording, f"desc-{params.label}_ave.fif")
    ctx.record_output(out)
    mne.write_evokeds(out, list(evokeds.values()), overwrite=True, verbose="error")

    values = [measure(ev, m, name) for m in params.measures for name, ev in evokeds.items()]
    table = None
    if values:
        table = derivative(ctx, recording, f"desc-{params.label}_erp.tsv")
        write_tsv(
            ctx,
            table,
            [
                "measure",
                "waveform",
                "kind",
                "channels",
                "tmin_s",
                "tmax_s",
                "value_uv",
                "latency_ms",
                "n_trials",
            ],
            [
                [
                    v.measure,
                    v.waveform,
                    v.kind,
                    ",".join(v.channels),
                    m.tmin,
                    m.tmax,
                    v.value_uv,
                    v.latency_ms,
                    v.n_trials,
                ]
                for m in params.measures
                for v in values
                if v.measure == m.name
            ],
        )
    return ErpOutput(
        epochs=relative(ctx, epochs_file),
        output=relative(ctx, out),
        table=None if table is None else relative(ctx, table),
        n_trials={name: int(ev.nave) for name, ev in evokeds.items() if name in epochs.event_id},
        waveforms=list(evokeds),
        measures=values,
        notes=notes,
    )


# --- eeg.spectral -------------------------------------------------------------------------


class SpectralInput(RecordingSelector):
    source: Literal["epochs", "continuous"] = Field(
        default="epochs",
        description="'epochs' (per condition, average of the epochs' spectra) or "
        "'continuous' (the whole cleaned recording, e.g. for resting state)",
    )
    label: str = _EPOCHS_LABEL
    method: SpectralMethod = Field(default="welch", description="'welch' or 'multitaper'")
    fmin: float = Field(default=1.0, ge=0, description="Lowest frequency, Hz")
    fmax: float = Field(default=45.0, gt=0, description="Highest frequency, Hz")
    bands: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_BANDS),
        description="Frequency bands, name -> [low, high) Hz",
    )
    conditions: list[str] | None = Field(default=None, description="Conditions; null for all")

    @model_validator(mode="after")
    def _check(self) -> SpectralInput:
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be below fmax")
        for name, (lo, hi) in self.bands.items():
            if lo >= hi:
                raise ValueError(f"band {name!r}: low edge must be below high edge")
        return self


class ConditionSpectrum(BaseModel):
    n_segments: int = Field(description="Epochs averaged (1 for continuous data)")
    peak_alpha_hz: float | None = Field(description="Peak alpha frequency, if there is a peak")
    bands: list[BandPower]


class SpectralOutput(BaseModel):
    input: str = Field(description="The epochs or recording analysed, relative to the dataset")
    output: str = Field(description="Spectra per condition (.npz), relative to the dataset")
    table: str = Field(description="Band power per channel and condition, relative path")
    method: SpectralMethod
    freq_resolution_hz: float
    conditions: dict[str, ConditionSpectrum]
    notes: list[str]


@tool(name="eeg.spectral", risk=Risk.COMPUTE)
def spectral(params: SpectralInput, ctx: RunContext) -> SpectralOutput:
    """Estimate power spectra (Welch or multitaper) and band power (delta, theta, alpha, beta,
    gamma, or chosen bands), per condition from epochs or for the whole continuous recording.
    Reports absolute (µV²) and relative band power and the peak alpha frequency."""
    import numpy as np

    from braino.eeg.spectral import band_power, peak_alpha, spectra, summarise_bands

    recording = find_recording(ctx.dataset, params)
    notes: list[str] = []
    if params.source == "epochs":
        inst, input_file = read_epochs(ctx, recording, params.label)
        label = params.label
        if params.method == "welch" and len(inst.times) / inst.info["sfreq"] < 2:
            notes.append(
                "Epochs shorter than 2 s give coarse frequency resolution; multitaper or "
                "longer epochs may be better."
            )
    else:
        inst, _, source = cleanest(ctx, recording)
        input_file = derivative(ctx, recording, f"desc-{source}_eeg.fif")
        label = "continuous"
        if params.conditions:
            raise ValueError("conditions apply to epochs; use source='epochs'")
    result = spectra(
        inst,
        method=params.method,
        fmin=params.fmin,
        fmax=params.fmax,
        conditions=params.conditions,
    )
    if not result:
        raise ValueError("no condition has epochs left")
    first = next(iter(result.values()))
    out = derivative(ctx, recording, f"desc-{label}_psd.npz")
    arrays: dict[str, Any] = {
        "freqs": first.freqs,
        "ch_names": np.array(first.ch_names),
        "conditions": np.array(list(result)),
        "unit": np.array("V^2/Hz"),
    }
    for i, s in enumerate(result.values()):
        arrays[f"data_{i}"] = s.data
    _save_npz(ctx, out, arrays)

    rows = []
    for name, s in result.items():
        absolute, rel = band_power(s, params.bands)
        for band in params.bands:
            for c, ch in enumerate(s.ch_names):
                rows.append(
                    [
                        name,
                        band,
                        ch,
                        round(float(absolute[band][c]), 5),
                        round(float(rel[band][c]), 5),
                    ]
                )
    table = derivative(ctx, recording, f"desc-{label}_bands.tsv")
    write_tsv(ctx, table, ["condition", "band", "channel", "power_uv2", "relative_power"], rows)
    freqs = first.freqs
    return SpectralOutput(
        input=relative(ctx, input_file),
        output=relative(ctx, out),
        table=relative(ctx, table),
        method=params.method,
        freq_resolution_hz=round(float(np.median(np.diff(freqs))), 4) if freqs.size > 1 else 0.0,
        conditions={
            name: ConditionSpectrum(
                n_segments=s.n_segments,
                peak_alpha_hz=peak_alpha(s),
                bands=summarise_bands(s, params.bands),
            )
            for name, s in result.items()
        },
        notes=notes,
    )


# --- eeg.tfr ------------------------------------------------------------------------------


class TfrInput(RecordingSelector):
    label: str = _EPOCHS_LABEL
    fmin: float = Field(
        default=6.0,
        gt=0,
        description="Lowest frequency, Hz. Low frequencies need long epochs: 6 Hz needs 0.8 s, "
        "4 Hz 1.2 s",
    )
    fmax: float = Field(default=40.0, gt=0, description="Highest frequency, Hz")
    n_freqs: int = Field(default=20, ge=2, le=200, description="Number of log-spaced frequencies")
    cycles_divisor: float = Field(
        default=2.0, gt=0, description="Wavelet cycles = frequency / this (at least 3)"
    )
    baseline: tuple[float | None, float | None] | None = Field(
        default=(None, 0.0), description="Baseline interval in s; null for none"
    )
    mode: BaselineMode = Field(
        default="logratio",
        description="Baseline correction: 'logratio' (dB), 'percent', 'zscore' or 'none' (µV²)",
    )
    decim: int = Field(default=1, ge=1, description="Keep every n-th time point of the result")
    conditions: list[str] | None = Field(default=None, description="Conditions; null for all")
    bands: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {k: v for k, v in DEFAULT_BANDS.items() if k != "delta"},
        description="Bands in which the summary reports the largest change",
    )

    @model_validator(mode="after")
    def _check(self) -> TfrInput:
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be below fmax")
        return self


class TfrOutput(BaseModel):
    epochs: str
    output: str = Field(description="Power per condition (.npz), relative to the dataset")
    unit: str
    n_epochs: dict[str, int]
    freqs_hz: tuple[float, float]
    times_s: tuple[float, float]
    largest_changes: list[TfrExtreme] = Field(
        description="Per condition and band, the largest change after the event (channel mean)"
    )
    notes: list[str]


@tool(name="eeg.tfr", risk=Risk.COMPUTE)
def tfr(params: TfrInput, ctx: RunContext) -> TfrOutput:
    """Time-frequency power with Morlet wavelets, per condition, baseline-corrected (dB by
    default). Shows when power in each frequency band rises or falls relative to the event.
    Run eeg.epoch first; written for group statistics."""
    import numpy as np

    from braino.eeg.tfr import extremes, frequencies, morlet_power

    recording = find_recording(ctx.dataset, params)
    epochs, epochs_file = read_epochs(ctx, recording, params.label)
    nyquist = float(epochs.info["sfreq"]) / 2
    if params.fmax >= nyquist:
        raise ValueError(f"fmax must be below the Nyquist frequency ({nyquist:g} Hz)")
    result, notes = morlet_power(
        epochs,
        freqs=frequencies(params.fmin, params.fmax, params.n_freqs),
        cycles_divisor=params.cycles_divisor,
        baseline=params.baseline,
        mode=params.mode,
        decim=params.decim,
        conditions=params.conditions,
    )
    if not result:
        raise ValueError("no condition has epochs left")
    first = next(iter(result.values()))
    unit = UNITS[params.mode if params.baseline is not None else "none"]
    out = derivative(ctx, recording, f"desc-{params.label}_tfr.npz")
    arrays: dict[str, Any] = {
        "freqs": first.freqs,
        "times": first.times,
        "ch_names": np.array(first.ch_names),
        "conditions": np.array(list(result)),
        "unit": np.array(unit),
    }
    for i, t in enumerate(result.values()):
        arrays[f"data_{i}"] = t.data.astype(np.float32)
    _save_npz(ctx, out, arrays)
    return TfrOutput(
        epochs=relative(ctx, epochs_file),
        output=relative(ctx, out),
        unit=unit,
        n_epochs={name: t.n_epochs for name, t in result.items()},
        freqs_hz=(round(float(first.freqs[0]), 2), round(float(first.freqs[-1]), 2)),
        times_s=(round(float(first.times[0]), 3), round(float(first.times[-1]), 3)),
        largest_changes=extremes(result, params.bands),
        notes=notes,
    )
