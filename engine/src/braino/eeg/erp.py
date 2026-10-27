"""Event-related potentials: condition averages, difference waves and amplitude measures.

Measures are taken from the average waveform of the measured channels:

- ``mean``: mean amplitude in the time window. The most robust ERP measure (Luck, 2014), and
  the one ERP CORE scores.
- ``peak_positive`` / ``peak_negative``: the most positive / negative point in the window, with
  its latency.

Amplitudes are in µV.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MeasureKind = Literal["mean", "peak_positive", "peak_negative"]


class Contrast(BaseModel):
    name: str = Field(description="Name of the difference wave, e.g. 'rare_minus_frequent'")
    plus: str = Field(description="Condition counted positively")
    minus: str = Field(description="Condition subtracted")


class Measure(BaseModel):
    name: str = Field(description="Name of the measure, e.g. 'P3'")
    channels: list[str] = Field(min_length=1, description="Channels averaged before measuring")
    tmin: float = Field(description="Window start, s")
    tmax: float = Field(description="Window end, s")
    kind: MeasureKind = Field(
        default="mean", description="'mean', 'peak_positive', 'peak_negative'"
    )

    @model_validator(mode="after")
    def _window(self) -> Measure:
        if self.tmin >= self.tmax:
            raise ValueError(f"measure {self.name!r}: tmin must be before tmax")
        return self


class MeasureValue(BaseModel):
    measure: str
    waveform: str = Field(description="Condition or contrast measured")
    kind: MeasureKind
    value_uv: float
    latency_ms: float | None = Field(default=None, description="Peak latency (peak measures)")
    channels: list[str]
    n_trials: int = Field(
        description="Trials averaged; for a difference wave, the effective number 1 / (1/n1 + 1/n2)"
    )


def condition_averages(
    epochs: Any, conditions: list[str] | None = None, contrasts: list[Contrast] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Average each condition and compute each contrast; returns ``{name: Evoked}`` and notes."""
    import mne

    notes: list[str] = []
    available = [c for c in epochs.event_id if len(epochs[c])]
    wanted = available if conditions is None else conditions
    unknown = [c for c in wanted if c not in epochs.event_id]
    if unknown:
        raise ValueError(
            f"no condition {', '.join(map(repr, unknown))}; the epochs have "
            f"{', '.join(epochs.event_id)}"
        )
    evokeds: dict[str, Any] = {}
    for name in wanted:
        if not len(epochs[name]):
            notes.append(f"Condition {name!r} has no epochs left; not averaged.")
            continue
        evoked = epochs[name].average(picks=["eeg", "eog"] if _has_eog(epochs) else ["eeg"])
        evoked.comment = name
        evokeds[name] = evoked
    for contrast in contrasts or []:
        if contrast.name in evokeds:
            raise ValueError(f"contrast name {contrast.name!r} is already a condition")
        missing = [c for c in (contrast.plus, contrast.minus) if c not in evokeds]
        if missing:
            notes.append(
                f"Contrast {contrast.name!r} skipped: no average for {', '.join(missing)}."
            )
            continue
        diff = mne.combine_evoked([evokeds[contrast.plus], evokeds[contrast.minus]], [1, -1])
        diff.comment = contrast.name
        evokeds[contrast.name] = diff
    return evokeds, notes


def measure(evoked: Any, spec: Measure, waveform: str) -> MeasureValue:
    """Take one measure from ``evoked``."""
    import numpy as np

    missing = [ch for ch in spec.channels if ch not in evoked.ch_names]
    if missing:
        raise ValueError(f"measure {spec.name!r}: no channel {', '.join(missing)}")
    times = evoked.times
    if spec.tmin < times[0] - 1e-9 or spec.tmax > times[-1] + 1e-9:
        raise ValueError(
            f"measure {spec.name!r}: window {spec.tmin:g}-{spec.tmax:g} s is outside the epoch "
            f"({times[0]:g}-{times[-1]:g} s)"
        )
    window = (times >= spec.tmin - 1e-9) & (times <= spec.tmax + 1e-9)
    trace = evoked.copy().pick(spec.channels).data.mean(axis=0)[window] * 1e6
    latency = None
    if spec.kind == "mean":
        value = float(trace.mean())
    else:
        k = int(np.argmax(trace) if spec.kind == "peak_positive" else np.argmin(trace))
        value = float(trace[k])
        latency = round(float(times[window][k]) * 1000, 1)
    return MeasureValue(
        measure=spec.name,
        waveform=waveform,
        kind=spec.kind,
        value_uv=round(value, 4),
        latency_ms=latency,
        channels=spec.channels,
        n_trials=int(evoked.nave),
    )


def _has_eog(epochs: Any) -> bool:
    return "eog" in epochs.get_channel_types()
