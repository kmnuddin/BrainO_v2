"""Cutting continuous EEG into epochs around events, with baseline correction and rejection of
epochs with artifacts.

Events are the recording's annotations; ``conditions`` groups their descriptions into named
conditions, e.g. ``{"rare": ["stimulus/11", "stimulus/22"], "frequent": [...]}``.

Rejection methods:

- ``autoreject``: autoreject's local method (Jas et al., 2017). Per channel, a peak-to-peak
  threshold is learned by cross-validation; bad channels in an epoch are interpolated when only
  a few are bad, and the epoch is dropped otherwise.
- ``autoreject_global``: one peak-to-peak threshold for all channels, learned by
  cross-validation (faster; epochs are only dropped, never repaired).
- ``peak_to_peak``: drop epochs whose peak-to-peak amplitude on any EEG channel exceeds a fixed
  threshold.
- ``none``: keep every epoch.

Needs the ``[eeg]`` extra; the autoreject methods also need autoreject.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from braino.eeg._quiet import quiet
from braino.eeg.preprocess import apply_reference

RejectMethod = Literal["autoreject", "autoreject_global", "peak_to_peak", "none"]

HEAVY_LOSS = 0.5
"""A condition losing this share of its epochs or more gets a warning in the notes."""

MIN_AUTOREJECT_EPOCHS = 20
"""Fewer epochs than this are too few for autoreject's cross-validation."""


class EpochParams(BaseModel):
    conditions: dict[str, list[str]] = Field(
        description="Condition name -> event descriptions (annotations) that belong to it, e.g. "
        "{'target': ['stimulus/11'], 'standard': ['stimulus/12', 'stimulus/13']}"
    )
    tmin: float = Field(default=-0.2, description="Epoch start relative to the event, s")
    tmax: float = Field(default=0.8, description="Epoch end relative to the event, s")
    baseline: tuple[float | None, float | None] | None = Field(
        default=(None, 0.0),
        description="Baseline interval in s ([null, 0] = from epoch start to the event); null "
        "for no baseline correction",
    )
    reject: RejectMethod = Field(
        default="autoreject",
        description="'autoreject' (learned per-channel thresholds, repairs or drops epochs), "
        "'autoreject_global' (one learned threshold), 'peak_to_peak' (fixed threshold) or "
        "'none'",
    )
    peak_to_peak_uv: float = Field(
        default=150.0, gt=0, description="Threshold for reject='peak_to_peak', µV"
    )
    interpolate_bads: bool = Field(
        default=True,
        description="Interpolate channels marked bad from their neighbours before epoching",
    )
    event_delay_s: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Measured delay between the event codes and what the participant saw or "
        "heard (e.g. a monitor's delay, measured with a photosensor), s. Events are moved "
        "this much later, to the earlier sample. ERP CORE: 0.026",
    )
    seed: int = Field(default=42, description="Random seed for autoreject")

    @model_validator(mode="after")
    def _check(self) -> EpochParams:
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be before tmax")
        if not self.conditions or any(not v for v in self.conditions.values()):
            raise ValueError("give each condition at least one event description")
        seen: dict[str, str] = {}
        for condition, descriptions in self.conditions.items():
            for d in descriptions:
                if d in seen and seen[d] != condition:
                    raise ValueError(f"event {d!r} is in both {seen[d]!r} and {condition!r}")
                seen[d] = condition
        return self


class ConditionCount(BaseModel):
    events: int
    kept: int
    rejected_fraction: float


class EpochReport(BaseModel):
    conditions: dict[str, ConditionCount]
    n_events: int
    n_kept: int
    rejected_fraction: float
    reject: RejectMethod
    threshold_uv: float | None = Field(
        default=None, description="Peak-to-peak threshold used (global or fixed methods)"
    )
    repaired_fraction: float | None = Field(
        default=None,
        description="autoreject: share of channel-epochs that were interpolated, of those kept",
    )
    interpolated_channels: list[str] = Field(description="Bad channels interpolated first")
    missing_events: list[str] = Field(description="Requested event descriptions not found")
    sfreq_hz: float
    notes: list[str] = Field(default_factory=list)


def make_epochs(
    raw: Any, params: EpochParams, *, reference: str | list[str] = "none"
) -> tuple[Any, EpochReport]:
    """Epoch ``raw`` (unchanged). ``reference`` is the data's reference, re-applied after bad
    channels are interpolated."""
    import mne

    notes: list[str] = []
    data = raw.copy().load_data(verbose="error")
    interpolated: list[str] = []
    eeg_bads = [ch for ch in data.info["bads"] if data.get_channel_types([ch])[0] == "eeg"]
    if params.interpolate_bads and eeg_bads:
        data.interpolate_bads(reset_bads=True, verbose="error")
        apply_reference(data, reference)
        interpolated = eeg_bads

    present = set(data.annotations.description)
    codes: dict[str, int] = {}
    for code, descriptions in enumerate(params.conditions.values(), start=1):
        for d in descriptions:
            codes[d] = code
    missing = [d for d in codes if d not in present]
    found = {d: c for d, c in codes.items() if d in present}
    if not found:
        known = ", ".join(sorted(present)[:20]) or "none"
        raise ValueError(f"none of the requested events are in the recording; it has: {known}")
    events, _ = mne.events_from_annotations(data, event_id=found, verbose="error")
    if params.event_delay_s:
        shift = int(params.event_delay_s * float(data.info["sfreq"]))  # earlier sample
        events = events.copy()
        events[:, 0] += shift
        notes.append(
            f"Events moved {shift / data.info['sfreq'] * 1000:.1f} ms later for the "
            f"{params.event_delay_s * 1000:g} ms stimulus delay."
        )
    names = list(params.conditions)
    event_id = {names[c - 1]: c for c in sorted(set(found.values()))}
    for name in names:
        if name not in event_id:
            notes.append(f"Condition {name!r} has no events in this recording.")

    epochs = mne.Epochs(
        data,
        events,
        event_id=event_id,
        tmin=params.tmin,
        tmax=params.tmax,
        baseline=params.baseline,
        picks=["eeg", "eog"] if "eog" in data.get_channel_types() else ["eeg"],
        reject_by_annotation=True,
        preload=True,
        verbose="error",
    )
    counts_before = {name: int((events[:, 2] == code).sum()) for name, code in event_id.items()}
    by_annotation = len(events) - len(epochs)
    if by_annotation:
        notes.append(f"{by_annotation} epochs overlapped segments annotated as bad.")

    method = params.reject
    if method in ("autoreject", "autoreject_global"):
        if len(epochs) < MIN_AUTOREJECT_EPOCHS:
            notes.append(
                f"Only {len(epochs)} epochs, too few for autoreject; used a fixed "
                f"{params.peak_to_peak_uv:g} µV threshold instead."
            )
            method = "peak_to_peak"
        elif method == "autoreject" and not _all_placed(epochs):
            notes.append("Not every EEG channel has a position; used global autoreject instead.")
            method = "autoreject_global"

    threshold_uv = repaired = None
    if method == "autoreject":
        epochs, repaired = _autoreject_local(epochs, params.seed)
    elif method == "autoreject_global":
        threshold_uv = _autoreject_global(epochs, params.seed)
    elif method == "peak_to_peak":
        threshold_uv = params.peak_to_peak_uv
        epochs.drop_bad(reject={"eeg": threshold_uv * 1e-6}, verbose="error")

    counts = {}
    for name in names:
        n_events = counts_before.get(name, 0)
        kept = len(epochs[name]) if name in epochs.event_id and n_events else 0
        counts[name] = ConditionCount(
            events=n_events,
            kept=kept,
            rejected_fraction=round(1 - kept / n_events, 4) if n_events else 0.0,
        )
    for name, count in counts.items():
        if count.events and count.rejected_fraction >= HEAVY_LOSS:
            threshold = f" (threshold {threshold_uv:.1f} µV)" if threshold_uv else ""
            notes.append(
                f"Warning: condition {name!r} lost {count.rejected_fraction:.0%} of its epochs"
                f"{threshold}; results for it rest on {count.kept} trials. Check the rejection "
                "settings (a fixed threshold or ICA may keep more)."
            )
    n_events = int(sum(counts_before.values()))
    n_kept = len(epochs)
    return epochs, EpochReport(
        conditions=counts,
        n_events=n_events,
        n_kept=n_kept,
        rejected_fraction=round(1 - n_kept / n_events, 4) if n_events else 0.0,
        reject=method,
        threshold_uv=None if threshold_uv is None else round(threshold_uv, 2),
        repaired_fraction=repaired,
        interpolated_channels=interpolated,
        missing_events=missing,
        sfreq_hz=float(epochs.info["sfreq"]),
        notes=notes,
    )


def _autoreject_local(epochs: Any, seed: int) -> tuple[Any, float]:
    from autoreject import AutoReject

    with quiet():
        ar = AutoReject(random_state=seed, n_jobs=1, verbose=False, picks="eeg")
        cleaned, log = ar.fit_transform(epochs, return_log=True)
    labels = log.labels[~log.bad_epochs]
    repaired = float((labels == 2).mean()) if labels.size else 0.0
    return cleaned, round(repaired, 4)


def _autoreject_global(epochs: Any, seed: int) -> float:
    from autoreject import get_rejection_threshold

    with quiet():
        reject = get_rejection_threshold(epochs, ch_types="eeg", random_state=seed, verbose=False)
    epochs.drop_bad(reject={"eeg": reject["eeg"]}, verbose="error")
    return float(reject["eeg"]) * 1e6


def _all_placed(epochs: Any) -> bool:
    import mne

    for i in mne.pick_types(epochs.info, eeg=True, exclude=[]):
        loc = epochs.info["chs"][i]["loc"][:3]
        if not ((loc == loc).all() and (loc != 0).any()):
            return False
    return True
