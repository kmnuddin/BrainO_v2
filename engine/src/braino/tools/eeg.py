"""EEG tools for one recording: quality screen, preprocessing, bad channels, ICA and epoching.

Each step writes new files to the derivatives and the next step reads them; see
:mod:`braino.tools.recordings` for the files. The measures taken from epochs (ERPs, spectra,
time-frequency) are in :mod:`braino.tools.eeg_measures`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.eeg.bad_channels import Method as BadChannelMethod
from braino.eeg.epochs import ConditionCount, EpochParams, RejectMethod
from braino.eeg.ica import DEFAULT_EXCLUDE, ICLABEL_LABELS, Component, IcaMethod, Labeler
from braino.eeg.preprocess import PreprocessParams
from braino.eeg.qc import ChannelQc
from braino.tools.base import Risk
from braino.tools.recordings import (
    CLEAN,
    LABEL_PATTERN,
    PREPROC,
    MissingStepError,
    RecordingNotFoundError,
    RecordingSelector,
    StaleStepError,
    bad_channels_stale,
    check_label,
    clean_stale,
    cleanest,
    derivative,
    find_recording,
    preprocessed,
    read_original,
    relative,
    require,
    sha256,
    write_json,
    write_raw,
    write_tsv,
)
from braino.tools.registry import tool

__all__ = [
    "MissingStepError",
    "RecordingNotFoundError",
    "RecordingSelector",
    "StaleStepError",
    "find_recording",
]

MAX_LISTED = 10
"""Longest channel list a summary gives; the per-channel table has them all."""


# --- eeg.qc_report ------------------------------------------------------------------------


class QcInput(RecordingSelector):
    pass


class QcSummary(BaseModel):
    recording: str = Field(description="The data file checked, relative to the dataset")
    duration_s: float
    n_channels_checked: int
    marked_bad_in_file: list[str]
    n_flat: int
    flat: list[str] = Field(description=f"Flat channels (at most {MAX_LISTED})")
    n_noisy: int
    noisy: list[dict[str, str | float]] = Field(
        description=f"Noisiest channels with reason and robust z (at most {MAX_LISTED})"
    )
    line_freq_hz: float | None
    line_noise_median_ratio: float | None
    n_strong_line_noise: int
    strong_line_noise: list[str] = Field(
        description=f"Channels with the strongest line noise (at most {MAX_LISTED})"
    )
    artifact_fraction: float = Field(
        description="Share of 1-s windows in which a checked channel exceeds the threshold"
    )
    artifact_threshold_uv: float
    report: str = Field(description="Per-channel table, relative to the dataset")
    notes: list[str]


@tool(name="eeg.qc_report", risk=Risk.COMPUTE)
def qc_report(params: QcInput, ctx: RunContext) -> QcSummary:
    """Screen one recording for data-quality problems: flat channels, noisy channels, power-line
    noise and the share of time with large artifacts. Changes nothing; the per-channel table is
    written to the derivatives."""
    from braino.eeg.qc import compute_qc

    recording = find_recording(ctx.dataset, params)
    raw = read_original(ctx, recording)
    result = compute_qc(raw, line_freq=recording.line_freq_hz)
    table = derivative(ctx, recording, "desc-qc_channels.tsv")
    _write_qc_table(ctx, table, result.channels)
    return QcSummary(
        recording=recording.path,
        duration_s=result.duration_s,
        n_channels_checked=result.n_channels_checked,
        marked_bad_in_file=result.marked_bad_in_file,
        n_flat=len(result.flat),
        flat=result.flat[:MAX_LISTED],
        n_noisy=len(result.noisy),
        noisy=[
            {"channel": n.channel, "reason": n.reason, "z": n.z} for n in result.noisy[:MAX_LISTED]
        ],
        line_freq_hz=result.line_freq_hz,
        line_noise_median_ratio=result.line_noise_median_ratio,
        n_strong_line_noise=len(result.strong_line_noise),
        strong_line_noise=result.strong_line_noise[:MAX_LISTED],
        artifact_fraction=result.artifact_fraction,
        artifact_threshold_uv=result.artifact_threshold_uv,
        report=relative(ctx, table),
        notes=result.notes,
    )


def _write_qc_table(ctx: RunContext, path: Path, channels: list[ChannelQc]) -> None:
    write_tsv(
        ctx,
        path,
        [
            "name",
            "median_std_uv",
            "flat_fraction",
            "artifact_fraction",
            "high_freq_ratio",
            "line_noise_ratio",
            "flags",
        ],
        [
            [
                c.channel,
                c.median_std_uv,
                c.flat_fraction,
                c.artifact_fraction,
                c.high_freq_ratio,
                c.line_noise_ratio,
                ",".join(c.flags) or None,
            ]
            for c in channels
        ],
    )


# --- eeg.preprocess -----------------------------------------------------------------------


class PreprocessInput(RecordingSelector, PreprocessParams):
    pass


class PreprocessOutput(BaseModel):
    recording: str = Field(description="The original recording, relative to the dataset")
    output: str = Field(description="The preprocessed copy, relative to the dataset")
    steps: list[str] = Field(description="What was done, in order")
    sfreq_hz: float
    duration_s: float
    channels: dict[str, int] = Field(description="Channels kept, per type")
    dropped_channels: list[str]
    bad_channels: list[str] = Field(description="Channels marked bad in the file")
    reference: str | list[str]
    positions: Literal["file", "template", "none"] = Field(
        description="Where electrode positions come from (needed for interpolation and RANSAC)"
    )
    notes: list[str]


@tool(name="eeg.preprocess", risk=Risk.COMPUTE)
def preprocess(params: PreprocessInput, ctx: RunContext) -> PreprocessOutput:
    """Filter (high-pass, low-pass, optional notch), resample and re-reference one recording.
    Writes a preprocessed copy to the derivatives; the original is not changed. Bad-channel
    detection, ICA and epoching all start from this copy, so run this first. Electrode
    positions are added from a standard layout if the file has none and the names identify
    one."""
    from braino.eeg.preprocess import preprocess as run_preprocess
    from braino.io.normalise import add_template_positions

    recording = find_recording(ctx.dataset, params)
    raw = read_original(ctx, recording)
    positions, template, unplaced, notes = add_template_positions(raw)
    settings = PreprocessParams.model_validate(params.model_dump(include=_PREPROCESS_FIELDS))
    out, report = run_preprocess(raw, settings, line_freq=recording.line_freq_hz)

    path = derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif")
    write_raw(ctx, out, path)
    write_json(
        ctx,
        {
            "Description": "Filtered, resampled and re-referenced EEG",
            "Sources": [recording.path],
            "Parameters": settings.model_dump(mode="json"),
            "Steps": report.steps,
            "Reference": report.reference,
            "SamplingFrequency": report.sfreq_hz,
            "PowerLineFrequency": recording.line_freq_hz,
            "Positions": positions,
            "PositionsTemplate": template,
        },
        path.with_suffix(".json"),
    )
    if unplaced and positions != "none":
        notes.append(f"No position for {len(unplaced)} EEG channels: {', '.join(unplaced[:10])}")
    return PreprocessOutput(
        recording=recording.path,
        output=relative(ctx, path),
        steps=report.steps,
        sfreq_hz=report.sfreq_hz,
        duration_s=report.duration_s,
        channels=report.channels,
        dropped_channels=report.dropped_channels,
        bad_channels=report.bad_channels,
        reference=report.reference,
        positions=positions,
        notes=notes + report.notes,
    )


_PREPROCESS_FIELDS = set(PreprocessParams.model_fields)


# --- eeg.detect_bad_channels (decision) ---------------------------------------------------


class DetectBadChannelsInput(RecordingSelector):
    method: BadChannelMethod = Field(
        default="pyprep",
        description="'pyprep' (PREP noisy-channel detection: deviation, correlation, "
        "high-frequency noise, RANSAC) or 'qc' (BrainO's flat/noisy screen)",
    )
    ransac: bool = Field(
        default=True, description="pyprep only: also run RANSAC (needs electrode positions)"
    )
    correlation_max_bad_time: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="pyprep only: share of 1-s windows with low correlation (< 0.4 with every "
        "other channel) that makes a channel bad. PREP uses 0.01, too strict for caps with "
        "fewer than 64 channels",
    )
    leave_out_blinks: bool = Field(
        default=True,
        description="Leave blinks (from the EOG, or Fp1/Fp2) out of the detection, so channels "
        "near the eyes are not marked bad for blinks, which ICA removes",
    )
    seed: int = Field(default=42, description="Random seed for RANSAC")


class BadChannelProposal(BaseModel):
    recording: str = Field(description="The preprocessed recording, relative to the dataset")
    channels: list[str] = Field(
        description="Channels to mark bad: left out of ICA and the average reference, and "
        "interpolated from their neighbours when epoching"
    )
    reasons: dict[str, list[str]] = Field(description="Why each channel was found bad")
    n_checked: int
    method: BadChannelMethod
    notes: list[str] = Field(default_factory=list)


class BadChannelsApplied(BaseModel):
    table: str = Field(description="Channels table with the approved status, relative path")
    bad_channels: list[str]
    n_good_eeg: int


def apply_bad_channels(proposal: BadChannelProposal, ctx: RunContext) -> BadChannelsApplied:
    """Write the approved bad channels to the channels table of the preprocessed recording."""
    import mne

    data_file = _derivative_file(ctx, proposal.recording, f"_desc-{PREPROC}_eeg.fif")
    ctx.record_input(data_file)
    info = mne.io.read_info(data_file, verbose="error")
    names = list(info.ch_names)
    unknown = [ch for ch in proposal.channels if ch not in names]
    if unknown:
        raise ValueError(f"no channel {', '.join(unknown)} in {data_file.name}")
    reference = _reference_channels(_read_sidecar(ctx, data_file))
    in_reference = [ch for ch in proposal.channels if ch in reference]
    if in_reference:
        raise ValueError(
            f"{', '.join(in_reference)} is the reference; a reference channel cannot be marked "
            "bad. Run eeg.preprocess again with another reference first."
        )
    types = info.get_channel_types()
    bad = set(proposal.channels)
    table = data_file.with_name(data_file.name.replace("_eeg.fif", "_channels.tsv"))
    rows = []
    for ch, kind in zip(names, types, strict=True):
        status = "bad" if ch in bad else "good"
        why = "; ".join(proposal.reasons.get(ch, ["marked by reviewer"])) if ch in bad else None
        rows.append([ch, kind.upper(), status, why])
    write_tsv(ctx, table, ["name", "type", "status", "status_description"], rows)
    write_json(
        ctx,
        {
            "Description": "Approved channel status of the preprocessed recording",
            "Sources": [relative(ctx, data_file)],
            "SourceSha256": sha256(data_file),
        },
        table.with_suffix(".json"),
    )
    n_good = sum(
        1 for ch, kind in zip(names, types, strict=True) if kind == "eeg" and ch not in bad
    )
    return BadChannelsApplied(
        table=relative(ctx, table),
        bad_channels=[ch for ch in names if ch in bad],
        n_good_eeg=n_good,
    )


@tool(name="eeg.detect_bad_channels", risk=Risk.DECISION, apply=apply_bad_channels)
def detect_bad_channels(params: DetectBadChannelsInput, ctx: RunContext) -> BadChannelProposal:
    """Find bad EEG channels (flat, noisy, poorly correlated with the others) and propose
    marking them bad. Checks the ICA-cleaned recording if there is one, else the preprocessed
    one. Running eeg.ica_clean first is best: eye activity then no longer makes channels near
    the eyes look bad, and a broken channel shows up in ICA as a 'channel noise' component.
    Once approved, bad channels are interpolated when epoching (and left out of an ICA run
    afterwards)."""
    from braino.eeg.bad_channels import find_bad_channels

    recording = find_recording(ctx.dataset, params)
    stale_clean = clean_stale(ctx, recording)
    if stale_clean:
        raw, sidecar = preprocessed(ctx, recording, with_decision=False)
        source = PREPROC
    else:
        raw, sidecar, source = cleanest(ctx, recording, with_decision=False)
    result = find_bad_channels(
        raw,
        method=params.method,
        ransac=params.ransac,
        seed=params.seed,
        line_freq=sidecar.get("PowerLineFrequency"),
        correlation_max_bad_time=params.correlation_max_bad_time,
        leave_out_blinks=params.leave_out_blinks,
    )
    path = derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif")
    notes = list(result.notes)
    if source == CLEAN:
        notes.append("Checked the ICA-cleaned recording.")
    elif stale_clean:
        notes.append(
            "Checked the preprocessed recording: the ICA-cleaned one is from an earlier "
            "preprocessing (run eeg.ica_clean again)."
        )
    else:
        notes.append("Checked the preprocessed recording (no ICA yet).")
    channels = list(result.channels)
    reference = _reference_channels(sidecar)
    flagged_reference = [ch for ch in channels if ch in reference]
    if flagged_reference:
        channels = [ch for ch in channels if ch not in reference]
        notes.append(
            f"Reference channel(s) {', '.join(flagged_reference)} look bad but cannot be marked "
            "bad; if they are, run eeg.preprocess again with another reference."
        )
    n_eeg = result.n_checked
    if n_eeg and len(channels) > n_eeg / 4:
        notes.append(
            f"{len(channels)} of {n_eeg} EEG channels found bad; that many usually "
            "means a problem with the recording or the preprocessing."
        )
    return BadChannelProposal(
        recording=relative(ctx, path),
        channels=channels,
        reasons=result.reasons,
        n_checked=result.n_checked,
        method=result.method,
        notes=notes,
    )


# --- eeg.ica_clean (decision) -------------------------------------------------------------


class IcaInput(RecordingSelector):
    n_components: int | float | None = Field(
        default=None,
        description="Number of components (>= 2), share of variance to keep (0-1, e.g. 0.99), "
        "or null for as many as the data allow",
    )
    method: IcaMethod = Field(
        default="picard",
        description="'picard' (extended infomax, fast), 'infomax' (the same, slower) or 'fastica'",
    )
    labeler: Labeler = Field(
        default="iclabel",
        description="'iclabel' (ICLabel classifier) or 'eog' (correlation with EOG channels)",
    )
    exclude_labels: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EXCLUDE),
        description=f"Component labels proposed for removal; ICLabel labels are "
        f"{', '.join(ICLABEL_LABELS)}",
    )
    min_probability: float = Field(
        default=0.8, ge=0, le=1, description="Least label probability for removal"
    )
    seed: int = Field(default=42, description="Random seed for ICA")


class IcaProposal(BaseModel):
    recording: str = Field(description="The preprocessed recording, relative to the dataset")
    ica_file: str = Field(description="The fitted ICA, relative to the dataset")
    bad_channels: list[str] = Field(description="Channels left out of the ICA")
    exclude: list[int] = Field(description="Components to remove (0-based indices)")
    components: list[Component] = Field(description="Every component with its label")
    n_components: int
    method: IcaMethod
    labeler: Labeler
    notes: list[str] = Field(default_factory=list)


class IcaApplied(BaseModel):
    output: str = Field(description="The cleaned recording, relative to the dataset")
    removed: list[int]
    removed_labels: dict[str, str]
    reference: str | list[str]


def apply_ica(proposal: IcaProposal, ctx: RunContext) -> IcaApplied:
    """Remove the approved components and write the cleaned recording."""
    import mne

    from braino.eeg.ica import remove_components

    data_file = _derivative_file(ctx, proposal.recording, f"_desc-{PREPROC}_eeg.fif")
    ica_file = _derivative_file(ctx, proposal.ica_file, f"_desc-{PREPROC}_ica.fif")
    ctx.record_input(data_file)
    ctx.record_input(ica_file)
    raw = mne.io.read_raw_fif(data_file, preload=True, verbose="error")
    sidecar = _read_sidecar(ctx, data_file)
    unknown = [ch for ch in proposal.bad_channels if ch not in raw.ch_names]
    if unknown:
        raise ValueError(f"no channel {', '.join(unknown)} in {data_file.name}")
    raw.info["bads"] = list(proposal.bad_channels)  # as when the ICA was fitted
    ica = mne.preprocessing.read_ica(ica_file, verbose="error")
    reference = sidecar.get("Reference", "none")
    clean = remove_components(raw, ica, proposal.exclude, reference)

    out = data_file.with_name(data_file.name.replace(f"_desc-{PREPROC}_", f"_desc-{CLEAN}_"))
    write_raw(ctx, clean, out)
    labels = {c.index: c.label for c in proposal.components}
    write_json(
        ctx,
        {
            "Description": "Preprocessed EEG with ICA components removed",
            "Sources": [relative(ctx, data_file), relative(ctx, ica_file)],
            "SourceSha256": sha256(data_file),
            "RemovedComponents": proposal.exclude,
            "RemovedLabels": {str(i): labels.get(i, "?") for i in proposal.exclude},
            "Reference": reference,
        },
        out.with_suffix(".json"),
    )
    return IcaApplied(
        output=relative(ctx, out),
        removed=list(proposal.exclude),
        removed_labels={str(i): labels.get(i, "?") for i in proposal.exclude},
        reference=reference,
    )


@tool(name="eeg.ica_clean", risk=Risk.DECISION, apply=apply_ica)
def ica_clean(params: IcaInput, ctx: RunContext) -> IcaProposal:
    """Fit ICA to the preprocessed recording (good channels only), label the components (eye,
    heart, muscle, brain, ...) and propose removing the artifact components. Run
    eeg.preprocess (and eeg.detect_bad_channels, if used) first. Once approved, the cleaned
    recording is what eeg.epoch reads."""
    from braino.eeg.ica import fit_ica

    if isinstance(params.n_components, float) and not 0 < params.n_components < 1:
        raise ValueError("a share of variance must be between 0 and 1")
    if isinstance(params.n_components, int) and params.n_components < 2:
        raise ValueError("ICA needs at least 2 components")
    recording = find_recording(ctx.dataset, params)
    stale = bad_channels_stale(ctx, recording)
    raw, _ = preprocessed(ctx, recording, with_decision=not stale)
    ica, result = fit_ica(
        raw,
        n_components=params.n_components,
        method=params.method,
        labeler=params.labeler,
        exclude_labels=params.exclude_labels,
        min_probability=params.min_probability,
        seed=params.seed,
    )
    ica_file = derivative(ctx, recording, f"desc-{PREPROC}_ica.fif")
    ctx.record_output(ica_file)
    ica.save(ica_file, overwrite=True, verbose="error")
    return IcaProposal(
        recording=relative(ctx, derivative(ctx, recording, f"desc-{PREPROC}_eeg.fif")),
        ica_file=relative(ctx, ica_file),
        bad_channels=list(raw.info["bads"]),
        exclude=result.proposed,
        components=result.components,
        n_components=result.n_components,
        method=result.method,
        labeler=result.labeler,
        notes=result.notes
        + (
            ["The approved bad channels were for an earlier preprocessing and were not used."]
            if stale
            else []
        ),
    )


# --- eeg.epoch ----------------------------------------------------------------------------


class EpochInput(RecordingSelector, EpochParams):
    label: str = Field(
        default="epochs",
        pattern=LABEL_PATTERN,
        description="Name for these epochs (letters and digits), used by later steps",
    )


class EpochOutput(BaseModel):
    recording: str = Field(description="The recording epoched (cleaned or preprocessed)")
    source: Literal["clean", "preproc"]
    output: str = Field(description="The epochs file, relative to the dataset")
    conditions: dict[str, ConditionCount]
    n_events: int
    n_kept: int
    rejected_fraction: float
    reject: RejectMethod
    threshold_uv: float | None
    repaired_fraction: float | None
    interpolated_channels: list[str]
    missing_events: list[str]
    notes: list[str]


@tool(name="eeg.epoch", risk=Risk.COMPUTE)
def epoch(params: EpochInput, ctx: RunContext) -> EpochOutput:
    """Cut the recording into epochs around events, grouped into named conditions, with
    baseline correction and rejection of epochs with artifacts (autoreject by default). Reads
    the ICA-cleaned recording if there is one, else the preprocessed one; bad channels are
    interpolated first."""
    from braino.eeg.epochs import make_epochs

    recording = find_recording(ctx.dataset, params)
    raw, sidecar, source = cleanest(ctx, recording)
    settings = EpochParams.model_validate(params.model_dump(include=set(EpochParams.model_fields)))
    epochs, report = make_epochs(raw, settings, reference=sidecar.get("Reference", "none"))
    path = derivative(ctx, recording, f"desc-{check_label(params.label, 'label')}_epo.fif")
    ctx.record_output(path)
    epochs.save(path, overwrite=True, verbose="error")
    notes = list(report.notes)
    if source == PREPROC:
        notes.append("No ICA-cleaned recording; epoched the preprocessed one.")
    data = derivative(ctx, recording, f"desc-{source}_eeg.fif")
    return EpochOutput(
        recording=relative(ctx, data),
        source="clean" if source == CLEAN else "preproc",
        output=relative(ctx, path),
        conditions=report.conditions,
        n_events=report.n_events,
        n_kept=report.n_kept,
        rejected_fraction=report.rejected_fraction,
        reject=report.reject,
        threshold_uv=report.threshold_uv,
        repaired_fraction=report.repaired_fraction,
        interpolated_channels=report.interpolated_channels,
        missing_events=report.missing_events,
        notes=notes,
    )


# --- helpers ------------------------------------------------------------------------------


def _derivative_file(ctx: RunContext, path: str, suffix: str) -> Path:
    """A file a proposal names: it must be one of BrainO's derivatives, of the expected kind."""
    target = (ctx.dataset / path).resolve()
    if not target.is_relative_to(ctx.derivatives_root) or not target.name.endswith(suffix):
        raise ValueError(f"{path} is not a BrainO derivative ending in {suffix}")
    return require(target, "the step that makes it")


def _read_sidecar(ctx: RunContext, data_file: Path) -> dict[str, Any]:
    from braino.tools.recordings import read_json

    return read_json(ctx, require(data_file.with_suffix(".json"), "eeg.preprocess"))


def _reference_channels(sidecar: dict[str, Any]) -> list[str]:
    reference = sidecar.get("Reference")
    return list(reference) if isinstance(reference, list) else []
