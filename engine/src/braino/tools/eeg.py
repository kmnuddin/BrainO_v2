"""EEG tools."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.eeg.qc import ChannelQc, compute_qc
from braino.io.dataset import RecordingSummary, scan_dataset
from braino.tools.base import Risk
from braino.tools.registry import tool


class RecordingSelector(BaseModel):
    subject: str = Field(description="Subject label without 'sub-', e.g. '001'")
    task: str | None = Field(default=None, description="Task label, if the subject has several")
    session: str | None = Field(default=None, description="Session label, if any")
    run: str | None = Field(default=None, description="Run number, if any")


class QcInput(RecordingSelector):
    pass


MAX_LISTED = 10
"""Longest channel list a summary gives; the per-channel table has them all."""


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
    import mne_bids

    recording = find_recording(ctx.dataset, params)
    data_file = ctx.dataset / recording.path
    for path in _recording_files(data_file):
        ctx.record_input(path)
    bids_path = mne_bids.get_bids_path_from_fname(data_file)
    raw = mne_bids.read_raw_bids(bids_path, verbose="error")
    result = compute_qc(raw, line_freq=recording.line_freq_hz)

    parts = [f"sub-{recording.subject}"]
    if recording.session:
        parts.append(f"ses-{recording.session}")
    out_dir = ctx.output_dir(*parts, recording.datatype)
    stem = data_file.name[: -len(data_file.name.split("_")[-1])]
    table = ctx.record_output(out_dir / f"{stem}desc-qc_channels.tsv")
    _write_table(table, result.channels)
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
        report=table.relative_to(ctx.dataset).as_posix(),
        notes=result.notes,
    )


class RecordingNotFoundError(LookupError):
    pass


def find_recording(root: Path, selector: RecordingSelector) -> RecordingSummary:
    """The one recording matching ``selector``; raises :class:`RecordingNotFoundError` if none
    or several match, listing the candidates."""
    recordings = scan_dataset(root).recordings
    matches = [
        r
        for r in recordings
        if r.subject == selector.subject
        and (selector.task is None or r.task == selector.task)
        and (selector.session is None or r.session == selector.session)
        and (selector.run is None or (r.run or "").lstrip("0") == selector.run.lstrip("0"))
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RecordingNotFoundError(
            f"no recording matches {selector.model_dump(exclude_none=True)}"
        )
    options = ", ".join(r.path for r in matches)
    raise RecordingNotFoundError(f"several recordings match; say which: {options}")


def _recording_files(data_file: Path) -> list[Path]:
    """The data file and its companions (e.g. BrainVision .eeg and .vmrk)."""
    stem = data_file.name.rsplit(".", 1)[0]
    return sorted(p for p in data_file.parent.glob(f"{stem}.*") if p.is_file())


def _write_table(path: Path, channels: list[ChannelQc]) -> None:
    rows = [c.model_dump() for c in channels]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "name",
                "median_std_uv",
                "flat_fraction",
                "artifact_fraction",
                "high_freq_ratio",
                "line_noise_ratio",
                "flags",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["channel"],
                    row["median_std_uv"],
                    row["flat_fraction"],
                    row["artifact_fraction"],
                    _na(row["high_freq_ratio"]),
                    _na(row["line_noise_ratio"]),
                    ",".join(row["flags"]) or "n/a",
                ]
            )


def _na(value: object) -> object:
    return "n/a" if value is None else value
