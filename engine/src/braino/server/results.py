"""A pipeline's results, read for the app to draw: ERPs, spectra and time-frequency power per
subject or averaged over subjects, scalp maps, the measures, and the group test.

Everything comes from the pipeline's state (what each step wrote), so only results of the
pipeline as it last ran are shown. Reads BrainO's derivatives only; changes nothing. Needs the
``[eeg]`` extra for the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from braino.context import RunContext
from braino.pipelines import PipelineSpec, load_state
from braino.pipelines.runner import GROUP

MEASURE_TOOLS = {"eeg.erp": "erp", "eeg.spectral": "psd", "eeg.tfr": "tfr"}
MAX_POINTS = 600
"""Longest waveform sent (points); longer ones are thinned."""


class ResultsError(ValueError):
    pass


def index(ctx: RunContext, spec: PipelineSpec) -> dict[str, Any]:
    """What results the pipeline has: its measure steps with the subjects that have results,
    the measures table, and each group step's result."""
    state = load_state(ctx, spec)
    steps = []
    for step in spec.steps:
        kind = MEASURE_TOOLS.get(step.tool)
        if kind is None:
            continue
        subjects = []
        names: list[str] = []
        for key, record in sorted(state.records.items()):
            subject, _, step_id = key.partition("/")
            if step_id != step.id or not record.result or subject == GROUP:
                continue
            subjects.append(subject)
            result = record.result
            for name in (
                result.get("waveforms")
                or list(result.get("conditions") or {})
                or list(result.get("n_epochs") or {})
            ):
                if name not in names:
                    names.append(name)
        windows = [
            {
                "name": m.get("name"),
                "tmin": m.get("tmin"),
                "tmax": m.get("tmax"),
                "channels": m.get("channels"),
            }
            for m in step.args.get("measures", []) or []
            if isinstance(m, dict)
        ]
        steps.append(
            {
                "id": step.id,
                "tool": step.tool,
                "kind": kind,
                "subjects": subjects,
                "names": names,
                "windows": windows,
            }
        )

    measures = []
    for key, record in sorted(state.records.items()):
        subject, _, step_id = key.partition("/")
        if record.tool != "eeg.erp" or not record.result:
            continue
        for m in record.result.get("measures", []):
            measures.append({"subject": subject, "step": step_id, **m})

    group = []
    for step in spec.group:
        found = state.records.get(f"{GROUP}/{step.id}")
        group.append(
            {
                "id": step.id,
                "tool": step.tool,
                "args": step.args,
                "result": found.result if found else None,
                "finished_at": found.finished_at.isoformat() if found else None,
            }
        )
    return {"name": spec.name, "steps": steps, "measures": measures, "group": group}


def data(
    ctx: RunContext,
    spec: PipelineSpec,
    step_id: str,
    *,
    subject: str | None = None,
    channel: str | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """One measure step's data at one channel, for one subject or averaged over the subjects
    that have it. For ERPs, also the scalp map of each waveform's mean in ``tmin``-``tmax``;
    for time-frequency, the map of condition ``name``."""
    step = spec.step(step_id)
    kind = MEASURE_TOOLS.get(step.tool)
    if kind is None:
        raise ResultsError(f"step {step_id!r} ({step.tool}) has no data to show")
    state = load_state(ctx, spec)
    files: dict[str, Path] = {}
    for key, record in state.records.items():
        s, _, sid = key.partition("/")
        if sid == step_id and record.result and (subject is None or s == subject):
            files[s] = ctx.dataset / record.result["output"]
    files = {s: f for s, f in sorted(files.items()) if f.is_file()}
    if not files:
        raise ResultsError(
            f"no results of {step_id!r} yet" + (f" for sub-{subject}" if subject else "")
        )
    if kind == "erp":
        return _erp(files, channel, tmin, tmax)
    if kind == "psd":
        return _psd(files, channel)
    return _tfr(files, channel, name)


def _thin(values: Any, step: int) -> list[float]:
    return [round(float(v), 4) for v in values[::step]]


def _erp(
    files: dict[str, Path], channel: str | None, tmin: float | None, tmax: float | None
) -> dict[str, Any]:
    import mne
    import numpy as np

    from braino.server.evidence import _layout

    per_subject: dict[str, dict[str, Any]] = {}
    for subject, path in files.items():
        per_subject[subject] = {e.comment: e for e in mne.read_evokeds(path, verbose="error")}
    first = next(iter(per_subject.values()))
    names = list(first)
    reference = first[names[0]]
    eeg = reference.copy().pick("eeg")
    channels = eeg.ch_names
    chosen = channel if channel in channels else ("Pz" if "Pz" in channels else channels[0])
    times = reference.times
    step = max(1, len(times) // MAX_POINTS)

    series: dict[str, list[float]] = {}
    maps: dict[str, list[float]] = {}
    window = (tmin, tmax) if tmin is not None and tmax is not None else None
    for name in names:
        stack = [ev[name].copy().pick("eeg").data for ev in per_subject.values() if name in ev]
        mean = np.mean(stack, axis=0) * 1e6  # channels x times, µV
        series[name] = _thin(mean[channels.index(chosen)], step)
        if window:
            mask = (times >= window[0]) & (times <= window[1])
            if mask.any():
                maps[name] = [round(float(v), 3) for v in mean[:, mask].mean(axis=1)]
    return {
        "kind": "erp",
        "subjects": list(files),
        "channels": channels,
        "channel": chosen,
        "x": _thin(times, step),
        "x_unit": "s",
        "unit": "µV",
        "series": series,
        "layout": _layout(eeg.info),
        "maps": maps,
        "window": list(window) if window else None,
    }


def _npz(path: Path) -> Any:
    import numpy as np

    return np.load(path, allow_pickle=False)


def _psd(files: dict[str, Path], channel: str | None) -> dict[str, Any]:
    import numpy as np

    loaded = [_npz(p) for p in files.values()]
    first = loaded[0]
    channels = [str(c) for c in first["ch_names"]]
    names = [str(c) for c in first["conditions"]]
    chosen = channel if channel in channels else ("Oz" if "Oz" in channels else channels[0])
    c = channels.index(chosen)
    series = {}
    for i, name in enumerate(names):
        stack = [f[f"data_{i}"][c] for f in loaded if f"data_{i}" in f]
        mean = np.mean(stack, axis=0) * 1e12  # V^2/Hz -> µV^2/Hz
        series[name] = [round(float(v), 4) for v in 10 * np.log10(np.maximum(mean, 1e-12))]
    return {
        "kind": "psd",
        "subjects": list(files),
        "channels": channels,
        "channel": chosen,
        "x": [round(float(f), 3) for f in first["freqs"]],
        "x_unit": "Hz",
        "unit": "dB (µV²/Hz)",
        "series": series,
    }


def _tfr(files: dict[str, Path], channel: str | None, name: str | None) -> dict[str, Any]:
    import numpy as np

    loaded = [_npz(p) for p in files.values()]
    first = loaded[0]
    channels = [str(c) for c in first["ch_names"]]
    names = [str(c) for c in first["conditions"]]
    chosen = channel if channel in channels else ("Cz" if "Cz" in channels else channels[0])
    condition = name if name in names else names[0]
    i = names.index(condition)
    c = channels.index(chosen)
    mean = np.mean([f[f"data_{i}"][c] for f in loaded], axis=0)  # freqs x times
    times = first["times"]
    step = max(1, len(times) // 200)
    return {
        "kind": "tfr",
        "subjects": list(files),
        "channels": channels,
        "channel": chosen,
        "names": names,
        "name": condition,
        "times": _thin(times, step),
        "freqs": [round(float(f), 2) for f in first["freqs"]],
        "unit": str(first["unit"]),
        "power": [[round(float(v), 4) for v in row[::step]] for row in mean],
    }
