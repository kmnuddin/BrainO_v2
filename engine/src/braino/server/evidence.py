"""What a person needs to see to decide a proposal, read from the recording it is about.

- **Bad channels:** where the electrodes are (a 2D layout), each channel's amplitude, and ten
  seconds of each proposed channel beside its nearest good neighbour.
- **ICA:** each component's weights at the electrodes (its map) and the power spectrum of its
  activity, with its label.

Reads only BrainO's own derivatives, never the original recording, and returns numbers for the
app to draw; it changes nothing. Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from braino.context import RunContext
from braino.provenance.models import ProposalRecord

SNIPPET_S = 10.0
SNIPPET_HZ = 100.0
SPECTRUM_FMAX = 45.0
SPECTRUM_SECONDS = 120.0
MAX_COMPONENTS = 40


def evidence(ctx: RunContext, record: ProposalRecord) -> dict[str, Any]:
    if record.tool == "eeg.detect_bad_channels":
        return _bad_channels(ctx, record)
    if record.tool == "eeg.ica_clean":
        return _ica(ctx, record)
    return {"kind": "none"}


def _layout(info: Any) -> dict[str, Any] | None:
    """2D electrode positions (0-1, nose up) of the EEG channels, if the recording has them."""
    import mne

    try:
        layout = mne.channels.make_eeg_layout(info, exclude=())
    except (RuntimeError, ValueError):
        return None
    centres = layout.pos[:, :2] + layout.pos[:, 2:] / 2
    return {
        "names": list(layout.names),
        "x": [round(float(v), 4) for v in centres[:, 0]],
        "y": [round(float(v), 4) for v in centres[:, 1]],
    }


def _bad_channels(ctx: RunContext, record: ProposalRecord) -> dict[str, Any]:
    import mne

    from braino.tools.eeg import _derivative_file
    from braino.tools.recordings import PREPROC

    proposal = record.proposal
    path = _derivative_file(ctx, proposal["recording"], f"_desc-{PREPROC}_eeg.fif")
    raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
    raw.pick("eeg", exclude=())
    middle = raw.times[-1] / 2
    start = max(0.0, middle - SNIPPET_S / 2)
    stop = min(raw.times[-1], start + SNIPPET_S)
    snippet = raw.copy().crop(start, stop).load_data(verbose="error")
    if snippet.info["sfreq"] > SNIPPET_HZ:
        snippet.resample(SNIPPET_HZ, verbose="error")
    data = snippet.get_data() * 1e6
    data -= np.median(data, axis=1, keepdims=True)
    names = snippet.ch_names
    amplitude = np.std(data, axis=1)

    proposed = [ch for ch in proposal.get("channels", []) if ch in names]
    positions = {ch["ch_name"]: np.asarray(ch["loc"][:3]) for ch in raw.info["chs"]}
    usable = [
        n for n in names
        if n not in proposed and n not in raw.info["bads"] and np.isfinite(positions[n]).all()
        and np.any(positions[n])
    ]  # fmt: skip

    def neighbour(channel: str) -> str | None:
        here = positions.get(channel)
        if here is None or not np.any(here) or not usable:
            return usable[0] if usable else None
        return str(min(usable, key=lambda n: float(np.linalg.norm(positions[n] - here))))

    def series(channel: str) -> list[float]:
        return [round(float(v), 2) for v in data[names.index(channel)]]

    channels = []
    for ch in proposed:
        other = neighbour(ch)
        channels.append(
            {
                "name": ch,
                "reasons": proposal.get("reasons", {}).get(ch, []),
                "signal": series(ch),
                "neighbour": other,
                "neighbour_signal": series(other) if other else [],
            }
        )
    return {
        "kind": "bad_channels",
        "layout": _layout(raw.info),
        "amplitude_uv": {n: round(float(a), 2) for n, a in zip(names, amplitude, strict=True)},
        "proposed": proposed,
        "snippet": {"start_s": round(start, 2), "sfreq": float(snippet.info["sfreq"])},
        "channels": channels,
    }


def _ica(ctx: RunContext, record: ProposalRecord) -> dict[str, Any]:
    import mne
    from scipy.signal import welch

    from braino.tools.eeg import _derivative_file
    from braino.tools.recordings import PREPROC

    proposal = record.proposal
    data_file = _derivative_file(ctx, proposal["recording"], f"_desc-{PREPROC}_eeg.fif")
    ica_file = _derivative_file(ctx, proposal["ica_file"], f"_desc-{PREPROC}_ica.fif")
    ica = mne.preprocessing.read_ica(ica_file, verbose="error")
    raw = mne.io.read_raw_fif(data_file, preload=False, verbose="error")
    raw.crop(0, min(raw.times[-1], SPECTRUM_SECONDS)).load_data(verbose="error")
    raw.info["bads"] = list(proposal.get("bad_channels", []))
    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]
    freqs, power = welch(sources, fs=sfreq, nperseg=min(sources.shape[1], int(2 * sfreq)))
    keep = (freqs >= 1) & (freqs <= SPECTRUM_FMAX)
    db = 10 * np.log10(np.maximum(power[:, keep], 1e-30))

    weights = ica.get_components()  # channels x components
    info = raw.copy().pick(ica.ch_names).info
    labels = {c["index"]: c for c in proposal.get("components", [])}
    excluded = set(proposal.get("exclude", []))
    components = []
    for index in range(min(weights.shape[1], MAX_COMPONENTS)):
        meta = labels.get(index, {})
        components.append(
            {
                "index": index,
                "label": meta.get("label", "?"),
                "probability": meta.get("probability"),
                "explained_variance": meta.get("explained_variance"),
                "proposed": index in excluded,
                "weights": [round(float(w), 5) for w in weights[:, index]],
                "psd_db": [round(float(v), 2) for v in db[index]],
            }
        )
    return {
        "kind": "ica",
        "layout": _layout(info),
        "channels": list(ica.ch_names),
        "freqs": [round(float(f), 2) for f in freqs[keep]],
        "components": components,
        "shown": len(components),
        "n_components": int(weights.shape[1]),
    }
