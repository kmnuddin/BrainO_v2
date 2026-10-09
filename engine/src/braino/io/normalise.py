"""Make recordings from different formats look the same: channel names, channel types, electrode
positions and events.

Every change is listed in a :class:`NormalisationReport`, so the user (and the provenance record)
can see exactly what import did to the data. Automatic choices can be overridden.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from braino.io.readers import Recording, _import_mne

POSITIONAL_TEMPLATE = "colin27_1005"
"""Layout for 10-05 system names (Fp1, Cz, ...). These names describe a place on the head, so
the layout is applied when at least :data:`MIN_TEMPLATE_COVERAGE` of the EEG channels have them."""

CAP_TEMPLATES = (
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "biosemi160",
    "biosemi256",
    "GSN-HydroCel-32",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-65_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-129",
    "GSN-HydroCel-256",
    "GSN-HydroCel-257",
    "EGI_256",
    "easycap-M1",
    "easycap-M10",
    "easycap-M43",
    "mgh60",
    "mgh70",
    "brainproducts-RNP-BA-128",
)
"""Layouts of specific caps. Their channel names (e.g. BioSemi A1-D32, EGI E1-E128) only mean
something for that cap, so a cap layout is applied only when every one of its channels is in the
recording, and caps named with plain numbers are never chosen automatically."""

TEMPLATE_MONTAGES = (POSITIONAL_TEMPLATE, *CAP_TEMPLATES)

MIN_TEMPLATE_COVERAGE = 0.5
"""Share of EEG channels that must have 10-05 names for the 10-05 layout to be applied."""

# Reference suffixes and type prefixes that clinical systems add to standard electrode names,
# e.g. "EEG Fp1-REF". They are removed only when what remains is a known electrode name.
_NAME_PREFIXES = ("EEG ", "EEG-", "EEG_")
_NAME_SUFFIXES = ("-REF", "-LE", "-RE", "-AVG", "-AV", "-CAR")

# Composite trigger channels, used instead of the per-bit trigger channels next to them.
_COMPOSITE_STIM_CHANNELS = ("STI 014", "STI101", "STI 101")

ChannelType = Literal["eeg", "eog", "ecg", "emg", "resp", "misc", "stim"]


class ChannelRetype(BaseModel):
    from_type: str
    to_type: str
    reason: str


class NormalisationReport(BaseModel):
    """What normalisation changed or found; compact enough to show to the user and the agent."""

    renamed: dict[str, str] = Field(default_factory=dict, description="Old name -> new name")
    retyped: dict[str, ChannelRetype] = Field(default_factory=dict)
    channel_types: dict[str, int] = Field(description="Number of channels of each type")
    bad_channels: list[str] = Field(description="Channels marked bad in the file")
    positions: Literal["file", "template", "none"] = Field(
        description="Where EEG electrode positions come from: the file, a standard template "
        "(approximate), or nowhere"
    )
    template: str | None = Field(default=None, description="Template used, if any")
    eeg_without_position: list[str] = Field(default_factory=list)
    events: dict[str, int] = Field(description="Number of events of each description")
    events_from_stim: dict[str, int] = Field(
        default_factory=dict, description="Events read from each trigger channel"
    )
    notes: list[str] = Field(default_factory=list)


def normalise(
    recording: Recording,
    *,
    rename: Mapping[str, str] | None = None,
    channel_types: Mapping[str, ChannelType] | None = None,
    montage: str | None = "auto",
    stim_channels: Sequence[str] | None = None,
) -> tuple[Recording, NormalisationReport]:
    """Normalise a recording read with :func:`~braino.io.read_raw`.

    1. **Names:** standard electrode names get their standard spelling (``FP1`` -> ``Fp1``,
       ``EEG Fz-REF`` -> ``Fz``). Other names are kept.
    2. **Types:** channels typed EEG whose names say otherwise become EOG, ECG, EMG, respiration
       or miscellaneous channels.
    3. **Positions:** positions from the file are kept. Otherwise the template that matches the
       most EEG channel names is applied (``montage="auto"``), or the named one; ``None`` skips
       this step.
    4. **Events:** events in trigger channels become annotations next to the file's own, unless
       the file already annotates them. ``stim_channels`` overrides which channels are read.

    ``rename`` and ``channel_types`` are applied last and win over the automatic choices.
    Returns a new :class:`Recording` (the original is unchanged) and the report.
    """
    mne = _import_mne()
    raw = recording.raw.copy()
    notes: list[str] = []

    renamed = _standardise_names(mne, raw, notes)
    retyped = _retype_channels(raw)
    if rename:
        raw.rename_channels(dict(rename))
        renamed.update(rename)
    if channel_types:
        for ch, new in channel_types.items():
            retyped[ch] = ChannelRetype(
                from_type=raw.get_channel_types(picks=[ch])[0], to_type=new, reason="requested"
            )
        raw.set_channel_types(dict(channel_types), on_unit_change="ignore")

    positions, template, unplaced = _set_positions(mne, raw, montage, notes)
    events_from_stim = _events_from_stim(mne, raw, stim_channels, notes)

    report = NormalisationReport(
        renamed=renamed,
        retyped=retyped,
        channel_types=dict(Counter(raw.get_channel_types())),
        bad_channels=list(raw.info["bads"]),
        positions=positions,
        template=template,
        eeg_without_position=unplaced,
        events=dict(Counter(str(d) for d in raw.annotations.description)),
        events_from_stim=events_from_stim,
        notes=notes,
    )
    normalised = dataclasses.replace(recording, raw=raw, notes=(*recording.notes, *notes))
    return normalised, report


def _known_names(mne: Any) -> dict[str, str]:
    """Lower-case name -> standard spelling, over all template montages."""
    known: dict[str, str] = {}
    for name in TEMPLATE_MONTAGES:
        for ch in mne.channels.make_standard_montage(name).ch_names:
            known.setdefault(ch.lower(), ch)
    return known


def _standardise_names(mne: Any, raw: Any, notes: list[str]) -> dict[str, str]:
    known = _known_names(mne)
    proposed: dict[str, str] = {}
    for i in mne.pick_types(raw.info, eeg=True, exclude=[]):
        old = raw.ch_names[i]
        standard = known.get(_strip_decorations(old).lower())
        if standard is not None and standard != old:
            proposed[old] = standard

    targets = Counter(proposed.values())
    others = set(raw.ch_names) - set(proposed)
    clashes = {old for old, new in proposed.items() if targets[new] > 1 or new in others}
    if clashes:
        notes.append(
            "Not renamed, because the standard name is already taken: " + ", ".join(sorted(clashes))
        )
    renames = {old: new for old, new in proposed.items() if old not in clashes}
    if renames:
        raw.rename_channels(renames)
    return renames


def _strip_decorations(name: str) -> str:
    stripped = name.strip()
    upper = stripped.upper()
    for prefix in _NAME_PREFIXES:
        if upper.startswith(prefix):
            stripped, upper = stripped[len(prefix) :], upper[len(prefix) :]
            break
    for suffix in _NAME_SUFFIXES:
        if upper.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    return stripped.strip()


def _guess_type(name: str) -> tuple[ChannelType, str] | None:
    upper = name.upper()
    if "EOG" in upper or upper in ("LOC", "ROC"):
        return "eog", "name indicates an eye (EOG) channel"
    if "ECG" in upper or "EKG" in upper:
        return "ecg", "name indicates a heart (ECG) channel"
    if "EMG" in upper or upper.startswith("CHIN"):
        return "emg", "name indicates a muscle (EMG) channel"
    if "RESP" in upper:
        return "resp", "name indicates a respiration channel"
    if upper.startswith("POL "):
        return "misc", "polygraphic channel ('POL' prefix)"
    return None


def _retype_channels(raw: Any) -> dict[str, ChannelRetype]:
    changes: dict[str, ChannelRetype] = {}
    for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True):
        guess = _guess_type(name) if kind == "eeg" else None
        if guess is not None:
            changes[name] = ChannelRetype(from_type=kind, to_type=guess[0], reason=guess[1])
    if changes:
        raw.set_channel_types(
            {name: change.to_type for name, change in changes.items()}, on_unit_change="ignore"
        )
    return changes


def _set_positions(
    mne: Any, raw: Any, montage: str | None, notes: list[str]
) -> tuple[Literal["file", "template", "none"], str | None, list[str]]:
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if not eeg:
        return "none", None, []
    placed = [ch for ch in eeg if _has_position(raw, ch)]
    if placed and montage == "auto":
        return "file", None, [ch for ch in eeg if ch not in placed]
    if montage is None:
        return ("file" if placed else "none"), None, [ch for ch in eeg if ch not in placed]

    if montage == "auto":
        chosen = _choose_template(mne, eeg)
        if chosen is None:
            notes.append(
                "No electrode positions in the file, and the channel names do not identify a "
                "standard layout; pass a layout name to add template positions."
            )
            return "none", None, eeg
        name = chosen
    else:
        name = montage

    layout = mne.channels.make_standard_montage(name)
    raw.set_montage(layout, on_missing="ignore")
    unplaced = [ch for ch in eeg if ch not in layout.ch_names]
    notes.append(
        f"Electrode positions are approximate, taken from the standard {name} layout "
        f"for {len(eeg) - len(unplaced)} of {len(eeg)} EEG channels."
    )
    return "template", name, unplaced


def _choose_template(mne: Any, eeg: list[str]) -> str | None:
    """The layout the EEG channel names identify, if any (see :data:`CAP_TEMPLATES`)."""
    names = set(eeg)
    positional = set(mne.channels.make_standard_montage(POSITIONAL_TEMPLATE).ch_names)
    if len(names & positional) >= MIN_TEMPLATE_COVERAGE * len(names):
        return POSITIONAL_TEMPLATE
    complete_caps = []
    for cap in CAP_TEMPLATES:
        cap_names = mne.channels.make_standard_montage(cap).ch_names
        if all(ch.isdigit() for ch in cap_names):
            continue
        if set(cap_names) <= names:
            complete_caps.append((len(cap_names), cap))
    return max(complete_caps)[1] if complete_caps else None  # the largest complete cap


def _has_position(raw: Any, ch: str) -> bool:
    loc = raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3]
    return bool((loc == loc).all() and (loc != 0).any())  # finite and not at the origin


def _events_from_stim(
    mne: Any, raw: Any, requested: Sequence[str] | None, notes: list[str]
) -> dict[str, int]:
    stim = [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True, exclude=[])]
    if requested is not None:
        channels = list(requested)
    else:
        composite = [ch for ch in _COMPOSITE_STIM_CHANNELS if ch in stim]
        channels = composite[:1] or stim
        annotated = set(raw.annotations.description)
        already = [ch for ch in channels if ch in annotated]
        if already:
            notes.append(
                "Trigger channels already annotated in the file, not read again: "
                + ", ".join(already)
            )
            channels = [ch for ch in channels if ch not in annotated]

    counts: dict[str, int] = {}
    for ch in channels:
        events = mne.find_events(
            raw,
            stim_channel=ch,
            shortest_event=1,
            consecutive=True,
            initial_event=True,
            verbose="error",
        )
        counts[ch] = len(events)
        # find_events and annotation onsets both count from the start of acquisition (sample
        # 0, not the first sample kept in the file, which is raw.first_samp).
        raw.annotations.append(
            onset=events[:, 0] / raw.info["sfreq"],
            duration=0.0,
            description=[str(int(code)) for code in events[:, 2]],
        )
    return counts
