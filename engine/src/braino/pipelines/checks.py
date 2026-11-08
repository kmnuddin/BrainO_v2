"""Plan checks: rules a pipeline spec must pass before it runs.

:func:`~braino.pipelines.spec.check_spec` checks each step on its own (the tool exists and
accepts the arguments). The plan checks look at the steps together, as an analyst reviewing
the plan would:

- **Order:** every step's input is made by an earlier step (epochs before ERPs, ERPs before a
  group test of ERPs), and cleaning steps come before epoching, which reads their output.
- **Names:** conditions, contrasts and epoch labels used by a step are defined by an earlier
  one; analysis windows lie inside the epochs.
- **Statistics:** a group test corrects for multiple comparisons, and has enough subjects for
  its permutations.
- **The data** (when a dataset is given): the subjects exist, the events of each condition
  occur in the recordings, the channels named exist, and filters fit the sampling rate.

Errors stop a plan; warnings are shown with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from braino.pipelines.spec import PipelineSpec, Step
from braino.stats.group import MIN_SUBJECTS
from braino.tools.registry import ToolNotFoundError, ToolRegistry, registry

Severity = Literal["error", "warning"]

CLEANING = ("eeg.ica_clean", "eeg.detect_bad_channels")
"""Steps that change the recording epoching reads; they must come before it."""
EPOCH_READERS = ("eeg.erp", "eeg.tfr")
"""Steps that read epochs (eeg.spectral too, unless it reads the continuous recording)."""
GROUP_SOURCE = {"erp": "eeg.erp", "psd": "eeg.spectral", "tfr": "eeg.tfr"}
"""The per-subject step whose results each kind of group test reads."""
MIN_CLUSTER_SUBJECTS = 7
#: Channel names an unknown-channel error lists (a 64-channel cap fits; 256 do not).
KNOWN_CHANNELS_LISTED = 70
"""Below this, a sign-flip permutation test has at most 2^6 = 64 permutations: its smallest
possible p is about 0.03, too coarse to correct for anything."""


class PlanFinding(BaseModel):
    severity: Severity
    step: str | None = Field(description="The step concerned, if one")
    message: str

    def __str__(self) -> str:
        where = f"{self.step}: " if self.step else ""
        return f"{self.severity}: {where}{self.message}"


class PlanCheck(BaseModel):
    findings: list[PlanFinding] = Field(default_factory=list)

    @property
    def errors(self) -> list[PlanFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[PlanFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def check_plan(
    spec: PipelineSpec,
    *,
    dataset: Path | None = None,
    subjects: list[str] | None = None,
    tools: ToolRegistry | None = None,
) -> PlanCheck:
    """Check ``spec`` as a whole; with ``dataset``, also against the data. ``subjects`` are
    the subjects it will run on (by default, every subject the spec selects)."""
    plan = _Plan(spec, registry if tools is None else tools)
    plan.check_order()
    plan.check_names()
    if dataset is not None:
        plan.check_data(dataset, subjects)
    else:
        plan.check_group_size(len(spec.subjects) if spec.subjects is not None else None)
    return PlanCheck(findings=plan.findings)


# --- the rules ----------------------------------------------------------------------------


class _Epochs(BaseModel):
    step: str
    conditions: dict[str, list[str]]
    tmin: float
    tmax: float


class _Plan:
    def __init__(self, spec: PipelineSpec, tools: ToolRegistry) -> None:
        self.spec = spec
        self.tools = tools
        self.findings: list[PlanFinding] = []
        self.steps = [(s, self._args(s, group=False)) for s in spec.steps]
        self.group = [(s, self._args(s, group=True)) for s in spec.group]

    def _args(self, step: Step, *, group: bool) -> dict[str, Any] | None:
        """The step's arguments with the tool's defaults (None if the tool is unknown)."""
        try:
            model = self.tools.get(step.tool).input_model
        except ToolNotFoundError:
            return None
        args = (
            self.spec.group_args(step, ["x", "y"]) if group
            else self.spec.recording_args(step, "x")
        )  # fmt: skip
        try:
            return model.model_validate(args).model_dump(mode="json")
        except ValidationError:
            return None  # check_spec reports it

    def add(self, severity: Severity, step: str | None, message: str) -> None:
        self.findings.append(PlanFinding(severity=severity, step=step, message=message))

    # order ------------------------------------------------------------------------------

    def check_order(self) -> None:
        seen: list[str] = []
        epoched = False
        for step, args in self.steps:
            tool = step.tool
            needs_preprocessing = tool.startswith("eeg.") and tool not in (
                "eeg.preprocess",
                "eeg.qc_report",
            )
            if needs_preprocessing and "eeg.preprocess" not in seen:
                self.add("error", step.id, f"{tool} needs eeg.preprocess before it")
            if tool in CLEANING and epoched:
                self.add(
                    "error",
                    step.id,
                    f"{tool} comes after eeg.epoch, so the epochs would not use it; put it "
                    "before eeg.epoch",
                )
            reads_epochs = tool in EPOCH_READERS or (
                tool == "eeg.spectral" and (args or {}).get("source") == "epochs"
            )
            if reads_epochs and "eeg.epoch" not in seen:
                self.add("error", step.id, f"{tool} reads epochs: put eeg.epoch before it")
            if tool == "eeg.epoch":
                epoched = True
            seen.append(tool)
        for step, args in self.group:
            if args is None:
                continue
            source = GROUP_SOURCE.get(args.get("data", "erp"))
            if source is not None and source not in seen:
                self.add(
                    "error",
                    step.id,
                    f"{step.tool} tests {source} results, but no step runs {source}",
                )

    # names and windows ------------------------------------------------------------------

    def check_names(self) -> None:
        epochs: dict[str, _Epochs] = {}
        contrasts: dict[str, set[str]] = {}  # epochs label -> contrast names
        spectra: dict[str, set[str]] = {}  # label -> conditions with spectra
        for step, args in self.steps:
            if args is None:
                continue
            tool = step.tool
            if tool == "eeg.epoch":
                epochs[args["label"]] = _Epochs(
                    step=step.id,
                    conditions=args["conditions"],
                    tmin=args["tmin"],
                    tmax=args["tmax"],
                )
                self._check_baseline(step.id, args)
                continue
            label = args.get("label", "epochs")
            if tool == "eeg.spectral" and args.get("source") == "continuous":
                spectra["continuous"] = {"continuous"}
                continue
            if tool not in (*EPOCH_READERS, "eeg.spectral"):
                continue
            ep = epochs.get(label)
            if ep is None:
                if epochs:
                    self.add(
                        "error",
                        step.id,
                        f"no epochs labelled {label!r}; the epochs are "
                        f"{', '.join(map(repr, epochs))}",
                    )
                continue
            names = set(ep.conditions)
            for c in args.get("conditions") or []:
                if c not in names:
                    self.add("error", step.id, _unknown("condition", c, names))
            if tool == "eeg.erp":
                for contrast in args.get("contrasts") or []:
                    for side in ("plus", "minus"):
                        if contrast[side] not in names:
                            self.add("error", step.id, _unknown("condition", contrast[side], names))
                contrasts.setdefault(label, set()).update(c["name"] for c in args["contrasts"])
                for m in args.get("measures") or []:
                    self._check_window(step.id, f"measure {m['name']!r}", m, ep)
            if tool == "eeg.spectral":
                spectra[label] = set(args.get("conditions") or names)
        for step, args in self.group:
            if args is None:
                continue
            self._check_group(step, args, epochs, contrasts, spectra)

    def _check_baseline(self, step_id: str, args: dict[str, Any]) -> None:
        baseline = args.get("baseline")
        if not baseline:
            return
        lo, hi = baseline
        if (lo is not None and lo < args["tmin"]) or (hi is not None and hi > args["tmax"]):
            self.add(
                "error",
                step_id,
                f"the baseline {baseline} is not inside the epoch "
                f"[{args['tmin']}, {args['tmax']}] s",
            )

    def _check_window(self, step_id: str, what: str, window: dict[str, Any], ep: _Epochs) -> None:
        tmin, tmax = window.get("tmin"), window.get("tmax")
        if (tmin is not None and tmin < ep.tmin) or (tmax is not None and tmax > ep.tmax):
            self.add(
                "error",
                step_id,
                f"{what}: the window [{tmin}, {tmax}] s is not inside the epochs "
                f"[{ep.tmin}, {ep.tmax}] s",
            )

    def _check_group(
        self,
        step: Step,
        args: dict[str, Any],
        epochs: dict[str, _Epochs],
        contrasts: dict[str, set[str]],
        spectra: dict[str, set[str]],
    ) -> None:
        data, label = args.get("data", "erp"), args.get("label", "epochs")
        if data == "psd":
            names = spectra.get(label)
        else:
            ep = epochs.get(label)
            names = None if ep is None else set(ep.conditions) | contrasts.get(label, set())
            if ep is not None:
                self._check_window(step.id, step.tool, args, ep)
        if names is not None:
            for c in args["conditions"]:
                if c not in names:
                    self.add("error", step.id, _unknown("condition or contrast", c, names))
        if step.tool == "stats.mass_univariate" and args.get("correction") == "none":
            self.add(
                "warning",
                step.id,
                "no correction for multiple comparisons: with many channels and time points, "
                "some will be significant by chance. Use 'fdr', 'bonferroni' or 'max_stat' "
                "unless this is exploration only.",
            )

    # the data ---------------------------------------------------------------------------

    def check_group_size(self, n_subjects: int | None) -> None:
        if n_subjects is None or not self.group:
            return
        for step, _ in self.group:
            if n_subjects < MIN_SUBJECTS:
                self.add(
                    "error",
                    step.id,
                    f"a group test needs at least {MIN_SUBJECTS} subjects; the plan has "
                    f"{n_subjects}",
                )
            elif step.tool == "stats.cluster_permutation" and n_subjects < MIN_CLUSTER_SUBJECTS:
                self.add(
                    "warning",
                    step.id,
                    f"with {n_subjects} subjects a cluster test has only {2**n_subjects} "
                    "sign-flip permutations, too few for small p-values; stats.mass_univariate "
                    "with FDR suits this design better",
                )

    def check_data(self, dataset: Path, subjects: list[str] | None) -> None:
        from braino.io.dataset import recording_channels, scan_dataset

        recordings = [
            r
            for r in scan_dataset(dataset).recordings
            if r.task == self.spec.task
            and (self.spec.session is None or r.session == self.spec.session)
            and (self.spec.run is None or r.run == self.spec.run)
        ]
        if not recordings:
            self.add("error", None, f"the dataset has no recording of task {self.spec.task!r}")
            return
        available = sorted({r.subject for r in recordings})
        wanted = subjects or self.spec.subjects or available
        missing = [s for s in wanted if s not in available]
        if missing:
            self.add("error", None, f"no {self.spec.task} recording for subject(s) "
                     f"{', '.join(missing)}; there are {', '.join(available)}")  # fmt: skip
        chosen = [r for r in recordings if r.subject in wanted]
        chosen = [r for r in chosen if r.subject not in self.spec.exclude_subjects]
        self.check_group_size(len({r.subject for r in chosen}))
        if not chosen:
            return
        events: dict[str, int] = {}
        for r in chosen:
            for name, n in r.events.items():
                events[name] = events.get(name, 0) + n
        channels: set[str] = set()
        for r in chosen:
            channels.update(recording_channels(dataset, r))
        rates = [r.sampling_rate_hz for r in chosen if r.sampling_rate_hz]
        for step, args in self.steps:
            if args is None:
                continue
            if step.tool == "eeg.epoch":
                self._check_events(step.id, args["conditions"], events)
            if step.tool == "eeg.preprocess":
                self._check_filters(step.id, args, min(rates) if rates else None)
                reference = args.get("reference")
                if isinstance(reference, list):
                    self._check_channels(step.id, "reference", reference, channels)
            if step.tool == "eeg.erp":
                for m in args.get("measures") or []:
                    self._check_channels(step.id, f"measure {m['name']!r}", m["channels"], channels)
        for step, args in self.group:
            if args is not None and args.get("channels"):
                self._check_channels(step.id, step.tool, args["channels"], channels)

    def _check_events(
        self, step_id: str, conditions: dict[str, list[str]], events: dict[str, int]
    ) -> None:
        if not events:
            return  # no events table: the epoching step reports what it finds
        for condition, descriptions in conditions.items():
            found = [d for d in descriptions if d in events]
            if not found:
                self.add(
                    "error",
                    step_id,
                    f"condition {condition!r}: none of its events "
                    f"({', '.join(descriptions)}) occur in the recordings; the events are "
                    f"{', '.join(sorted(events)[:20])}",
                )
            elif len(found) < len(descriptions):
                absent = [d for d in descriptions if d not in events]
                self.add(
                    "warning",
                    step_id,
                    f"condition {condition!r}: events {', '.join(absent)} do not occur",
                )

    def _check_filters(self, step_id: str, args: dict[str, Any], sfreq: float | None) -> None:
        rate = args.get("resample_hz") or sfreq
        h_freq, l_freq = args.get("h_freq"), args.get("l_freq")
        if rate is not None and h_freq is not None and h_freq >= rate / 2:
            self.add(
                "error",
                step_id,
                f"the low-pass at {h_freq:g} Hz is not below half the sampling rate ({rate:g} Hz)",
            )
        if h_freq is not None and l_freq is not None and l_freq >= h_freq:
            self.add("error", step_id, f"the high-pass ({l_freq:g} Hz) is not below the "
                     f"low-pass ({h_freq:g} Hz)")  # fmt: skip

    def _check_channels(self, step_id: str, what: str, names: list[str], known: set[str]) -> None:
        if not known:
            return
        lower = {k.lower() for k in known}
        unknown = [n for n in names if n.lower() not in lower]
        if unknown:
            # Name the channels there are, so the next try (often a model's) can pick from them.
            names_known = sorted(known, key=str.lower)
            listed = ", ".join(names_known[:KNOWN_CHANNELS_LISTED])
            more = len(names_known) - KNOWN_CHANNELS_LISTED
            if more > 0:
                listed += f" and {more} more"
            self.add(
                "error",
                step_id,
                f"{what}: no channel {', '.join(unknown)} in the recordings; they have {listed}",
            )


def _unknown(what: str, name: str, names: set[str]) -> str:
    return f"no {what} {name!r}; defined: {', '.join(sorted(names)) or 'none'}"
