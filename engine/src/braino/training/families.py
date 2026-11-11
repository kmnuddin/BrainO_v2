"""Families of training episodes: each makes a random episode of one kind of request.

A family picks a dataset (subjects, planted problems, event names, mains frequency), words a
request in one of several ways, and writes the script a careful analyst would follow, with
an answer written from the actual results. The evaluation's tasks are held out: requests
here are worded differently and use other settings, and :func:`check_held_out` refuses any
request that matches an evaluation request.

Every behaviour the agent needs is covered, not only the evaluation's gaps, so fine-tuning
on the gaps does not make the model forget the rest.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

from braino.evals.tasks import Check, DatasetSpec, ScriptedUser, SyntheticData
from braino.training.episodes import Action, Ask, Call, Decide, Episode, Results, Say

Family = Callable[[random.Random], Episode]

EEG = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]  # fmt: skip
#: Channels a planted problem may be on (not the frontal ones blinks already disturb, nor Pz,
#: which carries the P3).
PROBLEM_CHANNELS = ["F7", "F8", "T7", "T8", "C3", "C4", "P7", "P8", "O1", "O2", "Oz", "Cz"]
DEFAULT_EVENTS = {"rare": "stimulus/rare", "frequent": "stimulus/frequent"}
#: Event names that say which is which, with the words a user would use for them.
NAMED_EVENTS: list[tuple[dict[str, str], tuple[str, str]]] = [
    (DEFAULT_EVENTS, ("rare", "frequent")),
    ({"rare": "stimulus/target", "frequent": "stimulus/standard"}, ("target", "standard")),
    ({"rare": "stimulus/deviant", "frequent": "stimulus/standard"}, ("deviant", "standard")),
    ({"rare": "oddball", "frequent": "standard"}, ("oddball", "standard")),
]


# --- helpers -------------------------------------------------------------------------------


def labels(rng: random.Random, n: int) -> list[str]:
    """``n`` subject labels, sorted ("01" ... "12")."""
    return sorted(rng.sample([f"{i:02d}" for i in range(1, 13)], n))


def dataset(
    subjects: list[str],
    *,
    seconds: float = 60,
    events: dict[str, str] | None = None,
    **planted: Any,
) -> DatasetSpec:
    if events == DEFAULT_EVENTS:
        events = None
    return DatasetSpec(
        synthetic=SyntheticData(subjects=subjects, seconds=seconds, events=events, **planted)
    )


def listing(items: list[str], word: str = "and") -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {word} {items[-1]}"


def sub(rng: random.Random, label: str) -> str:
    """A subject as a user may write it."""
    return rng.choice(
        [f"subject {label}", f"sub-{label}", f"participant {label}", f"subject {label}"]
    )


def p_value(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def ms(seconds: float) -> str:
    return f"{round(seconds * 1000)}"


def window_words(rng: random.Random, tmin: float, tmax: float) -> str:
    return rng.choice(
        [
            f"{ms(tmin)}-{ms(tmax)} ms",
            f"between {ms(tmin)} and {ms(tmax)} ms",
            f"from {ms(tmin)} to {ms(tmax)} ms",
            f"{tmin:g}-{tmax:g} s",
        ]
    )


def notes_text(notes: list[str], limit: int = 2) -> str:
    """The tools' warnings, for the answer."""
    kept = [n for n in notes if n][:limit]
    return (" " + " ".join(kept)) if kept else ""


def capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def inspect() -> Call:
    return Call("dataset.inspect", {"max_recordings": 20}, id="ds")


# --- inspection ----------------------------------------------------------------------------


def inspect_dataset(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 6))
    events, _ = rng.choice(NAMED_EVENTS)
    seconds = rng.choice([40, 60])
    first = subjects[0]
    kind = rng.choice(["overview", "count", "events", "rate", "mains"])
    if kind == "overview":
        request = rng.choice(
            [
                "Give me an overview of this dataset.",
                "What does this dataset contain?",
                "Summarise the recordings in the open dataset.",
                "Tell me what's in the data I loaded.",
            ]
        )

        def answer(r: Results) -> str:
            ds = r["ds"]
            rec = ds["recordings"][0]
            evs = ", ".join(f"{k} ({v})" for k, v in rec["events"].items())
            return (
                f"The dataset has {ds['n_subjects']} subjects ({listing(ds['subjects'])}), "
                f"task '{ds['tasks'][0]}', {ds['n_recordings']} EEG recordings in all. Each "
                f"recording has {rec['channels']['eeg']} EEG channels and "
                f"{rec['channels'].get('eog', 0)} EOG channel, sampled at "
                f"{rec['sampling_rate_hz']:g} Hz, about {rec['duration_s']:g} s long. Events in "
                f"subject {rec['subject']}: {evs}."
            )

        terms = ["subjects"]
    elif kind == "count":
        request = rng.choice(
            [
                "How many participants are in this study?",
                "How many subjects do I have?",
                "Which subjects are in the dataset?",
            ]
        )

        def answer(r: Results) -> str:
            ds = r["ds"]
            return f"There are {ds['n_subjects']} subjects: {listing(ds['subjects'])}."

        terms = []
    elif kind == "events":
        request = rng.choice(
            [
                f"What event markers does {sub(rng, first)} have, and how many of each?",
                f"List the triggers in {sub(rng, first)}'s recording.",
                f"How many trials of each type are there for {sub(rng, first)}?",
            ]
        )

        def answer(r: Results) -> str:
            rec = next(x for x in r["ds"]["recordings"] if x["subject"] == first)
            evs = listing([f"{v} '{k}'" for k, v in rec["events"].items()])
            return f"Subject {first}'s recording has {evs} events."

        terms = list(events.values())
    elif kind == "rate":
        request = rng.choice(
            [
                "What is the sampling rate of the recordings, and how long are they?",
                "At what rate was the EEG sampled? How many seconds per recording?",
            ]
        )

        def answer(r: Results) -> str:
            recs = r["ds"]["recordings"]
            rates = sorted({x["sampling_rate_hz"] for x in recs})
            durations = sorted({x["duration_s"] for x in recs})
            return (
                f"All {len(recs)} recordings are sampled at "
                f"{listing([f'{x:g} Hz' for x in rates])} "
                f"and last {listing([f'{d:g} s' for d in durations])}."
            )

        terms = ["Hz"]
    else:
        line = rng.choice([50.0, 60.0])
        request = rng.choice(
            [
                "Were these data recorded with 50 Hz or 60 Hz mains?",
                "What is the power-line frequency of the recordings?",
            ]
        )

        def answer(r: Results) -> str:
            freqs = sorted({x["line_freq_hz"] for x in r["ds"]["recordings"]})
            listed = listing([f"{f:g} Hz" for f in freqs])
            return f"The recordings list a power-line frequency of {listed}."

        return Episode(
            family="inspect",
            request=request,
            dataset=dataset(subjects, seconds=seconds, events=events, line_freq=line),
            script=[inspect(), Say(answer)],
            checks=[
                Check(kind="tool_called", tool="dataset.inspect"),
                Check(kind="answer_number", min=line, max=line),
                Check(kind="max_calls", count=1),
            ],
        )
    checks = [Check(kind="tool_called", tool="dataset.inspect"), Check(kind="max_calls", count=1)]
    if terms:
        checks.append(Check(kind="answer_mentions", terms=terms))
    return Episode(
        family="inspect",
        request=request,
        dataset=dataset(subjects, seconds=seconds, events=events),
        script=[inspect(), Say(answer)],
        checks=checks,
    )


# --- quality ---------------------------------------------------------------------------------


def _qc_planted(rng: random.Random) -> tuple[str, dict[str, Any], str | None]:
    kind = rng.choice(["noisy", "flat", "line", "clean", "noisy"])
    if kind == "noisy":
        ch = rng.choice(PROBLEM_CHANNELS)
        return kind, {"noisy": [ch]}, ch
    if kind == "flat":
        ch = rng.choice(PROBLEM_CHANNELS)
        return kind, {"flat": [ch]}, ch
    if kind == "line":
        return (
            kind,
            {"line_noise_uv": rng.choice([10.0, 20.0]), "line_freq": rng.choice([50.0, 60.0])},
            None,
        )
    return kind, {}, None


def qc_findings(qc: dict[str, Any], label: str) -> str:
    parts = []
    if qc["n_flat"]:
        parts.append(f"{qc['n_flat']} flat channel(s): {listing(qc['flat'])}")
    if qc["n_noisy"]:
        noisy = [f"{n['channel']} ({n['reason']}, robust z {n['z']:g})" for n in qc["noisy"]]
        parts.append(f"{qc['n_noisy']} noisy channel(s): {listing(noisy)}")
    if qc.get("n_strong_line_noise"):
        parts.append(
            f"strong {qc['line_freq_hz']:g} Hz line noise on {qc['n_strong_line_noise']} channels "
            f"(median ratio {qc['line_noise_median_ratio']:g})"
        )
    found = "; ".join(parts) if parts else "no flat or noisy channels and no strong line noise"
    artifacts = f" {qc['artifact_fraction'] * 100:g}% of the recording has large artifacts."
    return f"Subject {label}: {found}.{artifacts if qc['artifact_fraction'] else ''}"


QC_NEXT = (
    " Nothing was changed. If you want, I can detect and mark bad channels, or filter out the "
    "line noise, as a next step."
)


def qc_subject(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    _, planted, ch = _qc_planted(rng)
    who = sub(rng, label)
    request = rng.choice(
        [
            f"Is {who}'s recording clean?",
            f"Run a quality check on {who}.",
            f"Any problems with the data of {who}?",
            f"Look at the signal quality of {who}: anything wrong?",
            f"QC {who} please.",
            f"Does {who} have flat or noisy electrodes?",
            f"Is there mains interference in {who}'s EEG?",
            f"Is {who}'s data fine to analyse?",
            f"Should I worry about anything in {who}'s EEG before the analysis?",
            f"Anything off with {who}?",
            f"How good is the recording of {who}?",
            f"Can I trust {who}'s recording, or are there bad channels?",
            f"Check {who} for problems before I start.",
        ]
    )
    # Half look at the dataset first, as a model often does. Its summary lists the channels
    # marked bad in the channels file, which only says what someone noted: the quality
    # screen measures the signals, and the answer comes from it.
    look_first = rng.random() < 0.5

    def answer(r: Results) -> str:
        text = qc_findings(r["qc"], label) + QC_NEXT
        if not look_first:
            return text
        rec = next(x for x in r["ds"]["recordings"] if x["subject"] == label)
        marked = (
            f"marks {listing(rec['bad_channels'])} as bad"
            if rec["bad_channels"]
            else "marks no channel as bad"
        )
        return (
            f"The dataset's channels file {marked}, but that is only what was noted at "
            f"recording; the quality screen measured the signals. {text}"
        )

    checks = [
        Check(kind="tool_called", tool="eeg.qc_report", args={"subject": label}),
        Check(kind="no_decision_requested"),
        Check(kind="tool_not_called", tool=["eeg.detect_bad_channels", "eeg.preprocess"]),
    ]
    if ch:
        checks.append(Check(kind="answer_mentions", terms=[ch]))
    script: list[Action] = [inspect()] if look_first else []
    script += [Call("eeg.qc_report", {"subject": label}, id="qc"), Say(answer)]
    return Episode(
        family="qc",
        request=request,
        dataset=dataset(subjects, seconds=rng.choice([30, 60]), **planted),
        script=script,
        checks=checks,
    )


def qc_many(rng: random.Random) -> Episode:
    """No subject named: ask which, or check them all when asked to."""
    subjects = labels(rng, rng.randint(2, 4))
    ch = rng.choice(PROBLEM_CHANNELS)
    planted: dict[str, Any] = {"noisy": [ch]}
    explicit = rng.random() < 0.4
    if explicit:
        request = rng.choice(
            [
                "Check the quality of every subject's recording.",
                "QC all the participants.",
                "Run the quality screen on each recording in the dataset.",
            ]
        )
        chosen = subjects
        script: list[Any] = [inspect()]
        user = ScriptedUser()
    else:
        request = rng.choice(
            [
                "Check the data quality.",
                "Are there noisy channels?",
                "Do a quality check.",
                "Is the EEG any good?",
                "Look for bad channels.",
                "Screen the recordings for problems.",
                "Find anything wrong with the data.",
                "Are the signals OK?",
            ]
        )
        all_of_them = rng.random() < 0.6
        chosen = subjects if all_of_them else [rng.choice(subjects)]
        reply = (
            rng.choice(["All of them.", "Every subject, please.", "Check them all."])
            if all_of_them
            else rng.choice([f"Just {chosen[0]}.", f"Only subject {chosen[0]}."])
        )
        script = [
            inspect(),
            Ask(
                lambda r: (
                    f"The dataset has subjects {listing(r['ds']['subjects'])}. Should I check "
                    "all of them, or which one?"
                )
            ),
        ]
        user = ScriptedUser(replies=[reply], default_reply=reply)
    for label in chosen:
        script.append(Call("eeg.qc_report", {"subject": label}, id=f"qc{label}"))

    def answer(r: Results) -> str:
        lines = [qc_findings(r[f"qc{label}"], label) for label in chosen]
        return "\n".join(lines) + QC_NEXT

    script.append(Say(answer))
    checks = [Check(kind="tool_called", tool="eeg.qc_report", args={"subject": s}) for s in chosen]
    checks += [Check(kind="no_decision_requested"), Check(kind="answer_mentions", terms=[ch])]
    return Episode(
        family="qc-many",
        request=request,
        dataset=dataset(subjects, seconds=30, **planted),
        script=script,
        checks=checks,
        user=user,
    )


# --- preprocessing -------------------------------------------------------------------------


def preprocess_settings(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    lo = rng.choice([0.1, 0.5, 1.0])
    hi = rng.choice([30.0, 40.0, 45.0])
    args: dict[str, Any] = {"subject": label, "l_freq": lo, "h_freq": hi}
    asks = [f"a {lo:g}-{hi:g} Hz band-pass"]
    if rng.random() < 0.4:
        rate = rng.choice([125.0, 200.0])
        args["resample_hz"] = rate
        asks.append(rng.choice([f"resample to {rate:g} Hz", f"downsample to {rate:g} Hz"]))
    ref_kind = rng.choice(["average", "average", "Cz", "pair"])
    if ref_kind == "average":
        args["reference"] = "average"
        asks.append(rng.choice(["average reference", "re-reference to the average"]))
    elif ref_kind == "Cz":
        args["reference"] = ["Cz"]
        asks.append(rng.choice(["reference to Cz", "Cz as the reference"]))
    else:
        pair = rng.choice([["T7", "T8"], ["P7", "P8"]])
        args["reference"] = pair
        asks.append(f"reference to the mean of {pair[0]} and {pair[1]}")
    if rng.random() < 0.3:
        args["notch_line_noise"] = True
        asks.append("a notch at the line frequency")
    rng.shuffle(asks)
    request = capital(
        rng.choice(
            [
                f"preprocess {who} with {listing(asks)}.",
                f"please filter {who}: {listing(asks)}.",
                f"for {who}, apply {listing(asks)}.",
            ]
        )
    )

    def answer(r: Results) -> str:
        pre = r["pre"]
        extra = notes_text(pre.get("notes", []))
        return (
            f"Preprocessed subject {label}: {'; '.join(pre['steps'])}. The result is at "
            f"{pre['sfreq_hz']:g} Hz, {pre['duration_s']:g} s, saved as {pre['output']}; the "
            f"original recording is unchanged.{extra}"
        )

    return Episode(
        family="preprocess",
        request=request,
        dataset=dataset(subjects, seconds=30),
        script=[Call("eeg.preprocess", args, id="pre"), Say(answer)],
        checks=[
            Check(kind="tool_called", tool="eeg.preprocess", args=args),
            Check(kind="max_calls", count=2),
        ],
    )


#: The reference the user asks for, the channels tried, and the word the answer uses: the
#: user's word, whichever channels stand for it (A1 and A2 are mastoid or ear channels).
MISSING_REFERENCES: list[tuple[str, list[str], str]] = [
    ("the mastoids", ["M1", "M2"], "mastoid"),
    ("the mastoids", ["A1", "A2"], "mastoid"),
    ("linked mastoids", ["TP9", "TP10"], "mastoid"),
    ("linked mastoids", ["A1", "A2"], "mastoid"),
    ("the mean of both mastoids", ["M1", "M2"], "mastoid"),
    ("the earlobes", ["A1", "A2"], "earlobe"),
    ("linked ears", ["A1", "A2"], "ear"),
    ("the nose tip", ["Nz"], "nose"),
]


def preprocess_missing_reference(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    words, channels, term = rng.choice(MISSING_REFERENCES)
    request = rng.choice(
        [
            f"Preprocess {who} and reference to {words}.",
            f"Re-reference {who} to {words}.",
            f"I need {who} referenced to {words}, with the usual filters.",
            f"Use {words} as the reference for {who}.",
            f"Filter {who} and set the reference to {words}.",
        ]
    )
    choice = rng.choice(["average", "average", "Cz", "pair"])
    reference: str | list[str]
    if choice == "average":
        reply, reference, used = (
            "Use the average reference then.",
            "average",
            "the average reference",
        )
    elif choice == "Cz":
        reply, reference, used = "Cz is fine.", ["Cz"], "Cz"
    else:
        reply, reference, used = "Take T7 and T8 instead.", ["T7", "T8"], "the mean of T7 and T8"
    missing = listing(channels)
    script: list[Action] = [
        Call(
            "eeg.preprocess", {"subject": label, "reference": channels}, id="try", expect_error=True
        ),
        Ask(
            f"The recording has no {term} channel ({missing}); its EEG channels are "
            f"{', '.join(EEG)}. Which reference should I use instead: the average reference, "
            "or other channels?"
        ),
        Call("eeg.preprocess", {"subject": label, "reference": reference}, id="pre"),
        Say(
            lambda r: (
                f"Subject {label}'s recording has no {term} channel ({missing}), so as you chose I "
                f"referenced it to {used} instead. Preprocessing: {'; '.join(r['pre']['steps'])}; "
                f"saved as {r['pre']['output']}."
            )
        ),
    ]
    return Episode(
        family="missing-reference",
        request=request,
        dataset=dataset(subjects, seconds=30),
        script=script,
        checks=[
            Check(kind="tool_called", tool="eeg.preprocess", args={"reference": reference}),
            Check(kind="answer_mentions", terms=[term, f"{term}s"], any_term=True),
            Check(kind="answer_mentions", terms=[channels[0]]),
        ],
        user=ScriptedUser(replies=[reply], default_reply=reply),
    )


# --- data the dataset does not have ---------------------------------------------------------


def missing_subject(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 4))
    absent = rng.choice([f"{i:02d}" for i in range(13, 30)])
    who = sub(rng, absent)
    asked = rng.choice(["qc", "preprocess"])
    request = (
        rng.choice([f"Check the quality of {who}.", f"Is {who}'s recording noisy?"])
        if asked == "qc"
        else rng.choice([f"Filter {who}'s data.", f"Preprocess {who}."])
    )
    go_on = rng.random() < 0.5
    script: list[Any] = [inspect()]
    user = ScriptedUser()
    checks = [Check(kind="answer_mentions", terms=[absent])]
    if go_on:
        other = rng.choice(subjects)
        reply = f"Oh, I meant {other}."
        user = ScriptedUser(replies=[reply], default_reply=reply)
        script.append(
            Ask(
                lambda r: (
                    f"There is no subject {absent} in this dataset; it has subjects "
                    f"{listing(r['ds']['subjects'])}. Which one did you mean?"
                )
            )
        )

        def done(r: Results) -> str:
            if asked == "qc":
                return qc_findings(r["qc"], other)
            return f"Preprocessed: {'; '.join(r['pre']['steps'])}."

        if asked == "qc":
            script.append(Call("eeg.qc_report", {"subject": other}, id="qc"))
        else:
            script.append(Call("eeg.preprocess", {"subject": other}, id="pre"))
        script.append(
            Say(
                lambda r: (
                    f"There is no subject {absent}, so as you said I used subject {other}. "
                    f"{done(r)}"
                )
            )
        )
        checks.append(Check(kind="answer_mentions", terms=[other]))
    else:
        script.append(
            Say(
                lambda r: (
                    f"There is no subject {absent} in this dataset: it has subjects "
                    f"{listing(r['ds']['subjects'])}. Tell me which of them you want, and I will "
                    "do it."
                )
            )
        )
        checks.append(Check(kind="tool_not_called", tool=["eeg.qc_report", "eeg.preprocess"]))
    return Episode(
        family="missing-subject",
        request=request,
        dataset=dataset(subjects, seconds=30),
        script=script,
        checks=checks,
        user=user,
    )


MISSING_INFO: list[tuple[str, str]] = [
    ("What was the mean reaction time on target trials?", "response"),
    ("How accurate were the participants?", "response"),
    ("Did responses get slower over the session?", "response"),
    ("Compare button-press latencies between the two conditions.", "response"),
    ("How old were the participants?", "age"),
    ("What is the handedness of each subject?", "handedness"),
    ("Show me the eye-tracking data for the first subject.", "eye-tracking"),
    ("What was the heart rate during the task?", "ECG"),
]


def missing_info(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 4))
    events, _ = rng.choice(NAMED_EVENTS)
    request, kind = rng.choice(MISSING_INFO)

    def answer(r: Results) -> str:
        rec = r["ds"]["recordings"][0]
        evs = listing(list(rec["events"]))
        channels = ", ".join(f"{v} {k.upper()}" for k, v in rec["channels"].items())
        if kind == "response":
            return (
                f"The recordings have no response events, only {evs}, so there are no reaction "
                "times or accuracies to report. If responses were logged elsewhere, add them to "
                "the events files and I can use them."
            )
        if kind == "ECG":
            return (
                f"There is no ECG channel: the recordings have {channels} channels, so I cannot "
                "give a heart rate."
            )
        if kind == "eye-tracking":
            return (
                f"The dataset has no eye-tracking data: only EEG recordings ({channels} channels)."
            )
        return (
            f"The dataset summary has no {kind} information: it lists only the recordings "
            f"({listing(r['ds']['subjects'])}) with their channels and events."
        )

    return Episode(
        family="missing-info",
        request=request,
        dataset=dataset(subjects, seconds=30, events=events),
        script=[inspect(), Say(answer)],
        checks=[
            Check(kind="tool_called", tool="dataset.inspect"),
            Check(kind="answer_mentions", terms=["no", "not", "cannot"], any_term=True),
            Check(
                kind="tool_not_called",
                tool=["eeg.epoch", "eeg.erp", "analysis.erp_comparison", "eeg.preprocess"],
            ),
        ],
    )


REFUSALS: list[tuple[str, str]] = [
    ("Delete the recording of subject {s}.", "delete data"),
    ("Remove {who} from the study folder.", "delete data"),
    ("Rename subject {s} to subject 99 in the dataset.", "rename subjects"),
    ("Upload the results to my Google Drive.", "upload files"),
    ("Email the ERP figure to my supervisor.", "send email"),
    ("Overwrite the raw file of {who} with the filtered data.", "change the original recordings"),
    ("Edit the events of {who} so the first trial is a target.", "edit the original events"),
    ("Train a deep-learning classifier on these recordings.", "train classifiers"),
    ("Do source localisation of the P3 for {who}.", "do source localisation"),
]


def refuse(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 4))
    label = rng.choice(subjects)
    template, what = rng.choice(REFUSALS)
    request = template.format(s=label, who=sub(rng, label))
    original = (
        "BrainO never changes the original recordings, and "
        if "change" in what or "delete" in what or "edit" in what or "rename" in what
        else ""
    )
    answer = (
        f"I cannot {what}: {original}none of my tools does that. I can preprocess, clean, "
        "epoch and analyse the data, and everything I make is saved as new files under "
        "derivatives/."
    )
    return Episode(
        family="refuse",
        request=request,
        dataset=dataset(subjects, seconds=30),
        script=[Say(answer)],
        checks=[
            Check(kind="answer_mentions", terms=["cannot"]),
            Check(kind="max_calls", count=1),
            Check(
                kind="tool_not_called",
                tool=["eeg.preprocess", "eeg.detect_bad_channels", "eeg.ica_clean", "eeg.epoch"],
            ),
        ],
    )


# --- decisions -----------------------------------------------------------------------------


def bad_channels(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    ch = rng.choice(PROBLEM_CHANNELS)
    request = rng.choice(
        [
            f"Find the broken electrodes in {who} and mark them bad.",
            f"Which channels of {who} should be marked bad? Mark them.",
            f"Flag the bad channels of {who}.",
            f"Detect bad channels for {who} and interpolate them later.",
        ]
    )
    verdict = rng.choice(["approve", "approve", "edit", "reject"])
    user = ScriptedUser(decisions="reject" if verdict == "reject" else "approve")
    if verdict == "edit":
        user = ScriptedUser(edits={"eeg.detect_bad_channels": {"channels": [ch]}})
    method = rng.choice(["qc", "qc", "pyprep"])

    def message(r: Results) -> str:
        prop = r["bads"]["proposal"]
        reasons = "; ".join(f"{c}: {', '.join(v)}" for c, v in prop["reasons"].items())
        return (
            f"The check proposes marking {listing(prop['channels'])} as bad in subject {label} "
            f"({reasons}). Approve to mark them in the preprocessed data, edit the list, or reject."
        )

    def answer(r: Results) -> str:
        d = r["bads_decision"]
        proposed = r["bads"]["proposal"]["channels"]
        if not d["approved"]:
            return (
                f"You rejected the proposal ({listing(proposed)}), so nothing was changed: no "
                f"channel of subject {label} is marked bad."
            )
        applied = d["applied"]["bad_channels"]
        dropped = [c for c in proposed if c not in applied]
        also = (
            f" {listing(dropped)} {'was' if len(dropped) == 1 else 'were'} proposed but not "
            "marked, as you chose."
            if dropped
            else ""
        )
        return (
            f"With your approval, {listing(applied)} {'is' if len(applied) == 1 else 'are'} marked "
            f"bad in subject {label}; {d['applied']['n_good_eeg']} good EEG channels remain.{also} "
            "Later steps interpolate them."
        )

    checks = [
        Check(kind="order", before="eeg.preprocess", after="eeg.detect_bad_channels"),
        Check(kind="decision_requested", tool="eeg.detect_bad_channels"),
    ]
    if verdict == "reject":
        checks.append(Check(kind="answer_mentions", terms=["nothing was changed"]))
    else:
        checks.append(Check(kind="answer_mentions", terms=[ch]))
    return Episode(
        family="bad-channels",
        request=request,
        dataset=dataset(subjects, seconds=60, noisy=[ch]),
        script=[
            Call("eeg.preprocess", {"subject": label}, id="pre"),
            Call("eeg.detect_bad_channels", {"subject": label, "method": method}, id="bads"),
            Decide("bads", message),
            Say(answer),
        ],
        checks=checks,
        user=user,
    )


def ica(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    request = rng.choice(
        [
            f"Get rid of the eye artifacts in {who}.",
            f"Run ICA on {who} to remove blinks.",
            f"{capital(who)} blinks a lot; clean that up.",
            f"Remove ocular artifacts from {who}'s recording with ICA.",
        ]
    )
    reject = rng.random() < 0.25
    labeler = rng.choice(["eog", "iclabel"])

    def message(r: Results) -> str:
        prop = r["ica"]["proposal"]
        comps = {c["index"]: c for c in prop["components"]}
        chosen = [
            f"{i} ({comps[i]['label']}, p {comps[i]['probability']:g})" for i in prop["exclude"]
        ]
        return (
            f"ICA on subject {label} proposes removing component(s) {listing(chosen)}. Approve "
            "to remove them from the preprocessed data, edit the list, or reject."
        )

    def answer(r: Results) -> str:
        d = r["ica_decision"]
        if not d["approved"]:
            return (
                f"You rejected the ICA proposal, so nothing was changed in subject {label}'s data."
            )
        removed = d["applied"]["removed"]
        what = ", ".join(f"{k} ({v})" for k, v in d["applied"]["removed_labels"].items())
        return (
            f"With your approval, {len(removed)} component(s) were removed from subject {label}: "
            f"{what}. The cleaned recording is {d['applied']['output']}."
        )

    return Episode(
        family="ica",
        request=request,
        dataset=dataset(subjects, seconds=60, blinks=rng.choice([30, 40])),
        script=[
            Call("eeg.preprocess", {"subject": label}, id="pre"),
            Call("eeg.ica_clean", {"subject": label, "labeler": labeler}, id="ica"),
            Decide("ica", message),
            Say(answer),
        ],
        checks=[
            Check(kind="order", before="eeg.preprocess", after="eeg.ica_clean"),
            Check(kind="decision_requested", tool="eeg.ica_clean"),
        ],
        user=ScriptedUser(decisions="reject" if reject else "approve"),
    )


def ica_then_erp(rng: random.Random) -> Episode:
    """Clean the blinks, wait for the decision, then measure an ERP on the cleaned data."""
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    channels = _channels(rng)
    tmin, tmax = rng.choice(WINDOWS)
    where = listing(channels) if len(channels) == 1 else f"the average of {listing(channels)}"
    win = window_words(rng, tmin, tmax)
    request = rng.choice(
        [
            f"Remove {who}'s blinks with ICA and then give me the {a} minus {b} amplitude at "
            f"{where}, {win}.",
            f"Run ICA on {who} to take out eye artifacts, then measure {a} versus {b} at "
            f"{where} ({win}).",
            f"Once the ocular artifacts are cleaned from {who}, what is the {a}-minus-{b} "
            f"difference at {where} {win}?",
            f"Clean {who}'s eye movements first, then measure the P3 ({a} minus {b}) at {where}, "
            f"{win}.",
        ]
    )
    labeler = rng.choice(["eog", "iclabel"])
    diff = f"{a}_minus_{b}"

    def message(r: Results) -> str:
        prop = r["ica"]["proposal"]
        comps = {c["index"]: c for c in prop["components"]}
        chosen = [
            f"{i} ({comps[i]['label']}, p {comps[i]['probability']:g})" for i in prop["exclude"]
        ]
        return (
            f"ICA on subject {label} proposes removing component(s) {listing(chosen)}. Approve "
            "to remove them before I epoch and measure, edit the list, or reject."
        )

    def answer(r: Results) -> str:
        removed = r["ica_decision"]["applied"]["removed"]
        value = next(m for m in r["erp"]["measures"] if m["waveform"] == diff)
        n = r["erp"]["n_trials"]
        return (
            f"With your approval, {len(removed)} ICA component(s) were removed from subject "
            f"{label}. On the cleaned data, the mean {a}-minus-{b} difference at {where}, "
            f"{ms(tmin)}-{ms(tmax)} ms, is {value['value_uv']:g} µV ({n[a]} {a} and {n[b]} {b} "
            f"epochs).{notes_text(r['erp'].get('notes', []), 1)}"
        )

    return Episode(
        family="ica-then-erp",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events, blinks=rng.choice([30, 40])),
        script=[
            Call("eeg.preprocess", {"subject": label}, id="pre"),
            Call("eeg.ica_clean", {"subject": label, "labeler": labeler}, id="ica"),
            Decide("ica", message),
            Call(
                "eeg.epoch",
                {
                    "subject": label,
                    "conditions": {a: [events["rare"]], b: [events["frequent"]]},
                    "reject": "peak_to_peak",
                    "peak_to_peak_uv": 100.0,
                },
                id="ep",
            ),
            Call(
                "eeg.erp",
                {
                    "subject": label,
                    "contrasts": [{"name": diff, "plus": a, "minus": b}],
                    "measures": [{"name": "P3", "channels": channels, "tmin": tmin, "tmax": tmax}],
                },
                id="erp",
            ),
            Say(answer),
        ],
        checks=[
            Check(kind="decision_requested", tool="eeg.ica_clean"),
            Check(kind="order", before="eeg.ica_clean", after="eeg.epoch"),
            Check(kind="tool_called", tool="eeg.erp", args={"subject": label}),
        ],
        user=ScriptedUser(decisions="approve"),
    )


# --- ERPs ----------------------------------------------------------------------------------

CHANNEL_SETS = [["Pz"], ["Pz"], ["Cz"], ["P3", "Pz", "P4"], ["CP"], ["Pz", "Cz"]]
WINDOWS = [(0.3, 0.5), (0.3, 0.6), (0.25, 0.5), (0.35, 0.55), (0.3, 0.45)]


def _channels(rng: random.Random) -> list[str]:
    chosen = rng.choice(CHANNEL_SETS)
    return ["P3", "Pz", "P4"] if chosen == ["CP"] else chosen


def _plan_message(r: Results, what: str) -> str:
    prop = r["plan"]["proposal"]
    steps = "; ".join(prop["steps"])
    group = f" Group test: {'; '.join(prop['group_steps'])}." if prop["group_steps"] else ""
    warn = f" Warnings: {' '.join(prop['warnings'])}" if prop["warnings"] else ""
    return (
        f"Plan '{prop['name']}' {what} for subject(s) {listing(prop['subjects'])}: {steps}.{group}"
        f"{warn} Approve to save the plan, then I run it."
    )


def _run_name(r: Results) -> dict[str, Any]:
    return {"name": r["plan_decision"]["applied"]["name"]}


def erp_one_subject(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 4))
    label = rng.choice(subjects)
    who = sub(rng, label)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    channels = _channels(rng)
    tmin, tmax = rng.choice(WINDOWS)
    peak = rng.random() < 0.35
    # A loosely worded P3 question names no channel or window: the usual Pz, 300-500 ms.
    loose = not peak and rng.random() < 0.35
    if loose:
        channels, (tmin, tmax) = ["Pz"], (0.3, 0.5)
    where = listing(channels) if len(channels) == 1 else f"the average of {listing(channels)}"
    win = window_words(rng, tmin, tmax)
    if loose:
        request = rng.choice(
            [
                f"Does {who} have a P300 effect? How large is it?",
                f"Measure the P3 of {who}.",
                f"What's the size of {who}'s P3?",
                f"Is the {a} vs {b} P3 effect there in {who}, and how big?",
                f"How large is the P300 difference between {a} and {b} trials for {who}?",
            ]
        )
    elif peak:
        request = rng.choice(
            [
                f"Where does {who}'s {a}-minus-{b} difference peak at {where} ({win}), and how "
                "high?",
                f"Find the peak latency and amplitude of the {a} vs {b} difference in {who} at "
                f"{where}, {win}.",
                f"For {who}, what is the latency of the largest positive deflection of {a} minus "
                f"{b} at {where}, {win}?",
            ]
        )
    else:
        request = rng.choice(
            [
                f"What is {who}'s {a} minus {b} amplitude at {where}, {win}?",
                f"For {who} only, measure {a} versus {b} at {where} {win}.",
                f"How much more positive are {a} than {b} trials in {who} at {where} ({win})?",
                f"Compare {a} and {b} ERPs at {where} for {who}, {win}.",
            ]
        )
    measure = "peak_positive" if peak else "mean"
    settings = {
        "subjects": [label],
        "condition_a": a,
        "events_a": [events["rare"]],
        "condition_b": b,
        "events_b": [events["frequent"]],
        "test_scope": "measured",
        "channels": channels,
        "tmin": tmin,
        "tmax": tmax,
        "measure": measure,
        "correction": "fdr",
    }
    diff = f"{a}_minus_{b}"

    def answer(r: Results) -> str:
        run = r["run"]
        row = next(x for x in run["per_subject"] if x["subject"] == label)
        kept = row.get("epochs_kept", {})
        trials = f" ({kept.get(a)} {a} and {kept.get(b)} {b} epochs kept)" if kept else ""
        at = f" at {where}, {ms(tmin)}-{ms(tmax)} ms"
        if peak:
            lat = row[f"{diff}_latency_ms"]
            text = (
                f"Subject {label}'s {a}-minus-{b} difference{at} peaks at {lat:g} ms, with an "
                f"amplitude of {row[diff]:g} {run['unit']}{trials}."
            )
        else:
            text = (
                f"In subject {label}, the mean {a}-minus-{b} difference{at} is {row[diff]:g} "
                f"{run['unit']} ({a} {row[a]:g}, {b} {row[b]:g}){trials}."
            )
        if loose:
            text += " I used Pz and 300-500 ms, the usual P3 channel and window."
        return text + notes_text(run.get("notes", []), 1)

    return Episode(
        family="erp-one-subject",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("analysis.erp_comparison", settings, id="plan"),
            Decide("plan", lambda r: _plan_message(r, f"measures the {a}-minus-{b} difference")),
            Call("analysis.run", _run_name, id="run"),
            Say(answer),
        ],
        checks=[
            Check(
                kind="tool_called",
                tool="analysis.erp_comparison",
                args={"subjects": [label], "measure": measure, "channels": channels},
            ),
            Check(kind="tool_called", tool="analysis.run"),
        ],
    )


CORRECTIONS = {"fdr": "FDR", "bonferroni": "Bonferroni", "max_stat": "max-statistic"}


def erp_group(rng: random.Random) -> Episode:
    n = rng.choice([4, 5, 6])
    subjects = labels(rng, n)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    channels = _channels(rng)
    tmin, tmax = rng.choice(WINDOWS)
    where = listing(channels) if len(channels) == 1 else f"the average of {listing(channels)}"
    win = window_words(rng, tmin, tmax)
    style = rng.choice(["default", "named", "named", "cluster", "everywhere"])
    correction, test, scope = "fdr", "mass_univariate", "measured"
    asked = ""
    if style == "named":
        correction = rng.choice(["bonferroni", "max_stat", "fdr"])
        asked = rng.choice(
            [
                f", corrected with {CORRECTIONS[correction]}",
                f" ({CORRECTIONS[correction]} correction)",
            ]
        )
    elif style == "cluster":
        test = "cluster_permutation"
        asked = rng.choice([" with a cluster permutation test", ", using a cluster-based test"])
    elif style == "everywhere":
        scope = "all"
        asked = rng.choice(
            [" at every channel and time point", " across the whole scalp and epoch"]
        )
    request = rng.choice(
        [
            f"Is there a significant {a} vs {b} difference at {where}, {win}, across "
            f"subjects{asked}?",
            f"Run the group comparison of {a} and {b} at {where} ({win}){asked}.",
            f"Test {a} against {b} over all participants at {where}, {win}{asked}.",
        ]
    )
    settings = {
        "condition_a": a,
        "events_a": [events["rare"]],
        "condition_b": b,
        "events_b": [events["frequent"]],
        "test_scope": scope,
        "channels": channels,
        "tmin": tmin,
        "tmax": tmax,
        "measure": "mean",
        "correction": correction,
        "test": test,
    }
    diff = f"{a}_minus_{b}"

    def answer(r: Results) -> str:
        run = r["run"]
        mean = run["mean"]
        g = run["group_test"]
        head = (
            f"Across {run['n_subjects']} subjects, the mean {a}-minus-{b} difference at {where} "
            f"({ms(tmin)}-{ms(tmax)} ms) is {mean[diff]:g} {run['unit']}."
        )
        if g is None:
            return head + notes_text(run.get("notes", []))
        if "clusters" in g:
            sig = [c for c in g["clusters"] if c["p"] <= g["alpha"]]
            if sig:
                c = sig[0]
                t0, t1 = c["ranges"]["time"]
                test_text = (
                    f" The cluster permutation test finds {len(sig)} significant cluster(s); the "
                    f"first is {c['sign']}, p = {p_value(c['p'])}, over "
                    f"{listing(c['channels'][:6])} "
                    f"from {ms(t0)} to {ms(t1)} ms."
                )
            else:
                best = min((c["p"] for c in g["clusters"]), default=None)
                lowest = f" (lowest cluster p = {p_value(best)})" if best is not None else ""
                test_text = f" The cluster permutation test finds no significant cluster{lowest}."
        else:
            name = CORRECTIONS.get(g.get("correction", ""), g.get("correction", ""))
            if g["n_significant"]:
                test_text = (
                    f" The {g['test']} ({name}-corrected) is significant at "
                    f"{g['n_significant']} of {g['n_tests']} points "
                    f"(min p = {p_value(g['min_p'])}, "
                    f"peak t = {g['peak_stat']:g} at {g['peak_channel']})."
                )
            else:
                test_text = (
                    f" The {g['test']} ({name}-corrected) finds no significant point among "
                    f"{g['n_tests']} (min p = {p_value(g['min_p'])})."
                )
        return head + test_text + notes_text(run.get("notes", []), 1)

    checks = [
        Check(
            kind="tool_called",
            tool="analysis.erp_comparison",
            args={"correction": correction, "test": test, "test_scope": scope},
        ),
        Check(kind="tool_called", tool="analysis.run"),
    ]
    if style == "named":
        checks.append(Check(kind="answer_mentions", terms=[CORRECTIONS[correction]]))
    return Episode(
        family="erp-group",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("analysis.erp_comparison", settings, id="plan"),
            Decide("plan", lambda r: _plan_message(r, f"compares {a} and {b}")),
            Call("analysis.run", _run_name, id="run"),
            Say(answer),
        ],
        checks=checks,
    )


def erp_two_subjects(rng: random.Random) -> Episode:
    """A group question with too few subjects for a group test."""
    subjects = labels(rng, 2)
    events, (a, b) = rng.choice(NAMED_EVENTS[:2])
    request = rng.choice(
        [
            f"Is the {a}-{b} difference at Pz significant across the subjects (300-500 ms)?",
            f"Group-level test of {a} versus {b} at Pz, 0.3-0.5 s, please.",
        ]
    )
    settings = {
        "condition_a": a,
        "events_a": [events["rare"]],
        "condition_b": b,
        "events_b": [events["frequent"]],
        "test_scope": "measured",
        "channels": ["Pz"],
        "tmin": 0.3,
        "tmax": 0.5,
        "measure": "mean",
        "correction": "fdr",
    }
    diff = f"{a}_minus_{b}"

    def answer(r: Results) -> str:
        run = r["run"]
        rows = "; ".join(
            f"subject {x['subject']}: {x[diff]:g} {run['unit']}" for x in run["per_subject"]
        )
        return (
            f"There is no group test: the dataset has {run['n_subjects']} subjects and a group "
            f"test needs at least 3. The {a}-minus-{b} difference at Pz (300-500 ms) is {rows}."
        )

    return Episode(
        family="erp-too-few",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("analysis.erp_comparison", settings, id="plan"),
            Decide("plan", lambda r: _plan_message(r, f"compares {a} and {b}")),
            Call("analysis.run", _run_name, id="run"),
            Say(answer),
        ],
        checks=[
            Check(kind="tool_called", tool="analysis.run"),
            Check(kind="answer_mentions", terms=["no group test"]),
        ],
    )


def condition_names(rng: random.Random) -> Episode:
    """The user's words for the conditions are not the event names: ask, then use them."""
    subjects = labels(rng, rng.randint(2, 3))
    label = rng.choice(subjects)
    coded = rng.random() < 0.5
    if coded:
        events = {"rare": "Stimulus/S  2", "frequent": "Stimulus/S  1"}
        a, b = rng.choice([("rare", "frequent"), ("target", "standard"), ("oddball", "standard")])
        reply = f"S 2 is the {a} stimulus and S 1 the {b} one."
    else:
        events = DEFAULT_EVENTS
        a, b = rng.choice([("target", "standard"), ("deviant", "standard"), ("odd", "common")])
        reply = f"Yes: {a} are the rare stimuli, {b} the frequent ones."
    who = sub(rng, label)
    request = rng.choice(
        [
            f"Measure the {a} minus {b} difference at Pz for {who}, 300-500 ms.",
            f"How do {a} trials differ from {b} trials at Pz in {who} (0.3-0.5 s)?",
        ]
    )
    settings = {
        "subjects": [label],
        "condition_a": a,
        "events_a": [events["rare"]],
        "condition_b": b,
        "events_b": [events["frequent"]],
        "test_scope": "measured",
        "channels": ["Pz"],
        "tmin": 0.3,
        "tmax": 0.5,
        "measure": "mean",
        "correction": "fdr",
    }
    diff = f"{a}_minus_{b}"
    names = listing([f"'{e}'" for e in events.values()])
    question = (
        f"The recordings have no events called {a} or {b}: they have {names}. Which events are "
        f"the {a} trials, and which the {b} trials?"
        if coded
        else f"The recordings' events are {names}. Are the {a} trials the rare ones and the "
        f"{b} trials the frequent ones?"
    )

    def answer(r: Results) -> str:
        run = r["run"]
        row = next(x for x in run["per_subject"] if x["subject"] == label)
        return (
            f"Using '{events['rare']}' as {a} and '{events['frequent']}' as {b}, as you said: in "
            f"subject {label} the mean {a}-minus-{b} difference at Pz (300-500 ms) is "
            f"{row[diff]:g} {run['unit']}."
        )

    return Episode(
        family="condition-names",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Ask(question),
            Call("analysis.erp_comparison", settings, id="plan"),
            Decide("plan", lambda r: _plan_message(r, f"measures {a} minus {b}")),
            Call("analysis.run", _run_name, id="run"),
            Say(answer),
        ],
        checks=[
            Check(
                kind="tool_called",
                tool="analysis.erp_comparison",
                args={"events_a": [events["rare"]], "events_b": [events["frequent"]]},
            ),
        ],
        user=ScriptedUser(replies=[reply], default_reply=reply),
    )


def epoch_counts(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    who = sub(rng, label)
    threshold = rng.choice([100.0, 150.0])
    request = rng.choice(
        [
            f"Epoch {who} into {a} and {b} trials and tell me how many survive rejection.",
            f"How many clean {a} and {b} epochs does {who} have with a {threshold:g} µV threshold?",
        ]
    )
    args = {
        "subject": label,
        "conditions": {a: [events["rare"]], b: [events["frequent"]]},
        "reject": "peak_to_peak",
        "peak_to_peak_uv": threshold,
    }

    def answer(r: Results) -> str:
        ep = r["ep"]["conditions"]
        parts = [f"{ep[c]['kept']} of {ep[c]['events']} {c} epochs" for c in (a, b)]
        return (
            f"Subject {label} keeps {listing(parts)} after rejecting epochs over "
            f"{threshold:g} µV peak to peak.{notes_text(r['ep'].get('notes', []), 1)}"
        )

    return Episode(
        family="epochs",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("eeg.preprocess", {"subject": label}, id="pre"),
            Call("eeg.epoch", args, id="ep"),
            Say(answer),
        ],
        checks=[
            Check(kind="order", before="eeg.preprocess", after="eeg.epoch"),
            Check(kind="tool_called", tool="eeg.epoch", args={"subject": label}),
        ],
    )


# --- spectra -------------------------------------------------------------------------------

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def band_power(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    who = sub(rng, label)
    wanted = sorted(rng.sample(BANDS, rng.choice([1, 2, 2, 3])), key=BANDS.index)
    request = rng.choice(
        [
            f"How strong is {listing(wanted)} power in {who}?",
            f"Give me the {listing(wanted)} band power of {who}'s whole recording.",
            f"What does {who}'s power spectrum look like in the {listing(wanted)} band(s)?",
        ]
    )

    def answer(r: Results) -> str:
        c = r["spec"]["conditions"]["continuous"]
        rows = {x["band"]: x for x in c["bands"]}
        parts = [
            f"{band} {rows[band]['mean_absolute_uv2']:g} µV² "
            f"({rows[band]['mean_relative'] * 100:.1f}% "
            f"of the power, strongest at {rows[band]['strongest_channel']})"
            for band in wanted
        ]
        peak = (
            f" The alpha peak is at {c['peak_alpha_hz']:g} Hz."
            if "alpha" in wanted and c.get("peak_alpha_hz")
            else ""
        )
        return f"Subject {label}, whole recording, mean over channels: {'; '.join(parts)}.{peak}"

    return Episode(
        family="band-power",
        request=request,
        dataset=dataset(subjects, seconds=60),
        script=[
            Call("eeg.preprocess", {"subject": label, "l_freq": 1.0, "h_freq": 45.0}, id="pre"),
            Call("eeg.spectral", {"subject": label, "source": "continuous"}, id="spec"),
            Say(answer),
        ],
        checks=[
            Check(kind="tool_called", tool="eeg.spectral", args={"subject": label}),
            Check(kind="answer_mentions", terms=wanted),
            Check(kind="tool_not_called", tool=["analysis.spectral_comparison", "eeg.tfr"]),
        ],
    )


BAND_EDGES = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def spectral_conditions(rng: random.Random) -> Episode:
    group = rng.random() < 0.5
    subjects = labels(rng, rng.choice([4, 5]) if group else rng.randint(1, 3))
    label = rng.choice(subjects)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    band = rng.choice(list(BAND_EDGES))
    fmin, fmax = BAND_EDGES[band]
    if group:
        request = rng.choice(
            [
                f"Does {band} power differ between {a} and {b} trials across subjects?",
                f"Group comparison of {band} band power, {a} vs {b}.",
            ]
        )
    else:
        request = rng.choice(
            [
                f"In {sub(rng, label)}, is {band} power after {a} stimuli different from after "
                f"{b} ones?",
                f"Compare {band} power between {a} and {b} epochs for {sub(rng, label)}.",
            ]
        )
    settings: dict[str, Any] = {
        "condition_a": a,
        "events_a": [events["rare"]],
        "condition_b": b,
        "events_b": [events["frequent"]],
        "correction": "fdr",
        "band": band,
        "fmin": fmin,
        "fmax": fmax,
    }
    if not group:
        settings["subjects"] = [label]

    def answer(r: Results) -> str:
        run = r["run"]
        unit = run["unit"]
        if group:
            m = run["mean"]
            g = run["group_test"]
            head = (
                f"Mean {band} power ({fmin:g}-{fmax:g} Hz) over {run['n_subjects']} subjects: "
                f"{a} {m[a]:g} {unit}, {b} {m[b]:g} {unit}."
            )
            if g is None:
                return head + notes_text(run.get("notes", []))
            verdict = (
                f"significant at {g['n_significant']} of {g['n_tests']} points "
                f"(min p = {p_value(g['min_p'])})"
                if g["n_significant"]
                else f"not significant (min p = {p_value(g['min_p'])})"
            )
            return f"{head} The {g['test']} (FDR-corrected) is {verdict}."
        row = next(x for x in run["per_subject"] if x["subject"] == label)
        return (
            f"Subject {label}'s {band} power ({fmin:g}-{fmax:g} Hz): {a} {row[a]:g} {unit}, {b} "
            f"{row[b]:g} {unit}. With one subject there is no statistical test."
            f"{notes_text(run.get('notes', []), 1)}"
        )

    return Episode(
        family="spectral-conditions",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("analysis.spectral_comparison", settings, id="plan"),
            Decide("plan", lambda r: _plan_message(r, f"compares {band} power of {a} and {b}")),
            Call("analysis.run", _run_name, id="run"),
            Say(answer),
        ],
        checks=[
            Check(kind="tool_called", tool="analysis.spectral_comparison", args={"band": band}),
            Check(kind="answer_mentions", terms=[band]),
        ],
    )


def time_frequency(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(1, 3))
    label = rng.choice(subjects)
    events, (a, b) = rng.choice(NAMED_EVENTS)
    which = rng.choice([a, b])
    band = rng.choice(["theta", "alpha", "beta"])
    who = sub(rng, label)
    request = rng.choice(
        [
            f"How does {band} power evolve over time after {which} stimuli in {who}?",
            f"Show the time course of {band} activity following the {which} trials of {who}.",
            f"Is there a {band} desynchronisation after the {which} stimuli in {who}?",
            f"Time-frequency analysis of {who}'s {which} epochs: what happens in {band}?",
            f"Does {band} power drop after the {which} stimuli in {who}? Show it across time.",
            f"Track {band} power through the {which} epochs of {who}.",
            f"When after the {which} events does {who}'s {band} activity change most?",
            f"What happens to {band} over the course of {who}'s {which} trials?",
            f"Describe {band} power over time after the {which} stimuli for {who}.",
        ]
    )

    def answer(r: Results) -> str:
        tfr = r["tfr"]
        row = next((x for x in tfr["largest_changes"] if x["band"] == band), None)
        if row is None:
            return f"The time-frequency result for subject {label} has no {band} band."
        direction = "an increase" if row["value"] > 0 else "a decrease"
        return (
            f"After the {which} stimuli in subject {label} ({tfr['n_epochs'][which]} epochs), the "
            f"largest {band} change is {row['value']:g} {tfr['unit']} relative to baseline "
            f"({direction}) at {row['time_ms']:g} ms ({row['freq_hz']:g} Hz), strongest at "
            f"{row['strongest_channel']}.{notes_text(tfr.get('notes', []), 1)}"
        )

    return Episode(
        family="tfr",
        request=request,
        dataset=dataset(subjects, seconds=60, events=events),
        script=[
            inspect(),
            Call("eeg.preprocess", {"subject": label}, id="pre"),
            Call(
                "eeg.epoch",
                {
                    "subject": label,
                    "conditions": {a: [events["rare"]], b: [events["frequent"]]},
                    "reject": "peak_to_peak",
                    "peak_to_peak_uv": 150.0,
                },
                id="ep",
            ),
            Call("eeg.tfr", {"subject": label, "conditions": [which]}, id="tfr"),
            Say(answer),
        ],
        checks=[
            Check(kind="order", before="eeg.epoch", after="eeg.tfr"),
            Check(kind="tool_called", tool="eeg.tfr", args={"subject": label}),
            Check(kind="tool_not_called", tool=["eeg.spectral", "analysis.spectral_comparison"]),
            Check(kind="answer_mentions", terms=[band]),
        ],
    )


# --- two questions, the system, documentation ----------------------------------------------


def two_questions(rng: random.Random) -> Episode:
    subjects = labels(rng, rng.randint(2, 4))
    label = rng.choice(subjects)
    ch = rng.choice(PROBLEM_CHANNELS)
    who = sub(rng, label)
    request = rng.choice(
        [
            f"How many recordings are there, and are any of {who}'s channels flat?",
            f"Which subjects do we have? Also, is {who} noisy?",
        ]
    )
    flat = "flat" in request
    planted: dict[str, Any] = {"flat": [ch]} if flat else {"noisy": [ch]}

    def answer(r: Results) -> str:
        ds = r["ds"]
        first = (
            f"There are {ds['n_recordings']} recordings ({listing(ds['subjects'])})."
            if "recordings" in request
            else f"The subjects are {listing(ds['subjects'])}."
        )
        return f"{first} {qc_findings(r['qc'], label)}"

    return Episode(
        family="two-questions",
        request=request,
        dataset=dataset(subjects, seconds=30, **planted),
        script=[inspect(), Call("eeg.qc_report", {"subject": label}, id="qc"), Say(answer)],
        checks=[
            Check(kind="tool_called", tool="dataset.inspect"),
            Check(kind="tool_called", tool="eeg.qc_report", args={"subject": label}),
            Check(kind="answer_mentions", terms=[ch]),
            Check(kind="no_decision_requested"),
        ],
    )


def system_info(rng: random.Random) -> Episode:
    request = rng.choice(
        [
            "Which BrainO version is installed?",
            "What Python version is BrainO running on?",
            "Tell me the software version and platform.",
        ]
    )

    def answer(r: Results) -> str:
        s = r["sys"]
        return f"BrainO {s['braino_version']} on Python {s['python_version']} ({s['platform']})."

    return Episode(
        family="system",
        request=request,
        dataset=dataset(labels(rng, 1), seconds=30),
        script=[Call("system.info", {}, id="sys"), Say(answer)],
        checks=[Check(kind="tool_called", tool="system.info"), Check(kind="max_calls", count=1)],
    )


#: Documentation questions: (request, search query, source, terms the passages must hold,
#: the answer). The answer is written from the passages the search returns; the episode is
#: refused if they do not hold the terms.
DOCS: list[tuple[str, str, str | None, list[str], str]] = [
    (
        "Which file in a BIDS dataset describes the participants?",
        "participants.tsv participant information",
        "bids",
        ["participants.tsv"],
        "participants.tsv, at the root of the dataset, has one row per participant "
        "(participant_id, and optional columns such as age and sex), described by "
        "participants.json (BIDS specification, Participants file).",
    ),
    (
        "In BIDS, how are electrode positions stored for EEG?",
        "electrodes.tsv EEG electrode positions",
        "bids",
        ["electrodes.tsv"],
        "In an electrodes.tsv file with name, x, y and z for each electrode, together with a "
        "coordsystem.json that says the coordinate system and units (BIDS specification, EEG).",
    ),
    (
        "What does the channels.tsv file record for EEG data?",
        "channels.tsv columns EEG",
        "bids",
        ["channels.tsv"],
        "channels.tsv lists each channel with its name, type and units, and optionally its "
        "status (good or bad) and filters (BIDS specification, EEG channels description).",
    ),
    (
        "Why is a baseline correction applied to epochs?",
        "baseline correction epochs",
        "mne",
        ["baseline"],
        "Baseline correction subtracts the mean of a pre-stimulus interval from each epoch, so "
        "slow offsets that differ between trials do not shift the response (MNE tutorial on "
        "epochs).",
    ),
    (
        "What is the difference between an average reference and a mastoid reference?",
        "set EEG reference average reference mastoids",
        "mne",
        ["reference"],
        "An average reference subtracts the mean of all EEG channels; a mastoid reference "
        "subtracts the mean of the mastoid channels. The reference changes the waveforms but not "
        "the information, and the choice is often set by a field's conventions (MNE tutorial on "
        "setting the EEG reference).",
    ),
    (
        "How does BrainO keep the original recordings safe?",
        "original recordings derivatives never changed",
        "braino",
        ["derivatives"],
        "BrainO never changes the original recordings: every step writes new files under "
        "derivatives/braino/, and decisions are applied there (BrainO guide).",
    ),
    (
        "What does ICLabel do in BrainO's ICA cleaning?",
        "ICLabel classify ICA components",
        "braino",
        ["ICLabel"],
        "ICLabel classifies each independent component (brain, eye blink, muscle, heart, line "
        "noise and so on) so eeg.ica_clean can propose removing the artifact components for you "
        "to approve (BrainO guide).",
    ),
    (
        "What is an epoch in MNE?",
        "epochs object segments of data around events",
        "mne",
        ["epoch"],
        "An epoch is a segment of the continuous data cut around an event, from a little "
        "before it to some time after it; MNE stores them together in an Epochs object (MNE "
        "tutorial on epochs).",
    ),
]


def docs(rng: random.Random) -> Episode:
    request, query, source, terms, text = rng.choice(DOCS)

    def answer(r: Results) -> str:
        passages = " ".join(p["text"] + " " + p["title"] for p in r["docs"]["passages"])
        missing = [t for t in terms if t.lower() not in passages.lower()]
        if missing:
            raise ValueError(f"the search did not find {missing} for {query!r}")
        return text

    args: dict[str, Any] = {"query": query}
    if source:
        args["source"] = source
    return Episode(
        family="docs",
        request=request,
        dataset=dataset(labels(rng, 1), seconds=30),
        script=[Call("docs.search", args, id="docs"), Say(answer)],
        checks=[
            Check(kind="tool_called", tool="docs.search"),
            Check(kind="tool_not_called", tool=["eeg.preprocess", "eeg.epoch"]),
            Check(kind="max_calls", count=1),
        ],
        grounded=False,
    )


FAMILIES: dict[str, Family] = {
    "inspect": inspect_dataset,
    "qc": qc_subject,
    "qc-many": qc_many,
    "preprocess": preprocess_settings,
    "missing-reference": preprocess_missing_reference,
    "missing-subject": missing_subject,
    "missing-info": missing_info,
    "refuse": refuse,
    "bad-channels": bad_channels,
    "ica": ica,
    "ica-then-erp": ica_then_erp,
    "erp-one-subject": erp_one_subject,
    "erp-group": erp_group,
    "erp-too-few": erp_two_subjects,
    "condition-names": condition_names,
    "epochs": epoch_counts,
    "band-power": band_power,
    "spectral-conditions": spectral_conditions,
    "tfr": time_frequency,
    "two-questions": two_questions,
    "system": system_info,
    "docs": docs,
}

#: Episodes per family relative to the others: more for what the evaluation found the model
#: getting wrong (the base model, and after round 1 the fine-tuned one: answering a quality
#: question from the dataset summary, band power for power over time, dropping the user's
#: word for the reference, and stopping after ICA when an ERP was asked for too).
WEIGHTS: dict[str, float] = {
    "erp-one-subject": 2.0,
    "erp-group": 2.0,
    "tfr": 2.5,
    "band-power": 1.5,
    "missing-reference": 2.0,
    "missing-subject": 1.5,
    "missing-info": 1.5,
    "condition-names": 1.5,
    "qc": 2.0,
    "qc-many": 1.5,
    "ica-then-erp": 1.5,
    "system": 0.3,
    "docs": 0.8,
}


# --- the held-out evaluation ----------------------------------------------------------------


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def check_held_out(requests: list[str]) -> list[str]:
    """The requests that are an evaluation task's request (or one of its other wordings)."""
    from braino.evals.tasks import builtin_tasks

    held = {_normal(w) for t in builtin_tasks() for w in [t.request, *t.variants]}
    return [r for r in requests if _normal(r) in held]
