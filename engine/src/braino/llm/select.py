"""Which tool schemas the first step gets, from the words of the request.

All the tool schemas are about 8,000 tokens: 100 s of reading for the 4B model on a CPU. A
request needs a few of them, so the conversation starts with the tools its words point to (at
most :data:`MAX_PRELOAD`), and the model loads others itself when it needs them
(``load_tools``; the system prompt lists every tool in one line each). A wrong guess costs a
step, not an answer.
"""

from __future__ import annotations

import re

from braino.tools.registry import ToolRegistry

MAX_PRELOAD = 4

#: Word stems that point to a tool. A tool missing here is still listed for the model.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "analysis.erp_comparison": (
        "compar",
        "versus",
        "vs",
        "minus",
        "differ",
        "across subjects",
        "all subjects",
        "every subject",
        "group",
        "significan",
        "erp",
        "p3",
        "p300",
        "n170",
        "n400",
        "mmn",
    ),
    "analysis.spectral_comparison": (
        "compar",
        "versus",
        "vs",
        "differ",
        "across subjects",
        "all subjects",
        "group",
        "band power",
        "alpha power",
        "theta power",
        "beta power",
    ),
    "dataset.inspect": (
        "dataset",
        "subject",
        "participant",
        "recording",
        "session",
        "task",
        "event",
        "trigger",
        "channel",
        "sampling",
        "overview",
        "inspect",
        "contain",
        "what is in",
    ),
    "docs.search": (
        "bids",
        "mne",
        "documentation",
        "docs",
        "explain",
        "why",
        "what does",
        "what is a ",
        "what is an ",
        "what are",
        "meaning",
        "definition",
        "define",
        "recommend",
        "should i",
        "best practice",
        "guideline",
        "how do i",
        "how does",
        "how should",
        "specification",
        "sidecar",
        "column",
        "tsv",
        "json",
        "metadata field",
    ),
    "eeg.qc_report": ("quality", "qc", "noise", "noisy", "flat", "line noise", "artifact"),
    "eeg.preprocess": (
        "preprocess",
        "filter",
        "high-pass",
        "highpass",
        "low-pass",
        "lowpass",
        "notch",
        "resampl",
        "downsampl",
        "re-referenc",
        "rereferenc",
        "reference",
    ),
    "eeg.detect_bad_channels": ("bad channel", "bad electrode", "bad sensor", "interpolat"),
    "eeg.ica_clean": ("ica", "independent component", "blink", "eye movement", "ocular"),
    "eeg.epoch": ("epoch", "trial", "segment", "baseline"),
    "eeg.erp": (
        "erp",
        "event-related potential",
        "evoked",
        "p3",
        "p300",
        "n1",
        "n170",
        "n2",
        "n400",
        "mmn",
        "mismatch",
        "lrp",
        "amplitude",
        "latency",
        "peak",
        "difference wave",
        "waveform",
        "ms",
        "µv",
        "microvolt",
    ),
    "eeg.spectral": (
        "spectr",
        "psd",
        "band power",
        "power",
        "alpha",
        "theta",
        "beta",
        "delta",
        "gamma",
        "frequenc",
        "oscillat",
        "resting",
    ),
    "eeg.tfr": ("time-frequency", "time frequency", "tfr", "wavelet", "morlet", "ersp"),
    "stats.mass_univariate": (
        "statistic",
        "significan",
        "t-test",
        "ttest",
        "anova",
        "fdr",
        "bonferroni",
        "multiple comparison",
        "group",
        "across subjects",
        "differ",
    ),
    "stats.cluster_permutation": ("cluster", "permutation", "statistic", "significan"),
    "system.info": ("version", "braino version", "python version", "operating system"),
}


def _pattern(stem: str) -> re.Pattern[str]:
    # Stems match at a word start: "filter" matches "filtering", "p3" does not match "mp3".
    return re.compile(r"(?<![a-z0-9])" + re.escape(stem))


_PATTERNS = {name: [_pattern(s) for s in stems] for name, stems in KEYWORDS.items()}


def select_tools(request: str, tools: ToolRegistry, *, limit: int = MAX_PRELOAD) -> list[str]:
    """Up to ``limit`` tools the request's words point to, best match first; the dataset
    summary when nothing matches (the usual first step)."""
    text = request.lower()
    scores: list[tuple[int, int, str]] = []
    order = [t.name for t in tools]
    for name in order:
        hits = sum(1 for p in _PATTERNS.get(name, ()) if p.search(text))
        if hits:
            scores.append((-hits, order.index(name), name))
    chosen = [name for _, _, name in sorted(scores)[:limit]]
    if not chosen and "dataset.inspect" in tools:
        chosen = ["dataset.inspect"]
    return sorted(chosen, key=order.index)
