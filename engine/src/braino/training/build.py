"""Building a training set: sample episodes from the families, play them, keep the clean ones."""

from __future__ import annotations

import random
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from braino.training.episodes import Example, Rejected, run_episode, write_jsonl
from braino.training.families import FAMILIES, WEIGHTS, check_held_out


class BuildReport(BaseModel):
    seed: int
    attempted: int
    kept: dict[str, int] = Field(description="Examples kept per family")
    rejected: dict[str, int] = Field(description="Episodes refused per family")
    reasons: list[str] = Field(description="Why episodes were refused (first 200)")
    held_out_clashes: list[str] = Field(description="Requests that were evaluation requests")


def sample_families(n: int, rng: random.Random, only: list[str] | None = None) -> list[str]:
    names = only or list(FAMILIES)
    weights = [WEIGHTS.get(name, 1.0) for name in names]
    return rng.choices(names, weights=weights, k=n)


def build(
    n: int,
    *,
    seed: int = 0,
    families: list[str] | None = None,
    workdir: Path | None = None,
    progress: Callable[[int, Example | Rejected], None] | None = None,
) -> tuple[list[Example], BuildReport]:
    """Play ``n`` episodes (families sampled by weight, each episode seeded from ``seed``)."""
    unknown = [f for f in families or [] if f not in FAMILIES]
    if unknown:
        raise ValueError(f"no family {', '.join(unknown)}; there are {', '.join(FAMILIES)}")
    rng = random.Random(seed)
    base = workdir or Path(tempfile.mkdtemp(prefix="braino-train-"))
    examples: list[Example] = []
    kept: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    reasons: list[str] = []
    clashes: list[str] = []
    seen: set[str] = set()
    for i, family in enumerate(sample_families(n, rng, families)):
        episode = FAMILIES[family](random.Random(f"{seed}-{i}"))
        if check_held_out([episode.request]):
            clashes.append(episode.request)
            continue
        key = f"{family}|{episode.request}|{episode.dataset.model_dump_json()}"
        if key in seen:
            continue
        seen.add(key)
        outcome = run_episode(episode, base / f"{i:05d}-{family}")
        if isinstance(outcome, Example):
            examples.append(outcome)
            kept[family] += 1
        else:
            refused[family] += 1
            if len(reasons) < 200:
                reasons.append(f"{family}: {outcome.request} -> {outcome.reason}")
        if progress:
            progress(i, outcome)
    report = BuildReport(
        seed=seed,
        attempted=n,
        kept=dict(sorted(kept.items())),
        rejected=dict(sorted(refused.items())),
        reasons=reasons,
        held_out_clashes=clashes,
    )
    return examples, report


def split(
    examples: list[Example], valid_share: float, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Training and validation examples (validation for the training loss only; the
    evaluation tasks are the real test)."""
    order = list(examples)
    random.Random(seed).shuffle(order)
    n_valid = round(len(order) * valid_share)
    return order[n_valid:], order[:n_valid]


__all__ = ["BuildReport", "build", "split", "write_jsonl"]
