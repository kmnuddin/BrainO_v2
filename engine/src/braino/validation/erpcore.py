"""Compare BrainO's ERP CORE results with the published ones.

BrainO's pipeline counts as reproducing a component when:

1. **Group level:** the 95% confidence interval of BrainO's mean difference includes the published
   mean, and the effect is significant in the published direction (one-sample t-test, p < .001).
2. **Participant level:** BrainO's per-participant scores correlate with the published ones at
   r >= :data:`MIN_CORRELATION` over the participants both include.

Which participants each side excluded is reported too, but is not a pass/fail check: the
authors also cleaned some artifacts by hand, so exclusions may differ for good reasons.

:func:`run` runs BrainO's built-in ERP CORE pipeline (``braino pipeline show erpcore-p3``) on
the downloaded data and makes the comparison; ``braino validate erpcore P3`` does the same from
the command line.

Needs SciPy (the ``[eeg]`` extra).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from braino.datasets.erpcore import references

MIN_CORRELATION = 0.9
"""Confirmed on the full ERP CORE runs: r = 0.95 (P3, 30 participants) and 0.97 (N170, 37)."""
MAX_P = 0.001


class ExclusionComparison(BaseModel):
    both: list[str]
    only_braino: list[str]
    only_published: list[str]


class ValidationReport(BaseModel):
    component: str
    n_braino: int
    n_compared: int = Field(description="Participants with both a BrainO and a published score")
    braino_mean_uv: float
    braino_sd_uv: float
    braino_ci95_uv: tuple[float, float]
    braino_dz: float
    t: float
    p: float
    published_mean_uv: float
    published_sd_uv: float
    correlation: float | None = Field(description="Pearson r over the participants compared")
    mean_abs_difference_uv: float | None
    exclusions: ExclusionComparison
    checks: dict[str, bool]
    passed: bool


def compare(
    component: str, scores_uv: Mapping[str, float], *, excluded: Iterable[str] = ()
) -> ValidationReport:
    """Compare BrainO's per-participant scores (subject label -> µV, for the participants BrainO
    kept) with the published ``component`` result. ``excluded`` lists the participants BrainO
    excluded."""
    from scipy import stats

    published = references()[component]
    values = list(scores_uv.values())
    if len(values) < 3:
        raise ValueError("need scores for at least 3 participants")
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half_width = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    ci = (mean - half_width, mean + half_width)
    t, p = stats.ttest_1samp(values, 0.0)

    common = sorted(set(scores_uv) & set(published.subject_scores_uv))
    correlation = mean_abs = None
    if len(common) >= 3:
        ours = [scores_uv[s] for s in common]
        theirs = [published.subject_scores_uv[s] for s in common]
        correlation = float(stats.pearsonr(ours, theirs).statistic)
        mean_abs = sum(abs(a - b) for a, b in zip(ours, theirs, strict=True)) / len(common)

    same_direction = math.copysign(1, mean) == math.copysign(1, published.mean_uv)
    checks = {
        "ci_includes_published_mean": ci[0] <= published.mean_uv <= ci[1],
        "significant_in_published_direction": same_direction and float(p) < MAX_P,
        "participant_correlation": correlation is not None and correlation >= MIN_CORRELATION,
    }
    ours_out, theirs_out = set(excluded), set(published.excluded_subjects)
    return ValidationReport(
        component=component,
        n_braino=n,
        n_compared=len(common),
        braino_mean_uv=round(mean, 3),
        braino_sd_uv=round(sd, 3),
        braino_ci95_uv=(round(ci[0], 3), round(ci[1], 3)),
        braino_dz=round(mean / sd, 3) if sd else float("inf"),
        t=round(float(t), 3),
        p=float(p),
        published_mean_uv=published.mean_uv,
        published_sd_uv=published.sd_uv,
        correlation=None if correlation is None else round(correlation, 3),
        mean_abs_difference_uv=None if mean_abs is None else round(mean_abs, 3),
        exclusions=ExclusionComparison(
            both=sorted(ours_out & theirs_out),
            only_braino=sorted(ours_out - theirs_out),
            only_published=sorted(theirs_out - ours_out),
        ),
        checks=checks,
        passed=all(checks.values()),
    )


MAX_REJECTED = 0.25
"""ERP CORE excluded participants with more than 25% of trials rejected; so does BrainO here."""


class ValidationRun(BaseModel):
    """A run of the built-in ERP CORE pipeline, and its comparison with the published result."""

    component: str
    run_id: str
    dataset: str = Field(description="Where the ERP CORE data and BrainO's derivatives are")
    subjects: list[str]
    scores_uv: dict[str, float] = Field(description="BrainO's score per participant")
    rejected_fraction: dict[str, float]
    excluded: list[str] = Field(description=f"Participants with > {MAX_REJECTED:.0%} rejected")
    failed: dict[str, str] = Field(description="Participants whose pipeline did not finish")
    report: ValidationReport | None = Field(description="None if under 3 participants scored")


def run(
    component: str,
    subjects: Iterable[str] | None = None,
    *,
    progress: Callable[[Any], None] | None = None,
) -> ValidationRun:
    """Download the ERP CORE data of ``component`` (all participants by default), run the
    built-in ``erpcore-<component>`` pipeline on it, and compare the scores with the published
    ones.

    Decisions (ICA components, bad channels) are approved as proposed, in the name of the
    person running this, and recorded as auto-approved. Results are cached: running again only
    redoes what changed.
    """
    from braino.datasets.erpcore import fetch
    from braino.pipelines import load_spec, load_state, run_pipeline
    from braino.tools import RunContext, current_user

    published = references()[component]
    wanted = None if subjects is None else [str(s) for s in subjects]
    root = fetch(published.task, wanted)
    spec = load_spec(f"erpcore-{component.lower()}")
    if wanted is not None:
        spec = spec.model_copy(update={"subjects": wanted})
    ctx = RunContext(dataset_root=root)
    report = run_pipeline(
        spec,
        ctx,
        auto_approve=f"{current_user()} (auto-approved: ERP CORE validation)",
        progress=progress,
    )
    state = load_state(ctx, spec)
    failed = {o.subject: f"{o.step}: {o.message}" for o in report.outcomes if o.status == "failed"}
    scores: dict[str, float] = {}
    rejected: dict[str, float] = {}
    for subject in report.subjects:
        erp = state.records.get(f"{subject}/erp")
        epochs = state.records.get(f"{subject}/epochs")
        if subject in failed or erp is None or epochs is None:
            continue
        values = [
            m["value_uv"]
            for m in (erp.result or {}).get("measures", [])
            if m["measure"] == component and m["waveform"] == "difference"
        ]
        if values:
            scores[subject] = float(values[0])
            rejected[subject] = float((epochs.result or {}).get("rejected_fraction", 0.0))
    excluded = sorted(s for s, r in rejected.items() if r > MAX_REJECTED)
    kept = {s: v for s, v in scores.items() if s not in excluded}
    return ValidationRun(
        component=component,
        run_id=ctx.run_id,
        dataset=str(root),
        subjects=report.subjects,
        scores_uv=scores,
        rejected_fraction=rejected,
        excluded=excluded,
        failed=failed,
        report=compare(component, kept, excluded=excluded) if len(kept) >= 3 else None,
    )
