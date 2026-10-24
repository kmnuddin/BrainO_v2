"""Compare BrainO's ERP CORE results with the published ones.

BrainO's pipeline counts as reproducing a component when:

1. **Group level:** the 95% confidence interval of BrainO's mean difference includes the published
   mean, and the effect is significant in the published direction (one-sample t-test, p < .001).
2. **Participant level:** BrainO's per-participant scores correlate with the published ones at
   r >= :data:`MIN_CORRELATION` over the participants both include.

Which participants each side excluded is reported too, but is not a pass/fail check: the
authors also cleaned some artifacts by hand, so exclusions may differ for good reasons.

Needs SciPy (the ``[eeg]`` extra).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field

from braino.datasets.erpcore import references

MIN_CORRELATION = 0.9
"""Provisional: to be confirmed after the first full run of the pipeline."""
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
