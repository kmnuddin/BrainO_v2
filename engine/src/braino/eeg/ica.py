"""Independent component analysis (ICA) for removing eye, heart and other artifacts.

ICA is extended infomax (Lee et al., 1999), solved with Picard (Ablin et al., 2018), which
reaches the same solution as MNE's infomax many times faster. It is fitted on a copy of the
data that is average-referenced and high-passed at 1 Hz, which gives more stable components
(Winkler et al., 2015) and is what ICLabel was trained on. The components are then classified:

- ``iclabel``: ICLabel (Pion-Tonachini et al., 2019, via mne-icalabel) gives each component a
  probability of being brain, muscle, eye, heart, line noise, channel noise or other activity.
- ``eog``: components that correlate with the EOG channels are eye components (no extra
  dependencies, but needs EOG channels).

Components whose label is one of ``exclude_labels`` with at least ``min_probability`` are
proposed for removal. Removing them is a separate, approved step (:func:`remove_components`).

Needs the ``[eeg]`` extra; ``iclabel`` also needs mne-icalabel with onnxruntime or PyTorch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from braino.eeg._quiet import quiet
from braino.eeg.preprocess import apply_reference

Labeler = Literal["iclabel", "eog"]
IcaMethod = Literal["infomax", "fastica", "picard"]

ICLABEL_LABELS = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)
DEFAULT_EXCLUDE = ("eye blink", "heart beat")
FIT_HIGH_PASS_HZ = 1.0
FIT_LOW_PASS_HZ = 100.0


class Component(BaseModel):
    index: int
    label: str
    probability: float = Field(description="Classifier's probability for the label (0-1)")
    explained_variance: float = Field(description="Share of the fitted data's variance")


class IcaResult(BaseModel):
    n_components: int
    method: IcaMethod
    labeler: Labeler
    components: list[Component]
    proposed: list[int] = Field(description="Components proposed for removal")
    notes: list[str] = Field(default_factory=list)


def fit_ica(
    raw: Any,
    *,
    n_components: int | float | None = None,
    method: IcaMethod = "picard",
    labeler: Labeler = "iclabel",
    exclude_labels: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
    min_probability: float = 0.8,
    seed: int = 42,
) -> tuple[Any, IcaResult]:
    """Fit ICA to the good EEG channels of ``raw`` (unchanged) and classify the components.

    ``n_components`` is a number of components, a share of variance to keep (e.g. 0.99), or
    ``None`` for as many as the data's rank allows.
    """
    import mne

    notes: list[str] = []
    fit_data = _fit_copy(raw, notes)
    rank = int(mne.compute_rank(fit_data, rank=None, verbose="error").get("eeg", 0))
    if rank < 2:
        raise ValueError("ICA needs at least two good EEG channels")
    if n_components is None:
        n_components = rank
    elif isinstance(n_components, int) and n_components > rank:
        notes.append(f"Asked for {n_components} components; the data's rank allows {rank}.")
        n_components = rank

    fit_params: dict[str, bool] | None = None
    if method == "picard":
        fit_params = {"ortho": False, "extended": True}  # extended infomax, solved faster
    elif method == "infomax":
        fit_params = {"extended": True}
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        rng=seed,
        max_iter="auto",
    )
    with quiet():
        ica.fit(fit_data, picks="eeg", reject_by_annotation=True, verbose="error")

    share = [
        float(ica.get_explained_variance_ratio(fit_data, components=[i], ch_type="eeg")["eeg"])
        for i in range(ica.n_components_)
    ]

    if labeler == "iclabel":
        labels, probabilities = _iclabel(fit_data, ica)
    else:
        labels, probabilities = _eog_labels(fit_data, ica, notes)

    components = [
        Component(
            index=i,
            label=labels[i],
            probability=round(float(probabilities[i]), 3),
            explained_variance=round(float(share[i]), 4),
        )
        for i in range(ica.n_components_)
    ]
    wanted = set(exclude_labels)
    proposed = [
        c.index for c in components if c.label in wanted and c.probability >= min_probability
    ]
    return ica, IcaResult(
        n_components=int(ica.n_components_),
        method=method,
        labeler=labeler,
        components=components,
        proposed=proposed,
        notes=notes,
    )


def remove_components(raw: Any, ica: Any, exclude: list[int], reference: str | list[str]) -> Any:
    """A copy of ``raw`` with the ``exclude`` components removed, in ``reference``.

    The components were fitted on average-referenced data, so they are removed from
    average-referenced data, and the result is then re-referenced to ``reference``. With
    ``reference="none"`` or ``"average_with_eog"`` they are removed from the data as they are:
    the removed activity sums to zero over the EEG channels, so it leaves an average reference
    as it was (and EOG channels must keep their relation to the EEG for that reference).
    """
    out = raw.copy().load_data(verbose="error")
    bad = [i for i in exclude if not 0 <= i < ica.n_components_]
    if bad:
        raise ValueError(f"no components {bad}; the ICA has components 0-{ica.n_components_ - 1}")
    if reference in ("none", "average_with_eog"):
        ica.apply(out, exclude=list(exclude), verbose="error")
        return out
    if reference != "none":
        apply_reference(out, "average")
    ica.apply(out, exclude=list(exclude), verbose="error")
    apply_reference(out, reference)
    return out


def _fit_copy(raw: Any, notes: list[str]) -> Any:
    """Average-referenced, 1-100 Hz copy of the EEG and EOG channels, for fitting."""
    keep = [
        ch
        for ch, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind in ("eeg", "eog")
    ]
    data = raw.copy().pick(keep).load_data(verbose="error")
    nyquist = float(data.info["sfreq"]) / 2
    h_freq = FIT_LOW_PASS_HZ if nyquist - 1 > FIT_LOW_PASS_HZ else None
    current_hp = float(data.info["highpass"] or 0.0)
    l_freq = FIT_HIGH_PASS_HZ if current_hp < FIT_HIGH_PASS_HZ else None
    if l_freq is not None or h_freq is not None:
        data.filter(l_freq, h_freq, verbose="error")
    if h_freq is None:
        notes.append("Sampling rate too low to low-pass at 100 Hz for fitting.")
    apply_reference(data, "average")
    return data


def _iclabel(data: Any, ica: Any) -> tuple[list[str], Any]:
    from mne_icalabel import label_components

    with quiet():
        out = label_components(data, ica, method="iclabel")
    return list(out["labels"]), out["y_pred_proba"]


def _eog_labels(data: Any, ica: Any, notes: list[str]) -> tuple[list[str], Any]:
    import numpy as np

    n = ica.n_components_
    labels = ["other"] * n
    probabilities = np.zeros(n)
    if "eog" not in data.get_channel_types():
        notes.append("No EOG channels: eye components could not be identified.")
        return labels, probabilities
    indices, scores = ica.find_bads_eog(data, verbose="error")
    scores = np.atleast_2d(np.abs(np.asarray(scores)))
    strongest = scores.max(axis=0)
    for i in indices:
        labels[i] = "eye blink"
    for i in range(n):
        probabilities[i] = strongest[i] if labels[i] == "eye blink" else 1 - strongest[i]
    notes.append("EOG labels: 'probability' is the absolute correlation with the EOG.")
    return labels, probabilities
