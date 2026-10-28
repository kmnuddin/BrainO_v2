"""ERP CORE: public EEG data and published results that BrainO's pipeline is validated against.

ERP CORE (Kappenman et al., 2021, NeuroImage 225:117465; CC-BY-4.0) has 40 young adults and
seven ERP components. BrainO uses the BIDS-format raw data of the P3 and N170 tasks, and the
authors' published per-participant scores as the results to reproduce.

Files are downloaded from OSF on demand, checked against the SHA-256 hashes pinned in
``erpcore_manifest.json``, and cached under ``$BRAINO_DATA/erpcore/<task>/`` (default
``~/.cache/braino``). Each task folder is a BIDS dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.paths import DATA_ENV, data_root

__all__ = ["DATA_ENV", "DownloadError", "Reference", "citation", "fetch", "files", "references"]

TASKS = ("P3", "N170")
SUBJECTS = tuple(f"{i:03d}" for i in range(1, 41))
_CHUNK = 1 << 20


class Reference(BaseModel):
    """A published ERP CORE result: what to measure, and what the authors found."""

    task: str
    contrast: str
    conditions: list[str]
    electrode: str
    window_ms: tuple[float, float]
    reference: str = Field(description="EEG reference the authors used")
    measure: str
    n: int
    mean_uv: float
    sd_uv: float
    excluded_subjects: list[str]
    subject_scores_uv: dict[str, float]
    url: str = Field(description="The authors' score file on OSF")


class DownloadError(RuntimeError):
    pass


def data_dir() -> Path:
    return data_root() / "erpcore"


@cache
def manifest() -> dict[str, Any]:
    path = Path(__file__).with_name("erpcore_manifest.json")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@cache
def references() -> dict[str, Reference]:
    """The published results, by component (``"P3"``, ``"N170"``)."""
    path = Path(__file__).with_name("erpcore_references.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: Reference.model_validate(r) for name, r in data["components"].items()}


def citation() -> str:
    return str(manifest()["citation"])


def files(
    task: str, subjects: Iterable[str] | None = None, *, stimuli: bool = False
) -> list[dict[str, Any]]:
    """Manifest entries for ``task``: the dataset-level files, plus those of ``subjects``
    (labels like ``"001"``; all 40 if ``None``). Stimulus images only if ``stimuli``."""
    if task not in TASKS:
        raise ValueError(f"unknown ERP CORE task {task!r}; choose from {', '.join(TASKS)}")
    wanted = set(SUBJECTS if subjects is None else (_label(s) for s in subjects))
    selected = []
    for entry in manifest()["tasks"][task]["files"]:
        top = entry["path"].split("/", 1)[0]
        if top.startswith("sub-"):
            if top[4:] in wanted:
                selected.append(entry)
        elif "/" not in entry["path"] or stimuli:
            selected.append(entry)
    return selected


def fetch(
    task: str,
    subjects: Iterable[str] | None = None,
    *,
    stimuli: bool = False,
    verify: bool = False,
) -> Path:
    """Download (or reuse) the BIDS data of ``task`` for ``subjects``; returns the BIDS root.

    Files already in the cache are reused if their size matches; ``verify=True`` re-checks
    their hashes too. Raises :class:`DownloadError` if a download does not match its hash.
    """
    root = data_dir() / task
    for entry in files(task, subjects, stimuli=stimuli):
        target = root / entry["path"]
        cached = target.is_file() and target.stat().st_size == entry["size"]
        if cached and (not verify or _sha256(target) == entry["sha256"]):
            continue
        _download(entry, target)
    return root


def _label(subject: str | int) -> str:
    label = str(subject).removeprefix("sub-")
    label = f"{int(label):03d}" if label.isdigit() else label
    if label not in SUBJECTS:
        raise ValueError(f"ERP CORE has subjects 001-040, not {subject!r}")
    return label


def _download(entry: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(entry["url"], timeout=120) as response, partial.open("wb") as f:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != entry["sha256"]:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{entry['path']}: downloaded file does not match its SHA-256")
    os.replace(partial, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
