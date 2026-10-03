"""The context a tool runs in: which dataset, where outputs go, which run it belongs to.

A *run* is one unit of work that may call several tools: a CLI invocation, a pipeline
execution or an agent session. Every tool call in a run shares one :class:`RunContext`.

The context is supplied by the caller (CLI, server or agent runtime), never by the language
model: it is not part of any tool's input schema, so the model cannot choose where data is read
from or written to.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from braino import __version__

DERIVATIVES_NAME = "braino"
"""Folder name of BrainO's outputs under ``<dataset>/derivatives/``."""

BIDS_VERSION = "1.10.0"
"""BIDS specification version that BrainO's derivatives follow."""


class NoDatasetError(RuntimeError):
    """Raised when a tool needs a dataset but the run has none."""


def new_run_id() -> str:
    """A sortable, unique run ID such as ``20261003T142501Z-3fa9c2``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class RunContext:
    """Where a run reads its data from and writes its outputs to."""

    dataset_root: Path | None = None
    """Root of the BIDS dataset, or ``None`` for runs that do not touch a dataset."""

    run_id: str = field(default_factory=new_run_id)

    def __post_init__(self) -> None:
        if self.dataset_root is None:
            return
        root = Path(self.dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"dataset root {root} is not a directory")
        object.__setattr__(self, "dataset_root", root)

    @property
    def dataset(self) -> Path:
        """The dataset root; raises :class:`NoDatasetError` if the run has none."""
        if self.dataset_root is None:
            raise NoDatasetError("this tool needs a dataset; pass one with --dataset")
        return self.dataset_root

    @property
    def derivatives_root(self) -> Path:
        """``<dataset>/derivatives/braino``, a BIDS-Derivatives dataset holding all outputs."""
        return self.dataset / "derivatives" / DERIVATIVES_NAME

    def output_dir(self, *parts: str) -> Path:
        """Create and return a folder under the derivatives root, e.g.
        ``ctx.output_dir("sub-01", "eeg")``.

        Also writes the derivatives ``dataset_description.json`` the first time it is needed.
        """
        root = self.derivatives_root
        target = root.joinpath(*parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"output path {target} is outside {root}")
        target.mkdir(parents=True, exist_ok=True)
        _ensure_dataset_description(root)
        return target


def _ensure_dataset_description(root: Path) -> None:
    path = root / "dataset_description.json"
    if path.exists():
        return
    description = {
        "Name": "BrainO derivatives",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "BrainO", "Version": __version__}],
    }
    path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
