"""Storage of decision-tool proposals.

Proposals live at the dataset level, ``<dataset>/derivatives/braino/proposals/<id>.json``, so
a proposal made in one run (e.g. a CLI call) can be decided in a later one. Runs without a
dataset keep proposals in memory.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from braino.provenance.models import ProposalRecord

PROPOSALS_DIR = "proposals"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ProposalNotFoundError(KeyError):
    pass


class ProposalStore:
    """Proposals of one dataset.

    ``directory`` is where proposal files are read from; ``prepare`` creates it (and the
    derivatives dataset around it) before the first write.
    """

    def __init__(
        self, directory: Path | None = None, prepare: Callable[[], Path] | None = None
    ) -> None:
        self._directory = directory
        self._prepare = prepare
        self._memory: dict[str, ProposalRecord] = {}

    def save(self, record: ProposalRecord) -> None:
        path = self._path(record.proposal_id)
        if path is None:
            self._memory[record.proposal_id] = record
            return
        if self._prepare is not None:
            self._prepare()
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def get(self, proposal_id: str) -> ProposalRecord:
        path = self._path(proposal_id)
        if path is None:
            if proposal_id not in self._memory:
                raise ProposalNotFoundError(proposal_id)
            return self._memory[proposal_id]
        if not path.is_file():
            raise ProposalNotFoundError(proposal_id)
        return ProposalRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ProposalRecord]:
        """All proposals, oldest first."""
        if self._directory is None:
            records = list(self._memory.values())
        elif self._directory.is_dir():
            records = [
                ProposalRecord.model_validate_json(p.read_text(encoding="utf-8"))
                for p in self._directory.glob("*.json")
            ]
        else:
            records = []
        return sorted(records, key=lambda r: (r.created_at, r.proposal_id))

    def _path(self, proposal_id: str) -> Path | None:
        if not _ID_PATTERN.fullmatch(proposal_id):
            raise ProposalNotFoundError(f"invalid proposal ID {proposal_id!r}")
        return None if self._directory is None else self._directory / f"{proposal_id}.json"
