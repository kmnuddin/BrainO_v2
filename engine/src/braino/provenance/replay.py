"""Re-running an exported analysis script (see :mod:`braino.provenance.export`).

An exported script calls the same tools with the same arguments, replays each recorded
decision, and checks after every step that the new result matches the recorded one. Files
themselves are not compared byte by byte (MNE stores the time of writing in every FIF file), but
every number a tool reported is.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from braino import __version__
from braino.context import RunContext
from braino.provenance.recorder import installed_versions
from braino.tools.base import PendingProposal
from braino.tools.decisions import current_user, decide
from braino.tools.registry import registry

REL_TOL = 1e-4
"""Relative tolerance for numbers: results differ in the last digits between machines."""
ABS_TOL = 1e-9
IGNORED_KEYS = frozenset({"proposal_id", "run_id"})
"""Keys that differ by construction between the recorded run and the replay."""
MAX_SHOWN = 5


@dataclass
class Difference:
    step: int
    where: str
    recorded: Any
    replayed: Any

    def __str__(self) -> str:
        return f"step {self.step}, {self.where}: recorded {self.recorded!r}, now {self.replayed!r}"


def compare(recorded: Any, replayed: Any, where: str = "result") -> list[tuple[str, Any, Any]]:
    """Differences between two JSON-like values (numbers within tolerance)."""
    if isinstance(recorded, Mapping) and isinstance(replayed, Mapping):
        out = []
        for key in sorted(set(recorded) | set(replayed)):
            if key in IGNORED_KEYS:
                continue
            if key not in recorded or key not in replayed:
                out.append((f"{where}.{key}", recorded.get(key), replayed.get(key)))
                continue
            out.extend(compare(recorded[key], replayed[key], f"{where}.{key}"))
        return out
    if isinstance(recorded, list) and isinstance(replayed, list):
        if len(recorded) != len(replayed):
            return [(f"{where} (length)", len(recorded), len(replayed))]
        out = []
        for i, (a, b) in enumerate(zip(recorded, replayed, strict=True)):
            out.extend(compare(a, b, f"{where}[{i}]"))
        return out
    if _is_number(recorded) and _is_number(replayed):
        a, b = float(recorded), float(replayed)
        if (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=REL_TOL,
                                                              abs_tol=ABS_TOL):  # fmt: skip
            return []
        return [(where, recorded, replayed)]
    return [] if recorded == replayed else [(where, recorded, replayed)]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Replay:
    """The state of a running exported script."""

    context: RunContext
    recorded: Mapping[int, Mapping[str, Any]]
    differences: list[Difference] = field(default_factory=list)
    n_steps: int = 0
    out: Any = None

    @classmethod
    def from_command_line(
        cls,
        argv: Sequence[str] | None,
        *,
        description: str,
        recorded: Mapping[int, Mapping[str, Any]],
        inputs: Mapping[str, str],
        environment: Mapping[str, Any],
    ) -> Replay:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("dataset", type=Path, help="Root of the BIDS dataset")
        parser.add_argument(
            "--no-verify", action="store_true", help="Run even if input files differ"
        )
        args = parser.parse_args(argv)
        _check_environment(environment)
        if not args.no_verify:
            changed = check_inputs(args.dataset, inputs)
            if changed:
                shown = "\n  ".join(changed[:MAX_SHOWN])
                raise SystemExit(
                    f"{len(changed)} input files differ from the recorded ones:\n  {shown}\n"
                    "Use --no-verify to run anyway."
                )
        context = RunContext(dataset_root=args.dataset)
        print(f"replaying into {context.derivatives_root} (run {context.run_id})")
        return cls(context=context, recorded=recorded)

    def run(self, tool: str, arguments: Mapping[str, Any], *, step: int) -> BaseModel:
        """Run one tool call as recorded, and compare its result."""
        self.n_steps += 1
        result = registry.get(tool).run(dict(arguments), self.context)
        recorded = self.recorded.get(step, {})
        if isinstance(result, PendingProposal):
            self._compare(step, recorded.get("proposal"), result.proposal, "proposal")
        else:
            self._compare(step, recorded.get("result"), result.model_dump(mode="json"), "result")
        print(f"{step:>4}  {tool}")
        return result

    def decide(
        self,
        pending: BaseModel,
        *,
        step: int,
        approve: bool,
        edits: Mapping[str, Any] | None = None,
        decided_by: str,
        note: str | None = None,
    ) -> None:
        """Make the recorded decision again on the new proposal: the same approval or
        rejection, with the same edits."""
        if not isinstance(pending, PendingProposal):
            raise TypeError(f"step {step} did not return a proposal")
        record = decide(
            pending.proposal_id,
            self.context,
            approve=approve,
            decided_by=f"{current_user()} (replaying a decision by {decided_by})",
            edits=dict(edits or {}),
            note=note,
        )
        recorded = self.recorded.get(step, {})
        if approve and record.decision is not None:
            self._compare(
                step, recorded.get("apply_result"), record.decision.apply_result, "applied"
            )

    def finish(self) -> int:
        """Print whether the results were reproduced; returns the exit code."""
        if not self.differences:
            print(f"reproduced: all {self.n_steps} steps gave the recorded results")
            return 0
        steps = sorted({d.step for d in self.differences})
        print(f"{len(self.differences)} differences in {len(steps)} of {self.n_steps} steps:")
        for d in self.differences[:20]:
            print(f"  {d}")
        return 1

    def _compare(self, step: int, recorded: Any, replayed: Any, label: str) -> None:
        if recorded is None:
            return
        for where, a, b in compare(recorded, replayed, label):
            self.differences.append(Difference(step, where, a, b))


def check_inputs(root: Path, inputs: Mapping[str, str]) -> list[str]:
    """Input files (relative to ``root``) that are missing or have another hash."""
    changed = []
    for relative, expected in inputs.items():
        path = root / relative
        if not path.is_file():
            changed.append(f"{relative} (missing)")
        elif sha256(path) != expected:
            changed.append(f"{relative} (different content)")
    return changed


def _check_environment(environment: Mapping[str, Any]) -> None:
    recorded = environment.get("braino_version")
    if recorded and recorded != __version__:
        print(f"note: recorded with BrainO {recorded}, running {__version__}")
    now = installed_versions()
    for name, version in sorted(dict(environment.get("packages", {})).items()):
        if now.get(name) != version:
            print(f"note: recorded with {name} {version}, running {now.get(name, 'none')}")
