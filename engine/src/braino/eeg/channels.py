"""Channel names as people and models write them.

The spelling of a channel's name varies (``Fz``, ``FZ``, ``fz``) but no recording has two
channels whose names differ only in case, so a name is found whatever its case, and the
recording's own spelling is used from then on. The plan check matches names the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def match_channels(
    requested: Iterable[str], available: Sequence[str]
) -> tuple[list[str], list[str]]:
    """``requested`` in the recording's spelling, each once (a model listed Fz twice, and MNE
    refuses to pick a channel twice), and the names not found at all."""
    by_lower = {name.lower(): name for name in available}
    found: list[str] = []
    missing: list[str] = []
    for name in requested:
        if name in available:
            match = name
        elif name.lower() in by_lower:
            match = by_lower[name.lower()]
        else:
            missing.append(name)
            continue
        if match not in found:
            found.append(match)
    return found, missing
