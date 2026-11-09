"""Whether the numbers in an answer come from what the tools returned.

The agent may only report numbers it got from a tool: a small model copying ``-14.45`` from
a result can write ``-14.46``. The agent loop checks every answer before giving it to the
user, and the evaluation harness scores answers with the same rule.

A number is *grounded* when it is a number from a tool result, an applied decision, the
user's replies or the request, rounded to the decimals it is written with (``6.384`` may be
written ``6.38``), possibly as a percentage (``0.12`` -> ``12%``) or in thousandths
(``0.3 s`` -> ``300 ms``). Small integers (counts such as "two conditions") need no source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from braino.agent.session import Session

# Numbers in text, but not digits inside names such as "T8", "P3", "sub-01" or "Fp1". A number
# after a hyphen counts only when the hyphen follows a digit, as in a range ("0.1-40 Hz").
# The sign may be a hyphen, a minus sign or an en dash (models write negative t values with
# one; in a range with an en dash, the dash follows a digit and is not a sign); thousands may
# be separated by commas ("5,020").
MINUS = chr(0x2212)
EN_DASH = chr(0x2013)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])(?<![A-Za-z_]-)[-"
    + MINUS
    + EN_DASH
    + r"]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![0-9])"
)
SMALL_INTEGERS = 3
"""Integers up to this size (counts like "two conditions") need no grounding."""
SCALES = (1.0, 100.0, 1000.0)
"""A result may be stated as a percentage (0.12 -> 12%) or in ms (0.3 s -> 300 ms)."""


def numbers_in(text: str) -> list[tuple[float, int]]:
    """The numbers in ``text``, with their number of decimals."""
    out = []
    for m in _NUMBER.finditer(text):
        token = m.group(0).replace(MINUS, "-").replace(EN_DASH, "-").replace(",", "")
        decimals = len(token.split(".", 1)[1]) if "." in token else 0
        out.append((float(token), decimals))
    return out


def numbers_from_session(session: Session) -> Iterator[float]:
    """Every number in the tool results, decisions and user replies of the session, and in the
    arguments of the calls that ran: the settings used ("from 300 to 500 ms") are facts about
    what was done, which the answer may state."""
    for call in session.calls:
        yield from leaves(call.result)
        if call.ok:
            yield from leaves(call.arguments)
    for decision in session.decisions:
        yield from leaves(decision.applied)
        yield from leaves(decision.edits)
    for q in session.questions:
        yield from (n for n, _ in numbers_in(q.reply))


def ungrounded(text: str, sources: Iterable[float]) -> list[float]:
    """The numbers in ``text`` that are not one of ``sources`` (rounded or rescaled)."""
    allowed = list(sources)
    invented = []
    for value, decimals in numbers_in(text):
        if decimals == 0 and abs(value) <= SMALL_INTEGERS:
            continue
        if not any(same(value, decimals, v) for v in allowed):
            invented.append(value)
    return invented


def ungrounded_in_answer(text: str, session: Session, request: str = "") -> list[float]:
    """The numbers in an answer that no tool result, decision, reply or the request gave."""
    sources = list(numbers_from_session(session)) + [n for n, _ in numbers_in(request)]
    return ungrounded(text, sources)


def same(value: float, decimals: int, source: float) -> bool:
    """Whether ``value``, written with ``decimals`` decimals, is ``source`` rounded (possibly
    as a percentage or in thousandths)."""
    tolerance = 0.5 * 10.0**-decimals + 1e-9
    return any(abs(source * scale - value) <= tolerance for scale in SCALES)


def leaves(data: Any) -> Iterator[float]:
    """Every number in a JSON-like value, including numbers written in its strings."""
    if isinstance(data, dict):
        for v in data.values():
            yield from leaves(v)
    elif isinstance(data, list | tuple):
        for v in data:
            yield from leaves(v)
    elif is_number(data):
        yield float(data)
    elif isinstance(data, str):
        yield from (n for n, _ in numbers_in(data))


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
