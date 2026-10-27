"""Scoring what an agent did against a task's checks."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from braino.agent.session import AgentAnswer, Session, ToolCallLog
from braino.evals.tasks import Check, EvalTask

# Numbers in text, but not digits inside names such as "T8", "P3", "sub-01" or "Fp1".
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-])-?\d+(?:\.\d+)?(?![0-9])")
SMALL_INTEGERS = 3
"""Integers up to this size (counts like "two conditions") need no grounding."""
SCALES = (1.0, 100.0, 1000.0)
"""A result may be stated as a percentage (0.12 -> 12%) or in ms (0.3 s -> 300 ms)."""


class CheckResult(BaseModel):
    kind: str
    description: str
    required: bool
    passed: bool
    detail: str = ""


def evaluate(task: EvalTask, session: Session, answer: AgentAnswer | None) -> list[CheckResult]:
    return [_evaluate(check, task, session, answer) for check in task.all_checks()]


def _evaluate(
    check: Check, task: EvalTask, session: Session, answer: AgentAnswer | None
) -> CheckResult:
    passed, detail = _CHECKS[check.kind](check, task, session, answer)
    return CheckResult(
        kind=check.kind,
        description=check.description or _default_description(check),
        required=check.required,
        passed=passed,
        detail=detail,
    )


def _default_description(check: Check) -> str:
    tools = " or ".join(check.tools)
    return {
        "tool_called": f"calls {tools}" + (f" with {check.args}" if check.args else ""),
        "tool_not_called": f"does not call {tools}",
        "order": f"runs {check.before} before {check.after}",
        "decision_requested": f"asks the user to decide on {tools}",
        "no_decision_requested": "asks for no decision",
        "answer_number": f"answer has a number in [{check.min}, {check.max}]",
        "answer_mentions": f"answer mentions {check.terms}",
        "answer_not_mentions": f"answer does not mention {check.terms}",
        "max_calls": f"at most {check.count} tool calls",
        "grounded": "every number in the answer comes from a tool result",
    }[check.kind]


# --- the checks ---------------------------------------------------------------------------

Outcome = tuple[bool, str]


def _tool_called(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    candidates = [c for c in s.calls if c.ok and c.tool in check.tools]
    if not candidates:
        failed = [c for c in s.calls if c.tool in check.tools]
        return False, f"{len(failed)} failed call(s)" if failed else "never called"
    for call in candidates:
        mismatches = [k for k, m in check.args.items() if not match(call.arguments.get(k), m)]
        if not mismatches:
            return True, f"call {call.index}"
    last = candidates[-1]
    shown = {k: last.arguments.get(k) for k in check.args}
    return False, f"called, but not with {check.args}; last call had {shown}"


def _tool_not_called(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    calls = [c.index for c in s.calls if c.tool in check.tools]
    return (not calls), (f"called at {calls}" if calls else "")


def _order(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    first = _first_ok(s.calls, check.before or "")
    second = _first_ok(s.calls, check.after or "")
    if first is None or second is None:
        missing = check.before if first is None else check.after
        return False, f"{missing} was not called"
    return first < second, f"{check.before} at {first}, {check.after} at {second}"


def _decision_requested(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    asked = [d for d in s.decisions if d.tool in check.tools]
    return bool(asked), f"{len(asked)} decision(s)" if asked else "never asked the user"


def _no_decision(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    return (not s.decisions), f"{len(s.decisions)} decision(s)"


def _answer_number(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    if a is None:
        return False, "no answer"
    lo = -math.inf if check.min is None else check.min
    hi = math.inf if check.max is None else check.max
    numbers = [n for n, _ in numbers_in(a.text)]
    hits = [n for n in numbers if lo <= n <= hi]
    return bool(hits), f"numbers in the answer: {numbers}"


def _mentions(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    if a is None:
        return False, "no answer"
    found = [t for t in check.terms if _has_term(a.text, t)]
    ok = bool(found) if check.any_term else len(found) == len(check.terms)
    missing = [t for t in check.terms if t not in found]
    return ok, f"missing: {missing}" if missing else ""


def _not_mentions(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    if a is None:
        return True, "no answer"
    found = [t for t in check.terms if _has_term(a.text, t)]
    return (not found), f"mentions {found}" if found else ""


def _max_calls(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    return len(s.calls) <= (check.count or 0), f"{len(s.calls)} calls"


def _grounded(check: Check, task: EvalTask, s: Session, a: AgentAnswer | None) -> Outcome:
    if a is None:
        return False, "no answer"
    allowed = list(numbers_from_session(s)) + [n for n, _ in numbers_in(task.request)]
    invented = []
    for value, decimals in numbers_in(a.text):
        if decimals == 0 and abs(value) <= SMALL_INTEGERS:
            continue
        if not any(_same(value, decimals, v) for v in allowed):
            invented.append(value)
    if invented:
        return False, f"not from any tool result: {invented}"
    return True, ""


_CHECKS = {
    "tool_called": _tool_called,
    "tool_not_called": _tool_not_called,
    "order": _order,
    "decision_requested": _decision_requested,
    "no_decision_requested": _no_decision,
    "answer_number": _answer_number,
    "answer_mentions": _mentions,
    "answer_not_mentions": _not_mentions,
    "max_calls": _max_calls,
    "grounded": _grounded,
}


# --- helpers ------------------------------------------------------------------------------


def match(value: Any, matcher: Any) -> bool:
    """Whether an argument value satisfies a matcher.

    - a plain value: equal (numbers within 1e-6; lists of strings in any order);
    - ``{not: x}``, ``{any: [..]}``, ``{range: [lo, hi]}``;
    - ``{contains: [..]}``: a list containing all of these;
    - ``{covers: [..]}``: every one of these strings appears somewhere in the value (e.g.
      event names among the conditions of an epoching call, whatever the conditions are named).
    """
    if isinstance(matcher, dict) and len(matcher) == 1:
        ((op, operand),) = matcher.items()
        if op == "not":
            return not match(value, operand)
        if op == "any":
            return any(match(value, m) for m in operand)
        if op == "range":
            return _is_number(value) and operand[0] <= value <= operand[1]
        if op == "contains":
            return isinstance(value, list) and all(any(match(v, m) for v in value)
                                                   for m in operand)  # fmt: skip
        if op == "covers":
            strings = set(_strings(value))
            return all(item in strings for item in operand)
    if _is_number(matcher) and _is_number(value):
        return math.isclose(float(value), float(matcher), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(matcher, list) and isinstance(value, list):
        if all(isinstance(x, str) for x in matcher + value):
            return sorted(matcher) == sorted(value)
        return len(matcher) == len(value) and all(
            match(v, m) for v, m in zip(value, matcher, strict=True)
        )
    return bool(value == matcher)


def numbers_in(text: str) -> list[tuple[float, int]]:
    """The numbers in ``text``, with their number of decimals."""
    out = []
    for m in _NUMBER.finditer(text):
        token = m.group(0)
        decimals = len(token.split(".", 1)[1]) if "." in token else 0
        out.append((float(token), decimals))
    return out


def numbers_from_session(session: Session) -> Iterator[float]:
    """Every number in the tool results, decisions and user replies of the session."""
    for call in session.calls:
        yield from _leaves(call.result)
    for decision in session.decisions:
        yield from _leaves(decision.applied)
        yield from _leaves(decision.edits)
    for q in session.questions:
        yield from (n for n, _ in numbers_in(q.reply))


def _same(value: float, decimals: int, source: float) -> bool:
    """Whether ``value``, written with ``decimals`` decimals, is ``source`` rounded (possibly
    as a percentage or in thousandths)."""
    tolerance = 0.5 * 10.0**-decimals + 1e-9
    return any(abs(source * scale - value) <= tolerance for scale in SCALES)


def _leaves(data: Any) -> Iterator[float]:
    if isinstance(data, dict):
        for v in data.values():
            yield from _leaves(v)
    elif isinstance(data, list | tuple):
        for v in data:
            yield from _leaves(v)
    elif _is_number(data):
        yield float(data)
    elif isinstance(data, str):
        yield from (n for n, _ in numbers_in(data))


def _strings(data: Any) -> Iterator[str]:
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for k, v in data.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(data, list | tuple):
        for v in data:
            yield from _strings(v)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I) is not None


def _first_ok(calls: list[ToolCallLog], tool: str) -> int | None:
    return next((c.index for c in calls if c.ok and c.tool == tool), None)


# --- templates (the oracle's answers) -----------------------------------------------------

_FIELD = re.compile(r"\{([^{}:]+)(?::([^{}]*))?\}")
_SEGMENT = re.compile(r"^(?P<name>[^\[\]]+)?(?:\[(?P<filter>[^\]]+)\])?$")


def render(template: str, results: dict[str, Any]) -> str:
    """Fill ``{step.path[key=value].field:.2f}`` fields from step results."""

    def field(m: re.Match[str]) -> str:
        value = resolve(results, m.group(1))
        return format(value, m.group(2) or "")

    return _FIELD.sub(field, template)


def resolve(results: dict[str, Any], path: str) -> Any:
    value: Any = results
    for segment in path.strip().split("."):
        parsed = _SEGMENT.match(segment)
        if parsed is None:
            raise KeyError(f"bad path segment {segment!r} in {path!r}")
        if parsed["name"]:
            value = (
                value[parsed["name"]] if isinstance(value, dict) else getattr(value, parsed["name"])
            )
        if parsed["filter"]:
            spec = parsed["filter"]
            if spec.lstrip("-").isdigit():
                value = value[int(spec)]
            else:
                key, _, wanted = spec.partition("=")
                value = next(item for item in value if str(item.get(key)) == wanted)
    return value
