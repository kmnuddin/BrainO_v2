"""A session whose user is the person at the command line.

Decision tools' proposals are shown with what the agent says about them, and the person
approves, edits or rejects them; the agent's questions are answered at the prompt. Edits are
checked against the proposal's model before anything is applied, so a typo asks again
instead of failing the decision.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer
from pydantic import ValidationError

from braino.agent.session import Session
from braino.context import RunContext
from braino.tools.decisions import current_user
from braino.tools.registry import ToolRegistry

Prompt = Callable[..., Any]
Echo = Callable[[str], None]

_APPROVE = {"a", "approve", "y", "yes"}
_REJECT = {"r", "reject", "n", "no"}
_EDIT = {"e", "edit"}


class ConsoleSession(Session):
    """Asks the person at the terminal (or whatever ``prompt`` and ``echo`` stand for)."""

    def __init__(
        self,
        context: RunContext,
        *,
        tools: ToolRegistry | None = None,
        prompt: Prompt = typer.prompt,
        echo: Echo = typer.echo,
    ) -> None:
        super().__init__(context, tools=tools)
        self.prompt = prompt
        self.echo = echo

    @property
    def user_name(self) -> str:
        return current_user()

    def _decide(
        self, tool: str, proposal: dict[str, Any], message: str
    ) -> tuple[bool, dict[str, Any], str | None]:
        self.echo("")
        self.echo(f"Decision needed: {tool} proposes")
        for key, value in proposal.items():
            shown = _show(value)
            self.echo(f"    {key}:{'' if shown.startswith(chr(10)) else ' '}{shown}")
        if message:
            self.echo("The agent says:")
            for line in message.strip().splitlines():
                self.echo(f"    {line}")
        while True:
            choice = str(self.prompt("Approve, edit or reject? [a/e/r]")).strip().lower()
            if choice in _APPROVE:
                return True, {}, self._note()
            if choice in _REJECT:
                return False, {}, self._note()
            if choice in _EDIT:
                edits = self._edits(tool, proposal)
                if edits is not None:
                    return True, edits, self._note()
            else:
                self.echo("Type a to approve, e to edit, or r to reject.")

    def _edits(self, tool: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Ask until the changes are valid; None (an empty line) goes back to the choice."""
        example = next(iter(proposal), "field")
        while True:
            text = str(
                self.prompt(
                    f'Fields to change, as JSON (e.g. {{"{example}": ...}}; empty to go back)',
                    default="",
                    show_default=False,
                )
            ).strip()
            if not text:
                return None
            edits = self._parse_edits(text, tool, proposal)
            if edits is not None:
                return edits

    def _parse_edits(self, text: str, tool: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
        try:
            edits = json.loads(text)
        except json.JSONDecodeError as exc:
            self.echo(f"That is not valid JSON ({exc.msg}).")
            return None
        if not isinstance(edits, dict) or not edits:
            self.echo("Give a JSON object of the fields to change.")
            return None
        unknown = [k for k in edits if k not in proposal]
        if unknown:
            self.echo(f"Unknown fields: {', '.join(unknown)}; choose from {', '.join(proposal)}.")
            return None
        try:
            self.tools.get(tool).output_model.model_validate({**proposal, **edits})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            self.echo(f"Those changes are not valid: {problems}")
            return None
        return edits

    def _note(self) -> str | None:
        note = str(self.prompt("Note for the record (optional)", default="", show_default=False))
        return note.strip() or None

    def _reply(self, question: str) -> str:
        self.echo("")
        self.echo(f"The agent asks: {question}")
        return str(self.prompt("Your answer")).strip()


def _show(value: Any) -> str:
    """A proposal field on one line."""
    if isinstance(value, list) and all(isinstance(v, str | int | float) for v in value):
        if any(len(str(v)) > LONG_ITEM for v in value):  # e.g. a plan's steps: one per line
            return "".join(f"\n        {_clip(str(v))}" for v in value)
        return ", ".join(map(str, value)) if value else "(none)"
    if isinstance(value, dict | list):
        return _clip(json.dumps(value, ensure_ascii=False))
    return str(value)


LONG_ITEM = 40
"""List items longer than this are shown one per line."""


def _clip(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
