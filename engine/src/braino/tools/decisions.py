"""Approving or rejecting what decision tools propose.

Only a person makes these calls, through the CLI or the app. The AI agent can run decision
tools, which only creates proposals, but it is never given :func:`decide`.
"""

from __future__ import annotations

import getpass
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from braino.context import RunContext
from braino.provenance.models import Decision, ProposalRecord
from braino.tools.registry import ToolRegistry, registry


class ProposalAlreadyDecidedError(RuntimeError):
    pass


def current_user() -> str:
    """The identity recorded for decisions made from this machine, e.g. ``user:alice``."""
    try:
        return f"user:{getpass.getuser()}"
    except (KeyError, OSError):
        return "user:unknown"


def decide(
    proposal_id: str,
    context: RunContext,
    *,
    approve: bool,
    decided_by: str,
    edits: Mapping[str, Any] | None = None,
    note: str | None = None,
    tools: ToolRegistry | None = None,
) -> ProposalRecord:
    """Approve or reject a pending proposal and record the decision.

    On approval, ``edits`` (top-level fields to change, e.g. a shorter list of channels) are
    merged into the proposal, the result is validated against the tool's output model, and the
    tool's ``apply`` function commits it. If ``apply`` fails, the proposal stays pending.
    """
    record = context.proposals.get(proposal_id)
    if record.status != "pending":
        raise ProposalAlreadyDecidedError(f"proposal {proposal_id} is already {record.status}")
    now = datetime.now(UTC)

    if not approve:
        decision = Decision(approved=False, decided_by=decided_by, decided_at=now, note=note)
        record = record.model_copy(update={"status": "rejected", "decision": decision})
        context.proposals.save(record)
        return record

    tool = (registry if tools is None else tools).get(record.tool)
    final = tool.output_model.model_validate({**record.proposal, **(edits or {})})
    result = tool.apply_approved(final, context, proposal_id=proposal_id)
    final_data = final.model_dump(mode="json")
    decision = Decision(
        approved=True,
        decided_by=decided_by,
        decided_at=now,
        note=note,
        final=final_data,
        edited=final_data != record.proposal,
        applied_in=f"{context.run_id}-{context.provenance.calls[-1].call_index}",
        apply_result=result.model_dump(mode="json"),
    )
    record = record.model_copy(update={"status": "approved", "decision": decision})
    context.proposals.save(record)
    return record
