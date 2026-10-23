"""Tools that describe the BIDS dataset of the run."""

from __future__ import annotations

from pydantic import BaseModel, Field

from braino.context import RunContext
from braino.io.dataset import DatasetSummary, scan_dataset
from braino.tools.base import Risk
from braino.tools.registry import tool


class InspectInput(BaseModel):
    max_recordings: int = Field(
        default=20, ge=0, le=500, description="How many recordings to list in detail"
    )


@tool(name="dataset.inspect", risk=Risk.READ)
def inspect_dataset(params: InspectInput, ctx: RunContext) -> DatasetSummary:
    """Summarise the dataset: subjects, sessions, tasks, and for each recording its sampling
    rate, duration, power-line frequency, channels per type, channels marked bad and events
    per type."""
    return scan_dataset(ctx.dataset, max_recordings=params.max_recordings)
