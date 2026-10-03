"""The provenance records written for every run and every tool call."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FileRecord(BaseModel):
    """A file a tool call read or wrote, identified by its content hash."""

    path: str = Field(
        description="POSIX path relative to the dataset root, or absolute if outside it"
    )
    sha256: str
    size: int = Field(description="Size in bytes")


class RunRecord(BaseModel):
    """The environment a run executed in, written once per run."""

    run_id: str
    started_at: datetime
    dataset_root: str | None
    braino_version: str
    python_version: str
    platform: str
    packages: dict[str, str] = Field(description="Versions of installed analysis libraries")


class ToolCallRecord(BaseModel):
    """One executed tool call."""

    run_id: str
    call_index: int = Field(description="1-based position of the call within its run")
    tool: str
    risk: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    status: Literal["ok", "error"]
    error: str | None = None
    started_at: datetime
    finished_at: datetime
    inputs: list[FileRecord] = Field(default_factory=list)
    outputs: list[FileRecord] = Field(default_factory=list)


class RunLog(BaseModel):
    """A run and all of its tool calls, as read back from disk."""

    run: RunRecord
    calls: list[ToolCallRecord]
