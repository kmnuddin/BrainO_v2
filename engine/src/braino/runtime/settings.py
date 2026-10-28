"""The user's choice of model and backend, and the last benchmark, kept between sessions."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from braino.paths import data_root
from braino.runtime.catalog import Backend
from braino.runtime.tiers import Placement, Tier

#: Context length the agent runs with: room for the system prompt, the tool schemas of a
#: step and a few tool results.
DEFAULT_CTX = 16384

Rating = Literal["fast", "usable", "slow"]


class Benchmark(BaseModel):
    model: str
    backend: Backend
    placement: Placement
    load_seconds: float = Field(description="Starting the server and loading the model")
    prompt_tokens: int
    prompt_per_second: float = Field(description="Reading speed: the slow part on a CPU")
    generated_tokens: int
    generated_per_second: float = Field(description="Writing speed")
    first_step_seconds: float = Field(description="Estimated first agent step of a request")
    next_step_seconds: float = Field(description="Estimated later step (prompt cached)")
    rating: Rating
    measured_at: datetime


class RuntimeSettings(BaseModel):
    model: str | None = None
    backend: Backend | None = None
    placement: Placement | None = None
    tier: Tier | None = None
    ctx_size: int = DEFAULT_CTX
    threads: int | None = Field(None, description="CPU threads; None lets llama.cpp choose")
    benchmark: Benchmark | None = None


def settings_path() -> Path:
    return data_root() / "runtime" / "settings.json"


def load_settings() -> RuntimeSettings:
    path = settings_path()
    if not path.is_file():
        return RuntimeSettings()
    return RuntimeSettings.model_validate_json(path.read_text(encoding="utf-8"))


def save_settings(settings: RuntimeSettings) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(settings.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
