"""The pinned llama.cpp build and the models BrainO can run (``catalog.json``).

Every file is pinned by size and SHA-256, and models by repository revision too, so a download
is exactly the file BrainO was tested with. Updating a pin is a code change, tested like any
other.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Backend = Literal["cpu", "vulkan", "cuda", "metal"]
BACKENDS: tuple[Backend, ...] = ("cpu", "vulkan", "cuda", "metal")


class PinnedFile(BaseModel):
    name: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeBuild(BaseModel):
    os: Literal["windows", "linux", "macos"]
    arch: Literal["x64", "arm64"]
    backend: Backend
    files: list[PinnedFile] = Field(min_length=1)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


class LlamaCpp(BaseModel):
    build: str
    released: str
    licence: str
    url: str = Field(description="Download URL with a {name} placeholder")
    builds: list[RuntimeBuild]

    def url_of(self, file: PinnedFile) -> str:
        return self.url.format(name=file.name)


class ModelSpec(BaseModel):
    id: str
    name: str
    params_b: float = Field(description="Parameters, in billions")
    active_params_b: float = Field(description="Parameters used per token (less for MoE)")
    moe: bool
    cpu_ok: bool = Field(description="Fast enough to recommend on a CPU alone")
    quality: int = Field(description="Higher is a stronger model")
    licence: str
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"

    @property
    def size_gb(self) -> float:
        return self.size / 2**30

    def pinned(self) -> PinnedFile:
        return PinnedFile(name=self.file, size=self.size, sha256=self.sha256)


class Catalog(BaseModel):
    llama_cpp: LlamaCpp
    models: list[ModelSpec]

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        known = ", ".join(m.id for m in self.models)
        raise KeyError(f"unknown model {model_id!r}; choose from {known}")

    def build(self, os: str, arch: str, backend: str) -> RuntimeBuild:
        for b in self.llama_cpp.builds:
            if (b.os, b.arch, b.backend) == (os, arch, backend):
                return b
        options = ", ".join(self.backends(os, arch)) or "none"
        raise KeyError(f"no llama.cpp {backend!r} build for {os} {arch} (available: {options})")

    def backends(self, os: str, arch: str) -> list[Backend]:
        return [b.backend for b in self.llama_cpp.builds if (b.os, b.arch) == (os, arch)]


@cache
def catalog() -> Catalog:
    path = Path(__file__).with_name("catalog.json")
    return Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
