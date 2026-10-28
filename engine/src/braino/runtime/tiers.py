"""Which model to run on this computer, and where: the hardware tiers of the plan.

- ``minimum``: no GPU, 16 GB RAM: Qwen3.5 4B on the CPU (the design target).
- ``more-ram``: no GPU, 32 GB+ RAM: Qwen3.6 35B-A3B (mixture of experts) on the CPU.
- ``gpu``: a GPU with its own memory, or Apple silicon: the strongest model that fits on the
  GPU, or a mixture of experts split between GPU (shared layers) and RAM (experts).
- ``below-minimum``: less than 16 GB RAM and no GPU: the 4B model, with a warning.

The recommendation leaves ``RESERVE_GB`` of RAM for the operating system, the app and the
analyses themselves, and ``CONTEXT_GB`` for the model's working memory (its context). The user
can choose any other model that fits; ``options`` lists them all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from braino.runtime.catalog import Backend, Catalog, ModelSpec, catalog
from braino.runtime.hardware import Gpu, Hardware

Tier = Literal["below-minimum", "minimum", "more-ram", "gpu"]
Placement = Literal["cpu", "gpu", "partial"]

RESERVE_GB = 8.0
CONTEXT_GB = 1.5
MIN_RAM_GB = 15.0  # "16 GB" computers report a little less
#: Apple silicon lets the GPU use about two thirds of the shared memory.
APPLE_GPU_SHARE = 0.66
#: Below this, a GPU does not hold enough of a mixture-of-experts model to help.
MIN_PARTIAL_GPU_GB = 6.0


class Option(BaseModel):
    model: str
    placement: Placement = Field(description="cpu, all on the GPU, or split between the two")
    backend: Backend
    recommendable: bool = Field(
        description="Fast enough to propose (dense models on the CPU alone are not, past 4B)"
    )
    memory_gb: float = Field(description="RAM or GPU memory the model needs, with its context")


class Recommendation(BaseModel):
    tier: Tier
    model: str
    placement: Placement
    backend: Backend
    gpu: str | None = Field(description="The GPU used, if any")
    reasons: list[str]
    warnings: list[str] = []
    options: list[Option] = Field(description="Every model and placement that fits")


def options(hw: Hardware, cat: Catalog | None = None) -> list[Option]:
    """Every (model, placement) that fits this computer, strongest model first."""
    cat = cat or catalog()
    gpu = hw.best_gpu()
    ram = _ram_budget(hw)
    found = []
    for model in sorted(cat.models, key=lambda m: -m.quality):
        need = model.size_gb + CONTEXT_GB
        if gpu is not None:
            backend = gpu_backend(hw, gpu, cat)
            if gpu.unified:
                if need <= min(ram, gpu.memory_gb * APPLE_GPU_SHARE):
                    found.append(_option(model, "gpu", backend, True, need))
                    continue
            elif need <= gpu.memory_gb:
                found.append(_option(model, "gpu", backend, True, need))
                continue
            elif gpu.memory_gb >= MIN_PARTIAL_GPU_GB and need <= ram + gpu.memory_gb - 1.0:
                found.append(_option(model, "partial", backend, model.moe or model.cpu_ok, need))
                continue
        if need <= ram:
            backend = "metal" if hw.os == "macos" and hw.arch == "arm64" else "cpu"
            found.append(_option(model, "cpu", backend, model.cpu_ok, need))
    return found


def recommend(hw: Hardware, cat: Catalog | None = None) -> Recommendation:
    """Propose a tier and model for ``hw``."""
    cat = cat or catalog()
    fitting = options(hw, cat)
    gpu = hw.best_gpu()
    warnings = []
    chosen = next((o for o in fitting if o.recommendable), None)
    if chosen is None:
        smallest = min(cat.models, key=lambda m: m.size)
        backend: Backend = "metal" if hw.os == "macos" and hw.arch == "arm64" else "cpu"
        chosen = _option(smallest, "cpu", backend, False, smallest.size_gb + CONTEXT_GB)
        warnings.append(
            f"{_gb(hw.memory_gb)} of RAM is below the minimum BrainO is designed for (16 GB): "
            f"the model may not fit next to your analyses. Close other programs, or use the "
            f"engine without the agent."
        )
    model = cat.model(chosen.model)
    tier = tier_of(hw, chosen, cat)
    if tier == "below-minimum" and not warnings:
        warnings.append(f"{_gb(hw.memory_gb)} of RAM is below the 16 GB minimum.")
    return Recommendation(
        tier=tier,
        model=chosen.model,
        placement=chosen.placement,
        backend=chosen.backend,
        gpu=gpu.name if gpu is not None and chosen.placement != "cpu" else None,
        reasons=_reasons(hw, gpu, model, chosen),
        warnings=warnings,
        options=fitting,
    )


def tier_of(hw: Hardware, option: Option, cat: Catalog | None = None) -> Tier:
    """The tier that running ``option`` on ``hw`` represents (what eval reports record)."""
    cat = cat or catalog()
    if option.placement in ("gpu", "partial"):
        return "gpu"
    if hw.memory_gb is not None and hw.memory_gb < MIN_RAM_GB:
        return "below-minimum"
    smallest = min(m.quality for m in cat.models)
    return "more-ram" if cat.model(option.model).quality > smallest else "minimum"


def option_for(hw: Hardware, model_id: str, cat: Catalog | None = None) -> Option | None:
    """Where ``model_id`` would run on ``hw``, or ``None`` if it does not fit."""
    return next((o for o in options(hw, cat) if o.model == model_id), None)


def gpu_backend(hw: Hardware, gpu: Gpu, cat: Catalog | None = None) -> Backend:
    """The llama.cpp build for ``gpu``: Metal on Apple silicon, CUDA on NVIDIA (fastest at
    reading prompts, the slow part on a laptop), Vulkan otherwise (AMD, Intel, and any GPU
    without a CUDA build)."""
    cat = cat or catalog()
    available = cat.backends(hw.os, hw.arch)
    if gpu.unified and "metal" in available:
        return "metal"
    if gpu.vendor == "nvidia" and "cuda" in available:
        return "cuda"
    if "vulkan" in available:
        return "vulkan"
    return "cpu"


def _option(
    model: ModelSpec, placement: Placement, backend: Backend, ok: bool, need: float
) -> Option:
    return Option(
        model=model.id,
        placement=placement,
        backend=backend,
        recommendable=ok,
        memory_gb=round(need, 1),
    )


def _ram_budget(hw: Hardware) -> float:
    return (hw.memory_gb if hw.memory_gb is not None else 16.0) - RESERVE_GB


def _reasons(hw: Hardware, gpu: Gpu | None, model: ModelSpec, chosen: Option) -> list[str]:
    ram = _gb(hw.memory_gb)
    if chosen.placement == "cpu":
        where = f"on the CPU ({hw.cpu or 'unknown CPU'}, {ram} RAM)"
        if gpu is None:
            where += "; no GPU with its own memory was found"
    elif chosen.placement == "gpu" and gpu is not None:
        where = f"on the GPU ({gpu.name}, {gpu.memory_gb:g} GB)"
    else:
        assert gpu is not None
        where = (
            f"split between the GPU ({gpu.name}, {gpu.memory_gb:g} GB) and RAM ({ram}): "
            f"the shared layers on the GPU, the experts in RAM"
        )
    return [
        f"{model.name}, {model.size_gb:.1f} GB download, runs {where}.",
        f"Needs about {chosen.memory_gb:g} GB with its context, leaving room for your analyses.",
    ]


def _gb(value: float | None) -> str:
    return "unknown" if value is None else f"{value:g} GB"
