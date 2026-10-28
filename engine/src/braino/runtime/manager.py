"""Setting up the model runtime, and starting the server the user chose.

``setup`` is what the desktop app's first-run screen (and ``braino model setup``) does:
detect the hardware, take the recommended model (or the user's choice), install llama.cpp,
download the model once, measure the speed, and remember it all. ``start_server`` then starts
that model whenever the agent needs it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from braino.runtime.bench import run_benchmark
from braino.runtime.catalog import Backend, Catalog, catalog
from braino.runtime.download import Progress
from braino.runtime.hardware import Hardware, detect
from braino.runtime.install import (
    download_model,
    find_server,
    install_runtime,
    is_model_installed,
    model_path,
)
from braino.runtime.server import LlamaServer, ServerError
from braino.runtime.settings import RuntimeSettings, load_settings, save_settings
from braino.runtime.tiers import Option, option_for, recommend, tier_of


class SetupError(RuntimeError):
    pass


def choose(
    hw: Hardware,
    model: str | None = None,
    backend: Backend | None = None,
    cat: Catalog | None = None,
) -> tuple[Option, Backend]:
    """The option to run: the recommendation, or ``model`` (which must fit), on ``backend``
    (default: the one that suits where it runs)."""
    cat = cat or catalog()
    if model is None:
        rec = recommend(hw, cat)
        option = next((o for o in rec.options if o.model == rec.model), None)
        if option is None:  # below the minimum: the recommendation does not really fit
            option = Option(
                model=rec.model,
                placement=rec.placement,
                backend=rec.backend,
                recommendable=False,
                memory_gb=0.0,
            )
    else:
        cat.model(model)  # raises for an unknown model
        fitting = option_for(hw, model, cat)
        if fitting is None:
            gpu = hw.best_gpu()
            raise SetupError(
                f"{model} does not fit on this computer (RAM {hw.memory_gb} GB, "
                f"GPU {gpu.name if gpu else 'none'}); see 'braino model hardware' for the "
                f"models that do"
            )
        option = fitting
    chosen = backend or option.backend
    if chosen not in cat.backends(hw.os, hw.arch):
        raise SetupError(f"no llama.cpp {chosen} build for {hw.os} {hw.arch}")
    if chosen == "cpu" and option.placement != "cpu":
        option = option.model_copy(update={"placement": "cpu", "backend": "cpu"})
    return option, chosen


def setup(
    model: str | None = None,
    backend: Backend | None = None,
    *,
    benchmark: bool = True,
    progress: Callable[[str], Progress | None] | None = None,
    hw: Hardware | None = None,
    cat: Catalog | None = None,
) -> RuntimeSettings:
    """Install the runtime and model for this computer, benchmark it, and save the choice.

    ``progress(label)`` may return a callback for the download of ``label``.
    """
    cat = cat or catalog()
    hw = hw or detect()
    option, chosen_backend = choose(hw, model, backend, cat)
    spec = cat.model(option.model)

    def report(label: str) -> Progress | None:
        return progress(label) if progress else None

    install_runtime(hw, chosen_backend, progress=report(f"llama.cpp ({chosen_backend})"), cat=cat)
    download_model(spec, progress=report(spec.name))
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "model": spec.id,
            "backend": chosen_backend,
            "placement": option.placement,
            "tier": tier_of(hw, option, cat),
            "benchmark": None,
        }
    )
    save_settings(settings)
    if benchmark:
        settings = run_saved_benchmark(settings, cat=cat)
    return settings


def run_saved_benchmark(
    settings: RuntimeSettings | None = None, cat: Catalog | None = None
) -> RuntimeSettings:
    """Benchmark the chosen model and store the result with the settings."""
    settings = settings or load_settings()
    with start_server(settings, cat=cat) as server:
        result = run_benchmark(server, settings.model or "", settings.backend or "cpu")
    settings = settings.model_copy(update={"benchmark": result})
    save_settings(settings)
    return settings


def server_command(backend: Backend, cat: Catalog | None = None) -> Path | list[str] | None:
    """The ``llama-server`` executable for ``backend`` (or a command prefix that runs one)."""
    return find_server(backend, cat)


def start_server(
    settings: RuntimeSettings | None = None,
    *,
    cat: Catalog | None = None,
    port: int | None = None,
    start: bool = True,
) -> LlamaServer:
    """A server for the chosen model (started unless ``start=False``)."""
    cat = cat or catalog()
    settings = settings or load_settings()
    if settings.model is None or settings.backend is None:
        raise SetupError("no model chosen yet: run 'braino model setup'")
    spec = cat.model(settings.model)
    if not is_model_installed(spec):
        raise SetupError(f"{spec.id} is not downloaded: run 'braino model setup'")
    binary = server_command(settings.backend, cat)
    if binary is None:
        raise SetupError(
            f"llama.cpp ({settings.backend}) is not installed: run 'braino model setup'"
        )
    server = LlamaServer(
        model_path(spec),
        binary=binary,
        placement=settings.placement or "cpu",
        ctx_size=settings.ctx_size,
        threads=settings.threads,
        port=port,
        alias=spec.id,
    )
    if start:
        try:
            server.start()
        except ServerError as error:
            hint = ""
            if settings.backend in ("cuda", "vulkan"):
                hint = (
                    "\nIf the GPU build does not work on this computer, "
                    "'braino model setup --backend cpu' uses the CPU instead."
                )
            raise ServerError(f"{error}{hint}") from error
    return server
