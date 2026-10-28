"""The local model runtime: llama.cpp and a Qwen model on the user's own computer.

BrainO detects the hardware (:func:`detect`), proposes a model tier (:func:`recommend`),
installs the pinned llama.cpp build and downloads the model once, checked against its pinned
SHA-256 (:func:`setup`), measures how fast the agent will be (:func:`run_benchmark`), and
starts and stops the server itself (:class:`LlamaServer`, :func:`start_server`). The server
speaks the OpenAI-compatible API on localhost; the agent's client (plan item 17) talks to it.
"""

from braino.runtime.bench import describe, run_benchmark
from braino.runtime.catalog import Backend, Catalog, ModelSpec, catalog
from braino.runtime.download import DownloadError
from braino.runtime.hardware import Gpu, Hardware, detect
from braino.runtime.install import (
    download_model,
    find_server,
    install_runtime,
    installed_backends,
    installed_models,
    model_path,
)
from braino.runtime.manager import SetupError, choose, run_saved_benchmark, setup, start_server
from braino.runtime.server import LlamaServer, ServerError
from braino.runtime.settings import Benchmark, RuntimeSettings, load_settings, save_settings
from braino.runtime.tiers import Option, Recommendation, Tier, options, recommend

__all__ = [
    "Backend",
    "Benchmark",
    "Catalog",
    "DownloadError",
    "Gpu",
    "Hardware",
    "LlamaServer",
    "ModelSpec",
    "Option",
    "Recommendation",
    "RuntimeSettings",
    "ServerError",
    "SetupError",
    "Tier",
    "catalog",
    "choose",
    "describe",
    "detect",
    "download_model",
    "find_server",
    "install_runtime",
    "installed_backends",
    "installed_models",
    "load_settings",
    "model_path",
    "options",
    "recommend",
    "run_benchmark",
    "run_saved_benchmark",
    "save_settings",
    "setup",
    "start_server",
]
