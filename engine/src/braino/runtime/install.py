"""Installing llama.cpp and the models into BrainO's data folder.

Layout under ``$BRAINO_DATA`` (default ``~/.cache/braino``)::

    runtime/llama.cpp-<build>-<backend>/   one folder per installed build (CPU, Vulkan, ...)
    runtime/settings.json                  the chosen model, backend and benchmark
    models/<file>.gguf                     downloaded models

The desktop installers will bundle llama.cpp; until then, and for development, ``install``
downloads the pinned build. ``BRAINO_LLAMA_SERVER`` points BrainO at a ``llama-server`` of
your own instead (for example one built for the HPC).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

from braino.paths import data_root
from braino.runtime.catalog import BACKENDS, Backend, Catalog, ModelSpec, RuntimeBuild, catalog
from braino.runtime.download import DownloadError, Progress, download, is_downloaded
from braino.runtime.hardware import Hardware

SERVER_ENV = "BRAINO_LLAMA_SERVER"
_MARKER = ".installed.json"


def runtime_root() -> Path:
    return data_root() / "runtime"


def models_dir() -> Path:
    return data_root() / "models"


def server_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


# --- llama.cpp ---------------------------------------------------------------------------


def install_dir(build: RuntimeBuild, cat: Catalog | None = None) -> Path:
    cat = cat or catalog()
    return runtime_root() / f"llama.cpp-{cat.llama_cpp.build}-{build.backend}"


def install_runtime(
    hw: Hardware,
    backend: Backend,
    *,
    progress: Progress | None = None,
    cat: Catalog | None = None,
) -> Path:
    """Download and unpack the pinned llama.cpp ``backend`` build for this computer; returns
    the path of its ``llama-server``. Does nothing if it is already installed."""
    cat = cat or catalog()
    build = cat.build(hw.os, hw.arch, backend)
    target = install_dir(build, cat)
    server = _installed_server(target)
    if server is not None:
        return server
    downloads = runtime_root() / "downloads"
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for file in build.files:
        archive = download(
            cat.llama_cpp.url_of(file), downloads / file.name, file, progress=progress
        )
        _extract(archive, staging)
    server = _find_file(staging, server_name())
    if server is None:
        shutil.rmtree(staging)
        raise DownloadError(f"{server_name()} is missing from the llama.cpp {backend} build")
    if sys.platform != "win32":
        server.chmod(server.stat().st_mode | 0o111)
    marker = {
        "build": cat.llama_cpp.build,
        "backend": backend,
        "server": server.relative_to(staging).as_posix(),
        "files": [f.name for f in build.files],
    }
    (staging / _MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    for file in build.files:  # the archives are no longer needed
        (downloads / file.name).unlink(missing_ok=True)
    found = _installed_server(target)
    assert found is not None
    return found


def installed_backends(cat: Catalog | None = None) -> list[Backend]:
    """Backends whose pinned build is installed."""
    cat = cat or catalog()
    found: list[Backend] = []
    for path in sorted(runtime_root().glob(f"llama.cpp-{cat.llama_cpp.build}-*")):
        info = _marker(path)
        backend = info.get("backend") if info else None
        if backend in BACKENDS and backend not in found:
            found.append(backend)
    return found


def find_server(backend: Backend | None = None, cat: Catalog | None = None) -> Path | None:
    """The ``llama-server`` to run: ``$BRAINO_LLAMA_SERVER``, else the installed build of
    ``backend`` (any installed build if ``None``), else one on the PATH."""
    if os.environ.get(SERVER_ENV):
        return Path(os.environ[SERVER_ENV])
    cat = cat or catalog()
    backends = [backend] if backend else installed_backends(cat)
    for b in backends:
        root = runtime_root() / f"llama.cpp-{cat.llama_cpp.build}-{b}"
        server = _installed_server(root)
        if server is not None:
            return server
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else None


def remove_runtime(backend: Backend, cat: Catalog | None = None) -> bool:
    cat = cat or catalog()
    root = runtime_root() / f"llama.cpp-{cat.llama_cpp.build}-{backend}"
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def _installed_server(root: Path) -> Path | None:
    info = _marker(root)
    if info is None:
        return None
    server = root / info["server"]
    return server if server.is_file() else None


def _marker(root: Path) -> dict[str, str] | None:
    try:
        data: dict[str, str] = json.loads((root / _MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data


def _find_file(root: Path, name: str) -> Path | None:
    return next((p for p in sorted(root.rglob(name)) if p.is_file()), None)


def _extract(archive: Path, target: Path) -> None:
    """Unpack a zip or tar archive into ``target``, refusing entries that would land outside
    it (absolute paths, ``..``, links pointing out)."""
    root = target.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                dest = (target / member.filename).resolve()
                if not dest.is_relative_to(root):
                    raise DownloadError(f"{archive.name}: unsafe path {member.filename!r}")
            zf.extractall(target)
    else:
        try:
            with tarfile.open(archive) as tf:
                tf.extractall(target, filter="data")
        except tarfile.TarError as error:
            raise DownloadError(f"{archive.name}: {error}") from error


# --- models ------------------------------------------------------------------------------


def model_path(model: ModelSpec) -> Path:
    return models_dir() / model.file


def is_model_installed(model: ModelSpec, *, verify: bool = False) -> bool:
    return is_downloaded(model_path(model), model.pinned(), verify=verify)


def download_model(
    model: ModelSpec, *, progress: Progress | None = None, verify: bool = False
) -> Path:
    """Download ``model`` once (checked against its pinned SHA-256); returns its path.

    An existing file of the right size is reused; ``verify=True`` re-checks its hash and
    downloads it again if it does not match."""
    path = model_path(model)
    if verify and path.is_file() and not is_model_installed(model, verify=True):
        path.unlink()
    return download(model.url, path, model.pinned(), progress=progress)


def installed_models(cat: Catalog | None = None) -> list[ModelSpec]:
    cat = cat or catalog()
    return [m for m in cat.models if is_model_installed(m)]


def remove_model(model: ModelSpec) -> bool:
    path = model_path(model)
    if not path.exists():
        return False
    path.unlink()
    return True
