"""The local model runtime: hardware tiers, pinned downloads, llama.cpp install, the server
process and the benchmark.

The server tests run a stand-in ``llama-server`` (``fake_llama_server.py``), so they need no
model; ``test_real_llama_cpp`` runs the real thing when ``BRAINO_LLAMA_TESTS=1``.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from braino.cli import app
from braino.runtime import (
    Catalog,
    DownloadError,
    Gpu,
    Hardware,
    LlamaServer,
    ServerError,
    SetupError,
    catalog,
    choose,
    detect,
    manager,
    recommend,
)
from braino.runtime import install as install_mod
from braino.runtime.bench import describe, rate, run_benchmark
from braino.runtime.catalog import ModelSpec, PinnedFile
from braino.runtime.download import download
from braino.runtime.hardware import parse_nvidia_smi, vendor_of
from braino.runtime.settings import load_settings

FAKE_SERVER = Path(__file__).with_name("fake_llama_server.py")
FAKE_COMMAND = [sys.executable, str(FAKE_SERVER)]


@pytest.fixture(autouse=True)
def isolated_data(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each test gets an empty data folder, except the real llama.cpp test, which uses the
    normal one so the model is downloaded only once."""
    if request.node.get_closest_marker("llama") is None:
        monkeypatch.setenv("BRAINO_DATA", str(tmp_path / "braino-data"))
        monkeypatch.delenv("BRAINO_LLAMA_SERVER", raising=False)


# --- a local file server ------------------------------------------------------------------


class FileServer:
    """Serves ``files`` over HTTP with range requests; ``drop`` makes the first requests
    send half their body and then close the connection."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.drop = 0
        self.requests: list[dict[str, Any]] = []
        handler = self._handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                name = self.path.lstrip("/")
                outer.requests.append({"path": name, "range": self.headers.get("Range")})
                if name not in outer.files:
                    self.send_error(404)
                    return
                body = outer.files[name]
                start = 0
                if self.headers.get("Range"):
                    start = int(self.headers["Range"].split("=")[1].rstrip("-"))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                part = body[start:]
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                if outer.drop > 0:
                    outer.drop -= 1
                    self.wfile.write(part[: len(part) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                self.wfile.write(part)

        return Handler

    def add(self, name: str, data: bytes) -> PinnedFile:
        self.files[name] = data
        return PinnedFile(name=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def files() -> Iterator[FileServer]:
    server = FileServer()
    yield server
    server.httpd.shutdown()


# --- catalog and hardware -----------------------------------------------------------------


def test_catalog_pins_every_platform_and_model() -> None:
    cat = catalog()
    for os_name in ("windows", "linux", "macos"):
        for arch in ("x64", "arm64"):
            assert {"cpu", "metal"} & set(cat.backends(os_name, arch)), (os_name, arch)
    ids = [m.id for m in cat.models]
    assert len(ids) == len(set(ids))
    assert len({m.quality for m in cat.models}) == len(ids)
    for m in cat.models:
        assert m.revision in m.url and m.url.endswith(m.file)
        assert m.licence == "Apache-2.0"
    assert any(m.cpu_ok and not m.moe for m in cat.models)  # the minimum-tier model
    for b in cat.llama_cpp.builds:
        for f in b.files:
            assert cat.llama_cpp.build in cat.llama_cpp.url_of(f)


def test_parse_nvidia_smi_and_vendors() -> None:
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 2080 Ti, 11264, 551.86\nbroken line\n")
    assert len(gpus) == 1
    assert gpus[0].memory_gb == 11.0 and gpus[0].driver == "551.86"
    assert vendor_of("AMD Radeon RX 7800 XT") == "amd"
    assert vendor_of("Intel(R) Arc(TM) A770") == "intel"
    assert vendor_of("NVIDIA GeForce GTX 1080 Ti") == "nvidia"


def test_detect_runs_on_this_computer() -> None:
    hw = detect()
    assert hw.os in ("windows", "linux", "macos", "other")
    assert hw.memory_gb is None or hw.memory_gb > 0
    assert recommend(hw).model in [m.id for m in catalog().models]


def _hw(
    ram: float, gpu: tuple[str, float] | None = None, os: str = "windows", arch: str = "x64"
) -> Hardware:
    gpus = []
    if gpu is not None:
        name, memory = gpu
        unified = name.startswith("Apple")
        gpus.append(
            Gpu(name=name, vendor=vendor_of(name), memory_gb=memory, unified=unified, source="test")
        )
    return Hardware(
        platform="test",
        os=os,
        arch=arch,
        cpu="Test CPU",
        cpu_count=8,
        memory_gb=ram,
        gpus=gpus,
    )


@pytest.mark.parametrize(
    ("hw", "tier", "model", "placement", "backend"),
    [
        (_hw(15.7), "minimum", "qwen3.5-4b", "cpu", "cpu"),
        (_hw(15.7, ("Intel(R) UHD Graphics 630", 1.0)), "minimum", "qwen3.5-4b", "cpu", "cpu"),
        (_hw(8.0), "below-minimum", "qwen3.5-4b", "cpu", "cpu"),
        (_hw(31.8, os="linux"), "more-ram", "qwen3.6-35b-a3b", "cpu", "cpu"),
        (_hw(15.7, ("NVIDIA GeForce RTX 4060", 8.0)), "gpu", "qwen3.5-9b", "gpu", "cuda"),
        (_hw(31.8, ("NVIDIA GeForce GTX 1080 Ti", 11.0)), "gpu", "qwen3.6-35b-a3b", "partial",
         "cuda"),
        (_hw(15.7, ("AMD Radeon RX 6600", 6.0), os="linux"), "gpu", "qwen3.5-4b", "gpu",
         "vulkan"),
        (_hw(16.0, ("Apple M2", 16.0), os="macos", arch="arm64"), "gpu", "qwen3.5-9b", "gpu",
         "metal"),
        (_hw(64.0, ("Apple M3 Max", 64.0), os="macos", arch="arm64"), "gpu", "qwen3.6-35b-a3b",
         "gpu", "metal"),
    ],
)  # fmt: skip
def test_recommendation_by_tier(
    hw: Hardware, tier: str, model: str, placement: str, backend: str
) -> None:
    rec = recommend(hw)
    assert (rec.tier, rec.model, rec.placement, rec.backend) == (tier, model, placement, backend)
    assert rec.reasons
    assert bool(rec.warnings) == (tier == "below-minimum")


def test_choose_a_model_by_hand() -> None:
    hw = _hw(15.7)
    with pytest.raises(SetupError, match="does not fit"):
        choose(hw, "qwen3.6-35b-a3b")
    with pytest.raises(KeyError):
        choose(hw, "no-such-model")
    option, backend = choose(_hw(15.7, ("NVIDIA GeForce RTX 4060", 8.0)), backend="cpu")
    assert (option.placement, backend) == ("cpu", "cpu")
    with pytest.raises(SetupError, match=r"no llama\.cpp cuda build"):
        choose(_hw(16.0, ("Apple M2", 16.0), os="macos", arch="arm64"), backend="cuda")


# --- downloads ----------------------------------------------------------------------------


def test_download_checks_and_reuses(files: FileServer, tmp_path: Path) -> None:
    data = bytes(range(256)) * 5000
    pin = files.add("model.gguf", data)
    target = tmp_path / "out" / "model.gguf"
    seen: list[int] = []
    download(f"{files.url}/model.gguf", target, pin, progress=lambda d, t: seen.append(d))
    assert target.read_bytes() == data and seen[-1] == len(data)
    download(f"{files.url}/model.gguf", target, pin)  # already there: no request
    assert len(files.requests) == 1


def test_download_resumes_a_partial_file(files: FileServer, tmp_path: Path) -> None:
    data = b"0123456789" * 100_000
    pin = files.add("big.bin", data)
    target = tmp_path / "big.bin"
    target.with_name("big.bin.part").write_bytes(data[:300_000])
    download(f"{files.url}/big.bin", target, pin)
    assert target.read_bytes() == data
    assert files.requests[0]["range"] == "bytes=300000-"


def test_download_retries_dropped_connections(files: FileServer, tmp_path: Path) -> None:
    data = b"abcdefgh" * 200_000
    pin = files.add("flaky.bin", data)
    files.drop = 3
    target = download(f"{files.url}/flaky.bin", tmp_path / "flaky.bin", pin, pause=0.01)
    assert target.read_bytes() == data
    assert len(files.requests) >= 4
    assert any(r["range"] for r in files.requests[1:])


def test_download_rejects_a_wrong_hash(files: FileServer, tmp_path: Path) -> None:
    pin = files.add("bad.bin", b"real content")
    files.files["bad.bin"] = b"evil content"  # same size, different bytes
    target = tmp_path / "bad.bin"
    with pytest.raises(DownloadError, match="SHA-256"):
        download(f"{files.url}/bad.bin", target, pin)
    assert not target.exists() and not target.with_name("bad.bin.part").exists()


def test_download_fails_fast_on_missing_file(files: FileServer, tmp_path: Path) -> None:
    pin = PinnedFile(name="x", size=10, sha256="0" * 64)
    with pytest.raises(DownloadError, match="404"):
        download(f"{files.url}/missing.bin", tmp_path / "x", pin, pause=0.01)
    assert len(files.requests) == 1


def test_download_checks_disk_space(
    files: FileServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100, 100, 1000))
    pin = files.add("f.bin", b"x" * 2000)
    with pytest.raises(DownloadError, match="not enough disk space"):
        download(f"{files.url}/f.bin", tmp_path / "f.bin", pin)


# --- installing llama.cpp and models ------------------------------------------------------


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _tar(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _test_catalog(files: FileServer, archives: dict[str, bytes], hw: Hardware) -> Catalog:
    """The real catalog with the llama.cpp builds for ``hw`` served from ``files``."""
    pins = [files.add(name, data) for name, data in archives.items()]
    data = catalog().model_dump()
    data["llama_cpp"]["url"] = files.url + "/{name}"
    data["llama_cpp"]["builds"] = [
        {"os": hw.os, "arch": hw.arch, "backend": "cpu", "files": [p.model_dump() for p in pins]}
    ]
    return Catalog.model_validate(data)


def test_install_runtime_zip(files: FileServer) -> None:
    hw = _hw(16.0)
    name = install_mod.server_name()
    cat = _test_catalog(files, {"llama.zip": _zip({name: b"exe", "ggml.dll": b"lib"})}, hw)
    server = install_mod.install_runtime(hw, "cpu", cat=cat)
    assert server.name == name and server.read_bytes() == b"exe"
    assert install_mod.installed_backends(cat) == ["cpu"]
    assert install_mod.find_server("cpu", cat) == server
    assert not (install_mod.runtime_root() / "downloads" / "llama.zip").exists()
    install_mod.install_runtime(hw, "cpu", cat=cat)  # installed: nothing downloaded again
    assert len(files.requests) == 1
    assert install_mod.remove_runtime("cpu", cat)
    assert install_mod.installed_backends(cat) == []


def test_install_runtime_tar_with_two_archives(files: FileServer) -> None:
    hw = _hw(16.0, os="linux")
    name = install_mod.server_name()
    archives = {
        "llama.tar.gz": _tar({f"llama-b1/{name}": b"exe", "llama-b1/libllama.so": b"lib"}),
        "cudart.tar.gz": _tar({"libcudart.so": b"cuda"}),
    }
    server = install_mod.install_runtime(hw, "cpu", cat=_test_catalog(files, archives, hw))
    assert server.parent.name == "llama-b1"
    assert (server.parents[1] / "libcudart.so").is_file()


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_install_refuses_unsafe_archives(files: FileServer, kind: str) -> None:
    hw = _hw(16.0)
    entries = {"../escape.txt": b"x", install_mod.server_name(): b"exe"}
    archive = _zip(entries) if kind == "zip" else _tar(entries)
    with pytest.raises(DownloadError):
        install_mod.install_runtime(hw, "cpu", cat=_test_catalog(files, {f"a.{kind}": archive}, hw))
    assert not (install_mod.runtime_root().parent / "escape.txt").exists()


def test_server_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAINO_LLAMA_SERVER", str(tmp_path / "my-llama-server"))
    assert install_mod.find_server("cuda") == tmp_path / "my-llama-server"


def _serve_models(files: FileServer, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    """The real catalog with small stand-in model files served from ``files``."""
    data = catalog().model_dump()
    for m in data["models"]:
        pin = files.add(m["file"], b"GGUF" + m["id"].encode() * 1000)
        m["size"], m["sha256"] = pin.size, pin.sha256
    monkeypatch.setattr(ModelSpec, "url", property(lambda self: f"{files.url}/{self.file}"))
    return Catalog.model_validate(data)


def test_download_model(files: FileServer, monkeypatch: pytest.MonkeyPatch) -> None:
    cat = _serve_models(files, monkeypatch)
    model = cat.model("qwen3.5-4b")
    path = install_mod.download_model(model)
    assert path == install_mod.model_path(model) and path.read_bytes().startswith(b"GGUF")
    assert [m.id for m in install_mod.installed_models(cat)] == ["qwen3.5-4b"]
    path.write_bytes(b"X" * model.size)  # corrupted, same size
    assert not install_mod.is_model_installed(model, verify=True)
    install_mod.download_model(model, verify=True)  # replaced by a good copy
    assert install_mod.is_model_installed(model, verify=True)
    assert install_mod.remove_model(model) and not path.exists()


# --- the server process -------------------------------------------------------------------


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF")
    return path


def test_server_starts_serves_and_stops(
    model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args_file = tmp_path / "args.json"
    monkeypatch.setenv("FAKE_LLAMA_ARGS", str(args_file))
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "0.6")
    with LlamaServer(model_file, binary=FAKE_COMMAND, placement="cpu", alias="tiny") as server:
        assert server.running and server.load_seconds is not None
        assert server.load_seconds >= 0.5  # waited for the model to "load"
        assert server.get("/v1/models")["data"][0]["id"] == "tiny"
        with pytest.raises(urllib.error.HTTPError) as denied:  # no API key, no access
            urllib.request.urlopen(server.base_url + "/v1/models", timeout=5)
        assert denied.value.code == 401
    assert not server.running
    args = json.loads(args_file.read_text())
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--device" in args and "--no-webui" in args and "--api-key" in args


def test_gpu_placement_leaves_offloading_to_llama_cpp(model_file: Path) -> None:
    command = LlamaServer(model_file, binary=FAKE_COMMAND, placement="partial").command()
    assert "--device" not in command and "--n-gpu-layers" not in command


def test_server_start_failure_shows_the_log(
    model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_FAIL", "unsupported model architecture")
    server = LlamaServer(model_file, binary=FAKE_COMMAND)
    with pytest.raises(ServerError, match="unsupported model architecture"):
        server.start(timeout=30)
    assert not server.running
    with pytest.raises(ServerError, match="model file not found"):
        LlamaServer(model_file.with_name("missing.gguf"), binary=FAKE_COMMAND).start()


def test_server_start_timeout(model_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "60")
    server = LlamaServer(model_file, binary=FAKE_COMMAND)
    with pytest.raises(ServerError, match="did not become ready"):
        server.start(timeout=1.5)
    assert not server.running


def test_benchmark(model_file: Path) -> None:
    with LlamaServer(model_file, binary=FAKE_COMMAND) as server:
        result = run_benchmark(server, "qwen3.5-4b", "cpu")
    # the stand-in reads 50 tokens/s and writes 10: 2000/50 + 120/10 and 400/50 + 120/10
    assert (result.prompt_per_second, result.generated_per_second) == (50.0, 10.0)
    assert (result.first_step_seconds, result.next_step_seconds) == (52.0, 20.0)
    assert result.rating == "usable" and result.prompt_tokens > 500
    assert "52 s" in describe(result)
    assert (rate(10), rate(40), rate(200)) == ("fast", "usable", "slow")


# --- setup, end to end with the stand-in server -------------------------------------------


@pytest.fixture
def fake_setup(files: FileServer, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    """A catalog served locally, and the stand-in server in place of the installed one."""
    hw = _hw(15.7, os=detect().os, arch=detect().arch)
    cat = _serve_models(files, monkeypatch)
    runtime = _test_catalog(files, {"llama.zip": _zip({install_mod.server_name(): b"exe"})}, hw)
    # only the minimum-tier model: the stand-in files are tiny, so every model would "fit"
    cat = cat.model_copy(
        update={"llama_cpp": runtime.llama_cpp, "models": [cat.model("qwen3.5-4b")]}
    )
    import braino.runtime as runtime_pkg
    from braino.runtime import tiers

    catalog_module = sys.modules["braino.runtime.catalog"]
    for module in (runtime_pkg, catalog_module, install_mod, tiers, manager):
        monkeypatch.setattr(module, "catalog", lambda: cat)
    for module in (runtime_pkg, manager):
        monkeypatch.setattr(module, "detect", lambda: hw)
    monkeypatch.setattr(manager, "server_command", lambda backend, c=None: FAKE_COMMAND)
    return cat


def test_setup_installs_benchmarks_and_remembers(fake_setup: Catalog) -> None:
    settings = manager.setup()
    assert (settings.model, settings.backend, settings.placement, settings.tier) == (
        "qwen3.5-4b",
        "cpu",
        "cpu",
        "minimum",
    )
    assert settings.benchmark is not None and settings.benchmark.rating == "usable"
    assert load_settings() == settings
    with manager.start_server() as server:
        assert server.get("/v1/models")["data"][0]["id"] == "qwen3.5-4b"


def test_start_server_needs_setup() -> None:
    with pytest.raises(SetupError, match="braino model setup"):
        manager.start_server()


def test_cli_model_commands(fake_setup: Catalog) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "hardware", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recommendation"]["tier"] == "minimum"
    assert runner.invoke(app, ["model", "hardware"]).exit_code == 0
    assert "No model set up" in runner.invoke(app, ["model", "status"]).output
    result = runner.invoke(app, ["model", "setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Ready: qwen3.5-4b" in result.output and "first step" in result.output
    status = runner.invoke(app, ["model", "status"]).output
    assert "qwen3.5-4b on cpu" in status and "Speed:" in status
    assert "downloaded" in runner.invoke(app, ["model", "list"]).output
    result = runner.invoke(app, ["model", "bench"])
    assert result.exit_code == 0 and "tokens/s" in result.output
    assert runner.invoke(app, ["model", "remove", "nope"]).exit_code == 1
    assert "deleted" in runner.invoke(app, ["model", "remove", "qwen3.5-4b"]).output


def test_eval_reports_record_gpus() -> None:
    from braino.evals.runner import hardware

    info = hardware()
    assert "gpus" in info and "memory_gb" in info


# --- the real llama.cpp (opt-in) ----------------------------------------------------------


@pytest.mark.llama
def test_real_llama_cpp(tmp_path: Path) -> None:
    """Install the pinned llama.cpp CPU build and the minimum-tier model, start the server and
    benchmark it: the plan's minimum tier, on this computer's CPU."""
    settings = manager.setup("qwen3.5-4b", "cpu")
    assert settings.benchmark is not None
    print(describe(settings.benchmark))
    with manager.start_server(settings) as server:
        reply = server.post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
    assert reply["choices"][0]["message"]
