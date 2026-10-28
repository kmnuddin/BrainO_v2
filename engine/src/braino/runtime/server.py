"""Starting and stopping ``llama-server``, the local model server.

The server listens on ``127.0.0.1`` only, on a free port, with a fresh random API key per
start, so other programs and other users of the computer cannot use it. It serves one
conversation at a time (``--parallel 1``, the least memory) and reuses the prompt cache
between turns, since reading the prompt is the slow part on a CPU. Its log goes to
``runtime/logs/llama-server.log``; when it fails to start, the error includes the end of it.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from braino.paths import data_root
from braino.runtime.settings import DEFAULT_CTX
from braino.runtime.tiers import Placement

HOST = "127.0.0.1"
#: Tokens of a previous prompt that must match before llama.cpp shifts its cache to reuse them.
CACHE_REUSE = 256


class ServerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        port: int = s.getsockname()[1]
        return port


class LlamaServer:
    """A ``llama-server`` process serving one model on localhost.

    ``binary`` is the server executable, or a command prefix (used by the tests to run a
    stand-in server). ``placement="cpu"`` keeps everything off the GPU; otherwise llama.cpp
    fits as much of the model as the GPU holds (``--fit``, its default), which also puts a
    mixture-of-experts model's experts in RAM when the GPU is too small.

    Use it as a context manager, or call :meth:`start` and :meth:`stop`::

        with LlamaServer(model_path, binary=find_server()) as server:
            server.post("/completion", {"prompt": "Hello", "n_predict": 8})
    """

    def __init__(
        self,
        model: Path,
        *,
        binary: Path | Sequence[str],
        placement: Placement = "cpu",
        ctx_size: int = DEFAULT_CTX,
        threads: int | None = None,
        port: int | None = None,
        alias: str | None = None,
        extra_args: Sequence[str] = (),
        log_path: Path | None = None,
    ) -> None:
        self.model = Path(model)
        self.binary = [str(binary)] if isinstance(binary, Path | str) else list(binary)
        self.placement = placement
        self.ctx_size = ctx_size
        self.threads = threads
        self.port = port
        self.alias = alias or self.model.stem
        self.extra_args = list(extra_args)
        self.log_path = log_path or data_root() / "runtime" / "logs" / "llama-server.log"
        self.api_key = secrets.token_urlsafe(24)
        self.load_seconds: float | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise ServerError("the server has not been started")
        return f"http://{HOST}:{self.port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def command(self) -> list[str]:
        port = self.port if self.port is not None else 0
        args = [
            *self.binary,
            "--model",
            str(self.model),
            "--host",
            HOST,
            "--port",
            str(port),
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            "1",
            "--cache-reuse",
            str(CACHE_REUSE),
            "--alias",
            self.alias,
            "--api-key",
            self.api_key,
            "--no-webui",
        ]
        if self.placement == "cpu":
            args += ["--device", "none", "--n-gpu-layers", "0"]
        if self.threads:
            args += ["--threads", str(self.threads)]
        return args + self.extra_args

    def start(self, timeout: float = 600.0) -> Self:
        """Start the server and wait until the model is loaded (``/health`` answers 200)."""
        if self.running:
            return self
        if not self.model.is_file():
            raise ServerError(f"model file not found: {self.model}")
        if self.port is None:
            self.port = free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        started = time.monotonic()
        try:
            self._process = subprocess.Popen(
                self.command(),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            self._close_log()
            raise ServerError(f"could not start {self.binary[0]}: {error}") from error
        atexit.register(self.stop)
        while time.monotonic() - started < timeout:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._process = None
                self._close_log()
                raise ServerError(
                    f"llama-server stopped while starting (exit code {code}).\n{self.log_tail()}"
                )
            if self._healthy():
                self.load_seconds = round(time.monotonic() - started, 2)
                return self
            time.sleep(0.25)
        self.stop()
        raise ServerError(
            f"llama-server did not become ready within {timeout:.0f} s.\n{self.log_tail()}"
        )

    def stop(self, timeout: float = 15.0) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout)
        self._close_log()
        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)

    def get(self, path: str, timeout: float = 30.0) -> Any:
        return self._request("GET", path, None, timeout)

    def post(self, path: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
        return self._request("POST", path, payload, timeout)

    def log_tail(self, lines: int = 25) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no server log)"
        return "\n".join(text.splitlines()[-lines:])

    def _request(self, method: str, path: str, payload: Any, timeout: float) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            raise ServerError(f"{method} {path}: HTTP {error.code}: {body}") from error

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                return bool(response.status == 200)
        except (OSError, ValueError):
            return False  # refused while starting, or 503 while the model loads

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
