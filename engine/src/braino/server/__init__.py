"""The local engine service: what the desktop app talks to.

``braino serve`` starts it on the computer's own address (127.0.0.1) only. The app starts it,
passes a token of its choosing in ``BRAINO_SERVER_TOKEN`` (or reads the one printed), and stops
it by closing its standard input (``--exit-with-stdin``) or ending the process. When it is
ready it prints one line of JSON::

    {"braino_server": {"url": "http://127.0.0.1:53817", "pid": 4242}}

The routes are described in :mod:`braino.server.app`; long work runs as jobs
(:mod:`braino.server.jobs`). Needs the ``[server]`` extra (FastAPI and uvicorn).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
from collections.abc import Callable, Sequence

from braino.server.jobs import JobConflictError, JobInfo, JobNotFoundError, JobRunner

TOKEN_ENV = "BRAINO_SERVER_TOKEN"
LOOPBACK = ("127.0.0.1", "localhost", "::1")

__all__ = [
    "LOOPBACK",
    "TOKEN_ENV",
    "JobConflictError",
    "JobInfo",
    "JobNotFoundError",
    "JobRunner",
    "serve",
]


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    allow_origins: Sequence[str] = (),
    exit_with_stdin: bool = False,
    announce: Callable[[str], None] | None = None,
) -> None:
    """Run the service until it is stopped. ``port`` 0 lets the system choose a free one;
    the ready line says which."""
    import uvicorn

    from braino.server.app import create_app

    if host not in LOOPBACK:
        raise ValueError(
            f"the service only listens on this computer ({', '.join(LOOPBACK)}), not {host}: "
            "it can read and write your files"
        )
    token = os.environ.get(TOKEN_ENV)
    generated = not token
    token = token or secrets.token_hex(24)

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(128)  # connections wait in the backlog until the server takes them
    bound = sock.getsockname()[1]

    app = create_app(token=token, allow_origins=allow_origins)
    config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on")
    server = uvicorn.Server(config)

    if exit_with_stdin:
        _exit_when_stdin_closes(lambda: setattr(server, "should_exit", True))

    shown_host = f"[{host}]" if ":" in host else host
    ready: dict[str, object] = {"url": f"http://{shown_host}:{bound}", "pid": os.getpid()}
    if generated:
        ready["token"] = token
    (announce or _print)(json.dumps({"braino_server": ready}))
    server.run(sockets=[sock])


def _exit_when_stdin_closes(stop: Callable[[], None]) -> None:
    """Call ``stop`` when the standard input closes (the app that started us has ended).

    The pipe is read from a private copy, and standard input itself is pointed at the null
    device. On Windows, a thread reading standard input directly made the service hang the
    first time a request loaded numpy's native code (the evidence for a decision): Windows
    lets only one operation at a time use a pipe opened this way, and something in the
    loading waited for the read to finish.
    """
    private = os.dup(sys.stdin.fileno())
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115 (open for the process)

    def watch() -> None:
        with os.fdopen(private, "rb", buffering=0) as stream:
            while stream.read(65536):
                pass
        stop()

    threading.Thread(target=watch, daemon=True, name="braino-stdin").start()


def _print(line: str) -> None:
    print(line, flush=True)
