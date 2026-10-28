"""A stand-in for ``llama-server``, for testing BrainO's runtime without a model.

Accepts the same command line, answers ``/health`` (503 while "loading", then 200), and
``/completion`` and ``/v1/models`` with the API key, like the real server. Environment:

- ``FAKE_LLAMA_LOAD_S``: seconds of "loading" before healthy (default 0.3)
- ``FAKE_LLAMA_FAIL``: exit at once with this message (to test start failures)
- ``FAKE_LLAMA_ARGS``: write the command-line arguments to this file (JSON)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--alias", default="model")
    args, _ = parser.parse_known_args()

    if os.environ.get("FAKE_LLAMA_ARGS"):
        with open(os.environ["FAKE_LLAMA_ARGS"], "w", encoding="utf-8") as f:
            json.dump(sys.argv[1:], f)
    if os.environ.get("FAKE_LLAMA_FAIL"):
        print(f"error: {os.environ['FAKE_LLAMA_FAIL']}", flush=True)
        sys.exit(3)
    print(f"loading model {args.model}", flush=True)
    ready_at = time.monotonic() + float(os.environ.get("FAKE_LLAMA_LOAD_S", "0.3"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *a: Any) -> None:
            pass

        def send(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorised(self) -> bool:
            if args.api_key and self.headers.get("Authorization") != f"Bearer {args.api_key}":
                self.send(401, {"error": {"code": 401, "message": "Invalid API Key"}})
                return False
            return True

        def do_GET(self) -> None:
            if self.path == "/health":
                if time.monotonic() < ready_at:
                    self.send(503, {"error": {"code": 503, "message": "Loading model"}})
                else:
                    self.send(200, {"status": "ok"})
            elif self.path == "/v1/models" and self.authorised():
                self.send(200, {"object": "list", "data": [{"id": args.alias}]})
            elif self.path != "/v1/models":
                self.send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/completion":
                self.send(404, {"error": "not found"})
                return
            if not self.authorised():
                return
            prompt_n = max(1, len(payload.get("prompt", "")) // 4)
            predicted_n = int(payload.get("n_predict", 16))
            self.send(
                200,
                {
                    "content": " word" * predicted_n,
                    "timings": {
                        "prompt_n": prompt_n,
                        "prompt_ms": prompt_n / 50 * 1000,
                        "prompt_per_second": 50.0,
                        "predicted_n": predicted_n,
                        "predicted_ms": predicted_n / 10 * 1000,
                        "predicted_per_second": 10.0,
                    },
                },
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"server is listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
