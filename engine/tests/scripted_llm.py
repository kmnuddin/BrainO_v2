"""A scripted OpenAI-compatible server for tests of the LLM client and the agent loop: it
answers ``/v1/chat/completions`` with queued replies and records every request."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from braino.llm import ChatClient


class ScriptedServer:
    """Answers ``/v1/chat/completions`` with the queued replies, records every request."""

    def __init__(self) -> None:
        self.replies: list[Any] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.unavailable = 0  # answer 503 this many times first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                outer.headers.append(dict(self.headers))
                if outer.unavailable:
                    outer.unavailable -= 1
                    self._send(503, {"error": {"message": "Loading model"}})
                elif self.path == "/tokenize":
                    self._send(200, {"tokens": list(range(len(payload["content"].split())))})
                elif self.path == "/v1/chat/completions":
                    reply = outer.replies.pop(0)
                    if isinstance(reply, tuple):
                        self._send(*reply)
                    else:
                        self._send(200, _completion(reply, payload))
                else:
                    self._send(404, {"error": "not found"})

            def _send(self, code: int, body: Any) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def client(self, **kwargs: Any) -> ChatClient:
        return ChatClient(self.url, api_key="secret", retry_pause=0.01, **kwargs)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _completion(reply: Any, payload: dict[str, Any]) -> dict[str, Any]:
    content = reply if isinstance(reply, str) else json.dumps(reply)
    prompt = sum(len(m["content"]) for m in payload["messages"]) // 4
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": len(content) // 4},
        "timings": {"prompt_n": 10, "prompt_ms": 150.0, "predicted_ms": 900.0},
    }
