"""A client for the OpenAI-compatible chat API that llama.cpp, Ollama and vLLM serve.

BrainO's own server is llama.cpp (:mod:`braino.runtime`); :meth:`ChatClient.for_server`
connects to it. Any other compatible server works too (``ChatClient("http://host/v1",
llama_cpp=False)``), as long as it supports ``response_format`` with a JSON schema, which is
how the agent's replies are constrained.

With ``llama_cpp=True`` the client also asks the server to reuse its prompt cache and turns the
model's step-by-step reasoning on or off (Qwen3.5 reasons by default, which costs about 20 s
of writing per step on a CPU), and reads llama.cpp's timings, which say how much of the prompt
was read and how much came from the cache.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from braino.runtime.server import LlamaServer

Message = dict[str, Any]


class LLMError(RuntimeError):
    """The server could not be reached, refused the request or answered nonsense."""


class Usage(BaseModel):
    prompt_tokens: int = Field(description="Tokens in the whole prompt")
    completion_tokens: int
    prompt_read: int | None = Field(
        None, description="Prompt tokens read for this request; the rest came from the cache"
    )
    prompt_seconds: float | None = None
    completion_seconds: float | None = None
    seconds: float = Field(description="Wall time of the request")

    @property
    def cached_tokens(self) -> int | None:
        return None if self.prompt_read is None else self.prompt_tokens - self.prompt_read


class ChatReply(BaseModel):
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage


class ChatClient:
    """Talks to one model on one server.

    ``base_url`` ends in ``/v1`` (e.g. ``http://127.0.0.1:9931/v1``). Requests that fail to
    connect, or meet a server that is still loading (HTTP 503), are retried ``retries`` times.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str = "local",
        llama_cpp: bool = True,
        thinking: bool = False,
        timeout: float = 600.0,
        retries: int = 2,
        retry_pause: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.llama_cpp = llama_cpp
        self.thinking = thinking
        self.timeout = timeout
        self.retries = retries
        self.retry_pause = retry_pause

    @classmethod
    def for_server(cls, server: LlamaServer, **kwargs: Any) -> ChatClient:
        """A client for a started :class:`~braino.runtime.LlamaServer`."""
        return cls(
            server.base_url + "/v1",
            api_key=server.api_key,
            model=server.alias,
            llama_cpp=True,
            **kwargs,
        )

    def chat(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        thinking: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatReply:
        """One chat completion. With ``json_schema``, the reply is constrained to match it
        (on llama.cpp by a grammar, so it cannot do otherwise)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": True, "schema": json_schema},
            }
        if self.llama_cpp:
            think = self.thinking if thinking is None else thinking
            payload["cache_prompt"] = True
            payload["chat_template_kwargs"] = {"enable_thinking": think}
        payload.update(extra or {})

        started = time.monotonic()
        body = self._post("/chat/completions", payload)
        seconds = round(time.monotonic() - started, 3)
        try:
            choice = body["choices"][0]
            message = choice["message"]
            usage = body.get("usage") or {}
            timings = body.get("timings") or {}
            return ChatReply(
                content=message.get("content") or "",
                reasoning=message.get("reasoning_content") or None,
                finish_reason=choice.get("finish_reason"),
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    prompt_read=_int(timings.get("prompt_n")),
                    prompt_seconds=_seconds(timings.get("prompt_ms")),
                    completion_seconds=_seconds(timings.get("predicted_ms")),
                    seconds=seconds,
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMError(f"unexpected reply from the server: {str(body)[:300]}") from error

    def count_tokens(self, text: str) -> int | None:
        """Tokens in ``text`` by the model's tokenizer (llama.cpp only; else ``None``)."""
        if not self.llama_cpp:
            return None
        root = self.base_url.removesuffix("/v1")
        body = self._post("/tokenize", {"content": text}, url=root + "/tokenize")
        return len(body["tokens"])

    def _post(self, path: str, payload: dict[str, Any], url: str | None = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode()
        target = url or self.base_url + path
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(target, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read() or b"null")
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code != 503 or attempt == self.retries:
                    raise LLMError(f"HTTP {error.code} from {target}: {detail}") from error
            except TimeoutError as error:
                raise LLMError(
                    f"the model server did not answer within {self.timeout:g} s"
                ) from error
            except (urllib.error.URLError, ConnectionError) as error:
                if attempt == self.retries:
                    raise LLMError(
                        f"could not reach the model server at {target}: {error}"
                    ) from error
            except json.JSONDecodeError as error:
                raise LLMError(f"the server's answer is not JSON: {error}") from error
            time.sleep(self.retry_pause * (attempt + 1))
        raise AssertionError("unreachable")


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


def _seconds(ms: Any) -> float | None:
    return None if ms is None else round(float(ms) / 1000, 3)
