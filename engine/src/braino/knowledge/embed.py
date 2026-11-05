"""The embedding model, in its own ``llama-server`` on localhost.

It runs on the CPU: a question is a few dozen tokens, embedded in a fraction of a second, so
the GPU (if any) stays free for the chat model. It uses about 1 GB of RAM while it runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from braino.paths import data_root
from braino.runtime.catalog import Catalog, EmbeddingSpec, catalog
from braino.runtime.install import find_server, is_model_installed, model_path
from braino.runtime.server import LlamaServer, ServerError

#: Tokens the server reads in one batch; a passage must fit (they are about 250).
BATCH_TOKENS = 2048
#: Passages sent in one request while building the index.
PASSAGES_PER_REQUEST = 16


class ServerEmbedder:
    """Embeds questions and passages with the catalog's embedding model.

    Use it as a context manager, or :meth:`start` and :meth:`stop` it::

        with ServerEmbedder.create() as embedder:
            vector = embedder.embed_query("Which columns must events.tsv have?")
    """

    def __init__(self, spec: EmbeddingSpec, server: LlamaServer) -> None:
        self.spec = spec
        self.server = server
        self.model_id = spec.id

    @classmethod
    def create(cls, *, threads: int | None = None, cat: Catalog | None = None) -> ServerEmbedder:
        """An embedder for the downloaded model (not started yet); raises
        :class:`~braino.runtime.server.ServerError` when the model or llama.cpp is missing."""
        spec = (cat or catalog()).embedding
        if not is_model_installed(spec):
            raise ServerError(f"{spec.id} is not downloaded: run 'braino model setup'")
        binary = find_server("cpu") or find_server()
        if binary is None:
            raise ServerError("llama.cpp is not installed: run 'braino model setup'")
        server = LlamaServer(
            model_path(spec),
            binary=binary,
            placement="cpu",
            ctx_size=BATCH_TOKENS * 2,
            threads=threads,
            alias=spec.id,
            extra_args=[
                "--embeddings",
                "--pooling",
                spec.pooling,
                "--batch-size",
                str(BATCH_TOKENS),
                "--ubatch-size",
                str(BATCH_TOKENS),
            ],
            log_path=data_root() / "runtime" / "logs" / "llama-embedding.log",
        )
        return cls(spec, server)

    def start(self) -> Self:
        self.server.start()
        return self

    def stop(self) -> None:
        self.server.stop()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.spec.query_prefix + text])[0]

    def embed_passages(self, texts: Sequence[str]) -> list[Sequence[float]]:
        out: list[Sequence[float]] = []
        for i in range(0, len(texts), PASSAGES_PER_REQUEST):
            out += self._embed(list(texts[i : i + PASSAGES_PER_REQUEST]))
        return out

    def _embed(self, texts: list[str]) -> list[list[float]]:
        reply = self.server.post("/v1/embeddings", {"input": texts, "model": self.spec.id})
        rows = sorted(reply["data"], key=lambda d: d["index"])
        if len(rows) != len(texts):
            raise ServerError(f"asked for {len(texts)} embeddings, got {len(rows)}")
        return [_normalised(r["embedding"]) for r in rows]

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def _normalised(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class LazyEmbedder:
    """An embedder that starts its server at the first question, so a session that never
    searches the documentation never loads the model. If the server cannot start, it raises
    :class:`~braino.knowledge.index.EmbeddingError` once and search uses keywords alone."""

    def __init__(self, *, threads: int | None = None, cat: Catalog | None = None) -> None:
        self._cat = cat
        self._threads = threads
        self.model_id = (cat or catalog()).embedding.id
        self._inner: ServerEmbedder | None = None
        self._failed: str | None = None

    @property
    def started(self) -> bool:
        return self._inner is not None

    def embed_query(self, text: str) -> list[float]:
        return self._get().embed_query(text)

    def embed_passages(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return self._get().embed_passages(texts)

    def stop(self) -> None:
        if self._inner is not None:
            self._inner.stop()
            self._inner = None

    def _get(self) -> ServerEmbedder:
        from braino.knowledge.index import EmbeddingError

        if self._failed is not None:
            raise EmbeddingError(self._failed)
        if self._inner is None:
            try:
                self._inner = ServerEmbedder.create(threads=self._threads, cat=self._cat).start()
            except ServerError as error:
                self._failed = str(error)
                raise EmbeddingError(self._failed) from error
        return self._inner


def available(cat: Catalog | None = None) -> bool:
    """Whether the embedding model and llama.cpp are installed."""
    return is_model_installed((cat or catalog()).embedding) and find_server() is not None
