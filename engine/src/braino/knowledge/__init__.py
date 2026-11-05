"""Searching the documentation: BrainO's guide and tools, the BIDS specification and MNE's
EEG tutorials, in an index that ships with BrainO.

The agent looks things up with the ``docs.search`` tool instead of answering from what the
model remembers::

    from braino.knowledge import search

    for hit in search("Which columns must an events.tsv file have?"):
        print(hit.passage.title, hit.passage.url)

Search uses keywords, and also meaning when the embedding model is downloaded and set for
the session (:data:`embedder`, which the agent and ``braino docs search`` set).
"""

from __future__ import annotations

from contextvars import ContextVar

from braino.knowledge.index import (
    DocsIndex,
    Embedder,
    EmbeddingError,
    Hit,
    IndexMeta,
    SourceInfo,
    default_index,
)
from braino.knowledge.passages import Passage, Source

#: The embedder search uses in this context (``None``: keywords only).
embedder: ContextVar[Embedder | None] = ContextVar("docs_embedder", default=None)


def search(query: str, *, limit: int = 3, source: Source | None = None) -> list[Hit]:
    """The passages of the shipped index that best answer ``query``."""
    return default_index().search(query, limit=limit, source=source, embedder=embedder.get())


__all__ = [
    "DocsIndex",
    "Embedder",
    "EmbeddingError",
    "Hit",
    "IndexMeta",
    "Passage",
    "Source",
    "SourceInfo",
    "default_index",
    "embedder",
    "search",
]
