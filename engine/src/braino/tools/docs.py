"""Searching the documentation that ships with BrainO."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from braino.tools.base import Risk
from braino.tools.registry import tool

#: Passages a search returns: about 1,000 tokens, read in seconds on a CPU.
RESULTS = 3


class DocsSearchInput(BaseModel):
    query: str = Field(
        min_length=3,
        description="The question to look up, in plain words (e.g. 'which columns must "
        "events.tsv have', 'why high-pass filter before ICA')",
    )
    source: Literal["any", "braino", "bids", "mne"] = Field(
        "any",
        description="Search one source only: 'braino' (BrainO's guide and tools), 'bids' "
        "(the BIDS specification) or 'mne' (MNE-Python's EEG tutorials and glossary)",
    )


class DocPassage(BaseModel):
    source: str = Field(description="Where the passage comes from, with its version")
    title: str
    url: str
    text: str


class DocsSearchOutput(BaseModel):
    passages: list[DocPassage]
    note: str | None = None


@tool(name="docs.search", risk=Risk.READ)
def docs_search(params: DocsSearchInput) -> DocsSearchOutput:
    """Search the documentation shipped with BrainO: BrainO's guide and tools, the BIDS
    specification (1.11) and MNE-Python's EEG tutorials. Use it for questions about methods,
    file formats or how BrainO works, and answer from the passages, naming their source."""
    from braino.knowledge import default_index, search

    index = default_index()
    labels = {s.source: f"{s.name} {s.version}" for s in index.meta.sources}
    source = None if params.source == "any" else params.source
    hits = search(params.query, limit=RESULTS, source=source)
    passages = [
        DocPassage(
            source=labels.get(h.passage.source, h.passage.source),
            title=h.passage.title,
            url=h.passage.url,
            text=h.passage.text,
        )
        for h in hits
    ]
    note = None if passages else "Nothing in the documentation matches; try other words."
    return DocsSearchOutput(passages=passages, note=note)
