"""The documentation index: passages, their vectors, and search.

Search combines two rankings with reciprocal rank fusion:

- **keywords** (BM25 over the words of the title and text, lightly stemmed), which finds a
  field name, a tool or a term of art exactly, and needs no model;
- **meaning** (the cosine between the question's vector and each passage's), which finds a
  passage worded differently from the question. It needs the embedding model, which the
  agent starts when it is downloaded; without it, search uses the keywords alone.

The index ships with BrainO (``braino/knowledge/data``): ``index.json.gz`` holds the passages
and what they were built from, ``vectors.npy`` their vectors (float16, one row each).
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from braino.knowledge.passages import Passage, Source

if TYPE_CHECKING:
    import numpy as np

DATA_DIR = Path(__file__).with_name("data")
INDEX_FILE = "index.json.gz"
VECTORS_FILE = "vectors.npy"
#: Reciprocal rank fusion constant: rank r adds 1 / (RRF_K + r).
RRF_K = 60
#: Passages each ranking contributes to the fusion.
CANDIDATES = 40
#: Passages of one section a search returns at most.
PER_SECTION = 2


class EmbeddingError(RuntimeError):
    """The embedding model could not embed the question; search goes on by keywords."""


class Embedder(Protocol):
    """What search needs from the embedding model."""

    model_id: str

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_passages(self, texts: Sequence[str]) -> list[Sequence[float]]: ...


class SourceInfo(BaseModel):
    source: Source
    name: str
    version: str
    licence: str
    url: str
    passages: int = 0


class IndexMeta(BaseModel):
    format: Literal[1] = 1
    embedding: str | None = Field(description="Embedding model id the vectors come from")
    dims: int | None = None
    sources: list[SourceInfo]


class Hit(BaseModel):
    passage: Passage
    score: float = Field(description="Fused score: higher is better")
    keyword_rank: int | None = None
    semantic_rank: int | None = None


class DocsIndex:
    def __init__(
        self,
        meta: IndexMeta,
        passages: list[Passage],
        vectors: np.ndarray | None = None,
    ) -> None:
        if vectors is not None and len(vectors) != len(passages):
            raise ValueError(f"{len(vectors)} vectors for {len(passages)} passages")
        self.meta = meta
        self.passages = passages
        self.vectors = vectors
        self.by_id = {p.id: i for i, p in enumerate(passages)}
        self._bm25 = Bm25([tokens(p.title) * 2 + tokens(p.text) for p in passages])

    # --- files ------------------------------------------------------------------------

    @classmethod
    def load(cls, directory: Path = DATA_DIR) -> DocsIndex:
        with gzip.open(directory / INDEX_FILE, "rt", encoding="utf-8") as f:
            data = json.load(f)
        meta = IndexMeta.model_validate(data["meta"])
        passages = [Passage.model_validate(p) for p in data["passages"]]
        vectors = None
        if (directory / VECTORS_FILE).is_file():
            try:
                import numpy as np
            except ImportError:  # the core install: search by keywords alone
                np = None  # type: ignore[assignment]
            if np is not None:
                vectors = np.load(directory / VECTORS_FILE)
        return cls(meta, passages, vectors)

    def save(self, directory: Path = DATA_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": self.meta.model_dump(mode="json"),
            "passages": [p.model_dump(mode="json") for p in self.passages],
        }
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), indent=None)
        # mtime=0: the same index gives the same file, so git sees no change.
        with (
            (directory / INDEX_FILE).open("wb") as fh,
            gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz,
        ):
            gz.write(raw.encode("utf-8"))
        vectors_path = directory / VECTORS_FILE
        if self.vectors is not None:
            import numpy as np

            np.save(vectors_path, self.vectors.astype(np.float16))
        else:
            vectors_path.unlink(missing_ok=True)

    # --- search -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        source: Source | None = None,
        embedder: Embedder | None = None,
    ) -> list[Hit]:
        """The ``limit`` passages that best answer ``query``: by keywords and, with an
        ``embedder`` whose model made the vectors, by meaning too."""
        allowed = [i for i, p in enumerate(self.passages) if source is None or p.source == source]
        keyword = self._keyword_ranking(query, allowed)
        semantic = self._semantic_ranking(query, allowed, embedder)
        fused: dict[int, float] = defaultdict(float)
        for ranking in (keyword, semantic):
            for rank, i in enumerate(ranking[:CANDIDATES]):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        k_rank = {i: r + 1 for r, i in enumerate(keyword)}
        s_rank = {i: r + 1 for r, i in enumerate(semantic)}
        best: list[int] = []
        per_section: Counter[str] = Counter()
        for i in sorted(fused, key=lambda i: (-fused[i], i)):
            # Two passages of a section at most: room for another section's view.
            section = self.passages[i].id.split("~")[0]
            if per_section[section] < PER_SECTION:
                per_section[section] += 1
                best.append(i)
            if len(best) == limit:
                break
        return [
            Hit(
                passage=self.passages[i],
                score=round(fused[i], 5),
                keyword_rank=k_rank.get(i),
                semantic_rank=s_rank.get(i),
            )
            for i in best
        ]

    def semantic(self, embedder: Embedder | None) -> bool:
        """Whether search with ``embedder`` uses the vectors."""
        return (
            embedder is not None
            and self.vectors is not None
            and embedder.model_id == self.meta.embedding
        )

    def _keyword_ranking(self, query: str, allowed: list[int]) -> list[int]:
        scores = self._bm25.scores(tokens(query))
        return sorted((i for i in allowed if scores.get(i, 0) > 0), key=lambda i: (-scores[i], i))

    def _semantic_ranking(
        self, query: str, allowed: list[int], embedder: Embedder | None
    ) -> list[int]:
        if not self.semantic(embedder):
            return []
        import numpy as np

        assert embedder is not None and self.vectors is not None
        try:
            q = np.asarray(embedder.embed_query(query), dtype=np.float32)
        except EmbeddingError:
            return []
        q /= np.linalg.norm(q) or 1.0
        vectors = self.vectors[allowed].astype(np.float32)
        sims = vectors @ q
        order = np.argsort(-sims, kind="stable")
        return [allowed[int(j)] for j in order]


# --- keywords ------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9µ]+(?:[._/-][a-z0-9µ]+)*")
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could do does doing did for from had has have
    how i if in into is it its me my of on or our should so than that the their them then there
    these they this those to too was we were what when where which while who whom why will with
    would you your about any each other some such only own same very just also yes no nor
    want need tell explain please""".split()  # noqa: SIM905 (easier to read as words)
)


def stem(word: str) -> str:
    """A light stemmer, the same for questions and passages: filtering, filtered, filters ->
    filter; reference, references, referencing -> referenc; frequencies -> frequency."""
    if len(word) <= 4 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith(("ches", "shes", "xes")):
        word = word[:-2]
    elif word.endswith("ing") and len(word) >= 6:
        word = _undouble(word[:-3])
    elif word.endswith("ed") and len(word) >= 5:
        word = _undouble(word[:-2])
    elif word.endswith("s"):
        word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def _undouble(stem: str) -> str:
    # labelled -> label, stopped -> stop; not "ss" (passed -> pass) or short stems (called).
    if len(stem) >= 5 and stem[-1] == stem[-2] and stem[-1] in "bdglmnprt":
        return stem[:-1]
    return stem


def tokens(text: str) -> list[str]:
    """Search terms: words and numbers, lower-cased and stemmed, without stopwords. A
    compound (events.tsv, eeg.preprocess, high-pass) counts as itself and as its parts."""
    out: list[str] = []
    for match in _WORD.finditer(text.lower()):
        word = match.group()
        parts = re.split(r"[._/-]", word)
        if len(parts) > 1:
            out.append(word)
        out += [stem(p) for p in parts if p and p not in STOPWORDS]
    return out


class Bm25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in docs]
        self.mean_length = sum(self.lengths) / len(docs) if docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                self.postings[term].append((i, count))
        n = len(docs)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query):
            for i, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.lengths[i] / (self.mean_length or 1)
                out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out


@cache
def default_index() -> DocsIndex:
    """The index that ships with BrainO."""
    return DocsIndex.load()


def describe_sources(meta: IndexMeta) -> list[dict[str, Any]]:
    return [s.model_dump() for s in meta.sources]
