"""Building the documentation index (a development step: the index ships with BrainO).

``braino docs build`` downloads the pinned BIDS and MNE releases once (to the cache), cuts
them and BrainO's own guide and tools into passages, and embeds the passages that are new or
changed; the vectors of unchanged passages are reused, so a change to BrainO's docs costs
seconds, not the quarter of an hour a full build takes on a CPU.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from braino.knowledge.index import DATA_DIR, DocsIndex, Embedder, IndexMeta, SourceInfo
from braino.knowledge.passages import Passage
from braino.knowledge.sources import (
    ARCHIVES,
    BIDS,
    BRAINO_SITE,
    MNE,
    Archive,
    bids_passages,
    braino_passages,
    mne_passages,
)
from braino.paths import data_root
from braino.runtime.catalog import PinnedFile

Log = Callable[[str], None]


def repo_docs() -> Path:
    """The ``docs/`` folder of the BrainO checkout this package runs from."""
    return Path(__file__).resolve().parents[4] / "docs"


def sources_cache() -> Path:
    return data_root() / "knowledge" / "sources"


def fetch(archive: Archive, cache: Path | None = None) -> Path:
    """Download and unpack ``archive`` once; returns the unpacked folder."""
    from braino.runtime.download import download
    from braino.runtime.install import _extract

    cache = cache or sources_cache()
    target = cache / archive.root
    if (target / ".complete").is_file():
        return target
    name = f"{archive.root}.tar.gz"
    pin = PinnedFile(name=name, size=archive.size, sha256=archive.sha256)
    tarball = download(archive.url, cache / name, pin)
    if target.exists():
        shutil.rmtree(target)
    _extract(tarball, cache)
    if not target.is_dir():
        raise FileNotFoundError(f"{name} did not unpack to {archive.root}/")
    (target / ".complete").write_text(archive.sha256, encoding="utf-8")
    return target


def collect(
    docs_dir: Path | None = None, *, cache: Path | None = None, log: Log | None = None
) -> list[Passage]:
    """Every passage of the index, BrainO's first."""
    say = log or (lambda _: None)
    passages = braino_passages(docs_dir or repo_docs())
    say(f"BrainO: {len(passages)} passages")
    bids = bids_passages(fetch(BIDS, cache))
    say(f"BIDS {BIDS.version}: {len(bids)} passages")
    mne = mne_passages(fetch(MNE, cache))
    say(f"MNE {MNE.version}: {len(mne)} passages")
    passages = passages + bids + mne
    seen: set[str] = set()
    for p in passages:
        if p.id in seen:
            raise ValueError(f"two passages have the id {p.id!r}")
        seen.add(p.id)
    return passages


def source_infos(passages: list[Passage]) -> list[SourceInfo]:
    from braino import __version__

    counts = {s: sum(p.source == s for p in passages) for s in ("braino", "bids", "mne")}
    infos = [
        SourceInfo(
            source="braino",
            name="BrainO documentation",
            version=__version__,
            licence="GPL-3.0-or-later",
            url=f"{BRAINO_SITE}/",
            passages=counts["braino"],
        )
    ]
    for a in ARCHIVES:
        infos.append(
            SourceInfo(
                source=a.source,
                name=a.name,
                version=a.version,
                licence=a.licence,
                url=a.homepage,
                passages=counts[a.source],
            )
        )
    return infos


def build(
    passages: list[Passage],
    embedder: Embedder | None,
    *,
    previous: DocsIndex | None = None,
    log: Log | None = None,
) -> DocsIndex:
    """An index of ``passages``, with vectors from ``embedder`` (``None``: keywords only).
    Vectors of ``previous`` are reused for passages whose text has not changed."""
    import numpy as np

    say = log or (lambda _: None)
    meta = IndexMeta(embedding=None, dims=None, sources=source_infos(passages))
    if embedder is None:
        return DocsIndex(meta, passages)
    model_id = embedder.model_id
    reuse: dict[str, np.ndarray] = {}
    if (
        previous is not None
        and previous.vectors is not None
        and previous.meta.embedding == model_id
    ):
        reuse = {p.digest: previous.vectors[i] for i, p in enumerate(previous.passages)}
    todo = [i for i, p in enumerate(passages) if p.digest not in reuse]
    say(f"Embedding {len(todo)} new or changed passages ({len(passages) - len(todo)} reused)")
    fresh: dict[int, np.ndarray] = {}
    step = 64
    for start in range(0, len(todo), step):
        batch = todo[start : start + step]
        vectors = embedder.embed_passages([passages[i].embed_text() for i in batch])
        for i, v in zip(batch, vectors, strict=True):
            fresh[i] = np.asarray(v, dtype=np.float32)
        say(f"  {min(start + step, len(todo))}/{len(todo)}")
    rows = [fresh[i] if i in fresh else reuse[p.digest] for i, p in enumerate(passages)]
    matrix = np.vstack(rows).astype(np.float16) if rows else np.zeros((0, 0), np.float16)
    meta = meta.model_copy(update={"embedding": model_id, "dims": int(matrix.shape[1])})
    return DocsIndex(meta, passages, matrix)


def build_and_save(
    docs_dir: Path | None = None,
    *,
    embedder: Embedder | None,
    out_dir: Path = DATA_DIR,
    cache: Path | None = None,
    log: Log | None = None,
) -> DocsIndex:
    passages = collect(docs_dir, cache=cache, log=log)
    previous = None
    if (out_dir / "index.json.gz").is_file():
        previous = DocsIndex.load(out_dir)
    index = build(passages, embedder, previous=previous, log=log)
    index.save(out_dir)
    return index


def stale_braino_passages(index: DocsIndex, docs_dir: Path | None = None) -> list[str]:
    """Ids of BrainO passages that differ between ``index`` and the current docs and tools
    (added, changed or removed): the index needs rebuilding when this is not empty."""
    current = {p.id: p.digest for p in braino_passages(docs_dir or repo_docs())}
    indexed = {p.id: p.digest for p in index.passages if p.source == "braino"}
    return sorted(k for k in current.keys() | indexed.keys() if current.get(k) != indexed.get(k))


__all__ = [
    "build",
    "build_and_save",
    "collect",
    "fetch",
    "repo_docs",
    "sources_cache",
    "stale_braino_passages",
]
