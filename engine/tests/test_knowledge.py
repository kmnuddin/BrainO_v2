"""Documentation search: cutting documents into passages, the index, the shipped index, the
docs.search tool, the CLI, and the agent looking something up."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from scripted_llm import ScriptedServer
from typer.testing import CliRunner

from braino import knowledge
from braino.cli import app
from braino.knowledge.build import build, repo_docs, stale_braino_passages
from braino.knowledge.index import DocsIndex, EmbeddingError, IndexMeta, SourceInfo, stem, tokens
from braino.knowledge.passages import (
    LEAD_WORDS,
    MAX_WORDS,
    SEP,
    Passage,
    Source,
    markdown_sections,
    rst_to_markdown,
    section_passages,
)
from braino.knowledge.sources import (
    _admonitions,
    _render_macro,
    gallery_text,
    glossary_passages,
    tool_passages,
)
from braino.runtime.catalog import catalog
from braino.tools import registry
from braino.tools.docs import DocsSearchOutput

# --- cutting documents ---------------------------------------------------------------------

MARKDOWN = """# Pipelines

A pipeline is an analysis written down.

## Plan checks

Before a pipeline runs, its steps are checked together.

```yaml
# not a heading
steps: []
```

### Windows fit the epochs

A measure at 300-1200 ms in epochs that end at 800 ms is an error.
"""


def test_markdown_sections_keep_headings_and_code() -> None:
    sections = markdown_sections(MARKDOWN, title="Pipelines")
    assert [s.path for s in sections] == [
        ["Pipelines"],
        ["Pipelines", "Plan checks"],
        ["Pipelines", "Plan checks", "Windows fit the epochs"],
    ]
    assert sections[2].title == SEP.join(sections[2].path)
    assert [s.anchor for s in sections] == ["", "plan-checks", "windows-fit-the-epochs"]
    assert "# not a heading" in sections[1].paragraphs[1]  # code stays whole
    passages = section_passages(sections, source="braino", id_prefix="braino/p", url="u/")
    assert [p.id for p in passages] == [
        "braino/p",
        "braino/p#plan-checks",
        "braino/p#windows-fit-the-epochs",
    ]
    assert passages[1].url == "u/#plan-checks"


def test_long_sections_are_cut_between_paragraphs() -> None:
    text = "# Doc\n\n" + "\n\n".join(" ".join([f"w{i}"] * 60) for i in range(8))
    passages = section_passages(markdown_sections(text), source="braino", id_prefix="d", url="u")
    assert len(passages) > 1 and all(len(p.text.split()) <= MAX_WORDS for p in passages)
    assert [p.id for p in passages][:2] == ["d#doc", "d#doc~1"]
    assert " ".join(p.text for p in passages).split() == " ".join(text.split()[2:]).split()


def test_rst_to_markdown() -> None:
    rst = """
==================
Repairing with ICA
==================

.. _tut-ica:

See :ref:`the overview <tut-overview>`, :func:`mne.preprocessing.ICA` and
:class:`~mne.Epochs`; ``picks`` matter :footcite:`Smith2020`.

.. note::
   Look at the artifacts first.

.. NOTE: an RST comment
   that goes on.

Filtering
^^^^^^^^^

A 1 Hz high-pass helps ICA::

    raw.filter(l_freq=1.0)

.. image:: picture.png
   :width: 50%

Done.
"""
    md = rst_to_markdown(rst)
    assert "# Repairing with ICA" in md and "## Filtering" in md
    flat = " ".join(md.split())
    assert "See the overview, `mne.preprocessing.ICA` and `Epochs`; `picks` matter." in flat
    assert "Note: Look at the artifacts first." in md
    assert "A 1 Hz high-pass helps ICA:\n\n" in md and "raw.filter" not in md
    assert "comment" not in md and "picture.png" not in md and "\n\nDone." in md


def test_gallery_text_keeps_text_and_drops_code() -> None:
    script = '''"""
Title
=====

What this covers.
"""
# License: BSD-3-Clause

# %%
import mne

# %%
# First we filter.
#
# Then we epoch.

raw.filter(1, 40)
'''
    text = gallery_text(script)
    assert "What this covers." in text and "First we filter.\n\nThen we epoch." in text
    assert "import mne" not in text and "raw.filter" not in text and "License" not in text


def test_glossary_passages() -> None:
    rst = """Glossary
========

.. glossary::
    :sorted:

    BEM
    boundary element model
        BEM is the acronym for boundary element model. See :term:`forward`.

    epochs
        Epochs are equal-length segments of data.
"""
    passages = glossary_passages(rst)
    assert [p.title for p in passages] == [
        f"MNE glossary{SEP}BEM (boundary element model)",
        f"MNE glossary{SEP}epochs",
    ]
    assert passages[0].text == "BEM is the acronym for boundary element model. See `forward`."
    assert passages[1].url.endswith("glossary.html#term-epochs")


def test_bids_tables_are_written_out() -> None:
    schema: dict[str, Any] = {
        "rules": {
            "sidecars": {
                "eeg": {
                    "EEGRequired": {
                        "fields": {
                            "PowerLineFrequency": "required",
                            "SamplingFrequency": {
                                "level": "required",
                                "description_addendum": "Per channel in channels.tsv.",
                            },
                        }
                    }
                }
            },
            "tabular_data": {"events": {"Events": {"columns": {"onset": "required"}}}},
        },
        "objects": {
            "metadata": {
                "PowerLineFrequency": {"description": "Frequency of the\n[power grid](x.md)."},
                "SamplingFrequency": {"description": "Sampling rate."},
            },
            "columns": {"onset": {"name": "onset", "description": "Onset in seconds."}},
        },
    }
    table = _render_macro(schema, "make_sidecar_table", '"eeg.EEGRequired"')
    assert table.splitlines() == [
        "- `PowerLineFrequency` (REQUIRED): Frequency of the power grid.",
        "- `SamplingFrequency` (REQUIRED): Sampling rate. Per channel in channels.tsv.",
    ]
    assert _render_macro(schema, "make_columns_table", '"events.Events"') == (
        "- `onset` (REQUIRED): Onset in seconds."
    )
    assert _render_macro(schema, "make_filetree_example", "{'a': 1}") == ""
    assert _admonitions('!!! note "Precision"\n\n    Use seconds.\n\nNext.') == (
        "Precision:\n\nUse seconds.\n\nNext."
    )


def test_tool_passages() -> None:
    passages = tool_passages(registry.get("eeg.preprocess"))
    assert passages[0].id == "braino/tool/eeg.preprocess"
    assert passages[0].url.endswith("/reference/tools/#eegpreprocess")
    text = "\n".join(p.text for p in passages)
    assert "`l_freq` (default 0.1)" in text and "`subject`" not in text


# --- search --------------------------------------------------------------------------------


def test_tokens_and_stems() -> None:
    assert stem("filtering") == stem("filtered") == stem("filters") == "filter"
    assert stem("reference") == stem("references") == stem("referencing") == "referenc"
    assert stem("frequencies") == "frequency" and stem("analysis") == "analysis"
    assert tokens("Which columns must events.tsv have?") == [
        "column",
        "must",
        "events.tsv",
        "event",
        "tsv",
    ]


def _passage(pid: str, text: str, source: Source = "braino", title: str = "") -> Passage:
    return Passage(id=pid, source=source, title=title or pid, url=f"u/{pid}", text=text)


class FakeEmbedder:
    """Vectors from a fixed vocabulary of meanings: words with the same meaning share a dim."""

    model_id = "fake-embedding"
    MEANINGS = (
        ("drift", "slow", "high-pass", "highpass"),
        ("blink", "eye", "ocular"),
        ("column", "onset", "duration"),
    )

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        words = text.lower().replace("?", " ").split()
        v = [float(sum(w in words for w in group)) for group in self.MEANINGS]
        return v if any(v) else [0.1, 0.1, 0.1]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_passages(self, texts: Sequence[str]) -> list[Sequence[float]]:
        self.calls += len(texts)
        return [self._vector(t) for t in texts]


PASSAGES = [
    _passage("a", "Remove slow drift with a high-pass filter before ICA."),
    _passage("b", "ICA finds eye blink components; reject them after review.", "mne"),
    _passage("c", "The events file has the columns onset and duration.", "bids"),
    _passage("c~1", "Further columns of the events file: trial_type, value.", "bids", "c"),
    _passage("c~2", "Yet more columns of the events file: response_time.", "bids", "c"),
]


def _index(embedder: FakeEmbedder | None) -> DocsIndex:
    return build(PASSAGES, embedder)


def test_keyword_search_and_two_passages_per_section() -> None:
    index = _index(None)
    hits = index.search("columns of the events file")
    # All of "c" matches, but a search returns two passages of a section at most.
    assert len(hits) == 2 and {h.passage.id.split("~")[0] for h in hits} == {"c"}
    assert all(h.semantic_rank is None for h in hits)
    assert [h.passage.id for h in index.search("ICA", source="mne")] == ["b"]
    assert index.search("nothing matches zebra") == []


def test_search_by_meaning_finds_other_words() -> None:
    embedder = FakeEmbedder()
    index = _index(embedder)
    assert index.meta.embedding == "fake-embedding" and index.meta.dims == 3
    # No passage has the word "ocular", but one means it.
    hits = index.search("ocular artifacts", embedder=embedder, limit=1)
    assert hits[0].passage.id == "b" and hits[0].keyword_rank is None
    assert hits[0].semantic_rank == 1
    # Another model's vectors cannot be compared: keywords only.
    other = FakeEmbedder()
    other.model_id = "other"
    assert index.search("ocular artifacts", embedder=other) == []


def test_search_goes_on_by_keywords_when_the_model_fails() -> None:
    class Broken(FakeEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise EmbeddingError("server did not start")

    embedder = Broken()
    index = _index(FakeEmbedder())
    hits = index.search("blink components", embedder=embedder)
    assert hits[0].passage.id == "b" and hits[0].semantic_rank is None


def test_save_load_and_reuse_vectors(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index = _index(embedder)
    index.save(tmp_path / "a")
    index.save(tmp_path / "b")
    assert (tmp_path / "a" / "index.json.gz").read_bytes() == (
        tmp_path / "b" / "index.json.gz"
    ).read_bytes()  # the same index is the same file
    loaded = DocsIndex.load(tmp_path / "a")
    assert [p.id for p in loaded.passages] == [p.id for p in PASSAGES]
    assert loaded.vectors is not None and str(loaded.vectors.dtype) == "float16"
    changed = [*PASSAGES[:3], _passage("d", "A new passage about eye blinks.")]
    embedder.calls = 0
    rebuilt = build(changed, embedder, previous=loaded)
    assert embedder.calls == 1  # only the new passage
    assert rebuilt.vectors is not None and rebuilt.vectors.shape == (4, 3)


# --- the shipped index ---------------------------------------------------------------------


def test_the_shipped_index() -> None:
    index = knowledge.default_index()
    counts = {s.source: s.passages for s in index.meta.sources}
    assert counts["braino"] > 50 and counts["bids"] > 150 and counts["mne"] > 300
    assert sum(counts.values()) == len(index.passages)
    assert len({p.id for p in index.passages}) == len(index.passages)
    licences = {s.source: s.licence for s in index.meta.sources}
    assert licences == {"braino": "GPL-3.0-or-later", "bids": "CC-BY-4.0", "mne": "BSD-3-Clause"}
    spec = catalog().embedding
    assert index.meta.embedding == spec.id and index.meta.dims == spec.dims
    assert index.vectors is not None and index.vectors.shape == (len(index.passages), spec.dims)
    assert max(len(p.text.split()) for p in index.passages) <= MAX_WORDS + LEAD_WORDS


def test_the_shipped_index_is_current() -> None:
    if not (repo_docs() / "guide").is_dir():
        pytest.skip("not a BrainO checkout")
    stale = stale_braino_passages(knowledge.default_index())
    assert not stale, f"the docs or tools changed: run 'braino docs build' ({stale[:5]})"


#: Questions, and a pattern that a passage answering them matches (in its title and text, or
#: its id with "id:"). Worded as users ask, often unlike the documentation.
QUESTIONS = [
    ("Which columns must an events.tsv file have?", r"`onset` \(REQUIRED\)"),
    ("What do I put in the field for mains electricity frequency?", r"PowerLineFrequency"),
    ("How do I get rid of eye blinks?", r"(?s)(ICA|regression|EOG).*(blink|ocular|EOG)"),
    ("How can I repair a broken electrode?", r"(?i)interpolat"),
    (
        "Why does my ERP look distorted after strong filtering?",
        r"id:mne/preprocessing/25_background_filtering",
    ),
    ("Where does BrainO keep the record of what it did?", r"(?i)provenance"),
    ("Why is the data high-pass filtered before ICA, and at what cutoff?", r"1 Hz"),
    ("How many subjects does BrainO need before it will run a group test?", r"3 subjects"),
    ("What does the task entity mean in BIDS?", r"id:bids/entity/task"),
    ("How should I choose the reference for EEG?", r"id:mne/preprocessing/55_setting_eeg"),
    ("What does autoreject do?", r"(?i)autoreject"),
    (
        "How does a cluster-based permutation test control false positives?",
        r"(?is)cluster.*(permutation|family-wise|FWER)",
    ),
    ("Which file formats can BrainO import?", r"id:braino/guide/importing"),
    ("What must dataset_description.json contain?", r"BIDSVersion"),
    ("How do I baseline-correct epochs?", r"(?i)baseline"),
    ("How does BrainO find bad channels?", r"(?i)(PREP|pyprep|RANSAC)"),
    ("How are ICA components labelled?", r"ICLabel"),
    ("Can I run BrainO without a GPU?", r"(?i)(no GPU|without a GPU|on the CPU|CPU only)"),
    (
        "In what order should I preprocess EEG?",
        r"id:(braino/guide/analysis#recommended-order|mne/preprocessing/10_preprocessing)",
    ),
    ("Where are processed files stored in a BIDS dataset?", r"derivatives/"),
    ("What is the difference between an electrode and a channel?", r"(?i)electrode = a single"),
    ("How does the agent make sure numbers are not made up?", r"(?i)(invented|every number)"),
]


def _answers(passage: Passage, pattern: str) -> bool:
    if pattern.startswith("id:"):
        return re.match(pattern[3:], passage.id) is not None
    return re.search(pattern, passage.embed_text()) is not None


def _found(query: str, pattern: str, embedder: Any = None) -> bool:
    hits = knowledge.default_index().search(query, embedder=embedder)
    return any(_answers(h.passage, pattern) for h in hits)


def test_keyword_search_of_the_shipped_index() -> None:
    found = [q for q, pattern in QUESTIONS if _found(q, pattern)]
    # Keywords alone answer most; the embedding model is for the rest (below).
    assert len(found) >= 17, sorted(set(q for q, _ in QUESTIONS) - set(found))


# --- the tool, the CLI, the agent ----------------------------------------------------------


def test_docs_search_tool() -> None:
    tool = registry.get("docs.search")
    out = tool.run({"query": "Which columns must an events.tsv file have?", "source": "bids"})
    assert isinstance(out, DocsSearchOutput) and len(out.passages) == 3
    assert all(p.source == "BIDS specification 1.11.2" for p in out.passages)
    empty = tool.run({"query": "zzqx qqzx"})
    assert isinstance(empty, DocsSearchOutput) and empty.passages == []
    assert "try other words" in (empty.note or "")


def test_the_search_uses_the_session_embedder() -> None:
    index = knowledge.default_index()

    class Spy:
        model_id = index.meta.embedding or ""

        def __init__(self) -> None:
            self.queries: list[str] = []

        def embed_query(self, text: str) -> Sequence[float]:
            self.queries.append(text)
            assert index.vectors is not None
            return [
                float(x) for x in index.vectors[index.by_id["braino/guide/pipelines#plan-checks"]]
            ]

        def embed_passages(self, texts: Sequence[str]) -> list[Sequence[float]]:
            raise AssertionError("not while searching")

    spy = Spy()
    token = knowledge.embedder.set(spy)
    try:
        hits = knowledge.search("group test subjects")
    finally:
        knowledge.embedder.reset(token)
    assert spy.queries == ["group test subjects"]
    # The passage whose vector the spy returned ranks first by meaning.
    best = next(h for h in hits if h.semantic_rank == 1)
    assert best.passage.id == "braino/guide/pipelines#plan-checks"


def test_cli() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["docs", "search", "spherical spline interpolation", "--keywords-only"]
    )
    assert result.exit_code == 0, result.output
    assert (
        "Bad channel repair via interpolation" in result.output
        and "mne.tools/1.13" in result.output
    )
    result = runner.invoke(app, ["docs", "search", "x", "--source", "nilearn"])
    assert result.exit_code == 1 and "unknown source" in result.output
    result = runner.invoke(app, ["docs", "sources"])
    assert "BIDS specification 1.11.2 (CC-BY-4.0)" in result.output
    assert "qwen3-embedding-0.6b" in result.output


@pytest.fixture
def server() -> Iterator[ScriptedServer]:
    s = ScriptedServer()
    yield s
    s.close()


def test_the_agent_looks_it_up(server: ScriptedServer, tmp_path: Path) -> None:
    from braino.agent import LLMAgent
    from braino.agent.session import Session
    from braino.tools import RunContext

    class Quiet(Session):
        @property
        def user_name(self) -> str:
            return "test:user"

        def _decide(
            self, tool: str, proposal: dict[str, Any], message: str
        ) -> tuple[bool, dict[str, Any], str | None]:
            raise AssertionError("no decisions here")

        def _reply(self, question: str) -> str:
            raise AssertionError("no questions here")

    query = "required columns of events.tsv"
    server.replies += [
        {"action": "call", "tool": "docs.search", "arguments": {"query": query, "source": "bids"}},
        {"action": "answer", "text": "Every events.tsv has onset and duration (BIDS 1.11.2)."},
    ]
    session = Quiet(RunContext(dataset_root=tmp_path, run_id="r1"))
    answer = LLMAgent(server.client()).solve(
        "Which columns must a BIDS events.tsv file have?", session
    )
    assert answer.text.startswith("Every events.tsv") and answer.stopped is None
    first = server.requests[0]["messages"]
    assert "docs.search" in first[2]["content"]  # preloaded: the request is a docs question
    assert "docs.search" in first[0]["content"]  # the system prompt says when to use it
    assert [c.tool for c in session.calls] == ["docs.search"] and session.calls[0].ok
    last = server.requests[1]["messages"][-1]["content"]  # the search result, nothing after it
    result = json.loads(last.split("\n", 1)[1].rsplit("\n", 1)[0])["result"]
    assert len(result["passages"]) == 3


def test_tools_named_in_the_docs_are_not_loaded(server: ScriptedServer, tmp_path: Path) -> None:
    from braino.agent import LLMAgent
    from braino.agent.session import Session
    from braino.tools import RunContext

    class Quiet(Session):
        user_name = "test:user"

        def _decide(
            self, tool: str, proposal: dict[str, Any], message: str
        ) -> tuple[bool, dict[str, Any], str | None]:
            raise AssertionError("no decisions here")

        def _reply(self, question: str) -> str:
            raise AssertionError("no questions here")

    search = {"query": "the recommended order of preprocessing steps", "source": "braino"}
    server.replies += [
        {"action": "call", "tool": "docs.search", "arguments": search},
        {"action": "answer", "text": "Filter first, then ICA, then epoch."},
    ]
    session = Quiet(RunContext(dataset_root=tmp_path, run_id="r1"))
    LLMAgent(server.client()).solve("What is the recommended order of steps?", session)
    last = server.requests[1]["messages"][-1]["content"]
    assert "eeg.ica_clean" in last  # the guide names tools...
    assert not last.startswith("<tool_response>\nLoaded.")  # ...which are not loaded for it


# --- with the real embedding model (opt-in) ------------------------------------------------


@pytest.mark.llama
def test_search_by_meaning_with_the_real_model() -> None:
    """Keywords and meaning together answer more questions than keywords alone (on this
    index: 21 of 22, against 18)."""
    from braino.knowledge.embed import ServerEmbedder

    with ServerEmbedder.create() as embedder:
        both = [q for q, pattern in QUESTIONS if _found(q, pattern, embedder)]
    keywords = [q for q, pattern in QUESTIONS if _found(q, pattern)]
    print(f"keywords {len(keywords)}/{len(QUESTIONS)}, with meaning {len(both)}/{len(QUESTIONS)}")
    assert len(both) >= len(QUESTIONS) - 2 and len(both) > len(keywords)


def test_index_meta_round_trip() -> None:
    meta = IndexMeta(
        embedding=None,
        sources=[SourceInfo(source="bids", name="B", version="1", licence="L", url="u")],
    )
    assert IndexMeta.model_validate_json(meta.model_dump_json()) == meta
