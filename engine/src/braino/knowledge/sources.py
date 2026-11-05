"""What the documentation index is made of, and how each source is cut into passages.

- **BrainO**: the user guide (``docs/``) and every tool's description and parameters.
- **BIDS**: the EEG chapter and the modality-agnostic chapters of the specification, with the
  tables the specification renders from its schema written out as lists.
- **MNE-Python**: the tutorials that apply to EEG (their text, not their code), the
  glossary and the notes on interpolation and units.

The BIDS and MNE sources are release archives pinned by size and SHA-256, so an index is
built from exactly the text it says. They are needed only to build the index, which ships with
BrainO; the user's computer never downloads them.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from braino.knowledge.passages import (
    SEP,
    Passage,
    Section,
    Source,
    markdown_sections,
    plain,
    rst_inline,
    rst_to_markdown,
    section_passages,
)

BRAINO_SITE = "https://kmnuddin.github.io/BrainO_v2"


class Archive(BaseModel):
    """A pinned source release."""

    source: Source
    name: str
    version: str
    licence: str
    homepage: str
    url: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    root: str = Field(description="The folder the archive unpacks to")


# GitHub's release archives: if GitHub ever recompresses them, the hash check fails and the
# pin is updated, as for any other change of source.
BIDS = Archive(
    source="bids",
    name="BIDS specification",
    version="1.11.2",
    licence="CC-BY-4.0",
    homepage="https://bids-specification.readthedocs.io/en/v1.11.2/",
    url="https://github.com/bids-standard/bids-specification/archive/refs/tags/v1.11.2.tar.gz",
    size=7780661,
    sha256="7affcc2e6b9544249a3b53fd18ea326fdb53b3b0858e26f4cf3c62dfbd328133",
    root="bids-specification-1.11.2",
)
MNE = Archive(
    source="mne",
    name="MNE-Python documentation",
    version="1.13.2",
    licence="BSD-3-Clause",
    homepage="https://mne.tools/1.13/",
    url="https://github.com/mne-tools/mne-python/archive/refs/tags/v1.13.2.tar.gz",
    size=62698080,
    sha256="b9d1e28d3e7c608975be64afc8cc3c145a32ba1992288f31b791456b2c8cef79",
    root="mne-python-1.13.2",
)
ARCHIVES = (BIDS, MNE)


# --- BrainO --------------------------------------------------------------------------------

#: The user guide pages, as (file under docs/, title). The plan and the reference pages
#: (generated from the code, and covered by the tool passages) are left out.
BRAINO_PAGES = (
    ("index.md", "BrainO"),
    ("getting-started.md", "Getting started"),
    ("examples.md", "Examples"),
    ("guide/importing.md", "Importing recordings"),
    ("guide/analysis.md", "Analysing EEG"),
    ("guide/pipelines.md", "Pipelines"),
    ("guide/reproducibility.md", "Reproducibility"),
    ("guide/validation.md", "Validation"),
    ("guide/agent.md", "Asking the agent"),
    ("guide/model.md", "The local AI model"),
    ("guide/evals.md", "Evaluating the agent"),
)
#: Parameters every recording tool takes; their descriptions add nothing to a passage.
_RECORDING_PARAMS = {"subject", "task", "session", "run"}


def braino_passages(docs_dir: Path, tools: Iterable[Any] | None = None) -> list[Passage]:
    """BrainO's user guide (from ``docs_dir``) and its tools (default: every registered one)."""
    out: list[Passage] = []
    for name, title in BRAINO_PAGES:
        text = (docs_dir / name).read_text(encoding="utf-8")
        page = name.removesuffix(".md")
        url = f"{BRAINO_SITE}/" if page == "index" else f"{BRAINO_SITE}/{page}/"
        sections = markdown_sections(plain(text), title=title)
        out += section_passages(sections, source="braino", id_prefix=f"braino/{page}", url=url)
    if tools is None:
        from braino.tools import registry

        tools = list(registry)
    for tool in tools:
        out += tool_passages(tool)
    return out


def tool_passages(tool: Any) -> list[Passage]:
    """A tool's description and parameters (a long list of parameters goes on in a second
    passage)."""
    schema = tool.input_model.model_json_schema()
    required = set(schema.get("required", []))
    text = tool.description.replace("\n", " ")
    if tool.requires_approval:
        text += " A decision tool: it proposes, and the proposal takes effect once approved."
    blocks = [text]
    params = {k: v for k, v in schema.get("properties", {}).items() if k not in _RECORDING_PARAMS}
    if params:
        lines = ["Parameters:"]
        for name, prop in params.items():
            default = (
                "required" if name in required else f"default {json.dumps(prop.get('default'))}"
            )
            lines.append(f"- `{name}` ({default}): {prop.get('description', '')}".rstrip(": "))
        blocks.append("\n".join(lines))
    anchor = re.sub(r"[^\w-]", "", tool.name.lower())
    section = Section(path=["Tools", tool.name], anchor=anchor, paragraphs=blocks)
    return [
        p.model_copy(update={"id": p.id.replace(f"#{anchor}", "")})
        for p in section_passages(
            [section],
            source="braino",
            id_prefix=f"braino/tool/{tool.name}",
            url=f"{BRAINO_SITE}/reference/tools/",
        )
    ]


# --- BIDS ----------------------------------------------------------------------------------

#: Chapters of the specification, as (file under src/, title).
BIDS_PAGES = (
    ("common-principles.md", "Common principles"),
    ("modality-agnostic-files/dataset-description.md", "Dataset description"),
    ("modality-agnostic-files/data-summary-files.md", "Participants, sessions and scans"),
    ("modality-agnostic-files/events.md", "Events"),
    ("modality-agnostic-files/code.md", "Code"),
    ("modality-agnostic-files/phenotypic-and-assessment-data.md", "Phenotypic data"),
    ("modality-specific-files/electroencephalography.md", "Electroencephalography"),
    ("derivatives/introduction.md", "Derivatives"),
    ("derivatives/common-data-types.md", "Derivatives: common data types"),
    ("appendices/units.md", "Units"),
)
_MACRO = re.compile(r"\{\{\s*MACROS___(\w+)\((.*?)\)\s*\}\}", re.DOTALL)
_ADMONITION = re.compile(r'^(!!!|\?\?\?\+?)\s+\w+(?:\s+"([^"]*)")?\s*$')


def bids_passages(spec_dir: Path) -> list[Passage]:
    """The BIDS chapters EEG analysis needs, and every entity, from an unpacked
    specification release (``spec_dir`` holds ``src/``)."""
    from bidsschematools import schema as bids_schema

    schema = bids_schema.load_schema(str(spec_dir / "src" / "schema"))
    out: list[Passage] = []
    for name, title in BIDS_PAGES:
        text = (spec_dir / "src" / name).read_text(encoding="utf-8")
        text = _MACRO.sub(lambda m: _render_macro(schema, m.group(1), m.group(2)), text)
        text = _admonitions(plain(text))
        page = name.removesuffix(".md")
        sections = markdown_sections(text, title=title)
        url = f"{BIDS.homepage}{page}.html"
        out += section_passages(sections, source="bids", id_prefix=f"bids/{page}", url=url)
    out += _entity_passages(schema)
    return out


def _admonitions(text: str) -> str:
    """``!!! note "Title"`` blocks as plain paragraphs: the title as a label, the body (the
    indented lines that follow) dedented."""
    lines: list[str] = []
    in_body = False
    for line in text.splitlines():
        m = _ADMONITION.match(line)
        if m:
            in_body = True
            lines.append(f"{m.group(2)}:" if m.group(2) else "")
            continue
        if in_body and (not line.strip() or line.startswith("    ")):
            lines.append(line[4:])
            continue
        in_body = False
        lines.append(line)
    return "\n".join(lines)


def _get(node: Any, path: str) -> Any:
    for key in path.split("."):
        node = node[key]
    return node


def _render_macro(schema: Any, name: str, args: str) -> str:
    try:
        call = ast.parse(f"f({args})", mode="eval").body
        assert isinstance(call, ast.Call)
        pos = [ast.literal_eval(a) for a in call.args]
        kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
    except (SyntaxError, ValueError, AssertionError):
        return ""
    if name == "render_text":
        return str(_get(schema, pos[0]))
    if name == "define_common_principles":
        principles = schema["objects"]["common_principles"]
        return "\n".join(
            f"- **{p['display_name']}**: {_one_line(p['description'])}" for p in principles.values()
        )
    if name in ("make_sidecar_table", "make_json_table"):
        tables = pos[0] if isinstance(pos[0], list) else [pos[0]]
        fields: dict[str, Any] = {}
        for table in tables:
            path = table if name == "make_json_table" else f"sidecars.{table}"
            fields.update(_get(schema["rules"], path)["fields"])
        return _fields(schema["objects"]["metadata"], fields)
    if name == "make_metadata_table":
        return _fields(schema["objects"]["metadata"], pos[0] if pos else kw.get("field_info", {}))
    if name == "make_columns_table":
        table = _get(schema["rules"]["tabular_data"], pos[0])
        return _fields(schema["objects"]["columns"], table["columns"])
    if name == "make_subobject_table":
        obj = _get(schema["objects"], pos[0])
        props = obj.get("properties", {})
        required = set(obj.get("required", []))
        return "\n".join(
            f"- `{k}` ({'REQUIRED' if k in required else 'OPTIONAL'}): "
            f"{_one_line(v.get('description', ''))}"
            for k, v in props.items()
        )
    return ""  # file trees, filename templates: pictures of what the text says


def _fields(objects: Mapping[str, Any], fields: Mapping[str, Any]) -> str:
    """One line per field: its name, level and the first sentence of its description (a list
    of whole descriptions buries the names a question asks about)."""
    out = []
    for key, info in fields.items():
        level, addendum = _level(info)
        obj = objects.get(key, {})
        label = obj.get("name", key)
        description = _first_sentence(obj.get("description", ""))
        if addendum:
            description += " " + _first_sentence(addendum)
        out.append(f"- `{label}` ({level}): {description}".rstrip())
    return "\n".join(out)


def _first_sentence(text: str) -> str:
    line = _one_line(text)
    return re.split(r"(?<=[\w)\]`'\"][.!?])\s+(?=[A-Z`(])", line, maxsplit=1)[0]


def _level(info: Any) -> tuple[str, str]:
    if isinstance(info, str):
        return info.upper(), ""
    if isinstance(info, tuple | list):
        return str(info[0]).upper(), str(info[1]) if len(info) > 1 else ""
    return str(info.get("level", "")).upper(), str(info.get("description_addendum", ""))


def _one_line(text: str) -> str:
    return " ".join(plain(str(text)).split())


def _entity_passages(schema: Any) -> list[Passage]:
    out = []
    for key, entity in schema["objects"]["entities"].items():
        name = entity["name"]
        fmt = "<index>" if entity.get("format") == "index" else "<label>"
        out.append(
            Passage(
                id=f"bids/entity/{key}",
                source="bids",
                title=f"Entities{SEP}{entity['display_name']} ({name}-{fmt})",
                url=f"{BIDS.homepage}appendices/entities.html#{name}",
                text=_one_line(entity["description"]),
            )
        )
    return out


# --- MNE -----------------------------------------------------------------------------------

#: Tutorials that apply to EEG, under tutorials/.
MNE_TUTORIALS = (
    "intro/10_overview.py",
    "intro/20_events_from_raw.py",
    "intro/30_info.py",
    "intro/40_sensor_locations.py",
    "raw/10_raw_overview.py",
    "raw/20_event_arrays.py",
    "raw/30_annotate_raw.py",
    "preprocessing/10_preprocessing_overview.py",
    "preprocessing/15_handling_bad_channels.py",
    "preprocessing/20_rejecting_bad_data.py",
    "preprocessing/25_background_filtering.py",
    "preprocessing/30_filtering_resampling.py",
    "preprocessing/35_artifact_correction_regression.py",
    "preprocessing/40_artifact_correction_ica.py",
    "preprocessing/55_setting_eeg_reference.py",
    "epochs/10_epochs_overview.py",
    "epochs/15_baseline_regression.py",
    "epochs/30_epochs_metadata.py",
    "epochs/60_make_fixed_length_epochs.py",
    "evoked/10_evoked_overview.py",
    "evoked/30_eeg_erp.py",
    "time-freq/10_spectrum_class.py",
    "time-freq/20_sensors_time_frequency.py",
    "stats-sensor-space/10_background_stats.py",
    "stats-sensor-space/20_erp_stats.py",
    "stats-sensor-space/40_cluster_1samp_time_freq.py",
    "stats-sensor-space/75_cluster_ftest_spatiotemporal.py",
    "io/20_reading_eeg_data.py",
)
#: Reference notes under doc/_includes/, shown on the implementation page.
MNE_NOTES = (
    ("channel_interpolation.rst", "Bad channel repair via interpolation"),
    ("units.rst", "Units"),
)


def mne_passages(src_dir: Path) -> list[Passage]:
    """MNE's EEG tutorials, glossary and notes, from an unpacked source release."""
    out: list[Passage] = []
    for name in MNE_TUTORIALS:
        text = rst_to_markdown(gallery_text((src_dir / "tutorials" / name).read_text("utf-8")))
        page = name.removesuffix(".py")
        sections = _sphinx_anchors(markdown_sections(text))
        url = f"{MNE.homepage}auto_tutorials/{page}.html"
        out += section_passages(sections, source="mne", id_prefix=f"mne/{page}", url=url)
    for name, title in MNE_NOTES:
        text = rst_to_markdown((src_dir / "doc" / "_includes" / name).read_text("utf-8"))
        sections = _sphinx_anchors(markdown_sections(text, title=title))
        url = f"{MNE.homepage}documentation/implementation.html"
        out += section_passages(
            sections, source="mne", id_prefix=f"mne/notes/{name.removesuffix('.rst')}", url=url
        )
    glossary = (src_dir / "doc" / "documentation" / "glossary.rst").read_text("utf-8")
    out += glossary_passages(glossary)
    return out


def gallery_text(source: str) -> str:
    """The text of a sphinx-gallery script: its docstring and the comment blocks that start
    its ``# %%`` cells (the code is left out)."""
    tree = ast.parse(source)
    parts = [ast.get_docstring(tree, clean=True) or ""]
    lines = source.splitlines()
    start = tree.body[0].end_lineno if tree.body and ast.get_docstring(tree) else 0
    in_text = False
    block: list[str] = []
    for line in lines[start or 0 :]:
        stripped = line.strip()
        if stripped.startswith("# %%") or re.fullmatch(r"#{20,}", stripped):
            in_text = True
            title = stripped[4:].strip() if stripped.startswith("# %%") else ""
            if title:
                block.append(f"{title}\n{'-' * len(title)}")
            continue
        if in_text and (stripped == "#" or line.startswith("# ")):
            block.append(line[2:] if line.startswith("# ") else "")
            continue
        if block:
            parts.append("\n".join(block))
            block = []
        in_text = False
    if block:
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _sphinx_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _sphinx_anchors(sections: list[Section]) -> list[Section]:
    # Sphinx makes ids from headings differently from mkdocs; the page title has none.
    for s in sections:
        s.anchor = _sphinx_id(s.path[-1]) if len(s.path) > 1 else ""
    return sections


def glossary_passages(text: str) -> list[Passage]:
    """One passage per term of MNE's ``.. glossary::`` (terms at 4 spaces, definitions at 8)."""
    body = text.split(".. glossary::", 1)[-1]
    entries: list[tuple[list[str], list[str]]] = []
    for line in body.splitlines():
        if re.match(r"^ {4}\S", line) and not line.strip().startswith(":"):
            if entries and not entries[-1][1]:
                entries[-1][0].append(line.strip())  # another name for the same term
            else:
                entries.append(([line.strip()], []))
        elif re.match(r"^ {8}", line) and entries:
            entries[-1][1].append(line.strip())
    out = []
    for names, definition in entries:
        text = " ".join(rst_inline(" ".join(definition)).split())
        if not text:
            continue
        name = names[0]
        title = name if len(names) == 1 else f"{name} ({', '.join(names[1:])})"
        out.append(
            Passage(
                id=f"mne/glossary/{_sphinx_id(name)}",
                source="mne",
                title=f"MNE glossary{SEP}{title}",
                url=f"{MNE.homepage}documentation/glossary.html#term-{_sphinx_id(name)}",
                text=text,
            )
        )
    return out


__all__ = [
    "ARCHIVES",
    "BIDS",
    "MNE",
    "Archive",
    "bids_passages",
    "braino_passages",
    "gallery_text",
    "glossary_passages",
    "mne_passages",
    "tool_passages",
]
