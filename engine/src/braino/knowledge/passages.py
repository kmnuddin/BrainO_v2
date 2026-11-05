"""Documentation passages, the units the search returns, and how documents are cut into them.

A document is cut at its headings; a section longer than :data:`MAX_WORDS` is cut again
between paragraphs, so that three passages fit in one tool result. Each passage keeps its
heading path as its title (Pipelines, then Plan checks: see :data:`SEP`), which tells the
model (and the search) what the text is about even when the text alone does not say.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

Source = Literal["braino", "bids", "mne"]
SOURCES: tuple[Source, ...] = ("braino", "bids", "mne")

#: Words in a passage at most (about 250 tokens), and a lead kept with the block after it.
MAX_WORDS = 170
LEAD_WORDS = 50
#: Passages shorter than this (a heading with a line under it) are left out.
MIN_WORDS = 6
#: Between the headings of a title (a single right-pointing angle quotation mark).
SEP = " › "  # noqa: RUF001


class Passage(BaseModel):
    id: str
    source: Source
    title: str
    url: str
    text: str

    @property
    def digest(self) -> str:
        """Changes when the passage does: its vector is reused while this stays the same."""
        return hashlib.sha256(self.embed_text().encode()).hexdigest()[:16]

    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass
class Section:
    path: list[str]
    anchor: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return SEP.join(self.path)


# --- Markdown ------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def slug(text: str) -> str:
    """The anchor mkdocs (Python-Markdown) gives a heading."""
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE).strip()
    return re.sub(r"[\s]+", "-", text)


def markdown_sections(text: str, *, title: str | None = None) -> list[Section]:
    """Sections of a Markdown document, one per heading. ``title`` replaces the document's
    first-level heading in the paths."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current = Section(path=[title] if title else [], anchor="")
    lines: list[str] = []
    in_code = False

    def flush() -> None:
        current.paragraphs = paragraphs("\n".join(lines))
        if current.paragraphs:
            sections.append(current)
        lines.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        heading = None if in_code else _HEADING.match(line)
        if heading is None:
            lines.append(line)
            continue
        flush()
        level, name = len(heading.group(1)), _plain_heading(heading.group(2))
        stack = [(lvl, n) for lvl, n in stack if lvl < level] + [(level, name)]
        names = [n for _, n in stack]
        if title:
            names = [title] + [n for lvl, n in stack if lvl > 1]
        current = Section(path=names, anchor=slug(name) if level > 1 or not title else "")
    flush()
    return sections


def _plain_heading(text: str) -> str:
    return plain(text).replace("`", "").strip("* ")


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks; a fenced code block stays one block."""
    blocks: list[str] = []
    block: list[str] = []
    in_code = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
        if not line.strip() and not in_code:
            if block:
                blocks.append("\n".join(block).strip("\n"))
                block = []
            continue
        block.append(line.rstrip())
    if block:
        blocks.append("\n".join(block).strip("\n"))
    return [b for b in blocks if b.strip()]


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_LINK_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def plain(text: str) -> str:
    """Markdown without link targets, images, HTML comments and reference definitions."""
    text = _COMMENT.sub("", text)
    text = _LINK_DEF.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    return _REF_LINK.sub(r"\1", text)


# --- cutting sections into passages --------------------------------------------------------


def words(text: str) -> int:
    return len(text.split())


def chunks(blocks: list[str], max_words: int = MAX_WORDS) -> Iterator[str]:
    """Join blocks into chunks of at most ``max_words`` words; a longer block is cut between
    its lines, then its sentences. A short lead ("The rules are:") stays with the block after
    it, so a chunk may reach ``max_words + LEAD_WORDS``."""
    current: list[str] = []
    size = 0
    for block in (piece for b in blocks for piece in _split(b, max_words)):
        n = words(block)
        if current and size + n > max_words and size >= LEAD_WORDS:
            yield "\n\n".join(current)
            current, size = [], 0
        current.append(block)
        size += n
    if current:
        yield "\n\n".join(current)


def _split(block: str, max_words: int) -> list[str]:
    if words(block) <= max_words:
        return [block]
    lines = block.splitlines()
    if len(lines) > 1:
        return [c.replace("\n\n", "\n") for c in chunks(lines, max_words)]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    if len(sentences) > 1:
        return [c.replace("\n\n", " ") for c in chunks(sentences, max_words)]
    split = block.split()
    return [" ".join(split[i : i + max_words]) for i in range(0, len(split), max_words)]


def section_passages(
    sections: list[Section], *, source: Source, id_prefix: str, url: str
) -> list[Passage]:
    """Passages of a document's sections. ``url`` is the page; a section's anchor is added."""
    out: list[Passage] = []
    seen: dict[str, int] = {}
    for section in sections:
        base = f"{id_prefix}#{section.anchor}" if section.anchor else id_prefix
        for text in chunks(section.paragraphs):
            if words(text) < MIN_WORDS:
                continue
            n = seen.get(base, 0)
            seen[base] = n + 1
            out.append(
                Passage(
                    id=base if n == 0 else f"{base}~{n}",
                    source=source,
                    title=section.title,
                    url=f"{url}#{section.anchor}" if section.anchor else url,
                    text=text.strip(),
                )
            )
    return out


# --- reStructuredText ----------------------------------------------------------------------

_UNDERLINE = re.compile(r"^([=\-^~\"'`#*+])\1{2,}\s*$")
_ROLE = re.compile(r":[\w:+.-]+:`([^`]*)`")
_DROP_ROLES = re.compile(r":(?:footcite|cite)(?::[a-z]+)?:`[^`]*`")
_RST_LINK = re.compile(r"`([^`<]+?)\s*(?:<[^>]*>)?`__?")
_LITERAL = re.compile(r"``(.+?)``")
_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$")
_TARGET = re.compile(r"^\s*\.\.\s+(_[^:]+:|\|[^|]+\|).*$")
_ADMONITIONS = {"note", "warning", "tip", "important", "hint", "caution", "attention", "seealso"}


def _role(match: re.Match[str]) -> str:
    inner = match.group(1)
    target = re.match(r"^(.*?)\s*<([^>]*)>$", inner, re.DOTALL)
    if target and target.group(1).strip():
        return target.group(1).strip()
    if target:
        inner = target.group(2)
    if inner.startswith("~"):
        inner = inner[1:].rsplit(".", 1)[-1]
    return f"`{inner}`"


def rst_inline(text: str) -> str:
    text = _DROP_ROLES.sub("", text)
    text = _ROLE.sub(_role, text)
    text = _LITERAL.sub(r"`\1`", text)
    text = _RST_LINK.sub(r"\1", text)
    return re.sub(r"[ \t]+([.,;:)])", r"\1", text)


def rst_to_markdown(text: str) -> str:
    """Enough of reStructuredText as Markdown to cut it into passages: headings, admonitions
    as plain paragraphs, roles and links as their text; images, code and other directives
    are left out."""
    lines = text.splitlines()
    out: list[str] = []
    levels: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # Overlined heading: ===\nTitle\n===
        if (
            _UNDERLINE.match(line)
            and i + 2 < len(lines)
            and _UNDERLINE.match(lines[i + 2])
            and lines[i + 1].strip()
            and not _UNDERLINE.match(lines[i + 1])
        ):
            out.append(_heading(lines[i + 1].strip(), "o" + line[0], levels))
            i += 3
            continue
        if (
            line.strip()
            and not line.startswith(" ")
            and _UNDERLINE.match(nxt)
            and len(nxt.strip()) >= len(line.strip())
        ):
            out.append(_heading(line.strip(), nxt[0], levels))
            i += 2
            continue
        if _TARGET.match(line):
            i += 1
            continue
        directive = _DIRECTIVE.match(line)
        if directive:
            indent, name, arg = directive.groups()
            body, i = _indented_block(lines, i + 1, len(indent))
            if name in _ADMONITIONS:
                label = "See also" if name == "seealso" else name.capitalize()
                body_text = "\n".join(b for b in body if not re.match(r"^\s*:\w[\w-]*:", b))
                out.append(f"{label}: {arg} {_dedent(body_text)}".replace("  ", " ").strip())
            out.append("")  # what follows is another paragraph
            continue
        if re.match(r"^\s*\.\.(\s|$)", line):  # a comment, with its indented lines
            _, i = _indented_block(lines, i + 1, len(line) - len(line.lstrip()))
            out.append("")
            continue
        if line.rstrip().endswith("::") and not line.strip().startswith(".."):
            # A literal block follows: keep the sentence, drop the code.
            out.append(line.rstrip()[:-1].rstrip())
            _, i = _indented_block(lines, i + 1, len(line) - len(line.lstrip()))
            out.append("")
            continue
        out.append(line)
        i += 1
    return rst_inline("\n".join(out))


def _heading(text: str, kind: str, levels: list[str]) -> str:
    if kind not in levels:
        levels.append(kind)
    return "#" * min(levels.index(kind) + 1, 6) + " " + rst_inline(text)


def _indented_block(lines: list[str], start: int, indent: int) -> tuple[list[str], int]:
    """The lines indented more than ``indent`` from ``start`` (blank lines included)."""
    body: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
        i += 1
    while body and not body[-1].strip():
        body.pop()
    return body, i


def _dedent(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
