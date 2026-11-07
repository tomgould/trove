"""
Markdown handling for ingestion.

Markdown files skip the generic Unstructured parser and go through this module:

- ``split_frontmatter`` reads a leading YAML frontmatter block. Its fields become
  chunk metadata (see ``frontmatter_metadata``), and the block is removed from
  the text that is indexed. Bad frontmatter never drops a file: it is logged
  and the whole file is indexed as text.
- ``strip_images`` turns ``![alt](url)`` into ``alt`` and drops lines that hold
  only images, so image URLs do not pollute keyword or vector search.
- ``split_sections`` cuts the text at headings and records each piece's heading
  path ("Page title > H1 > H2 > H3"), which is stored as the chunk's ``section``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import re

import yaml

logger = logging.getLogger(__name__)

MARKDOWN_EXTENSIONS = (".md", ".markdown")

# Frontmatter fields copied into chunk metadata. The page's own content hash is
# stored as page_content_hash, because content_hash already holds the chunk's hash.
FRONTMATTER_FIELDS = {
    "title": "title",
    "url": "url",
    "date": "date",
    "domain": "domain",
    "page_type": "page_type",
    "scraped_at": "scraped_at",
    "content_hash": "page_content_hash",
}

SECTION_SEPARATOR = " > "

# A section shorter than this is merged into the chunk before it, so a short
# headed section ("## Install" plus one line) is not dropped as a too-short chunk.
MIN_SECTION_CHARS = 80

# The stored page title is cut to this length; it is copied into every chunk.
MAX_TITLE_CHARS = 300

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*$")
_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")

# ![alt](src "title"), with the alt allowed to be empty
_INLINE_IMAGE = r"!\[([^\]]*)\]\((?:[^()\s]|\([^()]*\))*(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
# ![alt][ref] and ![alt][]
_REF_IMAGE = r"!\[([^\]]*)\]\[[^\]]*\]"
_LINKED_IMAGE_RE = re.compile(r"\[\s*" + _INLINE_IMAGE + r"\s*\]\([^)]*\)")
_IMAGE_RE = re.compile(_INLINE_IMAGE + "|" + _REF_IMAGE)
_IMAGE_ONLY_LINE_RE = re.compile(
    r"^\s*(?:(?:\[\s*" + _INLINE_IMAGE + r"\s*\]\([^)]*\)|" + _INLINE_IMAGE + "|" + _REF_IMAGE + r")\s*)+$"
)


def split_frontmatter(text: str, source: str = "") -> tuple[dict, str]:
    """Split a leading ``---`` YAML block from ``text``.

    Returns (frontmatter, body). With no frontmatter, or frontmatter that is not
    valid YAML mapping, returns ({}, text) so the whole file is still indexed.
    """
    stripped = text.lstrip("﻿")
    lines = stripped.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    for end in range(1, len(lines)):
        if lines[end].strip() in ("---", "..."):
            break
    else:
        logger.warning("Frontmatter in %s has no closing '---'; indexing the whole file", source or "document")
        return {}, text

    block = "\n".join(lines[1:end])
    try:
        data = yaml.safe_load(block) if block.strip() else {}
    except yaml.YAMLError as e:
        logger.warning("Could not parse frontmatter in %s (%s); indexing the whole file", source or "document", e)
        return {}, text
    if data is None:
        data = {}
    if not isinstance(data, dict):
        logger.warning("Frontmatter in %s is not a mapping; indexing the whole file", source or "document")
        return {}, text

    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return data, body


def _scalar(value):
    """Coerce a frontmatter value to something ChromaDB can store (str, int, float, bool)."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _text_value(value) -> str | None:
    """A frontmatter title or url as a string, or None when it is not usable.

    Only plain strings and numbers are used; NaN, infinity, booleans, lists,
    mappings, dates and blank strings are ignored.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip() or None
    return None


def clean_title(title: str | None) -> str | None:
    """A page title trimmed and cut to MAX_TITLE_CHARS, or None when empty."""
    if not title:
        return None
    return title.strip()[:MAX_TITLE_CHARS].strip() or None


def frontmatter_metadata(frontmatter: dict) -> dict:
    """The known frontmatter fields, renamed and coerced for chunk metadata.

    ``title`` and ``url`` are always strings (see ``_text_value``), and the
    title is cut to MAX_TITLE_CHARS characters.
    """
    meta = {}
    for key, meta_key in FRONTMATTER_FIELDS.items():
        value = frontmatter.get(key)
        if key in ("title", "url"):
            value = _text_value(value)
            if key == "title":
                value = clean_title(value)
            if value is not None:
                meta[meta_key] = value
            continue
        if value is None or value == "":
            continue
        meta[meta_key] = _scalar(value)
    return meta


def _fence_state(line: str, fence: tuple[str, int] | None) -> tuple[bool, tuple[str, int] | None]:
    """Track fenced code blocks one line at a time.

    ``fence`` is None outside a fence, or (character, length) of the opening
    fence. Returns (is_fence_line, new_fence). A fence closes only with the same
    character repeated at least as many times, and nothing else on the line;
    a fence that never closes runs to the end of the text.
    """
    if fence is None:
        m = _FENCE_RE.match(line)
        if m:
            return True, (m.group(1)[0], len(m.group(1)))
        return False, None
    m = _FENCE_CLOSE_RE.match(line)
    if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
        return True, None
    return False, fence


def strip_images(text: str) -> str:
    """Replace Markdown images with their alt text and drop image-only lines."""
    out = []
    fence = None
    for line in text.split("\n"):
        is_fence, fence = _fence_state(line, fence)
        if is_fence or fence is not None:
            out.append(line)
            continue
        if _IMAGE_ONLY_LINE_RE.match(line):
            continue
        line = _LINKED_IMAGE_RE.sub(lambda m: m.group(1), line)
        line = _IMAGE_RE.sub(lambda m: m.group(1) if m.group(1) is not None else (m.group(2) or ""), line)
        out.append(line)
    return "\n".join(out)


def _clean_heading(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> text
    text = re.sub(r"[*_`]+", "", text)
    return text.strip()


def split_sections(text: str, title: str | None = None) -> list[tuple[list[str], str]]:
    """Cut Markdown at headings.

    Returns a list of (heading_path, section_text). heading_path starts with
    ``title`` when given; a first H1 that repeats the title is not repeated.
    Headings inside fenced code blocks are ignored. Both ``#`` headings and
    underlined (``===`` / ``---``) headings are recognised.
    """
    lines = text.split("\n")
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    fence = None

    def path() -> list[str]:
        names = [name for _, name in stack]
        if title:
            if names and names[0].casefold() == title.casefold():
                names = names[1:]
            return [title] + names
        return names

    def start_section(level: int, name: str, heading_lines: list[str]):
        nonlocal current
        if any(line.strip() for line in current):
            sections.append((path(), current))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        current = list(heading_lines)

    i = 0
    while i < len(lines):
        line = lines[i]
        is_fence, fence = _fence_state(line, fence)
        if is_fence:
            current.append(line)
            i += 1
            continue
        if fence is None:
            m = _ATX_RE.match(line)
            if m and m.group(2):
                start_section(len(m.group(1)), _clean_heading(m.group(2)), [line])
                i += 1
                continue
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if (
                nxt is not None
                and line.strip()
                and _SETEXT_RE.match(nxt)
                and not _ATX_RE.match(line)
                and not line.lstrip().startswith((">", "-", "*", "+", "|"))
            ):
                level = 1 if nxt.strip().startswith("=") else 2
                start_section(level, _clean_heading(line), [line, nxt])
                i += 2
                continue
        current.append(line)
        i += 1

    if any(line.strip() for line in current):
        sections.append((path(), current))

    return [(p, "\n".join(body).strip()) for p, body in sections]


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def chunk_markdown(
    text: str,
    title: str | None = None,
    max_chars: int = 1500,
    overlap: int = 100,
) -> list[tuple[str, str]]:
    """Split Markdown into chunks that follow its headings.

    Each chunk is (content, section) where section is the heading path joined
    with " > ". Long sections are split further with the recursive splitter and
    keep their section; a short section is merged into the chunk before it when
    both fit in ``max_chars``, and the merged chunk gets the headings they share.
    """
    from core.chunking import chunk_by_recursive

    pieces: list[tuple[list[str], str]] = []
    for path, body in split_sections(text, title=title):
        if len(body) <= max_chars:
            pieces.append((path, body))
        else:
            for part in chunk_by_recursive(body, max_chars=max_chars, overlap=overlap):
                pieces.append((path, part))

    merged: list[tuple[list[str], str]] = []
    for path, body in pieces:
        if merged:
            prev_path, prev_body = merged[-1]
            if len(body) < MIN_SECTION_CHARS and len(prev_body) + len(body) + 2 <= max_chars:
                merged[-1] = (_common_prefix(prev_path, path) or prev_path[:1], prev_body + "\n\n" + body)
                continue
        merged.append((path, body))

    return [(body, SECTION_SEPARATOR.join(path)) for path, body in merged if body.strip()]


def first_heading(text: str) -> str | None:
    """The text of the first level-1 heading, used as the page title when frontmatter has none."""
    lines = text.split("\n")
    fence = None
    for i, line in enumerate(lines):
        is_fence, fence = _fence_state(line, fence)
        if is_fence or fence is not None:
            continue
        m = _ATX_RE.match(line)
        if m and len(m.group(1)) == 1 and m.group(2):
            return _clean_heading(m.group(2))
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and not m and re.match(r"^\s{0,3}=+\s*$", nxt):
            return _clean_heading(line)
    return None
