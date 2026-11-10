"""Deterministic local chunking for Markdown knowledge documents."""

import hashlib
import re

from customer_support_rag.knowledge.models import KnowledgeChunk, KnowledgeDocument

_LINE_BREAK = r"\r\n|(?<!\r)\n|\r(?!\n)"
_NEWLINE = re.compile(_LINE_BREAK)
_PARAGRAPH_END = re.compile(rf"(?:[ \t]*(?:{_LINE_BREAK})){{2,}}")
_ATX = re.compile(r"^( {0,3})(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_OPEN_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _line_ranges(text: str) -> list[tuple[int, int, str]]:
    """Return line offsets and contents without their newline characters."""
    ranges: list[tuple[int, int, str]] = []
    offset = 0
    for match in _NEWLINE.finditer(text):
        end = match.end()
        ranges.append((offset, end, text[offset : match.start()]))
        offset = end
    if offset < len(text):
        ranges.append((offset, len(text), text[offset:]))
    return ranges


def _heading_label(match: re.Match[str]) -> str:
    label = (match.group(3) or "").strip(" \t")
    if re.fullmatch(r"#+[ \t]*", label):
        return ""
    label = re.sub(r"[ \t]+#+[ \t]*$", "", label).strip(" \t")
    return label


def _sections(text: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """Find supported ATX sections while ignoring headings in fenced blocks."""
    lines = _line_ranges(text)
    headings: list[tuple[int, tuple[str, ...]]] = []
    stack: list[tuple[int, str]] = []
    fence_char = ""
    fence_size = 0

    for start, _end, line in lines:
        if fence_char:
            close = re.match(r"^ {0,3}(`+|~+)[ \t]*$", line)
            if (
                close
                and close.group(1)[0] == fence_char
                and len(close.group(1)) >= fence_size
            ):
                fence_char = ""
                fence_size = 0
            continue

        opening = _OPEN_FENCE.match(line)
        if opening:
            fence_char = opening.group(1)[0]
            fence_size = len(opening.group(1))
            continue

        heading = _ATX.match(line)
        if not heading:
            continue
        level = len(heading.group(2))
        stack = [(depth, label) for depth, label in stack if depth < level]
        stack.append((level, _heading_label(heading)))
        headings.append((start, tuple(label for _, label in stack)))

    if not headings:
        return [(0, len(text), ())] if text else []

    sections: list[tuple[int, int, tuple[str, ...]]] = []
    if headings[0][0] > 0:
        sections.append((0, headings[0][0], ()))
    for index, (start, path) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        sections.append((start, end, path))
    return sections


def _safe_hard_end(text: str, start: int, limit: int) -> int:
    end = _safe_cut(text, start, min(start + limit, len(text)))
    return end


def _safe_cut(text: str, start: int, end: int) -> int:
    """Move any candidate cut before the LF when it would split a CRLF pair."""
    if end < len(text) and end > start and text[end - 1 : end + 1] == "\r\n":
        end -= 1
    return end


def _candidate_ends(
    pattern: re.Pattern[str], text: str, start: int, limit: int
) -> list[int]:
    """Return boundaries within one chunk, with lookahead to keep CRLF intact."""
    ends: list[int] = []
    search_end = min(limit + 1, len(text))
    for match in pattern.finditer(text, start, search_end):
        if match.end() > limit:
            break
        ends.append(match.end())
    return ends


def _split_section(
    text: str, start: int, end: int, max_chars: int
) -> list[tuple[int, int]]:
    """Split a section at paragraph, line, then safe hard boundaries."""
    result: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        if end - cursor <= max_chars:
            result.append((cursor, end))
            break
        limit = cursor + max_chars
        paragraph_ends = _candidate_ends(_PARAGRAPH_END, text, cursor, limit)
        if paragraph_ends:
            next_cursor = paragraph_ends[-1]
        else:
            line_ends = _candidate_ends(_NEWLINE, text, cursor, limit)
            if line_ends:
                next_cursor = line_ends[-1]
            else:
                next_cursor = _safe_hard_end(text, cursor, max_chars)
        next_cursor = _safe_cut(text, cursor, next_cursor)
        if next_cursor <= cursor:
            next_cursor = _safe_hard_end(text, cursor, max_chars)
        result.append((cursor, next_cursor))
        cursor = next_cursor
    return result


def chunk_document(
    document: KnowledgeDocument, *, max_chars: int = 1200
) -> list[KnowledgeChunk]:
    """Split a document body without changing its text or reading external data.

    The character limit counts Python string code points, not model tokens.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2:
        raise ValueError("max_chars must be an integer of at least 2")

    body = document.body
    if not body:
        return []
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    chunks: list[KnowledgeChunk] = []
    for section_start, section_end, heading_path in _sections(body):
        for start, end in _split_section(body, section_start, section_end, max_chars):
            chunks.append(
                KnowledgeChunk(
                    metadata=document.metadata,
                    body_sha256=digest,
                    index=len(chunks),
                    char_start=start,
                    char_end=end,
                    text=body[start:end],
                    heading_path=heading_path,
                )
            )
    return chunks
