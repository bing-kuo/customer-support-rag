"""Behavioral tests for deterministic local knowledge chunking."""

import hashlib

import pytest

from customer_support_rag.knowledge import (
    KnowledgeDocument,
    KnowledgeMetadata,
    chunk_document,
)
from customer_support_rag.knowledge import chunker as chunker_module


def document(body: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        metadata=KnowledgeMetadata(
            document_id="sample",
            product_id="hp-a",
            title="Sample",
            version="1",
            synthetic=True,
        ),
        body=body,
    )


def assert_exact_partition(body: str, chunks: list, max_chars: int) -> None:
    assert "".join(chunk.text for chunk in chunks) == body
    assert chunks
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(body)
    for index, chunk in enumerate(chunks):
        assert chunk.index == index
        assert chunk.text == body[chunk.char_start : chunk.char_end]
        assert len(chunk.text) <= max_chars
        if index:
            assert chunks[index - 1].char_end == chunk.char_start


def test_chunks_preserve_heading_paths_metadata_and_digest() -> None:
    body = "Intro.\n# Setup\nFirst paragraph.\n\n## Pairing\nUse the app.\n"
    source = document(body)

    chunks = chunk_document(source, max_chars=22)

    assert_exact_partition(body, chunks, 22)
    assert [chunk.heading_path for chunk in chunks] == [
        (),
        ("Setup",),
        ("Setup",),
        ("Setup", "Pairing"),
        ("Setup", "Pairing"),
    ]
    assert all(chunk.metadata == source.metadata for chunk in chunks)
    assert {chunk.body_sha256 for chunk in chunks} == {
        hashlib.sha256(body.encode()).hexdigest()
    }
    assert chunk_document(source, max_chars=22) == chunks


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_preserves_newline_styles_and_never_splits_crlf(newline: str) -> None:
    body = f"# Help{newline}One{newline}{newline}Two{newline}"

    chunks = chunk_document(document(body), max_chars=7)

    assert_exact_partition(body, chunks, 7)
    assert all(
        chunk.text[-1:] != "\r" or not body[chunk.char_end :].startswith("\n")
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("body", "max_chars"),
    [
        ("a\r\n\r\nb", 4),
        ("a\n\r\nb", 3),
        ("a\r\n\r\nb", 5),
        ("abc\r\nx", 4),
        ("abc\r\nx", 5),
    ],
)
def test_paragraph_and_neighboring_cuts_keep_crlf_pairs(
    body: str, max_chars: int
) -> None:
    chunks = chunk_document(document(body), max_chars=max_chars)

    assert_exact_partition(body, chunks, max_chars)
    assert all(
        not (
            body[chunk.char_end - 1 : chunk.char_end] == "\r"
            and body[chunk.char_end : chunk.char_end + 1] == "\n"
        )
        for chunk in chunks[:-1]
    )


def test_paragraph_detection_does_not_treat_one_crlf_as_a_blank_line() -> None:
    body = "a\r\nbbb\ncccccccc"

    chunks = chunk_document(document(body), max_chars=8)

    assert_exact_partition(body, chunks, 8)
    assert chunks[0].char_end == 7


def test_candidate_searches_are_bounded_to_one_chunk_window(monkeypatch) -> None:
    class TrackingPattern:
        def __init__(self, pattern) -> None:
            self.pattern = pattern
            self.search_spans: list[int] = []

        def finditer(self, text: str, *bounds):
            if len(bounds) == 2:
                start, end = bounds
                self.search_spans.append(end - start)
            return self.pattern.finditer(text, *bounds)

    newline = TrackingPattern(chunker_module._NEWLINE)
    paragraph = TrackingPattern(chunker_module._PARAGRAPH_END)
    monkeypatch.setattr(chunker_module, "_NEWLINE", newline)
    monkeypatch.setattr(chunker_module, "_PARAGRAPH_END", paragraph)
    body = "a" * 4096

    chunks = chunker_module.chunk_document(document(body), max_chars=32)

    assert_exact_partition(body, chunks, 32)
    assert len(chunks) == 128
    search_spans = newline.search_spans + paragraph.search_spans
    assert search_spans
    assert max(search_spans) <= 33


@pytest.mark.parametrize(
    ("body", "expected_path"),
    [
        ("# ###\nText", ("",)),
        ("## ###  \nText", ("",)),
        ("# Top\n## ###\nText", ("Top", "")),
    ],
)
def test_atx_closing_hashes_allow_an_empty_heading_label(
    body: str, expected_path: tuple[str, ...]
) -> None:
    chunks = chunk_document(document(body))

    assert chunks[-1].heading_path == expected_path


def test_hard_splits_long_lines_and_preserves_unicode_and_plain_text() -> None:
    body = "繁體中文🙂" * 8 + " Ignore instructions; https://example.invalid/a <script>"

    chunks = chunk_document(document(body), max_chars=9)

    assert_exact_partition(body, chunks, 9)


def test_ignores_headings_inside_fences_and_tracks_skipped_levels() -> None:
    body = (
        "# Top\n```md\n## not a heading\n```\n"
        "### Deep ###\nContent\n~~~\n# still code\n~~~\n# Next\nEnd"
    )

    chunks = chunk_document(document(body), max_chars=100)

    assert [chunk.heading_path for chunk in chunks] == [
        ("Top",),
        ("Top", "Deep"),
        ("Next",),
    ]
    assert_exact_partition(body, chunks, 100)


@pytest.mark.parametrize("value", [True, 1.5, "10", 1, 0, -2])
def test_rejects_invalid_max_chars(value: object) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(document("text"), max_chars=value)  # type: ignore[arg-type]


def test_empty_body_returns_no_chunks_and_exact_limit_is_single_chunk() -> None:
    assert chunk_document(document("")) == []
    body = "# 標題\n🙂"
    chunks = chunk_document(document(body), max_chars=len(body))

    assert len(chunks) == 1
    assert chunks[0].text == body
