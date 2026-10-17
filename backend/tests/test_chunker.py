"""Unit tests for TextChunker."""

import pytest

from app.rag.chunker import TextChunker


def test_chunk_short_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    text = "Short policy text."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_chunk_overlapping_long_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "This is paragraph one with enough content to exceed the limit. "
        "Paragraph two continues describing refund procedures and requirements. "
        "Paragraph three adds final operational details for support staff."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert chunk.char_end > chunk.char_start


def test_chunk_empty_text():
    chunker = TextChunker()
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_markdown_table_preserves_rows_and_replicates_header():
    """Verify that table rows are kept intact and subsequent chunks retain column headers."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=20)
    table_text = (
        "| Subject | Grade | Points |\n"
        "| --- | --- | --- |\n"
        "| Mathematics | A1 | 10 |\n"
        "| Science | A2 | 09 |\n"
        "| English | A1 | 10 |\n"
        "| Social Studies | A1 | 10 |"
    )
    chunks = chunker.chunk_text(table_text, page_number=2)
    assert len(chunks) > 1

    # First chunk contains headers
    assert "| Subject | Grade | Points |" in chunks[0].content
    assert "| Mathematics | A1 | 10 |" in chunks[0].content

    # Every continuation chunk must also replicate the table header!
    for chunk in chunks[1:]:
        assert "| Subject | Grade | Points |" in chunk.content
        assert "| --- | --- | --- |" in chunk.content
        assert chunk.page_number == 2


def test_chunk_preserves_page_number():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "First long paragraph here to test page number preservation across chunks. And second sentence follows."
    chunks = chunker.chunk_text(text, start_index=5, page_number=4)
    assert len(chunks) > 1
    assert chunks[0].index == 5
    for c in chunks:
        assert c.page_number == 4
