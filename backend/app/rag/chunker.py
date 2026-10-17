"""Context-preserving and table-aware text chunker implementation.

Includes BaseChunker, TextChunk, and RecursiveTextChunker.
Features:
- Preserves Markdown table integrity: rows are never sliced in half.
- Header replication: multi-chunk tables automatically retain column headers.
- Paragraph and sentence awareness for clean semantic prose boundaries.
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a generated text chunk with positional metadata."""

    index: int
    content: str
    char_start: int
    char_end: int
    page_number: int = 1


def is_markdown_table_row(line: str) -> bool:
    """Check if a line represents a Markdown table row."""
    s = line.strip()
    return len(s) >= 2 and s.startswith("|") and s.endswith("|")


def is_markdown_table_sep(line: str) -> bool:
    """Check if a line represents a Markdown table header delimiter row (e.g. |---|---|)."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    inner = s.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return len(inner) == 0


class BaseChunker(ABC):
    """Abstract interface for document chunking algorithms."""

    @abstractmethod
    def chunk_text(self, text: str, start_index: int = 0, page_number: int = 1) -> list[TextChunk]:
        """Split input text into chunks preserving positional offsets and page numbers."""


class RecursiveTextChunker(BaseChunker):
    """Splits documents into overlapping chunks preserving structural context.

    Uses a table-aware and recursive fallback strategy:
    - Tables: rows are kept intact; continuation chunks replicate table headers.
    - Prose: splits on double newlines (paragraphs), single newlines, sentences, and words.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, start_index: int = 0, page_number: int = 1) -> list[TextChunk]:
        """Split input text into overlapping chunks with character offsets."""
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        text_len = len(clean_text)

        if text_len <= self.chunk_size:
            return [
                TextChunk(
                    index=start_index,
                    content=clean_text,
                    char_start=0,
                    char_end=text_len,
                    page_number=page_number,
                )
            ]

        lines = clean_text.split("\n")
        has_table = any(is_markdown_table_row(row_line) for row_line in lines)

        if has_table:
            return self._chunk_table_aware(clean_text, lines, start_index, page_number)
        else:
            return self._chunk_prose(clean_text, start_index, page_number)

    def _chunk_table_aware(
        self, clean_text: str, lines: list[str], start_index: int, page_number: int
    ) -> list[TextChunk]:
        """Chunk text preserving Markdown table rows and replicating headers across boundaries."""
        chunks: list[TextChunk] = []
        chunk_idx = start_index

        current_block: list[str] = []
        current_len = 0
        in_table = False
        table_header: list[str] = []
        block_start = 0

        for line in lines:
            stripped = line.strip()
            is_row = is_markdown_table_row(stripped)

            if is_row:
                if not in_table:
                    in_table = True
                    table_header = [line]
                elif len(table_header) == 1 and is_markdown_table_sep(stripped):
                    table_header.append(line)

                row_len = len(line) + 1
                if current_len + row_len > self.chunk_size and current_block:
                    content_str = "\n".join(current_block).strip()
                    if content_str:
                        chunks.append(
                            TextChunk(
                                index=chunk_idx,
                                content=content_str,
                                char_start=block_start,
                                char_end=block_start + len(content_str),
                                page_number=page_number,
                            )
                        )
                        chunk_idx += 1
                        block_start += len(content_str)

                    # Replicate table header for continuation chunk
                    current_block = list(table_header) if len(table_header) == 2 else []
                    current_len = sum(len(h) + 1 for h in current_block)

                current_block.append(line)
                current_len += row_len
            else:
                in_table = False
                table_header = []
                line_len = len(line) + 1
                if current_len + line_len > self.chunk_size and current_block:
                    content_str = "\n".join(current_block).strip()
                    if content_str:
                        chunks.append(
                            TextChunk(
                                index=chunk_idx,
                                content=content_str,
                                char_start=block_start,
                                char_end=block_start + len(content_str),
                                page_number=page_number,
                            )
                        )
                        chunk_idx += 1
                        block_start += len(content_str)
                    current_block = []
                    current_len = 0

                current_block.append(line)
                current_len += line_len

        if current_block:
            content_str = "\n".join(current_block).strip()
            if content_str:
                chunks.append(
                    TextChunk(
                        index=chunk_idx,
                        content=content_str,
                        char_start=block_start,
                        char_end=block_start + len(content_str),
                        page_number=page_number,
                    )
                )

        return chunks

    def _chunk_prose(self, clean_text: str, start_index: int, page_number: int) -> list[TextChunk]:
        """Standard recursive splitting for prose, articles, and unstructured text."""
        text_len = len(clean_text)
        chunks: list[TextChunk] = []
        start = 0
        chunk_idx = start_index

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                window = clean_text[start:end]
                split_point = -1

                for separator in ["\n\n", "\n", ". ", "? ", "! ", " "]:
                    pos = window.rfind(separator)
                    if pos > (self.chunk_size // 2):
                        split_point = pos + len(separator)
                        break

                if split_point != -1:
                    end = start + split_point

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunks.append(
                    TextChunk(
                        index=chunk_idx,
                        content=chunk_content,
                        char_start=start,
                        char_end=end,
                        page_number=page_number,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break

            step = max(1, (end - start) - self.chunk_overlap)
            start += step

        return chunks


# Alias for backward compatibility
TextChunker = RecursiveTextChunker

sys.modules["app.rag.chunking"] = sys.modules[__name__]
sys.modules["app.rag.chunking.base"] = sys.modules[__name__]
sys.modules["app.rag.chunking.recursive"] = sys.modules[__name__]
