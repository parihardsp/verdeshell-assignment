"""Document text extraction service utilizing PyMuPDF4LLM and PyMuPDF.

Extracts page-by-page text preserving Markdown tables, headings, lists,
and paragraph boundaries for optimal RAG semantic chunking and retrieval.
"""

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents text extracted from an individual page or section."""

    page_number: int
    text: str


@dataclass
class ExtractionResult:
    """Unified container for extracted document text and structural metadata."""

    file_name: str
    file_type: str
    total_pages: int
    pages: list[ExtractedPage]
    full_text: str


class DocumentExtractor:
    """Extracts structured text content and page-level metadata from diverse file formats."""

    @classmethod
    def extract_from_path(cls, file_path: Path) -> ExtractionResult:
        """Extract text from a saved disk file path based on its extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return cls.extract_pdf(file_path)
        elif suffix == ".csv":
            return cls.extract_csv(file_path)
        else:
            return cls.extract_plain_text(file_path)

    @classmethod
    def extract_pdf(cls, file_path: Path) -> ExtractionResult:
        """Extract page-by-page text from PDF preserving tables, paragraphs, and headings."""
        # 1. Primary: PyMuPDF4LLM structured Markdown extraction
        try:
            md_pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
            pages: list[ExtractedPage] = []
            full_text_parts = []

            for idx, page_item in enumerate(md_pages):
                page_text = page_item.get("text", "").strip()
                page_num = page_item.get("metadata", {}).get("page_number", idx + 1)
                if page_text:
                    pages.append(ExtractedPage(page_number=page_num, text=page_text))
                    full_text_parts.append(page_text)

            if pages:
                logger.info(
                    "PyMuPDF4LLM extracted %d structured Markdown page(s) from '%s'.",
                    len(pages),
                    file_path.name,
                )
                return ExtractionResult(
                    file_name=file_path.name,
                    file_type="pdf",
                    total_pages=len(pages),
                    pages=pages,
                    full_text="\n\n".join(full_text_parts),
                )
        except Exception as exc:
            logger.warning(
                "PyMuPDF4LLM extraction failed on %s (%s). Falling back to native table-aware extraction.",
                file_path.name,
                exc,
            )

        # 2. Fallback: Native PyMuPDF with table detection & coordinate-sorted blocks
        try:
            pages = []
            full_text_parts = []
            with pymupdf.open(str(file_path)) as doc:
                total_pages = len(doc)
                for page_index in range(total_pages):
                    page = doc[page_index]
                    page_num = page_index + 1
                    page_content_parts = []

                    # Check for tables on page
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            for tab in tabs.tables:
                                md_table = tab.to_markdown()
                                if md_table:
                                    page_content_parts.append(md_table)
                    except Exception:
                        pass

                    # Extract layout-ordered text blocks (sorted top-to-bottom, left-to-right)
                    blocks = page.get_text("blocks")
                    # Filter and sort text blocks by vertical coordinate (y0), then horizontal (x0)
                    text_blocks = [b for b in blocks if b[4].strip() and b[6] == 0]
                    text_blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
                    sorted_text = "\n\n".join(b[4].strip() for b in text_blocks)

                    if sorted_text:
                        page_content_parts.append(sorted_text)

                    final_page_text = "\n\n".join(page_content_parts).strip()
                    if final_page_text:
                        pages.append(ExtractedPage(page_number=page_num, text=final_page_text))
                        full_text_parts.append(final_page_text)

            if not full_text_parts:
                raise ValidationError(
                    f"PDF '{file_path.name}' contains no readable text or is image-only."
                )

            return ExtractionResult(
                file_name=file_path.name,
                file_type="pdf",
                total_pages=total_pages,
                pages=pages,
                full_text="\n\n".join(full_text_parts),
            )
        except ValidationError:
            raise
        except Exception as exc:
            logger.error("PyMuPDF fallback failed to parse %s: %s", file_path, exc)
            raise ValidationError(f"Failed to extract text from PDF '{file_path.name}': {exc!s}")

    @classmethod
    def extract_csv(cls, file_path: Path) -> ExtractionResult:
        """Extract CSV data and convert directly into a structured Markdown table."""
        raw_bytes = file_path.read_bytes()
        text = ""
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise ValidationError(f"Could not decode CSV '{file_path.name}'.")

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise ValidationError(f"CSV '{file_path.name}' is empty.")

        # Convert to Markdown Table
        header = rows[0]
        md_table_lines = [
            "| " + " | ".join(h.strip() for h in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in rows[1:]:
            # Pad or truncate row to header length
            padded = row + [""] * (len(header) - len(row))
            md_table_lines.append(
                "| " + " | ".join(c.strip() for c in padded[: len(header)]) + " |"
            )

        formatted_table = "\n".join(md_table_lines)
        return ExtractionResult(
            file_name=file_path.name,
            file_type="csv",
            total_pages=1,
            pages=[ExtractedPage(page_number=1, text=formatted_table)],
            full_text=formatted_table,
        )

    @classmethod
    def extract_plain_text(cls, file_path: Path) -> ExtractionResult:
        """Extract text from plain text or markdown files."""
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        raw_bytes = file_path.read_bytes()
        text = ""

        for enc in encodings:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise ValidationError(
                f"Could not decode text in '{file_path.name}' with supported encodings."
            )

        clean_text = text.strip()
        return ExtractionResult(
            file_name=file_path.name,
            file_type=file_path.suffix.lstrip("."),
            total_pages=1,
            pages=[ExtractedPage(page_number=1, text=clean_text)],
            full_text=clean_text,
        )
