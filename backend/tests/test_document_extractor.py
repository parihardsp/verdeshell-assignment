"""Unit and integration tests for DocumentExtractor across diverse document formats.

Tests extraction of .docx, .doc, .md, .txt, .csv, and verifies binary safety guards.
"""

import tempfile
from pathlib import Path

import docx
import pytest

from app.core.exceptions import ValidationError
from app.ingestion.extractor import DocumentExtractor


def test_extract_synthetic_docx(tmp_path: Path):
    """Test python-docx and PyMuPDF extraction of a freshly generated .docx file."""
    doc_path = tmp_path / "synthetic_test.docx"
    doc = docx.Document()
    doc.add_heading("Company Knowledge Base Policy", level=1)
    doc.add_paragraph("This is the main introduction to our engineering workflow.")
    doc.add_heading("Service Level Agreements", level=2)
    doc.add_paragraph("All high-severity incidents must be acknowledged within 15 minutes.")

    # Add a table
    table = doc.add_table(rows=3, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Severity"
    hdr_cells[1].text = "Response Time"

    row1_cells = table.rows[1].cells
    row1_cells[0].text = "P0 (Critical)"
    row1_cells[1].text = "15 minutes"

    row2_cells = table.rows[2].cells
    row2_cells[0].text = "P1 (High)"
    row2_cells[1].text = "1 hour"

    doc.save(str(doc_path))

    res = DocumentExtractor.extract_from_path(doc_path)
    assert res.file_type == "docx"
    assert res.total_pages >= 1
    assert "Company Knowledge Base Policy" in res.full_text
    assert "15 minutes" in res.full_text
    assert not res.full_text.startswith("PK")


def test_extract_markdown(tmp_path: Path):
    """Test extraction of Markdown files (.md)."""
    md_path = tmp_path / "guide.md"
    md_content = (
        "# Engineering Handbook\n\n"
        "## Principles\n"
        "- Zero hallucination\n"
        "- Hybrid vector search\n\n"
        "| Service | Port |\n"
        "|---|---|\n"
        "| Backend | 8000 |\n"
        "| Frontend | 5173 |\n"
    )
    md_path.write_text(md_content, encoding="utf-8")

    res = DocumentExtractor.extract_from_path(md_path)
    assert res.file_type == "md"
    assert res.total_pages == 1
    assert "Engineering Handbook" in res.full_text
    assert "Zero hallucination" in res.full_text
    assert "8000" in res.full_text


def test_extract_plain_text(tmp_path: Path):
    """Test extraction of Plain Text files (.txt)."""
    txt_path = tmp_path / "notes.txt"
    txt_content = "Customer Support Runbook\n1. Verify identity\n2. Inspect session ID\n3. Resolve ticket"
    txt_path.write_text(txt_content, encoding="utf-8")

    res = DocumentExtractor.extract_from_path(txt_path)
    assert res.file_type == "txt"
    assert res.total_pages == 1
    assert "Customer Support Runbook" in res.full_text
    assert "Verify identity" in res.full_text


def test_extract_csv(tmp_path: Path):
    """Test extraction and Markdown table conversion of CSV files."""
    csv_path = tmp_path / "metrics.csv"
    csv_content = "Metric,Value,Unit\nLatency,120,ms\nRecall,0.96,ratio\nPrecision,0.94,ratio\n"
    csv_path.write_text(csv_content, encoding="utf-8")

    res = DocumentExtractor.extract_from_path(csv_path)
    assert res.file_type == "csv"
    assert res.total_pages == 1
    assert "| Metric | Value | Unit |" in res.full_text
    assert "| Latency | 120 | ms |" in res.full_text


def test_binary_pk_zip_guard_routing(tmp_path: Path):
    """Verify that a DOCX file disguised or misnamed as .txt is NOT decoded as raw latin-1 bytes."""
    doc_path = tmp_path / "disguised.txt"
    doc = docx.Document()
    doc.add_heading("Disguised Document", level=1)
    doc.add_paragraph("This is hidden text inside a zip container.")
    doc.save(str(doc_path))

    # Should detect PK magic bytes and route to extract_docx, NOT raw latin-1 decode!
    res = DocumentExtractor.extract_from_path(doc_path)
    assert not res.full_text.startswith("PK")
    assert "Disguised Document" in res.full_text


def test_binary_garbage_rejection(tmp_path: Path):
    """Verify that arbitrary binary garbage in a .txt file is rejected with ValidationError."""
    bin_path = tmp_path / "corrupt.txt"
    # Write random binary bytes with high non-printable byte density
    bin_path.write_bytes(bytes([i % 256 for i in range(1000)]))

    with pytest.raises(ValidationError, match="unreadable binary"):
        DocumentExtractor.extract_plain_text(bin_path)


def test_extract_real_uploaded_docx():
    """Verify extraction on real uploaded .docx files if present on disk."""
    candidates = list(Path("../data/uploads").glob("**/*.docx"))
    if not candidates:
        pytest.skip("Sample docx file not found in data/uploads")

    sample_file = candidates[0]
    res = DocumentExtractor.extract_from_path(sample_file)
    assert res.file_type == "docx"
    assert res.total_pages >= 1
    assert not res.full_text.startswith("PK")
    assert len(res.full_text) > 50
