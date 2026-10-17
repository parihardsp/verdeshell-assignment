"""Unit tests for FileValidationService."""

import pytest

from app.core.exceptions import PayloadTooLargeError, ValidationError
from app.ingestion.validator import FileValidationService


def test_sanitize_and_hash():
    title, content, sha = FileValidationService.validate_content(
        "  Policy Title  ", "Content with\x00 null bytes and\r\n carriage returns. "
    )
    assert title == "Policy Title"
    assert "\x00" not in content
    assert "\r" not in content
    assert len(sha) == 64  # Valid SHA-256 hex string


def test_empty_title_rejection():
    with pytest.raises(ValidationError, match="title cannot be empty"):
        FileValidationService.validate_content("   ", "Valid content")


def test_empty_content_rejection():
    with pytest.raises(ValidationError, match="content cannot be empty"):
        FileValidationService.validate_content("Valid Title", "   ")


def test_oversized_payload_rejection():
    huge_content = "A" * (6 * 1024 * 1024)  # 6MB
    with pytest.raises(PayloadTooLargeError, match="exceeds the 5MB limit"):
        FileValidationService.validate_content("Title", huge_content)
