"""File and content validation service.

Enforces security guardrails, payload size limits, UTF-8 normalization,
null-byte stripping, and content deduplication via SHA-256 hashing.
"""

import hashlib
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import PayloadTooLargeError, ValidationError

_settings = get_settings()


class FileValidationService:
    """Security guardrail service for validating document contents."""

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Strip dangerous null bytes, carriage returns, and normalize unicode whitespace."""
        if "\x00" in text:
            text = text.replace("\x00", "")
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    @staticmethod
    def compute_sha256(content: str) -> str:
        """Calculate the SHA-256 hex digest of the normalized content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def validate_content(cls, title: str, content: str) -> tuple[str, str, str]:
        """Validate title and raw text content against length and security constraints.

        Returns:
            Tuple of (sanitized_title, sanitized_content, sha256_hash)
        """
        sanitized_title = cls.sanitize_text(title)
        if not sanitized_title:
            raise ValidationError("Document title cannot be empty.")
        if len(sanitized_title) > 255:
            raise ValidationError("Document title cannot exceed 255 characters.")

        sanitized_content = cls.sanitize_text(content)
        if not sanitized_content:
            raise ValidationError("Document content cannot be empty.")

        content_bytes_len = len(sanitized_content.encode("utf-8"))
        if content_bytes_len > _settings.MAX_FILE_SIZE_BYTES:
            raise PayloadTooLargeError(
                f"Document content size ({content_bytes_len} bytes) exceeds the 5MB limit "
                f"({_settings.MAX_FILE_SIZE_BYTES} bytes)."
            )

        content_hash = cls.compute_sha256(sanitized_content)
        return sanitized_title, sanitized_content, content_hash

    @classmethod
    def validate_uploaded_files(cls, files: list) -> None:
        """Validate a list of uploaded files against quantity, format, and size constraints."""
        if not files:
            raise ValidationError("At least one file must be selected for upload.")

        if len(files) > _settings.MAX_FILES_PER_UPLOAD:
            raise ValidationError(
                f"Exceeded maximum file upload limit. You can upload up to "
                f"{_settings.MAX_FILES_PER_UPLOAD} files at once (received {len(files)})."
            )

        for file in files:
            filename = getattr(file, "filename", "") or ""
            suffix = Path(filename).suffix.lower()

            if not suffix:
                raise ValidationError(f"File '{filename}' lacks an extension.")

            if suffix not in _settings.ALLOWED_EXTENSIONS:
                raise ValidationError(
                    f"File type '{suffix}' for '{filename}' is not permitted. "
                    f"Allowed types: {', '.join(_settings.ALLOWED_EXTENSIONS)}."
                )

            # Check size if available on upload object
            file_size = getattr(file, "size", None)
            if file_size is not None and file_size > _settings.MAX_FILE_SIZE_BYTES:
                raise PayloadTooLargeError(
                    f"File '{filename}' ({file_size} bytes) exceeds the 5MB limit "
                    f"({_settings.MAX_FILE_SIZE_BYTES} bytes)."
                )
