"""Pydantic v2 schemas for document management and multi-file uploads."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentCreate(BaseModel):
    """Schema for POST /documents direct JSON text request body."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Title of the knowledge document or article",
        examples=["Refund Policy"],
    )
    content: str = Field(
        ...,
        min_length=5,
        max_length=1_000_000,
        description="Full text body of the document",
        examples=["Customers may request a full refund within 30 days of purchase..."],
    )
    session_id: str | None = Field(
        default="default",
        description="Optional session identifier for grouping documents",
    )
    user_id: str | None = Field(
        default="default_user",
        description="Optional user identifier for multi-user document isolation",
    )

    @field_validator("title")
    @classmethod
    def validate_title_not_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Title cannot be empty or purely whitespace.")
        return clean

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Content cannot be empty or purely whitespace.")
        return clean


class DocumentResponse(BaseModel):
    """Schema for single document creation/fetch response."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(..., description="Unique document identifier")
    status: Literal["pending", "processing", "indexed", "failed"] = Field(
        ..., description="Current indexing status"
    )
    title: str | None = Field(default=None, description="Document title")
    chunk_count: int | None = Field(default=None, description="Number of indexed chunks")
    session_id: str | None = Field(default="default", description="Associated session ID")
    user_id: str | None = Field(default="default_user", description="Associated user ID")
    file_name: str | None = Field(default=None, description="Original filename if uploaded")
    total_pages: int | None = Field(default=None, description="Total pages if PDF")
    content: str | None = Field(default=None, description="Extracted textual content of the document")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")


class DocumentListItem(BaseModel):
    """Schema for individual document item in GET /documents listing."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    status: str | None = Field(default="indexed", description="Indexing status")
    chunk_count: int | None = Field(default=0, description="Number of chunks")
    session_id: str | None = Field(default="default", description="Associated session ID")
    user_id: str | None = Field(default="default_user", description="Associated user ID")
    file_name: str | None = Field(default=None, description="Original filename")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")


class DocumentListResponse(BaseModel):
    """Paginated response wrapper for document listings."""

    items: list[DocumentListItem]
    total: int = Field(..., description="Total documents matching query")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class UploadedFileSummary(BaseModel):
    """Summary of a single processed file in a multi-file upload."""

    document_id: str
    file_name: str
    status: str
    chunk_count: int
    total_pages: int
    saved_path: str
    error_message: str | None = None


class MultiFileUploadResponse(BaseModel):
    """Schema for POST /documents/upload response."""

    session_id: str = Field(..., description="Unique session identifier for this upload batch")
    user_id: str = Field(
        default="default_user", description="User identifier for this upload batch"
    )
    upload_date: str = Field(..., description="UTC date string (YYYY-MM-DD)")
    total_files: int = Field(..., description="Number of files processed")
    indexed_count: int = Field(default=0, description="Number of successfully indexed documents")
    failed_count: int = Field(default=0, description="Number of failed/skipped documents")
    documents: list[UploadedFileSummary] = Field(
        default_factory=list, description="Details of indexed documents"
    )


class AsyncIngestionJobResponse(BaseModel):
    """Schema returned on HTTP 202 Accepted for asynchronous document ingestion."""

    job_id: str = Field(..., description="Unique asynchronous job identifier")
    session_id: str = Field(..., description="Target session identifier")
    user_id: str = Field(..., description="Target user identifier")
    status: Literal["queued", "processing", "completed", "failed", "completed_with_errors"] = Field(
        ..., description="Current job status"
    )
    total_files: int = Field(..., description="Total files queued in this job")
    message: str = Field(..., description="Human-readable job status message")
    poll_url: str = Field(..., description="URL to poll job status JSON snapshot")
    stream_url: str = Field(..., description="URL to stream real-time SSE progress events")


class JobStatusResponse(BaseModel):
    """Schema returned by GET /documents/jobs/{job_id}."""

    job_id: str
    session_id: str
    user_id: str
    status: str
    stage: str
    percent: int
    message: str
    current_file: str | None = None
    processed_files: int
    total_files: int
    results: list[UploadedFileSummary] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    created_at: float
    updated_at: float

