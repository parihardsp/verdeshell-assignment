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


class MultiFileUploadResponse(BaseModel):
    """Schema for POST /documents/upload response."""

    session_id: str = Field(..., description="Unique session identifier for this upload batch")
    user_id: str = Field(
        default="default_user", description="User identifier for this upload batch"
    )
    upload_date: str = Field(..., description="UTC date string (YYYY-MM-DD)")
    total_files: int = Field(..., description="Number of files processed")
    documents: list[UploadedFileSummary] = Field(
        default_factory=list, description="Details of indexed documents"
    )
