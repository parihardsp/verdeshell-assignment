"""SQLAlchemy ORM models and declarative Base.

Consolidates Document, Chunk, and QuestionHistory entities.
"""

import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class Document(Base):
    """Knowledge document entity."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: f"doc_{uuid.uuid4().hex[:12]}"
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(
        String(64), default="default", index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(64), default="default_user", index=True, nullable=False
    )
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="indexed", index=True, nullable=False
    )  # 'pending', 'processing', 'indexed', 'failed'
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.id}', title='{self.title}', user_id='{self.user_id}', status='{self.status}')>"


class Chunk(Base):
    """Text chunk entity with character and page offsets."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: f"chk_{uuid.uuid4().hex[:12]}"
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(
        String(64), default="default", index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(64), default="default_user", index=True, nullable=False
    )
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, default=1, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Chunk(id='{self.id}', doc_id='{self.document_id}', user_id='{self.user_id}', index={self.chunk_index})>"


class QuestionHistory(Base):
    """Audit log entity for asked questions and generated answers."""

    __tablename__ = "question_history"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: f"qry_{uuid.uuid4().hex[:12]}"
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str] = mapped_column(
        String(64), default="default_user", index=True, nullable=False
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    client_ip: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<QuestionHistory(id='{self.id}', latency_ms={self.latency_ms})>"


# Backward compatibility aliases for deep imports
sys.modules["app.db.models.document"] = sys.modules[__name__]
sys.modules["app.db.models.chunk"] = sys.modules[__name__]
sys.modules["app.db.models.query_history"] = sys.modules[__name__]
