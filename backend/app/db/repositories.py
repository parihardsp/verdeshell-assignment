"""Consolidated repositories for database operations.

Contains BaseRepository, DocumentRepository, ChunkRepository, and QuestionHistoryRepository.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Chunk, Document, QuestionHistory

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository defining required data persistence contracts."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """Retrieve an entity by its primary key."""

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Persist a new entity instance."""

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Remove an entity by primary key."""


class DocumentRepository(BaseRepository[Document]):
    """Encapsulates all database operations for Document entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, entity_id: str) -> Document | None:
        """Fetch a document by its ID."""
        stmt = select(Document).where(Document.id == entity_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        """Fetch an indexed document by its SHA-256 content hash."""
        stmt = (
            select(Document)
            .where(Document.content_hash == content_hash, Document.status == "indexed")
            .order_by(Document.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_hash_and_tenant(
        self, content_hash: str, user_id: str, session_id: str
    ) -> Document | None:
        """Fetch an indexed document matching content hash, user_id, and session_id."""
        stmt = select(Document).where(
            Document.content_hash == content_hash,
            Document.user_id == user_id,
            Document.session_id == session_id,
            Document.status == "indexed",
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create(self, entity: Document) -> Document:
        """Persist a new document record."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[list[Document], int]:
        """List documents with pagination, optionally filtered by session_id and user_id, returning (items, total_count)."""
        count_stmt = select(func.count(Document.id))
        if session_id:
            count_stmt = count_stmt.where(Document.session_id == session_id)
        if user_id:
            count_stmt = count_stmt.where(Document.user_id == user_id)
        total_res = await self.session.execute(count_stmt)
        total = total_res.scalar_one() or 0

        stmt = select(Document).order_by(Document.created_at.desc())
        if session_id:
            stmt = stmt.where(Document.session_id == session_id)
        if user_id:
            stmt = stmt.where(Document.user_id == user_id)
        stmt = stmt.limit(limit).offset(offset)

        result = await self.session.execute(stmt)
        return list(result.scalars().all()), total

    async def update_status(
        self, document_id: str, status: str, chunk_count: int | None = None
    ) -> Document | None:
        """Update indexing status and optional chunk count."""
        doc = await self.get_by_id(document_id)
        if doc:
            doc.status = status
            if chunk_count is not None:
                doc.chunk_count = chunk_count
            await self.session.flush()
        return doc

    async def delete(self, entity_id: str) -> bool:
        """Delete document by ID (cascades to chunks)."""
        doc = await self.get_by_id(entity_id)
        if doc:
            await self.session.delete(doc)
            await self.session.flush()
            return True
        return False


class ChunkRepository(BaseRepository[Chunk]):
    """Encapsulates all database operations for Chunk entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, entity_id: str) -> Chunk | None:
        """Fetch a single chunk by ID."""
        stmt = select(Chunk).where(Chunk.id == entity_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, entity: Chunk) -> Chunk:
        """Persist a single chunk."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        """Persist a batch of chunks."""
        self.session.add_all(chunks)
        await self.session.flush()
        return chunks

    create_many = bulk_create

    async def get_by_document_id(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks belonging to a specific document."""
        stmt = (
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_chunks_with_embeddings(self) -> list[Chunk]:
        """Retrieve all chunks across documents for populating vector search indexes."""
        stmt = select(Chunk).options(selectinload(Chunk.document)).order_by(Chunk.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        stmt = delete(Chunk).where(Chunk.document_id == document_id)
        result = await self.session.execute(stmt)
        return result.rowcount or 0

    async def delete(self, entity_id: str) -> bool:
        """Delete chunk by ID."""
        chunk = await self.get_by_id(entity_id)
        if chunk:
            await self.session.delete(chunk)
            await self.session.flush()
            return True
        return False


class QuestionHistoryRepository(BaseRepository[QuestionHistory]):
    """Encapsulates database operations for QuestionHistory entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, entity_id: str) -> QuestionHistory | None:
        """Fetch audit record by ID."""
        stmt = select(QuestionHistory).where(QuestionHistory.id == entity_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, entity: QuestionHistory) -> QuestionHistory:
        """Persist a new Q&A audit record."""
        self.session.add(entity)
        await self.session.commit()
        return entity

    async def list_recent(
        self,
        limit: int = 50,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[QuestionHistory]:
        """List recent Q&A interactions, optionally scoped to session_id and user_id."""
        stmt = select(QuestionHistory).order_by(QuestionHistory.created_at.desc())
        if session_id:
            stmt = stmt.where(QuestionHistory.session_id == session_id)
        if user_id:
            stmt = stmt.where(QuestionHistory.user_id == user_id)
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, entity_id: str) -> bool:
        """Delete an audit record."""
        record = await self.get_by_id(entity_id)
        if record:
            await self.session.delete(record)
            await self.session.commit()
            return True
        return False

    async def count_records(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count Q&A history interactions matching tenant filters."""
        stmt = select(func.count(QuestionHistory.id))
        if session_id:
            stmt = stmt.where(QuestionHistory.session_id == session_id)
        if user_id:
            stmt = stmt.where(QuestionHistory.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one() or 0

    async def clear_history(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Purge Q&A history for a given tenant session or user."""
        stmt = delete(QuestionHistory)
        if session_id:
            stmt = stmt.where(QuestionHistory.session_id == session_id)
        if user_id:
            stmt = stmt.where(QuestionHistory.user_id == user_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount or 0

