"""Database access layer consolidating database engine, models, and repositories."""

from app.db.database import AsyncSessionLocal, engine, get_db, init_db
from app.db.models import Base, Chunk, Document, QuestionHistory
from app.db.repositories import (
    BaseRepository,
    ChunkRepository,
    DocumentRepository,
    QuestionHistoryRepository,
)

__all__ = [
    "AsyncSessionLocal",
    "Base",
    "BaseRepository",
    "Chunk",
    "ChunkRepository",
    "Document",
    "DocumentRepository",
    "QuestionHistory",
    "QuestionHistoryRepository",
    "engine",
    "get_db",
    "init_db",
]
