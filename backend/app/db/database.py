"""Async database session management and lifecycle hooks."""

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.db.models import Base

_settings = get_settings()

# Normalize database URL and ensure storage directory exists for SQLite
db_url = _settings.DATABASE_URL
if db_url.startswith("sqlite+aiosqlite:///"):
    sqlite_path = db_url.replace("sqlite+aiosqlite:///", "")
    if not os.path.isabs(sqlite_path):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        resolved_path = (root_dir / sqlite_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite+aiosqlite:///{resolved_path}"
    else:
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

engine: AsyncEngine = create_async_engine(
    db_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
)

# Enforce foreign key constraints in SQLite
if "sqlite" in db_url:

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def _migrate_sqlite_schema(sync_conn):
    """Safely migrate missing columns on existing SQLite tables without dropping data."""
    cursor = sync_conn.connection.cursor()
    try:
        # Check chunks table columns
        cursor.execute("PRAGMA table_info(chunks)")
        chunk_cols = {row[1] for row in cursor.fetchall()}
        if chunk_cols:
            if "session_id" not in chunk_cols:
                cursor.execute(
                    "ALTER TABLE chunks ADD COLUMN session_id VARCHAR(64) DEFAULT 'default' NOT NULL"
                )
            if "user_id" not in chunk_cols:
                cursor.execute(
                    "ALTER TABLE chunks ADD COLUMN user_id VARCHAR(64) DEFAULT 'default_user' NOT NULL"
                )
            if "file_name" not in chunk_cols:
                cursor.execute("ALTER TABLE chunks ADD COLUMN file_name VARCHAR(255)")
            if "page_number" not in chunk_cols:
                cursor.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER DEFAULT 1")

        # Check documents table columns
        cursor.execute("PRAGMA table_info(documents)")
        doc_cols = {row[1] for row in cursor.fetchall()}
        if doc_cols:
            if "session_id" not in doc_cols:
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN session_id VARCHAR(64) DEFAULT 'default' NOT NULL"
                )
            if "user_id" not in doc_cols:
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN user_id VARCHAR(64) DEFAULT 'default_user' NOT NULL"
                )
            if "file_name" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN file_name VARCHAR(255)")
            if "file_path" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN file_path VARCHAR(512)")
            if "total_pages" not in doc_cols:
                cursor.execute("ALTER TABLE documents ADD COLUMN total_pages INTEGER")

        # Migrate content_hash index from unique to non-unique to allow multi-tenant deduplication
        cursor.execute("DROP INDEX IF EXISTS ix_documents_content_hash")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents (content_hash)"
        )

        # Check question_history table columns
        cursor.execute("PRAGMA table_info(question_history)")
        qh_cols = {row[1] for row in cursor.fetchall()}
        if qh_cols:
            if "session_id" not in qh_cols:
                cursor.execute("ALTER TABLE question_history ADD COLUMN session_id VARCHAR(64)")
            if "user_id" not in qh_cols:
                cursor.execute(
                    "ALTER TABLE question_history ADD COLUMN user_id VARCHAR(64) DEFAULT 'default_user' NOT NULL"
                )
    finally:
        cursor.close()


async def init_db() -> None:
    """Create all registered database tables and run non-destructive auto-migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in db_url:
            await conn.run_sync(_migrate_sqlite_schema)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
