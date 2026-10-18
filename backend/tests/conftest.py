"""Pytest fixtures and test environment configuration."""

import chromadb
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
    get_vector_store,
)
from app.db.database import get_db, get_session_factory

from app.core.rate_limiter import global_rate_limiter
from app.db.models import Base
from app.main import app
from app.providers.embeddings import MockEmbeddingProvider
from app.providers.llm import MockLLMProvider
from app.providers.vector_store import ChromaVectorStore

# In-memory test database
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Create fresh schema for each test session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Reset in-memory rate limiter and vector store
    global_rate_limiter.reset()
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Yield an active test database session."""
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncClient:
    """Yield an HTTP test client overriding dependencies with test mocks."""
    test_vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    test_embedding_provider = MockEmbeddingProvider()
    test_llm_provider = MockLLMProvider()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_factory] = lambda: TestSessionLocal
    app.dependency_overrides[get_vector_store] = lambda: test_vector_store
    app.dependency_overrides[get_embedding_provider] = lambda: test_embedding_provider
    app.dependency_overrides[get_llm_provider] = lambda: test_llm_provider

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
