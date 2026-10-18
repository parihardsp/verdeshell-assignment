"""FastAPI dependency injection factories and singleton provider registries."""

import logging
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import LogPhase
from app.db.database import get_db
from app.db.repositories import (
    ChunkRepository,
    DocumentRepository,
    QuestionHistoryRepository,
)
from app.ingestion.storage import SessionFileStorageService
from app.providers.embeddings import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from app.providers.llm import (
    BaseLLMProvider,
    GeminiLLMProvider,
    GroqLLMProvider,
    LangChainLLMProvider,
    MockLLMProvider,
)
from app.providers.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    IndexedChunk,
)
from app.rag.chunker import RecursiveTextChunker, TextChunker
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import (
    BaseReranker,
    GroqLLMReranker,
    MockReranker,
    ReciprocalRankReranker,
)
from app.rag.retriever import BaseRetriever, HybridRetriever
from app.rag.synthesizer import BaseSynthesizer, GroundedSynthesizer
from app.services.chat_service import RAGService
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

# Singletons
_chroma_instance: ChromaVectorStore | None = None
_chunker_instance = RecursiveTextChunker()
_storage_service_instance = SessionFileStorageService()


def get_storage_service() -> SessionFileStorageService:
    """Return singleton session storage service."""
    return _storage_service_instance


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> BaseVectorStore:
    """Return the persistent ChromaDB vector store singleton."""
    global _chroma_instance
    curr_settings = settings or get_settings()

    if _chroma_instance is None:
        _chroma_instance = ChromaVectorStore(persist_directory=curr_settings.CHROMA_PERSIST_DIR)
    return _chroma_instance


def get_chunker() -> TextChunker:
    """Return the text chunker singleton."""
    return _chunker_instance


_embedding_cache: dict[tuple, BaseEmbeddingProvider] = {}
_llm_cache: dict[tuple, BaseLLMProvider] = {}
_reranker_cache: dict[tuple, BaseReranker] = {}


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> BaseEmbeddingProvider:
    """Factory returning the configured embedding provider with mock fallback and client reuse."""
    curr_settings = settings or get_settings()
    provider_type = curr_settings.EMBEDDING_PROVIDER.lower()
    cache_key = (provider_type, curr_settings.GEMINI_API_KEY, curr_settings.GEMINI_EMBEDDING_MODEL)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    provider: BaseEmbeddingProvider
    if provider_type == "gemini" and settings.GEMINI_API_KEY:
        try:
            provider = GeminiEmbeddingProvider(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_EMBEDDING_MODEL,
            )
            _embedding_cache[cache_key] = provider
            return provider
        except Exception as exc:
            logger.warning(
                "Embedding provider initialization failed, falling back to mock | provider=gemini error=%s",
                type(exc).__name__,
                extra={"phase": LogPhase.PROVIDER},
            )

    provider = MockEmbeddingProvider(dimension=384)
    _embedding_cache[cache_key] = provider
    return provider


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseLLMProvider:
    """Factory returning the configured LLM provider with LangSmith tracing, mock fallback, and client reuse."""
    provider_type = settings.LLM_PROVIDER.lower()
    cache_key = (
        provider_type,
        settings.GEMINI_API_KEY,
        settings.GEMINI_MODEL,
        settings.GROQ_API_KEY,
        settings.GROQ_MODEL,
        settings.OPENAI_API_KEY,
    )
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    provider: BaseLLMProvider
    if provider_type == "gemini" and settings.GEMINI_API_KEY:
        try:
            provider = LangChainLLMProvider(
                provider="gemini",
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
            )
            _llm_cache[cache_key] = provider
            return provider
        except Exception as exc:
            logger.warning(
                "LLM provider initialization fallback to direct | provider=gemini error=%s",
                type(exc).__name__,
                extra={"phase": LogPhase.PROVIDER},
            )
            try:
                provider = GeminiLLMProvider(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                )
                _llm_cache[cache_key] = provider
                return provider
            except Exception as e:
                logger.warning(
                    "LLM provider initialization fallback to mock | provider=gemini error=%s",
                    type(e).__name__,
                    extra={"phase": LogPhase.PROVIDER},
                )

    if provider_type == "groq" and settings.GROQ_API_KEY:
        try:
            provider = LangChainLLMProvider(
                provider="groq",
                api_key=settings.GROQ_API_KEY,
                model=settings.GROQ_MODEL,
            )
            _llm_cache[cache_key] = provider
            return provider
        except Exception as exc:
            logger.warning(
                "LLM provider initialization fallback to direct | provider=groq error=%s",
                type(exc).__name__,
                extra={"phase": LogPhase.PROVIDER},
            )
            try:
                provider = GroqLLMProvider(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL,
                    base_url=settings.GROQ_BASE_URL,
                )
                _llm_cache[cache_key] = provider
                return provider
            except Exception as e:
                logger.warning(
                    "LLM provider initialization fallback to mock | provider=groq error=%s",
                    type(e).__name__,
                    extra={"phase": LogPhase.PROVIDER},
                )

    if provider_type in ("openai", "openrouter") and settings.OPENAI_API_KEY:
        try:
            provider = LangChainLLMProvider(
                provider="openai",
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                base_url=settings.OPENAI_BASE_URL,
            )
            _llm_cache[cache_key] = provider
            return provider
        except Exception as exc:
            logger.warning(
                "LLM provider initialization fallback to mock | provider=openai error=%s",
                type(exc).__name__,
                extra={"phase": LogPhase.PROVIDER},
            )

    provider = MockLLMProvider()
    _llm_cache[cache_key] = provider
    return provider


# Repository Dependencies
def get_document_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


def get_chunk_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ChunkRepository:
    return ChunkRepository(session)


def get_query_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> QuestionHistoryRepository:
    return QuestionHistoryRepository(session)


# RAG Pipeline Stage Dependencies
def get_retriever(
    embedding_provider: Annotated[BaseEmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> BaseRetriever:
    return HybridRetriever(embedding_provider=embedding_provider, vector_store=vector_store)


def get_reranker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseReranker:
    reranker_type = settings.RERANKER_TYPE.lower()
    cache_key = (
        reranker_type,
        settings.GROQ_API_KEY,
        settings.GROQ_RERANKER_MODEL,
        settings.RETRIEVAL_TOP_K,
    )
    if cache_key in _reranker_cache:
        return _reranker_cache[cache_key]

    reranker: BaseReranker
    if reranker_type == "groq" and settings.GROQ_API_KEY:
        reranker = GroqLLMReranker(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_RERANKER_MODEL,
            rrf_k=60,
            max_candidates=settings.RETRIEVAL_TOP_K,
        )
    elif reranker_type == "mock":
        reranker = MockReranker()
    else:
        reranker = ReciprocalRankReranker(rrf_k=60)

    _reranker_cache[cache_key] = reranker
    return reranker


def get_synthesizer(
    llm_provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> BaseSynthesizer:
    return GroundedSynthesizer(llm_provider=llm_provider)


def get_rag_pipeline(
    retriever: Annotated[BaseRetriever, Depends(get_retriever)],
    reranker: Annotated[BaseReranker, Depends(get_reranker)],
    synthesizer: Annotated[BaseSynthesizer, Depends(get_synthesizer)],
    query_repo: Annotated[QuestionHistoryRepository, Depends(get_query_repo)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        synthesizer=synthesizer,
        query_repo=query_repo,
        top_k=settings.RETRIEVAL_TOP_K,
        top_n=settings.FINAL_TOP_N,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
    )


# Service Dependencies
def get_document_service(
    doc_repo: Annotated[DocumentRepository, Depends(get_document_repo)],
    chunk_repo: Annotated[ChunkRepository, Depends(get_chunk_repo)],
    embedding_provider: Annotated[BaseEmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    chunker: Annotated[TextChunker, Depends(get_chunker)],
    storage_service: Annotated[SessionFileStorageService, Depends(get_storage_service)],
) -> DocumentService:
    return DocumentService(
        document_repo=doc_repo,
        chunk_repo=chunk_repo,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunker=chunker,
        storage_service=storage_service,
    )


def get_rag_service(
    embedding_provider: Annotated[BaseEmbeddingProvider, Depends(get_embedding_provider)],
    llm_provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    query_repo: Annotated[QuestionHistoryRepository, Depends(get_query_repo)],
    reranker: Annotated[BaseReranker, Depends(get_reranker)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGService:
    return RAGService(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        vector_store=vector_store,
        query_repo=query_repo,
        top_k=settings.RETRIEVAL_TOP_K,
        top_n=settings.FINAL_TOP_N,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        reranker=reranker,
    )


async def hydrate_vector_store(session: AsyncSession) -> int:
    """Pre-populate the vector store from SQLite chunk records on startup."""
    chunk_repo = ChunkRepository(session)
    chunks = await chunk_repo.get_all_chunks_with_embeddings()
    vector_store = get_vector_store()
    settings = get_settings()

    expected_dim = 3072 if settings.EMBEDDING_PROVIDER.lower() == "gemini" else 384

    indexed_chunks = []
    for c in chunks:
        if c.embedding and len(c.embedding) != expected_dim:
            continue
        doc_title = c.document.title if c.document else "Knowledge Article"
        indexed_chunks.append(
            IndexedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                document_title=doc_title,
                chunk_index=c.chunk_index,
                content=c.content,
                embedding=c.embedding,
                session_id=c.session_id,
                user_id=getattr(c, "user_id", "default_user") or "default_user",
                file_name=c.file_name or "",
                page_number=c.page_number or 1,
            )
        )

    if indexed_chunks:
        await vector_store.add_chunks(indexed_chunks)
    logger.info(
        "Vector store hydrated from database | chunks=%d dimension=%d",
        len(indexed_chunks),
        expected_dim,
        extra={"phase": LogPhase.VECTOR_STORE},
    )
    return len(indexed_chunks)
