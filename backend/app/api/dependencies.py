"""FastAPI dependency injection factories and singleton provider registries."""

import logging
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
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


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseEmbeddingProvider:
    """Factory returning the configured embedding provider with mock fallback."""
    provider_type = settings.EMBEDDING_PROVIDER.lower()

    if provider_type == "gemini" and settings.GEMINI_API_KEY:
        try:
            return GeminiEmbeddingProvider(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_EMBEDDING_MODEL,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize Gemini embedding provider (%s). Falling back to mock.",
                exc,
            )

    return MockEmbeddingProvider(dimension=384)


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseLLMProvider:
    """Factory returning the configured LLM provider with LangSmith tracing and mock fallback."""
    provider_type = settings.LLM_PROVIDER.lower()

    if provider_type == "gemini" and settings.GEMINI_API_KEY:
        try:
            return LangChainLLMProvider(
                provider="gemini",
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LangChain Gemini provider (%s). Falling back to direct provider.",
                exc,
            )
            try:
                return GeminiLLMProvider(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                )
            except Exception as e:
                logger.warning("Gemini LLM initialization failed (%s). Falling back to mock.", e)

    if provider_type == "groq" and settings.GROQ_API_KEY:
        try:
            return LangChainLLMProvider(
                provider="groq",
                api_key=settings.GROQ_API_KEY,
                model=settings.GROQ_MODEL,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LangChain Groq provider (%s). Falling back to direct provider.",
                exc,
            )
            try:
                return GroqLLMProvider(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL,
                    base_url=settings.GROQ_BASE_URL,
                )
            except Exception as e:
                logger.warning("Groq LLM initialization failed (%s). Falling back to mock.", e)

    if provider_type in ("openai", "openrouter") and settings.OPENAI_API_KEY:
        try:
            return LangChainLLMProvider(
                provider="openai",
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                base_url=settings.OPENAI_BASE_URL,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LangChain OpenAI provider (%s). Falling back to mock.",
                exc,
            )

    return MockLLMProvider()


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
    if reranker_type == "groq" and settings.GROQ_API_KEY:
        return GroqLLMReranker(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_RERANKER_MODEL,
            rrf_k=60,
            max_candidates=settings.RETRIEVAL_TOP_K,
        )
    elif reranker_type == "mock":
        return MockReranker()
    return ReciprocalRankReranker(rrf_k=60)


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
        "Hydrated vector store with %d chunk embeddings from database (dim: %d).",
        len(indexed_chunks),
        expected_dim,
    )
    return len(indexed_chunks)
