"""Retriever stage for the RAG pipeline.

Includes BaseRetriever, RetrievedChunk, and VectorRetriever.
"""

import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langsmith import traceable
from rank_bm25 import BM25L

from app.core.logging import LogPhase
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with relevance scoring and origin metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    session_id: str
    user_id: str = "default_user"
    file_name: str | None = None
    page_number: int | None = 1
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    retrieval_channel: str = "dense"  # "dense", "bm25", or "hybrid"
    rrf_score: float | None = None
    reranker_score: float | None = None
    reranker_reason: str | None = None


class BaseRetriever(ABC):
    """Abstract interface for candidate chunk retrieval."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.20,
    ) -> list[RetrievedChunk]:
        """Fetch candidate chunks matching query, scoped to session_id and user_id if provided."""


class VectorRetriever(BaseRetriever):
    """Dense vector retriever using vector store provider with session and user scoping."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    @traceable(name="Vector_Retriever", run_type="retriever")
    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.20,
    ) -> list[RetrievedChunk]:
        start_time = time.perf_counter()
        provider_name = self.embedding_provider.__class__.__name__

        query_embedding = await self.embedding_provider.embed_text(query)
        raw_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            session_id=session_id,
            user_id=user_id,
        )

        retrieved: list[RetrievedChunk] = []
        for chunk_record, score in raw_results:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_record.chunk_id,
                    document_id=chunk_record.document_id,
                    document_title=chunk_record.document_title,
                    chunk_index=chunk_record.chunk_index,
                    content=chunk_record.content,
                    session_id=chunk_record.session_id,
                    user_id=chunk_record.user_id,
                    file_name=chunk_record.file_name,
                    page_number=chunk_record.page_number,
                    score=score,
                    dense_score=score,
                    lexical_score=0.0,
                    retrieval_channel="dense",
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Dense vector retrieval completed | provider=%s top_k=%d matches=%d latency_ms=%.1f",
            provider_name,
            top_k,
            len(retrieved),
            latency_ms,
            extra={"phase": LogPhase.RETRIEVAL},
        )
        return retrieved


class BM25Retriever(BaseRetriever):
    """Lexical keyword retriever using BM25Okapi over scoped document chunks."""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    STOPWORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
    }

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        filtered = [t for t in tokens if t not in cls.STOPWORDS and len(t) > 1]
        return filtered if filtered else tokens

    @traceable(name="BM25_Retriever", run_type="retriever")
    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        start_time = time.perf_counter()
        all_chunks = await self.vector_store.get_all_chunks(session_id=session_id, user_id=user_id)
        if not all_chunks:
            return []

        tokenized_corpus = [self._tokenize(c.content) for c in all_chunks]
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        try:
            bm25 = BM25L(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
        except Exception as exc:
            logger.warning("BM25 scoring failed, returning empty | error=%s", type(exc).__name__, extra={"phase": LogPhase.RETRIEVAL})
            return []

        scored_pairs = []
        for chunk, score in zip(all_chunks, scores):
            if score > threshold:
                scored_pairs.append((chunk, float(score)))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        retrieved: list[RetrievedChunk] = []
        for chunk, score in top_pairs:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    session_id=chunk.session_id,
                    user_id=chunk.user_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    score=score,
                    dense_score=0.0,
                    lexical_score=score,
                    retrieval_channel="bm25",
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Lexical retrieval completed | top_k=%d matches=%d latency_ms=%.1f",
            top_k,
            len(retrieved),
            latency_ms,
            extra={"phase": LogPhase.RETRIEVAL},
        )
        return retrieved


class HybridRetriever(BaseRetriever):
    """True Dual-Channel Hybrid Retriever combining Dense Vector and BM25 Lexical search."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.dense_retriever = VectorRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        self.lexical_retriever = BM25Retriever(vector_store=vector_store)

    @traceable(name="Hybrid_Retriever", run_type="retriever")
    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.20,
    ) -> list[RetrievedChunk]:
        """Execute parallel Dense Vector + BM25 Lexical retrieval and merge candidate pools."""
        import asyncio

        start_time = time.perf_counter()

        dense_task = self.dense_retriever.retrieve(
            query=query,
            session_id=session_id,
            user_id=user_id,
            top_k=top_k,
            threshold=threshold,
        )
        lexical_task = self.lexical_retriever.retrieve(
            query=query,
            session_id=session_id,
            user_id=user_id,
            top_k=top_k,
        )

        dense_results, lexical_results = await asyncio.gather(dense_task, lexical_task)

        # Merge candidate pools
        candidate_map: dict[str, RetrievedChunk] = {}
        for c in dense_results:
            c.retrieval_channel = "dense"
            c.dense_score = c.score
            candidate_map[c.chunk_id] = c

        both_count = 0
        lexical_only_count = 0
        for c in lexical_results:
            if c.chunk_id in candidate_map:
                both_count += 1
                existing = candidate_map[c.chunk_id]
                existing.lexical_score = c.score
                existing.retrieval_channel = "hybrid"
            else:
                lexical_only_count += 1
                c.retrieval_channel = "bm25"
                c.lexical_score = c.score
                c.dense_score = 0.0
                candidate_map[c.chunk_id] = c

        merged = list(candidate_map.values())
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Hybrid retrieval completed | dense=%d bm25=%d merged=%d overlap=%d latency_ms=%.1f",
            len(dense_results),
            len(lexical_results),
            len(merged),
            both_count,
            latency_ms,
            extra={"phase": LogPhase.RETRIEVAL},
        )
        return merged


# Backward compatibility aliases for deep imports
sys.modules["app.rag.retrieval"] = sys.modules[__name__]
sys.modules["app.rag.retrieval.base"] = sys.modules[__name__]
sys.modules["app.rag.retrieval.vector_retriever"] = sys.modules[__name__]
sys.modules["app.rag.retrieval.hybrid_retriever"] = sys.modules[__name__]
sys.modules["app.rag.retrieval.bm25_retriever"] = sys.modules[__name__]
