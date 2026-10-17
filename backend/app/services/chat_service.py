"""Chat and Question Answering Service.

Orchestrates decoupled Retrieval, Reciprocal Rank Fusion Reranking, and Grounded Synthesis.
Provides ChatService and aliases RAGService.
"""

import logging

from app.db.repositories import QuestionHistoryRepository
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.llm import BaseLLMProvider
from app.providers.vector_store import BaseVectorStore
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import BaseReranker, ReciprocalRankReranker
from app.rag.retriever import BaseRetriever, HybridRetriever
from app.rag.synthesizer import GroundedSynthesizer
from app.schemas.rag import AskResponse

logger = logging.getLogger(__name__)


class ChatService:
    """Facade orchestrating decoupled retrieval, reranking, and grounded synthesis via RAGPipeline."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
        vector_store: BaseVectorStore,
        query_repo: QuestionHistoryRepository,
        top_k: int = 15,
        top_n: int = 3,
        similarity_threshold: float = 0.20,
        reranker: BaseReranker | None = None,
        retriever: BaseRetriever | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.query_repo = query_repo

        # Initialize decoupled pipeline stages
        self.retriever = retriever or HybridRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        self.reranker = reranker or ReciprocalRankReranker(rrf_k=60)
        self.synthesizer = GroundedSynthesizer(llm_provider=llm_provider)

        # Wire pipeline orchestrator
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            reranker=self.reranker,
            synthesizer=self.synthesizer,
            query_repo=query_repo,
            top_k=top_k,
            top_n=top_n,
            similarity_threshold=similarity_threshold,
        )

    async def answer_question(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        client_ip: str = "127.0.0.1",
    ) -> AskResponse:
        """Execute end-to-end RAG pipeline for an incoming question, optionally scoped to session_id and user_id."""
        return await self.pipeline.run(
            question=question,
            session_id=session_id,
            user_id=user_id,
            client_ip=client_ip,
        )


# Backward-compatible alias
RAGService = ChatService
