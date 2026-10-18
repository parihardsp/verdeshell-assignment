"""RAG Pipeline Orchestrator connecting Retrieval, Reranking, and Synthesis."""

import logging
import time

from langsmith import traceable

from app.core.logging import LogPhase, set_request_context
from app.db.models import QuestionHistory
from app.db.repositories import QuestionHistoryRepository
from app.rag.reranker import BaseReranker
from app.rag.retriever import BaseRetriever
from app.rag.synthesizer import BaseSynthesizer
from app.schemas.rag import AskResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates end-to-end RAG workflow across decoupled retrieval, reranking, and synthesis stages."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        synthesizer: BaseSynthesizer,
        query_repo: QuestionHistoryRepository | None = None,
        top_k: int = 5,
        top_n: int = 3,
        similarity_threshold: float = 0.20,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.query_repo = query_repo
        self.top_k = top_k
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold

    @traceable(name="RAG_Pipeline", run_type="chain")
    async def run(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        client_ip: str = "127.0.0.1",
    ) -> AskResponse:
        """Execute end-to-end RAG pipeline with stage-level observability and user isolation."""
        set_request_context(user_id=user_id, session_id=session_id)

        start_time = time.perf_counter()
        logger.info(
            "RAG pipeline started | top_k=%d top_n=%d client_ip=%s",
            self.top_k,
            self.top_n,
            client_ip,
            extra={"phase": LogPhase.RETRIEVAL},
        )

        # Stage 1: Retrieval
        candidate_chunks = await self.retriever.retrieve(
            query=question,
            session_id=session_id,
            user_id=user_id,
            top_k=self.top_k,
            threshold=self.similarity_threshold,
        )

        # Stage 2: Reranking
        pruned_chunks = await self.reranker.rerank(
            query=question,
            candidates=candidate_chunks,
            top_n=self.top_n,
        )

        # Stage 3: Grounded Synthesis
        synthesis_result = await self.synthesizer.synthesize(
            question=question,
            context_chunks=pruned_chunks,
            session_id=session_id,
        )

        total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "RAG pipeline completed | citations=%d confidence=%.2f latency_ms=%d",
            len(synthesis_result.sources),
            synthesis_result.confidence_score,
            total_latency_ms,
            extra={"phase": LogPhase.RESPONSE},
        )

        # Audit Logging
        if self.query_repo:
            try:
                sources_payload = [s.model_dump() for s in synthesis_result.sources]
                history = QuestionHistory(
                    question=question,
                    answer=synthesis_result.answer,
                    sources=sources_payload,
                    session_id=session_id or "default",
                    user_id=user_id or "default_user",
                    latency_ms=total_latency_ms,
                    client_ip=client_ip,
                )
                await self.query_repo.create(history)
            except Exception as exc:
                logger.error(
                    "Audit log creation failed | error=%s",
                    type(exc).__name__,
                    extra={"phase": LogPhase.DATABASE},
                )

        return AskResponse(
            answer=synthesis_result.answer,
            sources=synthesis_result.sources,
            session_id=session_id,
            user_id=user_id,
            latency_ms=total_latency_ms,
            confidence_score=synthesis_result.confidence_score,
        )
