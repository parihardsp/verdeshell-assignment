"""Reranker stage for the RAG pipeline.

Combines semantic dense vector ranks with lexical query-token overlap ranks
using Cormack's Reciprocal Rank Fusion formula:
RRF_Score(d) = Σ [ 1 / (k + rank_m(d)) ] with k = 60.
"""

import logging
import re
import sys
import time
from abc import ABC, abstractmethod

from langsmith import traceable

from app.core.logging import LogPhase
from app.rag.prompts.reranker_prompts import (
    build_reranker_passages_text,
    build_reranker_prompt,
)
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for candidate chunk reranking and precision filtering."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 3,
    ) -> list[RetrievedChunk]:
        """Re-order and prune candidate chunks to return the top_n most relevant chunks."""


class PassThroughReranker(BaseReranker):
    """Pass-through reranker sorting by candidate score descending."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 3,
    ) -> list[RetrievedChunk]:
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        return sorted_candidates[:top_n]


class ReciprocalRankReranker(BaseReranker):
    """Hybrid reranker using Reciprocal Rank Fusion across dense and lexical ranks."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Calculate lexical term match score between query and candidate text."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return 0.0

        doc_terms = re.findall(r"\w+", text.lower())
        if not doc_terms:
            return 0.0

        matches = sum(1 for term in doc_terms if term in query_terms)
        return matches / (len(query_terms) + (len(doc_terms) * 0.05))

    @traceable(name="ReciprocalRank_Reranker", run_type="parser")
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 3,
    ) -> list[RetrievedChunk]:
        """Compute hybrid RRF scores and return top_n precision-boosted chunks."""
        start_time = time.perf_counter()
        # 1. Dense Ranking
        dense_sorted = sorted(
            candidates,
            key=lambda c: (
                c.dense_score
                if getattr(c, "dense_score", None) is not None and c.dense_score > 0
                else c.score
            ),
            reverse=True,
        )
        dense_ranks = {c.chunk_id: idx + 1 for idx, c in enumerate(dense_sorted)}

        # 2. Lexical Ranking (BM25 or token overlap fallback)
        lexical_scores = {}
        for c in candidates:
            if getattr(c, "lexical_score", None) is not None and c.lexical_score > 0.0:
                lexical_scores[c.chunk_id] = c.lexical_score
            else:
                lexical_scores[c.chunk_id] = self._lexical_score(query, c.content)

        lexical_sorted = sorted(candidates, key=lambda c: lexical_scores[c.chunk_id], reverse=True)
        lexical_ranks = {c.chunk_id: idx + 1 for idx, c in enumerate(lexical_sorted)}

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for c in candidates:
            cid = c.chunk_id
            score = (1.0 / (self.rrf_k + dense_ranks[cid])) + (
                1.0 / (self.rrf_k + lexical_ranks[cid])
            )
            rrf_scores[cid] = round(score, 6)

        reranked = sorted(candidates, key=lambda c: rrf_scores[c.chunk_id], reverse=True)

        for c in reranked:
            c.score = rrf_scores[c.chunk_id]
            c.rrf_score = rrf_scores[c.chunk_id]

        selected = reranked[:top_n]
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "RRF reranking completed | candidates=%d selected=%d k=%d latency_ms=%.1f",
            len(candidates),
            len(selected),
            self.rrf_k,
            latency_ms,
            extra={"phase": LogPhase.RERANKING},
        )
        return selected


class GroqLLMReranker(BaseReranker):
    """Precision reranker powered by high-throughput Groq LLMs.

    Orchestrates a 2-stage reranking funnel:
      Stage 2A: Reciprocal Rank Fusion (RRF) coarsens and orders incoming hybrid candidates down to a top candidate pool (e.g., 15).
      Stage 2B: High-speed Groq LLM evaluates candidate passages against the query, assigning deep precision relevance scores and rationale.
      Graceful Fallback: Transparently falls back to RRF rank order if Groq encounters any rate limits, timeouts, or network exceptions.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
        rrf_k: int = 60,
        max_candidates: int = 15,
        temperature: float = 0.0,
    ):
        self.groq_api_key = groq_api_key
        self.model = model
        self.rrf_k = rrf_k
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.rrf_reranker = ReciprocalRankReranker(rrf_k=rrf_k)

    @traceable(name="GroqLLM_Reranker", run_type="chain")
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 3,
    ) -> list[RetrievedChunk]:
        """Compute Groq LLM precision scores and return top_n most relevant chunks."""
        if not candidates:
            return []

        # Single candidate short-circuit
        if len(candidates) <= 1:
            return candidates[:top_n]

        # Stage 2A: Coarse cut via Reciprocal Rank Fusion down to max_candidates (e.g., 15)
        rrf_fused = await self.rrf_reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=max(len(candidates), self.max_candidates),
        )
        pool = rrf_fused[: self.max_candidates]

        if not self.groq_api_key:
            logger.warning(
                "Missing API key, returning RRF fusion ranks | provider=groq",
                extra={"phase": LogPhase.RERANKING},
            )
            return pool[:top_n]

        start_time = time.perf_counter()
        try:
            import json

            from langchain_groq import ChatGroq

            from app.core.logging import get_session_context, get_user_context

            u_ctx = get_user_context()
            s_ctx = get_session_context()

            chat_model = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=2000,
            )

            passages_str = build_reranker_passages_text(pool, max_chars=500)
            prompt = build_reranker_prompt(
                query=query, passages_str=passages_str, top_candidates_limit=top_n * 2
            )

            config = {
                "tags": [
                    f"user:{u_ctx}",
                    f"session:{s_ctx}",
                    "stage:reranker",
                    "provider:groq",
                ],
                "metadata": {
                    "user_id": u_ctx,
                    "session_id": s_ctx,
                    "model": self.model,
                    "candidates_count": len(pool),
                },
            }

            response = await chat_model.ainvoke(prompt, config=config)
            raw_content = response.content.strip()

            # Clean markdown code fences if present
            if "```" in raw_content:
                clean = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.MULTILINE)
                clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()
            else:
                clean = raw_content

            parsed_scores = json.loads(clean)

            # Map Groq scores back to candidate chunks
            scored_indices = set()
            groq_ranked_candidates: list[RetrievedChunk] = []

            for item in parsed_scores:
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(pool):
                    chunk = pool[idx]
                    groq_score = float(item.get("score", 0.0))
                    chunk.reranker_score = groq_score
                    chunk.reranker_reason = str(item.get("reason", ""))
                    chunk.score = round(groq_score / 10.0, 4)
                    scored_indices.add(idx)
                    groq_ranked_candidates.append(chunk)

            # Sort groq scored candidates descending
            groq_ranked_candidates.sort(
                key=lambda c: getattr(c, "reranker_score", 0.0), reverse=True
            )

            # Append any un-scored candidates from pool in their original RRF order
            for idx, c in enumerate(pool):
                if idx not in scored_indices:
                    c.reranker_score = 0.0
                    c.reranker_reason = "Not flagged by LLM as high relevance"
                    groq_ranked_candidates.append(c)

            final_selected = groq_ranked_candidates[:top_n]
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Neural reranking completed | model=%s candidates=%d selected=%d latency_ms=%.1f",
                self.model,
                len(pool),
                len(final_selected),
                latency_ms,
                extra={"phase": LogPhase.RERANKING},
            )
            return final_selected

        except Exception as exc:
            logger.warning(
                "Neural reranking failed, falling back to RRF | model=%s error=%s",
                self.model,
                type(exc).__name__,
                extra={"phase": LogPhase.RERANKING},
            )
            return pool[:top_n]


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit tests and offline benchmarking."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 3,
    ) -> list[RetrievedChunk]:
        for c in candidates:
            c.reranker_score = round(c.score * 10.0, 1)
            c.reranker_reason = "Mock reranker score"
        sorted_cand = sorted(candidates, key=lambda c: c.score, reverse=True)
        return sorted_cand[:top_n]


# Backward compatibility aliases for deep imports
sys.modules["app.rag.reranking"] = sys.modules[__name__]
sys.modules["app.rag.reranking.base"] = sys.modules[__name__]
sys.modules["app.rag.reranking.reciprocal_rank"] = sys.modules[__name__]
sys.modules["app.rag.reranking.pass_through"] = sys.modules[__name__]
sys.modules["app.rag.reranking.groq_reranker"] = sys.modules[__name__]
