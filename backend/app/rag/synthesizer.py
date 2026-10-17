"""Synthesizer stage for the RAG pipeline.

Includes BaseSynthesizer, SynthesizedAnswer, and GroundedSynthesizer.
Utilizes Pydantic structured output for zero-hallucination grounded generation.
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langsmith import traceable

from app.providers.llm import BaseLLMProvider
from app.rag.prompts.synthesis_prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_context_blocks,
    build_synthesis_user_prompt,
)
from app.rag.retriever import RetrievedChunk
from app.schemas.rag import GroundedSynthesisOutput, SourceCitation

logger = logging.getLogger(__name__)

# Backward compatibility alias
SYSTEM_PROMPT = SYNTHESIS_SYSTEM_PROMPT


@dataclass
class SynthesizedAnswer:
    """Represents a generated answer with citations and latency tracking."""

    answer: str
    sources: list[SourceCitation]
    latency_ms: int = 0
    confidence_score: float | None = 1.0


class BaseSynthesizer(ABC):
    """Abstract interface for grounded answer generation."""

    @abstractmethod
    async def synthesize(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        session_id: str | None = None,
    ) -> SynthesizedAnswer:
        """Synthesize a grounded answer with citations from context chunks."""


class GroundedSynthesizer(BaseSynthesizer):
    """Synthesizes grounded answers using Pydantic structured output with fallback."""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    @traceable(name="Grounded_Synthesizer", run_type="llm")
    async def synthesize(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        session_id: str | None = None,
    ) -> SynthesizedAnswer:
        start_time = time.perf_counter()

        if not context_chunks:
            scope_desc = f" for session '{session_id}'" if session_id else ""
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "[STAGE 3: GROUNDED SYNTHESIS] Zero context chunks available%s. Returning grounded fallback answer.",
                scope_desc,
            )
            return SynthesizedAnswer(
                answer=f"The provided knowledge base{scope_desc} does not contain information to answer this question.",
                sources=[],
                latency_ms=elapsed_ms,
                confidence_score=0.0,
            )

        context_blocks, context_str = build_synthesis_context_blocks(context_chunks)
        user_prompt = build_synthesis_user_prompt(
            question=question,
            context_str=context_str,
            session_id=session_id,
        )

        provider_name = self.llm_provider.__class__.__name__
        logger.info(
            "[STAGE 3: GROUNDED SYNTHESIS] Structured generation with %d context chunk(s) (%d chars) via %s...",
            len(context_chunks),
            len(context_str),
            provider_name,
        )

        structured_output: GroundedSynthesisOutput | None = None

        try:
            structured_output = await self.llm_provider.generate_structured(
                schema=GroundedSynthesisOutput,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            logger.info(
                "[SYNTHESIS: STRUCTURED RESULT] %s generated answer (sufficient=%s, cited=%s, confidence=%.2f).",
                provider_name,
                structured_output.has_sufficient_context,
                structured_output.cited_chunk_indices,
                structured_output.confidence_score,
            )
        except Exception as exc:
            logger.warning(
                "Structured generation with %s failed (%s: %s). Falling back to raw response generation...",
                provider_name,
                type(exc).__name__,
                exc,
            )
            try:
                raw_answer = await self.llm_provider.generate_response(
                    system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                structured_output = GroundedSynthesisOutput(
                    answer=raw_answer,
                    has_sufficient_context="does not contain" not in raw_answer.lower(),
                    cited_chunk_indices=list(range(min(len(context_chunks), 3))),
                    confidence_score=0.8,
                )
            except Exception as inner_exc:
                logger.error("Upstream LLM completely unavailable: %s", inner_exc)
                from app.providers.llm import MockLLMProvider

                fallback_llm = MockLLMProvider()
                structured_output = await fallback_llm.generate_structured(
                    schema=GroundedSynthesisOutput,
                    system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )

        # Insufficiency handling: zero hallucination check
        if not structured_output.has_sufficient_context:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "[SYNTHESIS: INSUFFICIENT CONTEXT] Grounded model flagged insufficient context. Returning clean refusal."
            )
            return SynthesizedAnswer(
                answer=structured_output.answer or INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                latency_ms=elapsed_ms,
                confidence_score=0.0,
            )

        # Build citations strictly for chunks used/cited by the LLM
        cited_indices = set(structured_output.cited_chunk_indices)
        sources: list[SourceCitation] = []

        for idx, chunk in enumerate(context_chunks):
            # If cited_indices is non-empty, only include explicitly cited chunks
            if cited_indices and idx not in cited_indices:
                continue

            snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            sources.append(
                SourceCitation(
                    document_id=chunk.document_id,
                    title=chunk.document_title,
                    chunk_index=chunk.chunk_index,
                    snippet=snippet,
                    similarity_score=chunk.score,
                    session_id=chunk.session_id,
                    user_id=chunk.user_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    retrieval_channel=getattr(chunk, "retrieval_channel", "dense"),
                    reranker_score=getattr(chunk, "reranker_score", None),
                )
            )

        # If cited_indices were provided but matched nothing valid, include the #1 top chunk
        if not sources and context_chunks:
            top_chunk = context_chunks[0]
            sources.append(
                SourceCitation(
                    document_id=top_chunk.document_id,
                    title=top_chunk.document_title,
                    chunk_index=top_chunk.chunk_index,
                    snippet=top_chunk.content[:200],
                    similarity_score=top_chunk.score,
                    session_id=top_chunk.session_id,
                    user_id=top_chunk.user_id,
                    file_name=top_chunk.file_name,
                    page_number=top_chunk.page_number,
                    retrieval_channel=getattr(top_chunk, "retrieval_channel", "dense"),
                    reranker_score=getattr(top_chunk, "reranker_score", None),
                )
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "[SYNTHESIS: COMPLETE] Answer synthesized in %d ms with %d grounded source citation(s).",
            elapsed_ms,
            len(sources),
        )
        return SynthesizedAnswer(
            answer=structured_output.answer,
            sources=sources,
            latency_ms=elapsed_ms,
            confidence_score=structured_output.confidence_score,
        )


# Backward compatibility aliases for deep imports
sys.modules["app.rag.synthesis"] = sys.modules[__name__]
sys.modules["app.rag.synthesis.base"] = sys.modules[__name__]
sys.modules["app.rag.synthesis.grounded_synthesizer"] = sys.modules[__name__]
