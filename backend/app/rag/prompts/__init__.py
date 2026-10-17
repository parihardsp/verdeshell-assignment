"""RAG pipeline prompt templates and builder utilities."""

from app.rag.prompts.reranker_prompts import (
    RERANKER_SYSTEM_PROMPT,
    build_reranker_passages_text,
    build_reranker_prompt,
)
from app.rag.prompts.synthesis_prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_context_blocks,
    build_synthesis_user_prompt,
)

__all__ = [
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "RERANKER_SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
    "build_reranker_passages_text",
    "build_reranker_prompt",
    "build_synthesis_context_blocks",
    "build_synthesis_user_prompt",
]
