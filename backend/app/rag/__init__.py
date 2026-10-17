"""RAG architecture layer: chunking, embedding, retrieval, reranking, synthesis, and evaluation."""

from app.rag.chunker import BaseChunker, RecursiveTextChunker, TextChunk
from app.rag.embedder import RAGEmbedder
from app.rag.evaluation import (
    BENCHMARK_SUITE,
    MetricScore,
    RAGPipelineEvaluator,
    SampleEvaluationResult,
    get_all_rag_metrics,
    get_generation_metrics,
    get_retrieval_metrics,
)
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import BaseReranker, PassThroughReranker, ReciprocalRankReranker
from app.rag.retriever import BaseRetriever, RetrievedChunk, VectorRetriever
from app.rag.synthesizer import BaseSynthesizer, GroundedSynthesizer, SynthesizedAnswer

__all__ = [
    "BENCHMARK_SUITE",
    "BaseChunker",
    "BaseReranker",
    "BaseRetriever",
    "BaseSynthesizer",
    "GroundedSynthesizer",
    "MetricScore",
    "PassThroughReranker",
    "RAGEmbedder",
    "RAGPipeline",
    "RAGPipelineEvaluator",
    "ReciprocalRankReranker",
    "RecursiveTextChunker",
    "RetrievedChunk",
    "SampleEvaluationResult",
    "SynthesizedAnswer",
    "TextChunk",
    "VectorRetriever",
    "get_all_rag_metrics",
    "get_generation_metrics",
    "get_retrieval_metrics",
]
