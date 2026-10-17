"""Embedder stage for the RAG pipeline.

Coordinates text embedding generation for document chunks and user queries.
"""

from app.providers.embeddings import BaseEmbeddingProvider


class RAGEmbedder:
    """Embedding stage coordinator for RAG pipeline components."""

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the underlying provider."""
        return self.provider.dimension

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return await self.provider.embed_text(query)

    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Batch embed chunk contents."""
        return await self.provider.embed_batch(texts)
