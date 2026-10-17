"""Embedding Provider abstractions and implementations.

Includes BaseEmbeddingProvider, GeminiEmbeddingProvider, and MockEmbeddingProvider.
"""

import hashlib
import logging
import re
import sys
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import UpstreamProviderError

logger = logging.getLogger(__name__)
_settings = get_settings()


class BaseEmbeddingProvider(ABC):
    """Contract for text embedding generation."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate a dense vector embedding for a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a batch of text strings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the generated vectors."""


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using LangChain's GoogleGenerativeAIEmbeddings wrapper with LangSmith tracing."""

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-2",
        timeout_seconds: float = 12.0,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required for GeminiEmbeddingProvider.")
        self.api_key = api_key
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.timeout = timeout_seconds
        self._dim = 3072  # gemini-embedding-2 dimensions

        self._embedder = None
        self._use_langchain = False
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._embedder = GoogleGenerativeAIEmbeddings(
                model=self.model,
                google_api_key=self.api_key,
            )
            self._use_langchain = True
            logger.info(
                "Initialized GeminiEmbeddingProvider using LangChain GoogleGenerativeAIEmbeddings (%s).",
                self.model,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LangChain GoogleGenerativeAIEmbeddings (%s). Using direct REST.",
                exc,
            )

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string using LangChain GoogleGenerativeAIEmbeddings."""
        if self._use_langchain and self._embedder:
            try:
                return await self._embedder.aembed_query(text)
            except Exception as exc:
                logger.warning("LangChain embed_query failed (%s), falling back to REST.", exc)

        return await self._embed_text_http(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using LangChain GoogleGenerativeAIEmbeddings."""
        if not texts:
            return []
        if self._use_langchain and self._embedder:
            try:
                return await self._embedder.aembed_documents(texts)
            except Exception as exc:
                logger.warning("LangChain embed_documents failed (%s), falling back to REST.", exc)

        return await self._embed_batch_http(texts)

    async def _embed_text_http(self, text: str) -> list[float]:
        """Embed a single text string via REST fallback."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 429:
                    raise UpstreamProviderError(
                        message="Gemini embedding API rate limit reached.",
                        status_code=503,
                        headers={"Retry-After": "30"},
                    )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", {}).get("values", [])
                if not embedding:
                    raise UpstreamProviderError("Gemini returned empty embedding values.")
                return embedding
            except httpx.TimeoutException as exc:
                logger.error("Gemini embedding request timed out: %s", exc)
                raise UpstreamProviderError(
                    "Timeout connecting to Gemini embedding service.", status_code=504
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Gemini embedding HTTP error %s: %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                raise UpstreamProviderError(
                    f"Gemini embedding API returned status {exc.response.status_code}.",
                    status_code=502,
                )
            except Exception as exc:
                logger.error("Unexpected error in Gemini embedding: %s", exc)
                raise UpstreamProviderError(f"Embedding generation failed: {exc!s}")

    async def _embed_batch_http(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using batchEmbedContents REST fallback."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {"model": self.model, "content": {"parts": [{"text": text}]}} for text in texts[:100]
        ]

        async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
            try:
                response = await client.post(url, json={"requests": requests_payload})
                if response.status_code == 429:
                    raise UpstreamProviderError(
                        message="Gemini batch embedding rate limit reached.",
                        status_code=503,
                        headers={"Retry-After": "30"},
                    )
                response.raise_for_status()
                data = response.json()
                embeddings = [item.get("values", []) for item in data.get("embeddings", [])]
                if len(texts) > 100:
                    remainder = await self._embed_batch_http(texts[100:])
                    embeddings.extend(remainder)
                return embeddings
            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                UpstreamProviderError,
            ):
                raise
            except Exception as exc:
                logger.warning(
                    "Batch embedding failed (%s), falling back to sequential calls.",
                    exc,
                )
                results = []
                for t in texts:
                    res = await self._embed_text_http(t)
                    results.append(res)
                return results


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic pseudo-semantic embeddings without network calls."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _generate_vector(self, text: str) -> list[float]:
        """Convert text into a deterministic, L2-normalized dense vector."""
        if not text:
            return [0.0] * self._dim

        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())

        if not tokens:
            return [0.0] * self._dim

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

        for token in tokens:
            if token in STOPWORDS:
                continue

            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = h % self._dim
            weight = 2.0 + (len(token) * 0.2)
            vector[index] += weight

            neighbor_index = (index + 7) % self._dim
            vector[neighbor_index] += weight * 0.2

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    async def embed_text(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]


# Backward compatibility aliases for deep imports
sys.modules["app.providers.embedding"] = sys.modules[__name__]
sys.modules["app.providers.embedding.base"] = sys.modules[__name__]
sys.modules["app.providers.embedding.gemini"] = sys.modules[__name__]
sys.modules["app.providers.embedding.mock"] = sys.modules[__name__]
