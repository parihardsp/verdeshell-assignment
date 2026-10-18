"""Embedding Provider abstractions and implementations.

Includes BaseEmbeddingProvider, GeminiEmbeddingProvider, and MockEmbeddingProvider.
"""

import asyncio
import hashlib
import logging
import random
import re
import sys
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import UpstreamProviderError
from app.core.logging import LogPhase, mask_sensitive_data

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
    """Generates embeddings using Google Generative Language API with retry backoff and LangSmith tracing."""

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-2",
        timeout_seconds: float = 15.0,
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
                "Embedding provider initialized | provider=gemini model=%s engine=langchain",
                self.model,
                extra={"phase": LogPhase.PROVIDER},
            )
        except Exception as exc:
            logger.warning(
                "Embedding provider initialization fallback | provider=gemini model=%s engine=direct_rest",
                self.model,
                extra={"phase": LogPhase.PROVIDER},
            )

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string with 2 retries and 5s exponential backoff."""
        max_retries = 2
        base_delay = 5.0

        for attempt in range(max_retries + 1):
            try:
                if self._use_langchain and self._embedder:
                    try:
                        return await self._embedder.aembed_query(text)
                    except Exception as lc_exc:
                        lc_err_str = str(lc_exc).lower()
                        is_quota = (
                            "429" in lc_err_str
                            or "resource_exhausted" in lc_err_str
                            or "quota" in lc_err_str
                        )
                        if is_quota:
                            if attempt < max_retries:
                                delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                                logger.warning(
                                    "Provider rate limit encountered, backing off | provider=gemini engine=langchain attempt=%d max_retries=%d backoff_s=%.1f",
                                    attempt + 1,
                                    max_retries,
                                    delay,
                                    extra={"phase": LogPhase.PROVIDER},
                                )
                                await asyncio.sleep(delay)
                                continue
                            raise UpstreamProviderError(
                                message="Gemini embedding API rate limit reached.",
                                status_code=429,
                                headers={"Retry-After": "30"},
                            )
                        logger.warning(
                            "Provider invocation failed, falling back to direct REST | provider=gemini engine=langchain error=%s",
                            type(lc_exc).__name__,
                            extra={"phase": LogPhase.PROVIDER},
                        )

                return await self._embed_text_http(text)

            except UpstreamProviderError as up_err:
                if up_err.status_code in (429, 503) and attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider rate limit encountered, backing off | provider=gemini engine=direct_rest attempt=%d max_retries=%d backoff_s=%.1f",
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider request timed out, retrying | provider=gemini engine=direct_rest attempt=%d max_retries=%d backoff_s=%.1f",
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamProviderError(
                    "Timeout connecting to Gemini embedding service.", status_code=504
                )
            except Exception as exc:
                if attempt < max_retries and (
                    "429" in str(exc) or "quota" in str(exc).lower()
                ):
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider rate limit encountered, backing off | provider=gemini attempt=%d max_retries=%d backoff_s=%.1f",
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Embedding generation failed | provider=gemini error=%s",
                    type(exc).__name__,
                    extra={"phase": LogPhase.PROVIDER},
                )
                raise UpstreamProviderError(
                    f"Embedding generation failed: {mask_sensitive_data(str(exc))}"
                )

        raise UpstreamProviderError(
            "Gemini embedding API rate limit reached after retries.", status_code=429
        )

    async def _embed_sub_batch_with_retry(
        self, batch: list[str], max_retries: int = 2, base_delay: float = 5.0
    ) -> list[list[float]]:
        """Embed a sub-batch of texts with 2 retries and exponential backoff (starting at 5s)."""
        for attempt in range(max_retries + 1):
            try:
                if self._use_langchain and self._embedder:
                    try:
                        return await self._embedder.aembed_documents(batch)
                    except Exception as lc_exc:
                        lc_err_str = str(lc_exc).lower()
                        is_quota = (
                            "429" in lc_err_str
                            or "resource_exhausted" in lc_err_str
                            or "quota" in lc_err_str
                        )
                        if is_quota:
                            if attempt < max_retries:
                                delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                                logger.warning(
                                    "Provider rate limit encountered, backing off | provider=gemini engine=langchain batch_size=%d attempt=%d max_retries=%d backoff_s=%.1f",
                                    len(batch),
                                    attempt + 1,
                                    max_retries,
                                    delay,
                                    extra={"phase": LogPhase.PROVIDER},
                                )
                                await asyncio.sleep(delay)
                                continue
                            raise UpstreamProviderError(
                                message="Gemini batch embedding rate limit reached.",
                                status_code=429,
                                headers={"Retry-After": "30"},
                            )
                        logger.warning(
                            "Provider batch invocation failed, falling back to direct REST | provider=gemini engine=langchain error=%s",
                            type(lc_exc).__name__,
                            extra={"phase": LogPhase.PROVIDER},
                        )

                # Direct REST fallback using header authentication (zero keys in URL)
                return await self._embed_batch_http(batch)

            except UpstreamProviderError as up_err:
                if up_err.status_code in (429, 503) and attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider rate limit encountered, backing off | provider=gemini engine=direct_rest batch_size=%d attempt=%d max_retries=%d backoff_s=%.1f",
                        len(batch),
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider request timed out, retrying | provider=gemini engine=direct_rest batch_size=%d attempt=%d max_retries=%d backoff_s=%.1f",
                        len(batch),
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamProviderError(
                    "Timeout connecting to Gemini embedding service.", status_code=504
                )
            except Exception as exc:
                if attempt < max_retries and (
                    "429" in str(exc) or "quota" in str(exc).lower()
                ):
                    delay = base_delay * (2**attempt) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Provider rate limit encountered, backing off | provider=gemini batch_size=%d attempt=%d max_retries=%d backoff_s=%.1f",
                        len(batch),
                        attempt + 1,
                        max_retries,
                        delay,
                        extra={"phase": LogPhase.PROVIDER},
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Batch embedding generation failed | provider=gemini error=%s",
                    type(exc).__name__,
                    extra={"phase": LogPhase.PROVIDER},
                )
                raise UpstreamProviderError(
                    f"Embedding generation failed: {mask_sensitive_data(str(exc))}"
                )

        raise UpstreamProviderError(
            "Gemini batch embedding rate limit reached after retries.", status_code=429
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings, partitioned into sub-batches with polite rate-limit pacing."""
        if not texts:
            return []

        # Google Gemini batchEmbedContents works best with <= 20 texts per request to respect RPM & TPM limits
        sub_batch_size = 20
        all_embeddings: list[list[float]] = []
        total_sub_batches = (len(texts) + sub_batch_size - 1) // sub_batch_size

        for idx in range(0, len(texts), sub_batch_size):
            batch = texts[idx : idx + sub_batch_size]
            sub_batch_num = (idx // sub_batch_size) + 1
            if total_sub_batches > 1:
                logger.info(
                    "Batch embedding sub-batch dispatched | provider=gemini sub_batch=%d/%d chunks=%d",
                    sub_batch_num,
                    total_sub_batches,
                    len(batch),
                    extra={"phase": LogPhase.EMBEDDING},
                )

            batch_embeddings = await self._embed_sub_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            # Pacing sleep between consecutive sub-batches to respect rate limits
            if idx + sub_batch_size < len(texts):
                await asyncio.sleep(0.35)

        return all_embeddings

    async def _embed_text_http(self, text: str) -> list[float]:
        """Embed a single text string via REST fallback using header authentication (no keys in URL)."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    raise UpstreamProviderError(
                        message="Gemini embedding API rate limit reached.",
                        status_code=429,
                        headers={"Retry-After": "30"},
                    )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", {}).get("values", [])
                if not embedding:
                    raise UpstreamProviderError("Gemini returned empty embedding values.")
                return embedding
            except httpx.TimeoutException:
                raise
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Embedding HTTP request failed | provider=gemini status=%d",
                    exc.response.status_code,
                    extra={"phase": LogPhase.PROVIDER},
                )
                raise UpstreamProviderError(
                    f"Gemini embedding API returned status {exc.response.status_code}.",
                    status_code=502,
                )

    async def _embed_batch_http(self, texts: list[str]) -> list[list[float]]:
        """Embed a sub-batch of text strings using batchEmbedContents REST fallback with header auth."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        requests_payload = [
            {"model": self.model, "content": {"parts": [{"text": text}]}}
            for text in texts
        ]

        async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
            response = await client.post(
                url, headers=headers, json={"requests": requests_payload}
            )
            if response.status_code == 429:
                raise UpstreamProviderError(
                    message="Gemini batch embedding rate limit reached.",
                    status_code=429,
                    headers={"Retry-After": "30"},
                )
            if response.status_code != 200:
                logger.error(
                    "Batch embedding HTTP request failed | provider=gemini status=%d",
                    response.status_code,
                    extra={"phase": LogPhase.PROVIDER},
                )
                raise UpstreamProviderError(
                    f"Gemini embedding API returned status {response.status_code}.",
                    status_code=502,
                )
            data = response.json()
            embeddings = [item.get("values", []) for item in data.get("embeddings", [])]
            return embeddings


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
