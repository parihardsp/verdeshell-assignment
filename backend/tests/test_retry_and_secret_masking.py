"""Tests for 429 rate limit exponential backoff retry and zero-secret logging redaction."""

import io
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.exceptions import UpstreamProviderError
from app.core.logging import (
    SecretMaskingFilter,
    SecretMaskingFormatter,
    mask_sensitive_data,
)
from app.providers.embeddings import GeminiEmbeddingProvider
from app.providers.llm import GeminiLLMProvider


def test_mask_sensitive_data_patterns():
    """Verify that all known sensitive keys and tokens are redacted."""
    raw_text = (
        "POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
        "?key=AQ.MockSecretKeyForAutomatedTesting1234567890 "
        "Authorization: Bearer gsk_1234567890abcdef1234567890abcdef"
    )
    scrubbed = mask_sensitive_data(raw_text)

    assert "AQ.MockSecretKey" not in scrubbed
    assert "gsk_1234567890abcdef" not in scrubbed
    assert "key=[REDACTED]" in scrubbed
    assert "[REDACTED" in scrubbed


def test_secret_masking_formatter_stream():
    """Verify that a logger using SecretMaskingFormatter scrubs logs before writing."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        SecretMaskingFormatter(fmt="%(levelname)s - %(message)s")
    )

    test_logger = logging.getLogger("test_masking_logger")
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False

    test_logger.info(
        "Request failed with key: AIzaSyD9876543210abcdefghijklmnop12345678"
    )

    output = stream.getvalue()
    assert "AIzaSyD9876543210abcdefghijklmnop12345678" not in output
    assert "[REDACTED_GOOGLE_KEY]" in output


@pytest.mark.asyncio
async def test_gemini_embedding_retries_on_429():
    """Verify that GeminiEmbeddingProvider retries 2 times on 429 and succeeds on 3rd attempt."""
    provider = GeminiEmbeddingProvider(
        api_key="AQ.mock_gemini_key_for_unit_tests",
        model="models/gemini-embedding-2",
    )
    provider._use_langchain = False

    # Mock responses: 429, 429, then 200 with embeddings
    mock_429 = httpx.Response(
        status_code=429,
        json={"error": {"code": 429, "message": "Resource exhausted"}},
        request=httpx.Request("POST", "https://mock.google.api"),
    )
    mock_200 = httpx.Response(
        status_code=200,
        json={"embeddings": [{"values": [0.1] * 3072}]},
        request=httpx.Request("POST", "https://mock.google.api"),
    )

    mock_post = AsyncMock(side_effect=[mock_429, mock_429, mock_200])

    with (
        patch("httpx.AsyncClient.post", mock_post),
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await provider._embed_sub_batch_with_retry(
            ["sample text chunk"], max_retries=2, base_delay=0.01
        )

        assert len(result) == 1
        assert len(result[0]) == 3072
        assert mock_post.call_count == 3
        # Should have slept twice (once after each 429)
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_gemini_embedding_exhausted_retries_raises_429():
    """Verify that if all 2 retries fail on 429, UpstreamProviderError is raised."""
    provider = GeminiEmbeddingProvider(
        api_key="AQ.mock_gemini_key_for_unit_tests",
        model="models/gemini-embedding-2",
    )
    provider._use_langchain = False

    mock_429 = httpx.Response(
        status_code=429,
        json={"error": {"code": 429, "message": "Resource exhausted"}},
        request=httpx.Request("POST", "https://mock.google.api"),
    )
    mock_post = AsyncMock(side_effect=[mock_429, mock_429, mock_429])

    with (
        patch("httpx.AsyncClient.post", mock_post),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(UpstreamProviderError) as exc_info:
            await provider._embed_sub_batch_with_retry(
                ["sample text chunk"], max_retries=2, base_delay=0.01
            )
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_gemini_embedding_batch_partitioning():
    """Verify that a batch of 45 texts is partitioned into sub-batches of 20, 20, 5."""
    provider = GeminiEmbeddingProvider(
        api_key="AQ.mock_gemini_key_for_unit_tests",
        model="models/gemini-embedding-2",
    )
    provider._use_langchain = False

    texts = [f"Text chunk #{i}" for i in range(45)]
    calls = []

    async def mock_sub_batch(batch, **kwargs):
        calls.append(len(batch))
        return [[0.05] * 3072 for _ in batch]

    with (
        patch.object(provider, "_embed_sub_batch_with_retry", side_effect=mock_sub_batch),
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        embeddings = await provider.embed_batch(texts)
        assert len(embeddings) == 45
        assert calls == [20, 20, 5]
        # Slept between sub-batches
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_header_auth_no_key_in_url():
    """Verify that REST calls to Gemini send x-goog-api-key in headers and not in URL query string."""
    provider = GeminiEmbeddingProvider(
        api_key="AQ.secret_gemini_api_key_12345",
        model="models/gemini-embedding-2",
    )
    captured_urls = []
    captured_headers = []

    mock_resp = httpx.Response(
        status_code=200,
        json={"embeddings": [{"values": [0.1] * 3072}]},
        request=httpx.Request("POST", "https://mock.google.api"),
    )

    async def mock_post(url, headers=None, json=None):
        captured_urls.append(url)
        captured_headers.append(headers or {})
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await provider._embed_batch_http(["test text"])

    assert len(captured_urls) == 1
    assert "key=" not in captured_urls[0]
    assert captured_headers[0].get("x-goog-api-key") == "AQ.secret_gemini_api_key_12345"
