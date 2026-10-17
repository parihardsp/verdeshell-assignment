"""Unit tests for SlidingWindowRateLimiter."""

import pytest

from app.core.exceptions import RateLimitExceededError
from app.core.rate_limiter import SlidingWindowRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    client_ip = "192.168.1.100"
    for _ in range(5):
        await limiter.check(client_ip)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    client_ip = "192.168.1.101"
    for _ in range(3):
        await limiter.check(client_ip)

    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.check(client_ip)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after > 0
    assert "Retry-After" in exc_info.value.headers
