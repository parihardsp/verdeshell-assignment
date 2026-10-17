"""Sliding window rate limiter for protecting endpoints against abusive bursts.

Calculates exact `Retry-After` seconds and supports per-client IP throttling.
"""

import asyncio
import time
from collections import defaultdict

from fastapi import Request

from app.core.config import get_settings
from app.core.exceptions import RateLimitExceededError


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter.

    Tracks request timestamps within a moving window per client key (IP address).
    Thread-safe and async-compatible.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, client_id: str) -> None:
        """Verify whether the client is within the permitted request volume.

        Raises:
            RateLimitExceededError: If the client has exceeded `max_requests` in `window_seconds`.
        """
        now = time.time()
        window_start = now - self.window_seconds

        async with self._lock:
            timestamps = self._history[client_id]

            # Prune timestamps outside current sliding window
            self._history[client_id] = [ts for ts in timestamps if ts > window_start]
            valid_timestamps = self._history[client_id]

            if len(valid_timestamps) >= self.max_requests:
                # The oldest timestamp in the window determines when a slot frees up
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1, int(oldest_timestamp + self.window_seconds - now))
                raise RateLimitExceededError(
                    retry_after=retry_after,
                    message=f"Rate limit exceeded: maximum {self.max_requests} requests per {self.window_seconds}s.",
                    details={
                        "limit": self.max_requests,
                        "window_seconds": self.window_seconds,
                        "current_usage": len(valid_timestamps),
                    },
                )

            # Record current valid request
            self._history[client_id].append(now)

    def reset(self) -> None:
        """Clear all rate limit history (useful for tests)."""
        self._history.clear()


# Default singleton instance using application settings
_settings = get_settings()
global_rate_limiter = SlidingWindowRateLimiter(
    max_requests=_settings.RATE_LIMIT_REQUESTS,
    window_seconds=_settings.RATE_LIMIT_WINDOW_SECONDS,
)


def get_client_ip(request: Request) -> str:
    """Extract client IP address considering proxy headers like X-Forwarded-For."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the leftmost IP from comma-separated list
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


async def rate_limit_ask(request: Request) -> None:
    """FastAPI dependency to rate limit requests to POST /ask."""
    client_ip = get_client_ip(request)
    await global_rate_limiter.check(client_ip)
