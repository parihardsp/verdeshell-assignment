"""Domain exceptions and RFC 7807 compliant error responses.

Provides standard HTTP status code mapping and prevents leaking internal stack
traces or provider API keys to external clients.
"""

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.headers = headers or {}


class EntityNotFoundError(AppException):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None):
        super().__init__(message=message, status_code=status.HTTP_404_NOT_FOUND, details=details)


class ValidationError(AppException):
    """Raised when business logic validation fails."""

    def __init__(self, message: str = "Validation failed", details: dict[str, Any] | None = None):
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST, details=details)


class PayloadTooLargeError(AppException):
    """Raised when an uploaded document exceeds size limits."""

    def __init__(
        self,
        message: str = "Payload exceeds maximum allowed size",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            details=details,
        )


class RateLimitExceededError(AppException):
    """Raised when a client exceeds the allowed request window."""

    def __init__(
        self,
        retry_after: int,
        message: str = "Rate limit exceeded. Please wait before retrying.",
        details: dict[str, Any] | None = None,
    ):
        headers = {"Retry-After": str(retry_after)}
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            headers=headers,
        )
        self.retry_after = retry_after


class UpstreamProviderError(AppException):
    """Raised when an external LLM or Embedding provider fails or times out."""

    def __init__(
        self,
        message: str = "Upstream AI provider unavailable or timed out.",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details, headers=headers)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions with structured error responses."""
    content = {
        "error": {
            "type": exc.__class__.__name__,
            "message": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
        }
    }
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Mask unexpected internal server errors to prevent leaking stack traces or sensitive keys."""
    content = {
        "error": {
            "type": "InternalServerError",
            "message": "An unexpected error occurred while processing your request.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "path": request.url.path,
        }
    }
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)
