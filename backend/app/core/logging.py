"""Centralized Context-Aware Logging System.

Provides request-, user-, and session-scoped logging using Python contextvars.
Every log entry automatically includes the active [user_id:session_id],
enabling seamless end-to-end tracing across upload, extraction, chunking,
retrieval, reranking, and LLM synthesis.
"""

import logging
import sys
import uuid
from contextvars import ContextVar

# Global context variables tracking active user and session identifiers
user_context: ContextVar[str] = ContextVar("user_context", default="system")
session_context: ContextVar[str] = ContextVar("session_context", default="global")


def get_user_context() -> str:
    """Retrieve the current user identifier from contextvars."""
    return user_context.get()


def set_user_context(user_id: str | None) -> None:
    """Set the user identifier for the current async task context."""
    if user_id and user_id.strip():
        user_context.set(user_id.strip())
    else:
        user_context.set("system")


def get_session_context() -> str:
    """Retrieve the current session identifier from contextvars."""
    return session_context.get()


def set_session_context(session_id: str | None) -> None:
    """Set the session identifier for the current async task context."""
    if session_id and session_id.strip():
        session_context.set(session_id.strip())
    else:
        session_context.set("global")


def set_request_context(user_id: str | None, session_id: str | None) -> None:
    """Simultaneously update both user and session contextvars for a request lifecycle."""
    set_user_context(user_id)
    set_session_context(session_id)


def generate_standard_user_id(prefix: str = "usr") -> str:
    """Generate a standard unique user identifier (e.g. 'usr_9f1a2b3c')."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def generate_standard_session_id(prefix: str = "sess") -> str:
    """Generate a standard unique session identifier (e.g. 'sess_4d5e6f7a')."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# Backward-compatible alias
def generate_friendly_session_id(prefix: str = "session") -> str:
    """Generate a standard unique session identifier."""
    return generate_standard_session_id(prefix=prefix)


class UserSessionContextFilter(logging.Filter):
    """Logging filter that injects current [user_id] and [session_id] into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = get_user_context()
        record.session_id = get_session_context()
        return True


# Backward-compatible alias
SessionContextFilter = UserSessionContextFilter


def setup_logging(level: int = logging.INFO) -> None:
    """Configure centralized application logging with standard user and session tagging."""
    log_format = "%(asctime)s [%(levelname)s] [%(user_id)s:%(session_id)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(UserSessionContextFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on reload
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Attach filter and handler to key application loggers
    for logger_name in ("app", "uvicorn", "uvicorn.error"):
        app_logger = logging.getLogger(logger_name)
        app_logger.handlers = []
        app_logger.addHandler(handler)
        app_logger.propagate = False
