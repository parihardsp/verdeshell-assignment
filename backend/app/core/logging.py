"""Enterprise Centralized Context-Aware Logging System.

Provides request-, user-, and session-scoped structured logging using Python contextvars.
Every log entry adheres to the standardized enterprise format:
Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata

Logs are scrubbed for sensitive credentials (API keys, tokens, passwords) and organized
into date-based directories (logs/YYYY-MM-DD/) with automatic 7-day retention cleanup.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import sys
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path

# ==============================================================================
# 1. Context Variables (Tenant & Request Tracing)
# ==============================================================================

user_context: ContextVar[str] = ContextVar("user_context", default="system")
session_context: ContextVar[str] = ContextVar("session_context", default="global")
request_id_context: ContextVar[str] = ContextVar("request_id_context", default="system")
phase_context: ContextVar[str] = ContextVar("phase_context", default="SYSTEM")


def get_user_context() -> str:
    """Retrieve active user_id from contextvars."""
    return user_context.get()


def set_user_context(user_id: str | None) -> None:
    """Set active user_id for current async context."""
    if user_id and user_id.strip():
        user_context.set(user_id.strip())
    else:
        user_context.set("system")


def get_session_context() -> str:
    """Retrieve active session_id from contextvars."""
    return session_context.get()


def set_session_context(session_id: str | None) -> None:
    """Set active session_id for current async context."""
    if session_id and session_id.strip():
        session_context.set(session_id.strip())
    else:
        session_context.set("global")


def get_request_id_context() -> str:
    """Retrieve active request_id from contextvars."""
    return request_id_context.get()


def set_request_id_context(request_id: str | None) -> None:
    """Set active request_id for current async context."""
    if request_id and request_id.strip():
        request_id_context.set(request_id.strip())
    else:
        request_id_context.set("system")


def get_phase_context() -> str:
    """Retrieve active phase from contextvars."""
    return phase_context.get()


def set_phase_context(phase: str | None) -> None:
    """Set active phase for current async context."""
    if phase and phase.strip():
        phase_context.set(phase.strip().upper())
    else:
        phase_context.set("SYSTEM")


def set_request_context(
    user_id: str | None,
    session_id: str | None,
    request_id: str | None = None,
) -> None:
    """Simultaneously update request_id, user_id, and session_id contextvars."""
    set_user_context(user_id)
    set_session_context(session_id)
    if request_id is not None:
        set_request_id_context(request_id)


@contextmanager
def log_phase(phase: str):
    """Context manager to temporarily set active phase in contextvars."""
    token = phase_context.set(phase.strip().upper())
    try:
        yield
    finally:
        phase_context.reset(token)


def generate_standard_user_id(prefix: str = "usr") -> str:
    """Generate a standard unique user identifier (e.g. 'usr_9f1a2b')."""
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def generate_standard_session_id(prefix: str = "sess") -> str:
    """Generate a standard unique session identifier (e.g. 'sess_4d5e6f')."""
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def generate_friendly_session_id(prefix: str = "session") -> str:
    """Generate a standard unique session identifier."""
    return generate_standard_session_id(prefix=prefix)



# ==============================================================================
# 2. Standard Application Phases
# ==============================================================================


class LogPhase:
    """Standardized lifecycle phase constants."""

    STARTUP = "STARTUP"
    REQUEST = "REQUEST"
    AUTH = "AUTH"
    VALIDATION = "VALIDATION"
    INGESTION = "INGESTION"
    EXTRACTION = "EXTRACTION"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    RETRIEVAL = "RETRIEVAL"
    RERANKING = "RERANKING"
    GENERATION = "GENERATION"
    DATABASE = "DATABASE"
    VECTOR_STORE = "VECTOR_STORE"
    PROVIDER = "PROVIDER"
    RESPONSE = "RESPONSE"
    BACKGROUND = "BACKGROUND"
    SHUTDOWN = "SHUTDOWN"


DEFAULT_MODULE_PHASES: dict[str, str] = {
    "app.main": LogPhase.STARTUP,
    "app.db": LogPhase.DATABASE,
    "app.ingestion.validator": LogPhase.VALIDATION,
    "app.ingestion.storage": LogPhase.INGESTION,
    "app.ingestion.extractor": LogPhase.EXTRACTION,
    "app.rag.chunker": LogPhase.CHUNKING,
    "app.rag.embedder": LogPhase.EMBEDDING,
    "app.rag.retriever": LogPhase.RETRIEVAL,
    "app.rag.reranker": LogPhase.RERANKING,
    "app.rag.synthesizer": LogPhase.GENERATION,
    "app.rag.pipeline": LogPhase.RETRIEVAL,
    "app.providers.vector_store": LogPhase.VECTOR_STORE,
    "app.providers.embeddings": LogPhase.PROVIDER,
    "app.providers.llm": LogPhase.PROVIDER,
    "app.services.document_service": LogPhase.INGESTION,
    "app.services.chat_service": LogPhase.RETRIEVAL,
    "app.core.rate_limiter": LogPhase.REQUEST,
    "app.access": LogPhase.REQUEST,
    "uvicorn.access": LogPhase.REQUEST,
    "uvicorn.error": LogPhase.STARTUP,
    "uvicorn": LogPhase.STARTUP,
}


# ==============================================================================
# 3. Zero-Secret Redaction Pipeline
# ==============================================================================

SECRET_PATTERNS = [
    # Google API Keys (e.g. AQ.xxx, AIzaxxx)
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{15,}\b"), "[REDACTED_GEMINI_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z-_]{30,}\b"), "[REDACTED_GOOGLE_KEY]"),
    # Groq API Keys (e.g. gsk_xxx)
    (re.compile(r"\bgsk_[0-9A-Za-z]{20,}\b"), "[REDACTED_GROQ_KEY]"),
    # OpenAI & OpenRouter API Keys (e.g. sk-xxx)
    (re.compile(r"\bsk-[0-9A-Za-z]{20,}\b"), "[REDACTED_OPENAI_KEY]"),
    # Anthropic API Keys (e.g. sk-ant-xxx)
    (re.compile(r"\bsk-ant-[0-9A-Za-z_\-]{20,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    # LangChain / LangSmith API Keys (e.g. lsv2_xxx)
    (re.compile(r"\blsv2_[0-9A-Za-z_\-]{20,}\b"), "[REDACTED_LANGCHAIN_KEY]"),
    # Query parameters containing keys or tokens in URLs
    (
        re.compile(
            r"(?i)([?&](?:key|api_key|apikey|token|secret|access_token|password|auth)=)[^&\s\"\'\`]+"
        ),
        r"\g<1>[REDACTED]",
    ),
    # HTTP Authorization Bearer tokens
    (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9_\-\.]{15,}\b"),
        r"\g<1>[REDACTED]",
    ),
    # Headers in logged dicts or text (x-goog-api-key, authorization, x-api-key)
    (
        re.compile(
            r"(?i)([\"']?(?:x-goog-api-key|authorization|x-api-key)[\"']?\s*[:=]\s*[\"']?)[^'\",\s\`]+([\"']?)"
        ),
        r"\g<1>[REDACTED]\g<2>",
    ),
    # JSON key-value pairs with sensitive credentials
    (
        re.compile(
            r"(?i)([\"'](?:password|api_key|apiKey|secret|access_token|token)[\"']\s*:\s*[\"'])[^\"']+([\"'])"
        ),
        r"\g<1>[REDACTED]\g<2>",
    ),
]


def mask_sensitive_data(text: str) -> str:
    """Scrub sensitive credentials, tokens, and API keys from a given text string."""
    if not text or not isinstance(text, str):
        return text

    sanitized = text
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    try:
        from app.core.config import get_settings

        settings = get_settings()
        for secret_val in (
            getattr(settings, "GEMINI_API_KEY", None),
            getattr(settings, "GROQ_API_KEY", None),
            getattr(settings, "OPENAI_API_KEY", None),
            getattr(settings, "LANGCHAIN_API_KEY", None),
        ):
            if secret_val and len(secret_val) > 8 and secret_val in sanitized:
                sanitized = sanitized.replace(secret_val, "[REDACTED_API_KEY]")
    except Exception:
        pass

    return sanitized


class SecretMaskingFilter(logging.Filter):
    """Logging filter that scrubs sensitive API keys, tokens, and secrets from all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_sensitive_data(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (mask_sensitive_data(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_sensitive_data(a) if isinstance(a, str) else a
                    for a in record.args
                )
        if hasattr(record, "exc_text") and record.exc_text:
            record.exc_text = mask_sensitive_data(record.exc_text)
        return True


class UserSessionContextFilter(logging.Filter):
    """Logging filter injecting request_id, user_id, and session_id into LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id") or not record.request_id:
            record.request_id = get_request_id_context() or "system"
        if not hasattr(record, "user_id") or not record.user_id:
            record.user_id = get_user_context() or "system"
        if not hasattr(record, "session_id") or not record.session_id:
            record.session_id = get_session_context() or "global"
        return True


SessionContextFilter = UserSessionContextFilter


# ==============================================================================
# 4. Standard Enterprise Formatter (Vibrant ANSI Colors)
# ==============================================================================

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_BRIGHT_CYAN = "\033[96m"
ANSI_GREEN = "\033[32m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_YELLOW = "\033[33m"
ANSI_BRIGHT_YELLOW = "\033[93m"
ANSI_RED = "\033[31m"
ANSI_BRIGHT_RED = "\033[91m"
ANSI_MAGENTA = "\033[35m"
ANSI_BRIGHT_MAGENTA = "\033[95m"
ANSI_BLUE = "\033[34m"
ANSI_BRIGHT_BLUE = "\033[94m"
ANSI_GRAY = "\033[90m"

LEVEL_COLORS = {
    "DEBUG": ANSI_MAGENTA,
    "INFO": ANSI_BRIGHT_GREEN,
    "WARNING": ANSI_BRIGHT_YELLOW,
    "ERROR": ANSI_BRIGHT_RED,
    "CRITICAL": f"{ANSI_BOLD}{ANSI_BRIGHT_RED}",
}


class StandardLogFormatter(logging.Formatter):
    """Formats logs to enterprise standard with vibrant ANSI color highlights:
    Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_color: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.custom_fmt = fmt
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Backward-compatibility for explicit custom formats without standard columns
        if self.custom_fmt and "%(asctime)s" not in self.custom_fmt and "%(phase)s" not in self.custom_fmt:
            formatted = super().format(record)
            return mask_sensitive_data(formatted)

        # 1. Timestamp with millisecond precision: YYYY-MM-DD HH:MM:SS.mmm
        t = datetime.fromtimestamp(record.created)
        timestamp_str = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 2. Level
        level_str = record.levelname

        # 3. Phase resolution
        phase_str = getattr(record, "phase", None)
        if not phase_str:
            ctx_phase = get_phase_context()
            if ctx_phase and ctx_phase != "SYSTEM":
                phase_str = ctx_phase
            else:
                for mod_prefix, def_phase in DEFAULT_MODULE_PHASES.items():
                    if record.name == mod_prefix or record.name.startswith(f"{mod_prefix}."):
                        phase_str = def_phase
                        break
                if not phase_str:
                    phase_str = ctx_phase or "SYSTEM"

        # 4. Context identifiers
        req_id = getattr(record, "request_id", None) or get_request_id_context() or "system"
        u_id = getattr(record, "user_id", None) or get_user_context() or "system"
        s_id = getattr(record, "session_id", None) or get_session_context() or "global"

        # 5. Module
        module_name = record.name

        # 6. Message & Structured Metadata
        raw_msg = record.getMessage()

        metadata_val = getattr(record, "metadata", None)
        metadata_str = ""
        if isinstance(metadata_val, dict):
            metadata_str = " ".join(f"{k}={v}" for k, v in metadata_val.items())
        elif isinstance(metadata_val, str) and metadata_val.strip():
            metadata_str = metadata_val.strip()

        if metadata_str:
            if " | " in raw_msg:
                composed_msg = f"{raw_msg} {metadata_str}"
            else:
                composed_msg = f"{raw_msg} | {metadata_str}"
        else:
            composed_msg = raw_msg

        if self.use_color:
            color_level = LEVEL_COLORS.get(level_str, ANSI_RESET)
            c_timestamp = f"{ANSI_CYAN}{timestamp_str}{ANSI_RESET}"
            c_level = f"{color_level}{level_str}{ANSI_RESET}"
            c_phase = f"{ANSI_BRIGHT_MAGENTA}{phase_str}{ANSI_RESET}"
            c_req = f"{ANSI_YELLOW}{req_id}{ANSI_RESET}"
            c_user = f"{ANSI_BRIGHT_CYAN}{u_id}{ANSI_RESET}"
            c_sess = f"{ANSI_BRIGHT_BLUE}{s_id}{ANSI_RESET}"
            c_mod = f"{ANSI_GRAY}{module_name}{ANSI_RESET}"
            sep = f"{ANSI_GRAY}|{ANSI_RESET}"
            log_line = (
                f"{c_timestamp} {sep} {c_level} {sep} {c_phase} {sep} {c_req} {sep} "
                f"{c_user} {sep} {c_sess} {sep} {c_mod} {sep} {composed_msg}"
            )
        else:
            log_line = (
                f"{timestamp_str} | {level_str} | {phase_str} | {req_id} | {u_id} | {s_id} | "
                f"{module_name} | {composed_msg}"
            )

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            log_line += f"\n{record.exc_text}"
        elif record.stack_info:
            log_line += f"\n{self.formatStack(record.stack_info)}"

        return mask_sensitive_data(log_line)


# Backward-compatible alias
SecretMaskingFormatter = StandardLogFormatter


# ==============================================================================
# 5. Persistent Date-Based File Handler & 7-Day Retention
# ==============================================================================


def cleanup_old_logs(logs_root: Path, retention_days: int = 7) -> int:
    """Purge date-based log subdirectories older than `retention_days`. Returns purged count."""
    if not logs_root.exists():
        return 0

    purged = 0
    cutoff = (datetime.now() - timedelta(days=retention_days)).date()
    try:
        for item in logs_root.iterdir():
            if item.is_dir():
                try:
                    item_date = datetime.strptime(item.name, "%Y-%m-%d").date()
                    if item_date < cutoff:
                        shutil.rmtree(item, ignore_errors=True)
                        purged += 1
                except ValueError:
                    continue
    except Exception:
        pass
    return purged


class DailyDatedFileHandler(logging.Handler):
    """Thread-safe logging handler routing records to date-partitioned files:
    logs/YYYY-MM-DD/{filename}
    Automatically creates daily folders upon date transition and executes retention cleanup.
    """

    def __init__(
        self,
        logs_root: Path,
        filename: str,
        retention_days: int = 7,
        min_level: int = logging.INFO,
        max_level: int | None = None,
    ):
        super().__init__(level=min_level)
        self.logs_root = Path(logs_root)
        self.filename = filename
        self.retention_days = retention_days
        self.max_level = max_level
        self._current_date: str | None = None
        self._stream: io.TextIOWrapper | None = None
        self._lock = threading.Lock()

    def _get_stream(self, today_str: str) -> io.TextIOWrapper:
        if self._stream is None or self._current_date != today_str:
            if self._stream is not None:
                try:
                    self._stream.flush()
                    self._stream.close()
                except Exception:
                    pass
            daily_dir = self.logs_root / today_str
            daily_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = daily_dir / self.filename
            self._stream = open(log_file_path, "a", encoding="utf-8")
            self._current_date = today_str
            cleanup_old_logs(self.logs_root, self.retention_days)
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        if self.max_level is not None and record.levelno > self.max_level:
            return
        try:
            msg = self.format(record)
            today_str = datetime.now().strftime("%Y-%m-%d")
            with self._lock:
                stream = self._get_stream(today_str)
                stream.write(msg + "\n")
                stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.flush()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        super().close()


# ==============================================================================
# 6. Centralized Logging Configuration
# ==============================================================================


def setup_logging(
    level: int = logging.INFO,
    logs_dir: Path | str | None = None,
    retention_days: int = 7,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """Configure centralized application logging with standard user, session, and request tracing."""
    formatter = StandardLogFormatter()
    context_filter = UserSessionContextFilter()
    secret_filter = SecretMaskingFilter()

    # Determine logs root directory
    if logs_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resolved_logs_dir = project_root / "logs"
    else:
        resolved_logs_dir = Path(logs_dir).resolve()

    resolved_logs_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_logs(resolved_logs_dir, retention_days=retention_days)

    handlers: list[logging.Handler] = []

    # 1. Console Stream Handler (Development / Standard Output)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        console_handler.addFilter(secret_filter)
        handlers.append(console_handler)

    # 2. Daily Application Log (INFO and above)
    app_file_handler = None
    if enable_file:
        app_file_handler = DailyDatedFileHandler(
            logs_root=resolved_logs_dir,
            filename="application.log",
            retention_days=retention_days,
            min_level=level,
        )
        app_file_handler.setFormatter(formatter)
        app_file_handler.addFilter(context_filter)
        app_file_handler.addFilter(secret_filter)
        handlers.append(app_file_handler)

    # 3. Daily Error Log (WARNING and ERROR/CRITICAL)
    if enable_file:
        error_file_handler = DailyDatedFileHandler(
            logs_root=resolved_logs_dir,
            filename="error.log",
            retention_days=retention_days,
            min_level=logging.WARNING,
        )
        error_file_handler.setFormatter(formatter)
        error_file_handler.addFilter(context_filter)
        error_file_handler.addFilter(secret_filter)
        handlers.append(error_file_handler)

    # 4. Daily Access Log (Dedicated for HTTP requests)
    access_handlers: list[logging.Handler] = []
    if enable_console:
        access_handlers.append(console_handler)
    if enable_file and app_file_handler:
        access_handlers.append(app_file_handler)

    if enable_file:
        access_file_handler = DailyDatedFileHandler(
            logs_root=resolved_logs_dir,
            filename="access.log",
            retention_days=retention_days,
            min_level=logging.INFO,
        )
        access_file_handler.setFormatter(formatter)
        access_file_handler.addFilter(context_filter)
        access_file_handler.addFilter(secret_filter)
        access_handlers.append(access_file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    for h in handlers:
        root_logger.addHandler(h)

    # Attach to key application loggers
    for logger_name in ("app", "uvicorn", "uvicorn.error"):
        al = logging.getLogger(logger_name)
        al.handlers = []
        for h in handlers:
            al.addHandler(h)
        al.propagate = False

    # Configure dedicated access logger
    for access_logger_name in ("app.access", "uvicorn.access"):
        acl = logging.getLogger(access_logger_name)
        acl.handlers = []
        for h in access_handlers:
            acl.addHandler(h)
        acl.propagate = False

    # Suppress verbose third-party HTTP transport loggers
    for noisy_logger in ("httpx", "httpcore"):
        nl = logging.getLogger(noisy_logger)
        nl.setLevel(logging.WARNING)
        nl.handlers = []
        for h in handlers:
            nl.addHandler(h)
        nl.propagate = False
