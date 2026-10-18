"""FastAPI Application Entry Point for Mini Knowledge Assistant.

Configures lifespan lifecycle hooks, CORS middleware, global RFC 7807 exception
handlers, and dual-mounted API routing (root and /api/v1).
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import hydrate_vector_store
from app.api.routes import api_router
from app.core.config import get_settings
from app.core.exceptions import (
    AppException,
    app_exception_handler,
    unhandled_exception_handler,
)
from app.core.logging import LogPhase, set_phase_context, set_request_context, setup_logging
from app.db import AsyncSessionLocal, engine, init_db

# Initialize centralized context-aware logging
setup_logging()
logger = logging.getLogger("app")
access_logger = logging.getLogger("app.access")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize database tables and hydrate vector index."""
    setup_logging()
    logger.info(
        "Service startup initiated | app=%s version=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        extra={"phase": LogPhase.STARTUP},
    )
    # 1. Create SQLite tables
    await init_db()
    logger.info("Database schema initialized | status=ready", extra={"phase": LogPhase.DATABASE})

    # 2. Pre-populate in-memory vector store from SQLite
    async with AsyncSessionLocal() as session:
        hydrated_count = await hydrate_vector_store(session)
        logger.info(
            "Vector store index ready | chunks=%d",
            hydrated_count,
            extra={"phase": LogPhase.STARTUP},
        )

    if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
        logger.info(
            "Distributed tracing active | project=%s endpoint=%s",
            settings.LANGCHAIN_PROJECT,
            settings.LANGCHAIN_ENDPOINT,
            extra={"phase": LogPhase.STARTUP},
        )

    yield

    # Teardown
    logger.info("Service shutdown initiated | engine=sqlite", extra={"phase": LogPhase.SHUTDOWN})
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade asynchronous RAG API for internal knowledge assistance.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After", "X-Request-ID", "X-User-ID", "X-Session-ID"],
)


@app.middleware("http")
async def user_session_context_middleware(request: Request, call_next):
    """Propagate request_id, user, and session identifiers into contextvars, request.state, cookies, and response headers."""
    # 1. Request ID (honor incoming X-Request-ID or generate new req_xxxx)
    request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("request-id")
        or f"req_{uuid.uuid4().hex[:8]}"
    )

    # 2. Extract User ID from headers, cookies, or query params
    user_id = (
        request.headers.get("x-user-id")
        or request.headers.get("user-id")
        or request.cookies.get("user_id")
        or request.query_params.get("user_id")
        or "default_user"
    )

    # 3. Extract Session ID from headers, cookies, or query params
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or request.cookies.get("session_id")
        or request.query_params.get("session_id")
        or "default"
    )

    # 4. Propagate into contextvars for centralized logging
    set_request_context(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
    )
    set_phase_context(LogPhase.REQUEST)

    # 5. Attach to request.state
    request.state.request_id = request_id
    request.state.user_id = user_id
    request.state.session_id = session_id

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        client_ip = request.client.host if request.client else "127.0.0.1"

        access_logger.info(
            "HTTP request completed | method=%s path=%s status=%d latency_ms=%.1f client_ip=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            client_ip,
            extra={"phase": LogPhase.REQUEST},
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        client_ip = request.client.host if request.client else "127.0.0.1"
        access_logger.error(
            "HTTP request failed | method=%s path=%s error=%s latency_ms=%.1f client_ip=%s",
            request.method,
            request.url.path,
            type(exc).__name__,
            duration_ms,
            client_ip,
            extra={"phase": LogPhase.REQUEST},
        )
        raise

    # 6. Echo context headers and set cookies for browser automatic identity persistence
    response.headers["X-Request-ID"] = request_id
    response.headers["X-User-ID"] = user_id
    if session_id != "default":
        response.headers["X-Session-ID"] = session_id
    if "user_id" not in request.cookies and user_id != "default_user":
        response.set_cookie("user_id", user_id, httponly=False, samesite="lax")
    if "session_id" not in request.cookies and session_id != "default":
        response.set_cookie("session_id", session_id, httponly=False, samesite="lax")
    return response


# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# 1. Versioned API routes (Primary documented interface in Swagger UI /docs)
app.include_router(api_router, prefix="/api/v1")

# 2. Hidden root compatibility routes (Avoids duplicate entries in Swagger docs)
app.include_router(api_router, include_in_schema=False)


@app.get("/", tags=["Root"], summary="API Root Overview")
async def root():
    """Welcome endpoint providing service metadata and documentation links."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "add_document": "POST /api/v1/documents",
            "list_documents": "GET /api/v1/documents",
            "ask_question": "POST /api/v1/ask",
            "health_check": "GET /api/v1/health",
        },
    }
