"""FastAPI Application Entry Point for Mini Knowledge Assistant.

Configures lifespan lifecycle hooks, CORS middleware, global RFC 7807 exception
handlers, and dual-mounted API routing (root and /api/v1).
"""

import logging
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
from app.core.logging import setup_logging
from app.db import AsyncSessionLocal, engine, init_db

# Initialize centralized context-aware logging
setup_logging()
logger = logging.getLogger("app")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize database tables and hydrate vector index."""
    logger.info("Initializing Mini Knowledge Assistant backend...")
    # 1. Create SQLite tables
    await init_db()

    # 2. Pre-populate in-memory vector store from SQLite
    async with AsyncSessionLocal() as session:
        hydrated_count = await hydrate_vector_store(session)
        logger.info("Vector index ready with %d chunks.", hydrated_count)

    if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
        logger.info(
            "LangSmith Tracing active: project='%s' | endpoint='%s'",
            settings.LANGCHAIN_PROJECT,
            settings.LANGCHAIN_ENDPOINT,
        )

    yield

    # Teardown
    logger.info("Shutting down database engine...")
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
    expose_headers=["Retry-After", "X-User-ID", "X-Session-ID"],
)


@app.middleware("http")
async def user_session_context_middleware(request: Request, call_next):
    """Propagate user and session identifiers into contextvars, request.state, cookies, and response headers."""
    from app.core.logging import (
        set_request_context,
    )

    # 1. Extract User ID from headers, cookies, or query params
    user_id = (
        request.headers.get("x-user-id")
        or request.headers.get("user-id")
        or request.cookies.get("user_id")
        or request.query_params.get("user_id")
        or "default_user"
    )

    # 2. Extract Session ID from headers, cookies, or query params
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or request.cookies.get("session_id")
        or request.query_params.get("session_id")
        or "default"
    )

    # 3. Propagate into contextvars for centralized logging
    set_request_context(
        user_id=user_id,
        session_id=session_id,
    )

    # 4. Attach to request.state
    request.state.user_id = user_id
    request.state.session_id = session_id

    response = await call_next(request)

    # 5. Echo context headers and set cookies for browser automatic identity persistence
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
