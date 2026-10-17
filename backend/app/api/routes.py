"""Consolidated API router exposing Health, Document, and Q&A endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_document_service,
    get_embedding_provider,
    get_llm_provider,
    get_rag_service,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.core.rate_limiter import get_client_ip, rate_limit_ask
from app.db.database import get_db
from app.db.models import Chunk, Document
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.llm import BaseLLMProvider
from app.providers.vector_store import BaseVectorStore
from app.schemas.common import HealthResponse
from app.schemas.document import (
    DocumentCreate,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    MultiFileUploadResponse,
)
from app.schemas.rag import AskRequest, AskResponse
from app.services.chat_service import RAGService
from app.services.document_service import DocumentService

api_router = APIRouter()


# ---------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------
@api_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="System health check",
)
async def health_check(
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_provider: Annotated[BaseEmbeddingProvider, Depends(get_embedding_provider)],
    llm_provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> HealthResponse:
    doc_count_res = await db.execute(select(func.count(Document.id)))
    total_docs = doc_count_res.scalar_one() or 0

    chunk_count_res = await db.execute(select(func.count(Chunk.id)))
    total_chunks = chunk_count_res.scalar_one() or 0

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        embedding_provider=embedding_provider.__class__.__name__,
        llm_provider=llm_provider.__class__.__name__,
        embedding_provider_mode=settings.EMBEDDING_PROVIDER,
        llm_provider_mode=settings.LLM_PROVIDER,
        vector_store_provider=vector_store.__class__.__name__,
        total_documents=total_docs,
        total_chunks=total_chunks,
    )


# ---------------------------------------------------------
# Document Endpoints
# ---------------------------------------------------------
@api_router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Add a knowledge article via direct text",
)
async def create_document(
    payload: DocumentCreate,
    request: Request,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    effective_user = payload.user_id or getattr(request.state, "user_id", "default_user")
    effective_session = payload.session_id or getattr(request.state, "session_id", "default")
    doc = await service.ingest_document(
        title=payload.title,
        content=payload.content,
        session_id=effective_session,
        user_id=effective_user,
    )
    return DocumentResponse(
        document_id=doc.id,
        status=doc.status,
        title=doc.title,
        chunk_count=doc.chunk_count,
        session_id=doc.session_id,
        user_id=doc.user_id,
        file_name=doc.file_name,
        total_pages=doc.total_pages,
        created_at=doc.created_at,
    )


@api_router.post(
    "/documents/upload",
    response_model=MultiFileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload multiple files (up to 5)",
)
async def upload_multiple_documents(
    request: Request,
    files: list[UploadFile] = File(
        ..., description="List of files to upload (max 5 files, 5MB each)"
    ),
    session_id: str | None = Form(
        None, description="Optional existing session ID to append documents into"
    ),
    user_id: str | None = Form(None, description="Optional user identifier for document isolation"),
    service: DocumentService = Depends(get_document_service),
) -> MultiFileUploadResponse:
    effective_user = user_id or getattr(request.state, "user_id", None)
    sess_from_state = getattr(request.state, "session_id", None)
    effective_session = session_id or (sess_from_state if sess_from_state != "default" else None)
    return await service.ingest_uploaded_files(
        files=files, session_id=effective_session, user_id=effective_user
    )


@api_router.get(
    "/documents",
    response_model=DocumentListResponse | list[DocumentListItem],
    tags=["Documents"],
    summary="List available articles",
)
async def list_documents(
    request: Request,
    service: Annotated[DocumentService, Depends(get_document_service)],
    limit: int = Query(default=20, ge=1, le=100, description="Page size limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    session_id: str | None = Query(
        default=None, description="Filter articles by session identifier"
    ),
    user_id: str | None = Query(default=None, description="Filter articles by user identifier"),
    as_list: bool = Query(default=False, description="Return raw JSON array for legacy consumers"),
):
    effective_user = user_id or request.headers.get("x-user-id") or request.headers.get("user-id")
    effective_session = (
        session_id or request.headers.get("x-session-id") or request.headers.get("session-id")
    )
    docs, total = await service.list_documents(
        limit=limit, offset=offset, session_id=effective_session, user_id=effective_user
    )
    items = [
        DocumentListItem(
            document_id=d.id,
            title=d.title,
            status=d.status,
            chunk_count=d.chunk_count,
            session_id=d.session_id,
            user_id=d.user_id,
            file_name=d.file_name,
            created_at=d.created_at,
        )
        for d in docs
    ]
    if as_list:
        return items
    return DocumentListResponse(items=items, total=total, limit=limit, offset=offset)


@api_router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
    summary="Get document details",
)
async def get_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    doc = await service.get_document(document_id)
    return DocumentResponse(
        document_id=doc.id,
        status=doc.status,
        title=doc.title,
        chunk_count=doc.chunk_count,
        session_id=doc.session_id,
        user_id=doc.user_id,
        file_name=doc.file_name,
        total_pages=doc.total_pages,
        created_at=doc.created_at,
    )


@api_router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> None:
    await service.delete_document(document_id)


# ---------------------------------------------------------
# Q&A Endpoint
# ---------------------------------------------------------
@api_router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_ask)],
    tags=["Q&A"],
    summary="Answer a question using grounded RAG",
)
async def ask_question(
    payload: AskRequest,
    request: Request,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> AskResponse:
    client_ip = get_client_ip(request)
    effective_user = payload.user_id or getattr(request.state, "user_id", "default_user")
    effective_session = payload.session_id or getattr(request.state, "session_id", "default")
    return await service.answer_question(
        question=payload.question,
        session_id=effective_session,
        user_id=effective_user,
        client_ip=client_ip,
    )
