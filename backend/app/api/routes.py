import asyncio
import json
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.dependencies import (
    get_document_service,
    get_embedding_provider,
    get_llm_provider,
    get_query_repo,
    get_rag_service,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.core.rate_limiter import get_client_ip, rate_limit_ask
from app.db.database import get_db, get_session_factory
from app.db.models import Chunk, Document
from app.db.repositories import ChunkRepository, DocumentRepository, QuestionHistoryRepository
from app.ingestion.validator import FileValidationService
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.llm import BaseLLMProvider
from app.providers.vector_store import BaseVectorStore
from app.schemas.common import HealthResponse
from app.schemas.document import (
    AsyncIngestionJobResponse,
    DocumentCreate,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    JobStatusResponse,
    MultiFileUploadResponse,
)
from app.schemas.rag import AskRequest, AskResponse, QuestionHistoryItem, QuestionHistoryResponse
from app.services.chat_service import RAGService
from app.services.document_service import DocumentService
from app.services.job_manager import global_job_manager

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
    response_model=MultiFileUploadResponse | AsyncIngestionJobResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload multiple files (up to 5) with synchronous or asynchronous processing",
)
async def upload_multiple_documents(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(
        ..., description="List of files to upload (max 5 files, 5MB each)"
    ),
    session_id: str | None = Form(
        None, description="Optional existing session ID to append documents into"
    ),
    user_id: str | None = Form(None, description="Optional user identifier for document isolation"),
    background: bool = Query(
        default=False, description="Process files asynchronously in background job"
    ),
    service: DocumentService = Depends(get_document_service),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> MultiFileUploadResponse | AsyncIngestionJobResponse:
    effective_user = (
        user_id
        or getattr(request.state, "user_id", None)
        or service.storage_service.generate_user_id()
    )
    sess_from_state = getattr(request.state, "session_id", None)
    effective_session = session_id or (
        sess_from_state
        if sess_from_state and sess_from_state != "default"
        else service.storage_service.generate_session_id()
    )

    if not background:
        return await service.ingest_uploaded_files(
            files=files, session_id=effective_session, user_id=effective_user
        )

    # 1. Quick File Validation (Quantity, formats, sizes)
    FileValidationService.validate_uploaded_files(files)

    # 2. Fast disk persistence before closing HTTP request socket
    saved_files: list[tuple[Path, str, int]] = []
    for file in files:
        saved_path, file_size = await service.storage_service.save_uploaded_file(
            file, effective_session, user_id=effective_user
        )
        saved_files.append((saved_path, file.filename or "", file_size))

    # 3. Register pending job with TTL manager
    job = global_job_manager.create_job(
        session_id=effective_session,
        user_id=effective_user,
        total_files=len(saved_files),
        file_names=[f[1] for f in saved_files],
    )

    # 4. Spawn background worker with dedicated async DB session
    async def _async_worker(
        job_id: str,
        files_to_process: list[tuple[Path, str, int]],
        sess_id: str,
        usr_id: str,
    ):
        async with session_factory() as db_session:
            try:
                doc_repo = DocumentRepository(db_session)
                chunk_repo = ChunkRepository(db_session)
                doc_service = DocumentService(
                    document_repo=doc_repo,
                    chunk_repo=chunk_repo,
                    embedding_provider=service.embedding_provider,
                    vector_store=service.vector_store,
                    chunker=service.chunker,
                    storage_service=service.storage_service,
                )
                await doc_service.ingest_pre_saved_files(
                    job_id=job_id,
                    saved_files=files_to_process,
                    session_id=sess_id,
                    user_id=usr_id,
                )
                await db_session.commit()
            except Exception as exc:
                await db_session.rollback()
                await global_job_manager.fail_job(
                    job_id, f"Background processing failed: {str(exc)}"
                )

    asyncio.create_task(_async_worker(job.job_id, saved_files, effective_session, effective_user))

    response.status_code = status.HTTP_202_ACCEPTED
    return AsyncIngestionJobResponse(
        job_id=job.job_id,
        session_id=effective_session,
        user_id=effective_user,
        status="queued",
        total_files=len(saved_files),
        message=f"Indexing job queued with {len(saved_files)} file(s).",
        poll_url=f"/api/v1/documents/jobs/{job.job_id}",
        stream_url=f"/api/v1/documents/jobs/{job.job_id}/events",
    )


@api_router.get(
    "/documents/jobs/{job_id}/events",
    tags=["Documents"],
    summary="Stream real-time ingestion progress via Server-Sent Events (SSE)",
)
async def stream_job_events(job_id: str):
    job = global_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or expired.",
        )

    async def event_generator():
        current_data = global_job_manager.get_job_dict(job_id)
        if current_data:
            yield f"data: {json.dumps(current_data)}\n\n"
            if current_data.get("status") in ("completed", "failed", "completed_with_errors"):
                return

        queue = await global_job_manager.subscribe(job_id)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                    if payload.get("status") in ("completed", "failed", "completed_with_errors"):
                        break
                except TimeoutError:
                    yield ": ping\n\n"
        finally:
            await global_job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get(
    "/documents/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Documents"],
    summary="Get ingestion job status snapshot",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job_dict = global_job_manager.get_job_dict(job_id)
    if not job_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or expired.",
        )
    return JobStatusResponse(**job_dict)


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
        content=doc.content,
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


@api_router.get(
    "/history",
    response_model=QuestionHistoryResponse,
    tags=["Q&A"],
    summary="Get Q&A history scoped to user and session",
)
async def get_question_history(
    request: Request,
    query_repo: Annotated[QuestionHistoryRepository, Depends(get_query_repo)],
    session_id: str | None = Query(default=None, description="Filter history by session identifier"),
    user_id: str | None = Query(default=None, description="Filter history by user identifier"),
    limit: int = Query(default=50, ge=1, le=100, description="Page limit for historical interactions"),
) -> QuestionHistoryResponse:
    effective_user = (
        user_id
        or request.headers.get("x-user-id")
        or request.headers.get("user-id")
        or getattr(request.state, "user_id", None)
    )
    effective_session = (
        session_id
        or request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or getattr(request.state, "session_id", None)
    )

    sess_filter = effective_session if effective_session and effective_session != "default" else None
    user_filter = effective_user if effective_user and effective_user != "default_user" else None

    records = await query_repo.list_recent(limit=limit, session_id=sess_filter, user_id=user_filter)
    total = await query_repo.count_records(session_id=sess_filter, user_id=user_filter)

    items = [
        QuestionHistoryItem(
            id=r.id,
            question=r.question,
            answer=r.answer,
            sources=r.sources or [],
            session_id=r.session_id,
            user_id=r.user_id,
            latency_ms=r.latency_ms,
            client_ip=r.client_ip,
            created_at=r.created_at,
        )
        for r in records
    ]
    return QuestionHistoryResponse(
        items=items,
        total=total,
        session_id=sess_filter,
        user_id=user_filter,
    )


@api_router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Q&A"],
    summary="Clear Q&A history for active session and user",
)
async def clear_question_history(
    request: Request,
    query_repo: Annotated[QuestionHistoryRepository, Depends(get_query_repo)],
    session_id: str | None = Query(default=None, description="Session identifier filter to clear"),
    user_id: str | None = Query(default=None, description="User identifier filter to clear"),
) -> None:
    effective_user = (
        user_id
        or request.headers.get("x-user-id")
        or request.headers.get("user-id")
        or getattr(request.state, "user_id", None)
    )
    effective_session = (
        session_id
        or request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or getattr(request.state, "session_id", None)
    )

    sess_filter = effective_session if effective_session and effective_session != "default" else None
    user_filter = effective_user if effective_user and effective_user != "default_user" else None

    if not sess_filter and not user_filter:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specify either session_id or user_id to clear history.",
        )

    await query_repo.clear_history(session_id=sess_filter, user_id=user_filter)

