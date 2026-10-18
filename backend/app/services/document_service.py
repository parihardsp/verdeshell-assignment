"""Document ingestion, multi-file extraction, and lifecycle service."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import EntityNotFoundError, ValidationError
from app.core.logging import LogPhase, mask_sensitive_data, set_request_context
from app.db.models import Chunk, Document
from app.db.repositories import ChunkRepository, DocumentRepository
from app.ingestion import (
    DocumentExtractor,
    FileValidationService,
    SessionFileStorageService,
)
from app.providers.embeddings import BaseEmbeddingProvider
from app.providers.vector_store import BaseVectorStore, IndexedChunk
from app.rag.chunker import TextChunker
from app.schemas.document import MultiFileUploadResponse, UploadedFileSummary
from app.services.job_manager import global_job_manager

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrates document validation, multi-format extraction, chunking, and vector indexing."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        chunker: TextChunker | None = None,
        storage_service: SessionFileStorageService | None = None,
    ):
        self.doc_repo = document_repo
        self.chunk_repo = chunk_repo
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.storage_service = storage_service or SessionFileStorageService()

    async def ingest_document(
        self,
        title: str,
        content: str,
        session_id: str = "default",
        user_id: str = "default_user",
    ) -> Document:
        """Process and index a direct text document synchronously with user and session scoping."""
        set_request_context(user_id=user_id, session_id=session_id)
        logger.info(
            "Direct document ingestion started | title=%s chars=%d",
            title,
            len(content),
            extra={"phase": LogPhase.INGESTION},
        )

        # 1. Validation and Sanitization Guardrails
        clean_title, clean_content, content_hash = FileValidationService.validate_content(
            title, content
        )

        # 2. Multi-Tenant Deduplication Check
        existing_user_doc = await self.doc_repo.get_by_hash_and_tenant(
            content_hash=content_hash, user_id=user_id, session_id=session_id
        )
        if existing_user_doc:
            logger.info(
                "Document already indexed in session | doc_id=%s sha256=%s",
                existing_user_doc.id,
                content_hash[:12],
                extra={"phase": LogPhase.VALIDATION},
            )
            return existing_user_doc

        # System-wide reuse: if indexed by another user/session, reuse precomputed embeddings
        system_doc = await self.doc_repo.get_by_content_hash(content_hash)
        if system_doc:
            logger.info(
                "Reusing precomputed embeddings | source_doc_id=%s target_user=%s",
                system_doc.id,
                user_id,
                extra={"phase": LogPhase.INGESTION},
            )
            existing_chunks = await self.chunk_repo.get_by_document_id(system_doc.id)
            if existing_chunks:
                saved_doc = Document(
                    title=clean_title,
                    content=clean_content,
                    content_hash=content_hash,
                    session_id=session_id,
                    user_id=user_id,
                    file_name=system_doc.file_name,
                    file_path=system_doc.file_path,
                    total_pages=system_doc.total_pages,
                    status="indexed",
                    chunk_count=len(existing_chunks),
                )
                saved_doc = await self.doc_repo.create(saved_doc)

                new_chunks: list[Chunk] = []
                indexed_chunks: list[IndexedChunk] = []
                for c in existing_chunks:
                    new_cid = f"chk_{uuid.uuid4().hex[:12]}"
                    new_chunks.append(
                        Chunk(
                            id=new_cid,
                            document_id=saved_doc.id,
                            session_id=session_id,
                            user_id=user_id,
                            file_name=saved_doc.file_name,
                            page_number=c.page_number,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            char_start=c.char_start,
                            char_end=c.char_end,
                            embedding=c.embedding,
                        )
                    )
                    indexed_chunks.append(
                        IndexedChunk(
                            chunk_id=new_cid,
                            document_id=saved_doc.id,
                            document_title=saved_doc.title,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            embedding=c.embedding,
                            session_id=session_id,
                            user_id=user_id,
                            file_name=saved_doc.file_name or "",
                            page_number=c.page_number or 1,
                        )
                    )

                await self.chunk_repo.create_many(new_chunks)
                await self.vector_store.add_chunks(indexed_chunks)
                logger.info(
                    "Document deduplication completed | doc_id=%s chunks_reused=%d",
                    saved_doc.id,
                    len(new_chunks),
                    extra={"phase": LogPhase.INGESTION},
                )
                return saved_doc

        # 3. Create Document Record
        document = Document(
            title=clean_title,
            content=clean_content,
            content_hash=content_hash,
            session_id=session_id,
            user_id=user_id,
            status="processing",
            chunk_count=0,
        )
        saved_doc = await self.doc_repo.create(document)
        logger.info(
            "Document record created | doc_id=%s title=%s",
            saved_doc.id,
            clean_title,
            extra={"phase": LogPhase.DATABASE},
        )

        try:
            # 4. Context-Preserving Chunking
            text_chunks = self.chunker.chunk_text(clean_content)
            if not text_chunks:
                raise ValidationError("Failed to produce text chunks from document content.")
            logger.info(
                "Document chunking completed | title=%s chunks=%d",
                clean_title,
                len(text_chunks),
                extra={"phase": LogPhase.CHUNKING},
            )

            chunk_texts = [tc.content for tc in text_chunks]

            # 5. Embedding Generation
            provider_name = self.embedding_provider.__class__.__name__
            embeddings = await self.embedding_provider.embed_batch(chunk_texts)
            logger.info(
                "Embeddings generated | title=%s count=%d dim=%d provider=%s",
                clean_title,
                len(embeddings),
                len(embeddings[0]) if embeddings else 0,
                provider_name,
                extra={"phase": LogPhase.EMBEDDING},
            )

            # 6. Persist Chunks in Database & Vector Store
            chunk_records: list[Chunk] = []
            indexed_chunks: list[IndexedChunk] = []

            for tc, emb in zip(text_chunks, embeddings):
                chunk_id = f"chk_{uuid.uuid4().hex[:12]}"
                chunk = Chunk(
                    id=chunk_id,
                    document_id=saved_doc.id,
                    session_id=session_id,
                    user_id=user_id,
                    file_name=None,
                    page_number=1,
                    chunk_index=tc.index,
                    content=tc.content,
                    char_start=tc.char_start,
                    char_end=tc.char_end,
                    embedding=emb,
                )
                chunk_records.append(chunk)

                indexed_chunks.append(
                    IndexedChunk(
                        chunk_id=chunk_id,
                        document_id=saved_doc.id,
                        document_title=saved_doc.title,
                        chunk_index=tc.index,
                        content=tc.content,
                        embedding=emb,
                        session_id=session_id,
                        user_id=user_id,
                        file_name="",
                        page_number=1,
                    )
                )

            await self.chunk_repo.bulk_create(chunk_records)
            await self.vector_store.add_chunks(indexed_chunks)

            await self.doc_repo.update_status(
                saved_doc.id, status="indexed", chunk_count=len(chunk_records)
            )
            saved_doc.status = "indexed"
            saved_doc.chunk_count = len(chunk_records)
            logger.info(
                "Document indexing completed | doc_id=%s chunks=%d store_total=%d",
                saved_doc.id,
                len(chunk_records),
                self.vector_store.count(),
                extra={"phase": LogPhase.INGESTION},
            )
            return saved_doc

        except Exception as exc:
            logger.error(
                "Document indexing failed | doc_id=%s error=%s",
                saved_doc.id,
                type(exc).__name__,
                exc_info=True,
                extra={"phase": LogPhase.INGESTION},
            )
            await self.doc_repo.update_status(saved_doc.id, status="failed")
            saved_doc.status = "failed"
            raise

    async def ingest_uploaded_files(
        self,
        files: list[UploadFile],
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> MultiFileUploadResponse:
        """Handle multi-file upload (up to 5 files) with user and session scoping.

        1. Validates files (count, size, extension)
        2. Saves files under data/uploads/{user_id}_{session_id}_{date}/
        3. Extracts structured text (preserving page coordinates and tables)
        4. Chunks text and computes embeddings
        5. Indexes chunks in vector store with user_id and session_id metadata
        """
        active_session_id = session_id or self.storage_service.generate_session_id()
        active_user_id = user_id or self.storage_service.generate_user_id()
        set_request_context(user_id=active_user_id, session_id=active_session_id)

        # 1. Quick File Validation (Quantity, formats, sizes)
        FileValidationService.validate_uploaded_files(files)
        logger.info(
            "Batch file validation passed | files=%d",
            len(files),
            extra={"phase": LogPhase.VALIDATION},
        )

        _, date_str = self.storage_service.get_session_directory(
            active_session_id, user_id=active_user_id
        )
        processed_summaries: list[UploadedFileSummary] = []

        for f_idx, file in enumerate(files, 1):
            saved_path, file_size = await self.storage_service.save_uploaded_file(
                file, active_session_id, user_id=active_user_id
            )
            logger.info(
                "File persisted to storage | file=%s size_kb=%.1f",
                file.filename,
                file_size / 1024,
                extra={"phase": LogPhase.INGESTION},
            )

            summary = await self._process_single_file(
                saved_path=saved_path,
                filename=file.filename or "",
                file_size=file_size,
                active_session_id=active_session_id,
                active_user_id=active_user_id,
                job_id=None,
                file_index=f_idx,
                total_files=len(files),
            )
            processed_summaries.append(summary)

        indexed_count = sum(1 for s in processed_summaries if s.status == "indexed")
        failed_count = sum(1 for s in processed_summaries if s.status == "failed")

        logger.info(
            "Batch ingestion completed | total_files=%d indexed=%d failed=%d",
            len(files),
            indexed_count,
            failed_count,
            extra={"phase": LogPhase.INGESTION},
        )

        # If only 1 file was uploaded and it failed, raise ValidationError for clean HTTP 400
        if len(files) == 1 and indexed_count == 0 and processed_summaries:
            first_err = processed_summaries[0].error_message or "File contains no readable text or is invalid."
            raise ValidationError(first_err)

        return MultiFileUploadResponse(
            session_id=active_session_id,
            user_id=active_user_id,
            upload_date=date_str,
            total_files=len(processed_summaries),
            indexed_count=indexed_count,
            failed_count=failed_count,
            documents=processed_summaries,
        )

    async def _process_single_file(
        self,
        saved_path: Path,
        filename: str,
        file_size: int,
        active_session_id: str,
        active_user_id: str,
        job_id: str | None = None,
        file_index: int = 1,
        total_files: int = 1,
    ) -> UploadedFileSummary:
        """Process a single document file from disk through extraction, chunking, and indexing."""
        file_base_pct = int(((file_index - 1) / total_files) * 100)
        file_share = int(100 / total_files)
        saved_doc: Document | None = None

        try:
            # 1. Extraction
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="EXTRACTION",
                    percent=file_base_pct + int(file_share * 0.15),
                    message=f"Extracting text from '{filename}'...",
                    current_file=filename,
                    processed_files=file_index - 1,
                )

            logger.info(
                "[Step 1/5 - INGESTION] Processing uploaded file | file=%s size_bytes=%d user_id=%s session_id=%s",
                filename,
                file_size,
                active_user_id,
                active_session_id,
                extra={"phase": LogPhase.INGESTION},
            )

            extraction = await asyncio.to_thread(DocumentExtractor.extract_from_path, saved_path)
            content_hash = FileValidationService.compute_sha256(extraction.full_text)
            logger.info(
                "[Step 2/5 - EXTRACTION] Multi-tier document parsing completed | file=%s pages=%d chars=%d",
                filename,
                extraction.total_pages,
                len(extraction.full_text),
                extra={"phase": LogPhase.EXTRACTION},
            )

            # Multi-Tenant Deduplication Check
            existing_user_doc = await self.doc_repo.get_by_hash_and_tenant(
                content_hash=content_hash,
                user_id=active_user_id,
                session_id=active_session_id,
            )
            if existing_user_doc:
                logger.info(
                    "File already indexed in session | file=%s doc_id=%s",
                    filename,
                    existing_user_doc.id,
                    extra={"phase": LogPhase.VALIDATION},
                )
                if job_id:
                    await global_job_manager.update_progress(
                        job_id=job_id,
                        stage="INDEXED",
                        percent=int((file_index / total_files) * 100),
                        message=f"Reused existing indexed copy of '{filename}'.",
                        current_file=filename,
                        processed_files=file_index,
                    )
                return UploadedFileSummary(
                    document_id=existing_user_doc.id,
                    file_name=filename,
                    status="indexed",
                    chunk_count=existing_user_doc.chunk_count,
                    total_pages=existing_user_doc.total_pages or 1,
                    saved_path=str(saved_path),
                )

            # System-wide reuse: if indexed by another user/session, reuse precomputed embeddings
            system_doc = await self.doc_repo.get_by_content_hash(content_hash)
            if system_doc:
                logger.info(
                    "Reusing precomputed embeddings | file=%s source_doc_id=%s",
                    filename,
                    system_doc.id,
                    extra={"phase": LogPhase.INGESTION},
                )
                existing_chunks = await self.chunk_repo.get_by_document_id(system_doc.id)
                if existing_chunks:
                    doc_title = Path(filename or "Uploaded Document").stem.replace("_", " ").title()
                    saved_doc = Document(
                        title=doc_title,
                        content=extraction.full_text,
                        content_hash=content_hash,
                        session_id=active_session_id,
                        user_id=active_user_id,
                        file_name=filename,
                        file_path=str(saved_path),
                        total_pages=extraction.total_pages,
                        status="indexed",
                        chunk_count=len(existing_chunks),
                    )
                    saved_doc = await self.doc_repo.create(saved_doc)

                    new_chunks: list[Chunk] = []
                    indexed_chunks: list[IndexedChunk] = []
                    for c in existing_chunks:
                        new_cid = f"chk_{uuid.uuid4().hex[:12]}"
                        new_chunks.append(
                            Chunk(
                                id=new_cid,
                                document_id=saved_doc.id,
                                session_id=active_session_id,
                                user_id=active_user_id,
                                file_name=saved_doc.file_name,
                                page_number=c.page_number,
                                chunk_index=c.chunk_index,
                                content=c.content,
                                char_start=c.char_start,
                                char_end=c.char_end,
                                embedding=c.embedding,
                            )
                        )
                        indexed_chunks.append(
                            IndexedChunk(
                                chunk_id=new_cid,
                                document_id=saved_doc.id,
                                document_title=saved_doc.title,
                                chunk_index=c.chunk_index,
                                content=c.content,
                                embedding=c.embedding,
                                session_id=active_session_id,
                                user_id=active_user_id,
                                file_name=saved_doc.file_name or "",
                                page_number=c.page_number or 1,
                            )
                        )

                    await self.chunk_repo.create_many(new_chunks)
                    await self.vector_store.add_chunks(indexed_chunks)
                    logger.info(
                        "File deduplication completed | file=%s doc_id=%s chunks_reused=%d",
                        filename,
                        saved_doc.id,
                        len(new_chunks),
                        extra={"phase": LogPhase.INGESTION},
                    )
                    if job_id:
                        await global_job_manager.update_progress(
                            job_id=job_id,
                            stage="INDEXED",
                            percent=int((file_index / total_files) * 100),
                            message=f"Indexed '{filename}' via precomputed embeddings.",
                            current_file=filename,
                            processed_files=file_index,
                        )
                    return UploadedFileSummary(
                        document_id=saved_doc.id,
                        file_name=filename,
                        status="indexed",
                        chunk_count=len(new_chunks),
                        total_pages=extraction.total_pages,
                        saved_path=str(saved_path),
                    )

            # Create document record
            doc_title = Path(filename or "Uploaded Document").stem.replace("_", " ").title()
            document = Document(
                title=doc_title,
                content=extraction.full_text,
                content_hash=content_hash,
                session_id=active_session_id,
                user_id=active_user_id,
                file_name=filename,
                file_path=str(saved_path),
                total_pages=extraction.total_pages,
                status="processing",
                chunk_count=0,
            )
            saved_doc = await self.doc_repo.create(document)
            logger.info(
                "Document record created | doc_id=%s file=%s",
                saved_doc.id,
                filename,
                extra={"phase": LogPhase.DATABASE},
            )

            # 2. Chunking
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="CHUNKING",
                    percent=file_base_pct + int(file_share * 0.40),
                    message=f"Chunking '{filename}' into semantic passages...",
                    current_file=filename,
                )

            all_text_chunks = []
            current_chunk_idx = 0

            for page in extraction.pages:
                page_chunks = self.chunker.chunk_text(
                    page.text,
                    start_index=current_chunk_idx,
                    page_number=page.page_number,
                )
                all_text_chunks.extend(page_chunks)
                current_chunk_idx += len(page_chunks)

            if not all_text_chunks:
                raise ValidationError(
                    f"No textual content could be parsed from '{filename}'."
                )

            logger.info(
                "[Step 3/5 - CHUNKING] Document partitioned into semantic chunks | file=%s chunks=%d pages=%d",
                filename,
                len(all_text_chunks),
                extraction.total_pages,
                extra={"phase": LogPhase.CHUNKING},
            )

            # 3. Embeddings
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="EMBEDDING",
                    percent=file_base_pct + int(file_share * 0.70),
                    message=f"Generating embeddings for {len(all_text_chunks)} chunks in '{filename}'...",
                    current_file=filename,
                )

            chunk_texts = [tc.content for tc in all_text_chunks]
            provider_name = self.embedding_provider.__class__.__name__
            embeddings = await self.embedding_provider.embed_batch(chunk_texts)
            logger.info(
                "[Step 4/5 - EMBEDDING] Embeddings generated | file=%s count=%d dim=%d provider=%s",
                filename,
                len(embeddings),
                len(embeddings[0]) if embeddings else 0,
                provider_name,
                extra={"phase": LogPhase.EMBEDDING},
            )

            # 4. Vector Store & Relational Commit
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="VECTOR_STORE",
                    percent=file_base_pct + int(file_share * 0.90),
                    message=f"Committing '{filename}' vectors to ChromaDB...",
                    current_file=filename,
                )

            chunk_records: list[Chunk] = []
            indexed_chunks: list[IndexedChunk] = []

            for tc, emb in zip(all_text_chunks, embeddings):
                chunk_id = f"chk_{uuid.uuid4().hex[:12]}"
                chunk = Chunk(
                    id=chunk_id,
                    document_id=saved_doc.id,
                    session_id=active_session_id,
                    user_id=active_user_id,
                    file_name=filename,
                    page_number=tc.page_number,
                    chunk_index=tc.index,
                    content=tc.content,
                    char_start=tc.char_start,
                    char_end=tc.char_end,
                    embedding=emb,
                )
                chunk_records.append(chunk)

                indexed_chunks.append(
                    IndexedChunk(
                        chunk_id=chunk_id,
                        document_id=saved_doc.id,
                        document_title=saved_doc.title,
                        chunk_index=tc.index,
                        content=tc.content,
                        embedding=emb,
                        session_id=active_session_id,
                        user_id=active_user_id,
                        file_name=filename or "",
                        page_number=tc.page_number,
                    )
                )

            await self.chunk_repo.bulk_create(chunk_records)
            await self.vector_store.add_chunks(indexed_chunks)

            await self.doc_repo.update_status(
                saved_doc.id, status="indexed", chunk_count=len(chunk_records)
            )
            await self.doc_repo.session.commit()

            logger.info(
                "[Step 5/5 - VECTOR_STORE] File indexing completed | file=%s doc_id=%s chunks=%d store_total=%d status=indexed",
                filename,
                saved_doc.id,
                len(chunk_records),
                self.vector_store.count(),
                extra={"phase": LogPhase.INGESTION},
            )

            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="INDEXED",
                    percent=int((file_index / total_files) * 100),
                    message=f"Indexed '{filename}' ({len(chunk_records)} chunks).",
                    current_file=filename,
                    processed_files=file_index,
                )

            return UploadedFileSummary(
                document_id=saved_doc.id,
                file_name=filename,
                status="indexed",
                chunk_count=len(chunk_records),
                total_pages=extraction.total_pages,
                saved_path=str(saved_path),
            )

        except ValidationError as val_err:
            err_text = str(val_err.message if hasattr(val_err, "message") else val_err)
            logger.warning(
                "File validation skipped | file=%s reason=%s",
                filename,
                err_text,
                extra={"phase": LogPhase.VALIDATION},
            )
            if saved_doc:
                await self.doc_repo.update_status(saved_doc.id, status="failed")
                await self.doc_repo.session.commit()
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="SKIPPED",
                    percent=int((file_index / total_files) * 100),
                    message=f"Skipped '{filename}': {err_text}",
                    current_file=filename,
                    processed_files=file_index,
                )
            return UploadedFileSummary(
                document_id=saved_doc.id if saved_doc else f"failed_{uuid.uuid4().hex[:8]}",
                file_name=filename,
                status="failed",
                chunk_count=0,
                total_pages=0,
                saved_path=str(saved_path) if saved_path else "",
                error_message=err_text,
            )
        except Exception as exc:
            err_text = mask_sensitive_data(str(exc))
            logger.error(
                "File processing failed | file=%s error=%s",
                filename,
                type(exc).__name__,
                exc_info=True,
                extra={"phase": LogPhase.INGESTION},
            )
            if saved_doc:
                await self.doc_repo.update_status(saved_doc.id, status="failed")
            if job_id:
                await global_job_manager.update_progress(
                    job_id=job_id,
                    stage="FAILED",
                    percent=int((file_index / total_files) * 100),
                    message=f"Failed '{filename}': {err_text}",
                    current_file=filename,
                    processed_files=file_index,
                )
            return UploadedFileSummary(
                document_id=saved_doc.id if saved_doc else f"failed_{uuid.uuid4().hex[:8]}",
                file_name=filename,
                status="failed",
                chunk_count=0,
                total_pages=0,
                saved_path=str(saved_path) if saved_path else "",
                error_message=err_text,
            )

    async def ingest_pre_saved_files(
        self,
        job_id: str,
        saved_files: list[tuple[Path, str, int]],
        session_id: str,
        user_id: str,
    ) -> list[UploadedFileSummary]:
        """Process pre-saved files sequentially in background with real-time SSE progress reporting."""
        active_session_id = session_id
        active_user_id = user_id
        set_request_context(user_id=active_user_id, session_id=active_session_id)

        processed_summaries: list[UploadedFileSummary] = []
        total_files = len(saved_files)

        for f_idx, (saved_path, original_filename, file_size) in enumerate(saved_files, 1):
            summary = await self._process_single_file(
                saved_path=saved_path,
                filename=original_filename,
                file_size=file_size,
                active_session_id=active_session_id,
                active_user_id=active_user_id,
                job_id=job_id,
                file_index=f_idx,
                total_files=total_files,
            )
            processed_summaries.append(summary)

        # Ensure final state is committed to SQLite before broadcasting terminal SSE event
        await self.doc_repo.session.commit()

        # Notify job manager of completion
        indexed_items = [s.model_dump() for s in processed_summaries if s.status == "indexed"]
        failed_items = [s.model_dump() for s in processed_summaries if s.status == "failed"]
        await global_job_manager.finish_job(
            job_id=job_id,
            results=indexed_items,
            errors=failed_items,
        )
        return processed_summaries

    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[list[Document], int]:
        """Fetch paginated documents list, optionally filtered by session_id and user_id."""
        return await self.doc_repo.list_documents(
            limit=limit, offset=offset, session_id=session_id, user_id=user_id
        )

    async def get_document(self, document_id: str) -> Document:
        """Fetch document by ID."""
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise EntityNotFoundError(f"Document with ID '{document_id}' not found.")
        return doc

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and purge from vector store."""
        await self.vector_store.remove_by_document_id(document_id)
        return await self.doc_repo.delete(document_id)
