"""Document ingestion, multi-file extraction, and lifecycle service."""

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import EntityNotFoundError, ValidationError
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

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrates document validation, multi-file PyMuPDF extraction, chunking, and vector indexing."""

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
        from app.core.logging import set_request_context

        set_request_context(user_id=user_id, session_id=session_id)
        logger.info(
            "=== [INGESTION: DIRECT TEXT START] Title: '%s' | User: '%s' | Session: '%s' (Chars: %d) ===",
            title,
            user_id,
            session_id,
            len(content),
        )

        # 1. Validation and Sanitization Guardrails
        logger.info("[STEP 1/5: VALIDATION] Validating content length and sanitizing null bytes...")
        clean_title, clean_content, content_hash = FileValidationService.validate_content(
            title, content
        )
        logger.info(
            "[STEP 1/5: VALIDATION] Content validated. SHA-256: %s...",
            content_hash[:16],
        )

        # 2. Multi-Tenant Deduplication Check
        existing_user_doc = await self.doc_repo.get_by_hash_and_tenant(
            content_hash=content_hash, user_id=user_id, session_id=session_id
        )
        if existing_user_doc:
            logger.info(
                "[DEDUPLICATION: USER-SESSION] Content already indexed for user '%s' in session '%s' (id: %s).",
                user_id,
                session_id,
                existing_user_doc.id,
            )
            return existing_user_doc

        # System-wide reuse: if indexed by another user/session, reuse precomputed embeddings
        system_doc = await self.doc_repo.get_by_content_hash(content_hash)
        if system_doc:
            logger.info(
                "[DEDUPLICATION: SYSTEM REUSE] Content matches indexed document '%s' (id: %s). Reusing precomputed embeddings for user '%s' / session '%s'...",
                system_doc.title,
                system_doc.id,
                user_id,
                session_id,
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
                    "=== [DEDUPLICATION SUCCESS] Reused %d precomputed embeddings. Document '%s' (id: %s) marked for user '%s' / session '%s'. ===",
                    len(new_chunks),
                    saved_doc.title,
                    saved_doc.id,
                    user_id,
                    session_id,
                )
                return saved_doc

        # 3. Create Document Record
        logger.info("[STEP 2/5: DB RECORD] Creating pending document in SQLite...")
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
        logger.info("[STEP 2/5: DB RECORD] Created document record (id: %s).", saved_doc.id)

        try:
            # 4. Context-Preserving Chunking
            logger.info(
                "[STEP 3/5: RECURSIVE CHUNKING] Splitting content (chunk_size=%d, overlap=%d)...",
                self.chunker.chunk_size,
                self.chunker.chunk_overlap,
            )
            text_chunks = self.chunker.chunk_text(clean_content)
            if not text_chunks:
                raise ValidationError("Failed to produce text chunks from document content.")
            logger.info(
                "[STEP 3/5: RECURSIVE CHUNKING] Generated %d chunk(s).",
                len(text_chunks),
            )

            chunk_texts = [tc.content for tc in text_chunks]

            # 5. Embedding Generation
            provider_name = self.embedding_provider.__class__.__name__
            logger.info(
                "[STEP 4/5: VECTOR EMBEDDING] Generating embeddings for %d chunk(s) via %s...",
                len(chunk_texts),
                provider_name,
            )
            embeddings = await self.embedding_provider.embed_batch(chunk_texts)
            logger.info(
                "[STEP 4/5: VECTOR EMBEDDING] Successfully computed %d vector embeddings (dim: %d).",
                len(embeddings),
                len(embeddings[0]) if embeddings else 0,
            )

            # 6. Persist Chunks in SQLite & ChromaDB
            logger.info(
                "[STEP 5/5: PERSISTENCE & CHROMADB] Persisting chunks to SQLite and indexing in ChromaDB..."
            )
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
                "=== [INGESTION SUCCESS] Document '%s' (id: %s) indexed with %d chunk(s) in ChromaDB (Total in store: %d). ===",
                saved_doc.title,
                saved_doc.id,
                len(chunk_records),
                self.vector_store.count(),
            )
            return saved_doc

        except Exception as exc:
            logger.error(
                "[INGESTION FAILED] Failed to ingest document %s: %s",
                saved_doc.id,
                exc,
                exc_info=True,
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
        3. Extracts text via PyMuPDF (preserving page numbers for PDFs)
        4. Chunks text and computes embeddings
        5. Indexes chunks in ChromaDB with user_id and session_id metadata
        """
        from app.core.logging import set_request_context

        active_session_id = session_id or self.storage_service.generate_session_id()
        active_user_id = user_id or self.storage_service.generate_user_id()
        set_request_context(user_id=active_user_id, session_id=active_session_id)
        logger.info("==================== [MULTI-FILE INGESTION START] ====================")
        logger.info(
            "User: [%s] | Session: [%s] | Total Files to Ingest: %d",
            active_user_id,
            active_session_id,
            len(files),
        )

        # 1. Quick File Validation (Quantity, formats, sizes)
        logger.info(
            "[STEP 1/6: FILE VALIDATION] Checking batch file constraints (limit <= 5, size <= 5MB, format in .pdf/.txt/.md)..."
        )
        FileValidationService.validate_uploaded_files(files)
        for f in files:
            logger.info("   ├─ Validated: '%s' (ContentType: %s)", f.filename, f.content_type)
        logger.info(
            "[STEP 1/6: FILE VALIDATION] All %d file(s) passed validation constraints.",
            len(files),
        )

        _, date_str = self.storage_service.get_session_directory(
            active_session_id, user_id=active_user_id
        )
        processed_summaries: list[UploadedFileSummary] = []

        for f_idx, file in enumerate(files, 1):
            logger.info(
                "--- Processing File [%d/%d]: '%s' ---",
                f_idx,
                len(files),
                file.filename,
            )

            # Step 2: Disk Storage
            logger.info(
                "[STEP 2/6: DISK PERSISTENCE] Persisting '%s' to folder under data/uploads/%s_%s_%s/...",
                file.filename,
                active_user_id,
                active_session_id,
                date_str,
            )
            saved_path, file_size = await self.storage_service.save_uploaded_file(
                file, active_session_id, user_id=active_user_id
            )
            logger.info(
                "[STEP 2/6: DISK PERSISTENCE] Persisted '%s' to '%s' (%.2f KB).",
                file.filename,
                saved_path,
                file_size / 1024,
            )

            # Step 3: Text Extraction with PyMuPDF
            logger.info("[STEP 3/6: TEXT EXTRACTION] Extracting text using PyMuPDF parser...")
            extraction = DocumentExtractor.extract_from_path(saved_path)
            content_hash = FileValidationService.compute_sha256(extraction.full_text)
            logger.info(
                "[STEP 3/6: TEXT EXTRACTION] PyMuPDF parsed %d page(s), %d total characters. Content SHA-256: %s...",
                extraction.total_pages,
                len(extraction.full_text),
                content_hash[:16],
            )

            # Multi-Tenant Deduplication Check
            existing_user_doc = await self.doc_repo.get_by_hash_and_tenant(
                content_hash=content_hash,
                user_id=active_user_id,
                session_id=active_session_id,
            )
            if existing_user_doc:
                logger.info(
                    "[DEDUPLICATION: USER-SESSION] '%s' already indexed for user '%s' in session '%s' (id: %s).",
                    file.filename,
                    active_user_id,
                    active_session_id,
                    existing_user_doc.id,
                )
                processed_summaries.append(
                    UploadedFileSummary(
                        document_id=existing_user_doc.id,
                        file_name=file.filename or "",
                        status="indexed",
                        chunk_count=existing_user_doc.chunk_count,
                        total_pages=existing_user_doc.total_pages or 1,
                        saved_path=str(saved_path),
                    )
                )
                continue

            # System-wide reuse: if indexed by another user/session, reuse precomputed embeddings
            system_doc = await self.doc_repo.get_by_content_hash(content_hash)
            if system_doc:
                logger.info(
                    "[DEDUPLICATION: SYSTEM REUSE] '%s' matches existing content hash %s. Reusing precomputed embeddings for user '%s' / session '%s'...",
                    file.filename,
                    content_hash[:16],
                    active_user_id,
                    active_session_id,
                )
                existing_chunks = await self.chunk_repo.get_by_document_id(system_doc.id)
                if existing_chunks:
                    doc_title = (
                        Path(file.filename or "Uploaded Document").stem.replace("_", " ").title()
                    )
                    saved_doc = Document(
                        title=doc_title,
                        content=extraction.full_text,
                        content_hash=content_hash,
                        session_id=active_session_id,
                        user_id=active_user_id,
                        file_name=file.filename,
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
                        "=== [DEDUPLICATION SUCCESS] Reused %d precomputed embeddings. Document '%s' (id: %s) marked for user '%s' / session '%s'. ===",
                        len(new_chunks),
                        saved_doc.title,
                        saved_doc.id,
                        active_user_id,
                        active_session_id,
                    )
                    processed_summaries.append(
                        UploadedFileSummary(
                            document_id=saved_doc.id,
                            file_name=file.filename or "",
                            status="indexed",
                            chunk_count=len(new_chunks),
                            total_pages=extraction.total_pages,
                            saved_path=str(saved_path),
                        )
                    )
                    continue

            # Create document record
            doc_title = Path(file.filename or "Uploaded Document").stem.replace("_", " ").title()
            document = Document(
                title=doc_title,
                content=extraction.full_text,
                content_hash=content_hash,
                session_id=active_session_id,
                user_id=active_user_id,
                file_name=file.filename,
                file_path=str(saved_path),
                total_pages=extraction.total_pages,
                status="processing",
                chunk_count=0,
            )
            saved_doc = await self.doc_repo.create(document)
            logger.info(
                "[STEP 3.5: DB RECORD] Created SQLite document '%s' (id: %s).",
                doc_title,
                saved_doc.id,
            )

            try:
                # Step 4: Chunking
                logger.info(
                    "[STEP 4/6: RECURSIVE CHUNKING] Running page-preserving chunker on %d page(s)...",
                    extraction.total_pages,
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
                        f"No textual content could be parsed from '{file.filename}'."
                    )

                logger.info(
                    "[STEP 4/6: RECURSIVE CHUNKING] Generated %d chunk(s) across %d page(s).",
                    len(all_text_chunks),
                    extraction.total_pages,
                )

                # Step 5: Embeddings
                chunk_texts = [tc.content for tc in all_text_chunks]
                provider_name = self.embedding_provider.__class__.__name__
                logger.info(
                    "[STEP 5/6: VECTOR EMBEDDING] Computing batch embeddings for %d chunk(s) via %s...",
                    len(chunk_texts),
                    provider_name,
                )
                embeddings = await self.embedding_provider.embed_batch(chunk_texts)
                logger.info(
                    "[STEP 5/6: VECTOR EMBEDDING] Successfully computed %d embeddings (dim: %d).",
                    len(embeddings),
                    len(embeddings[0]) if embeddings else 0,
                )

                # Step 6: ChromaDB Indexing
                logger.info(
                    "[STEP 6/6: PERSISTENCE & CHROMADB] Persisting chunks to SQLite and indexing in ChromaDB..."
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
                        file_name=file.filename,
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
                            file_name=file.filename or "",
                            page_number=tc.page_number,
                        )
                    )

                await self.chunk_repo.bulk_create(chunk_records)
                await self.vector_store.add_chunks(indexed_chunks)

                await self.doc_repo.update_status(
                    saved_doc.id, status="indexed", chunk_count=len(chunk_records)
                )

                logger.info(
                    "[INGESTION SUCCESS] File '%s' fully indexed: %d chunks across %d pages (ChromaDB collection total: %d vectors).",
                    file.filename,
                    len(chunk_records),
                    extraction.total_pages,
                    self.vector_store.count(),
                )

                processed_summaries.append(
                    UploadedFileSummary(
                        document_id=saved_doc.id,
                        file_name=file.filename or "",
                        status="indexed",
                        chunk_count=len(chunk_records),
                        total_pages=extraction.total_pages,
                        saved_path=str(saved_path),
                    )
                )

            except Exception as exc:
                logger.error(
                    "[INGESTION FAILED] Failed to process file '%s': %s",
                    file.filename,
                    exc,
                    exc_info=True,
                )
                await self.doc_repo.update_status(saved_doc.id, status="failed")
                processed_summaries.append(
                    UploadedFileSummary(
                        document_id=saved_doc.id,
                        file_name=file.filename or "",
                        status="failed",
                        chunk_count=0,
                        total_pages=extraction.total_pages,
                        saved_path=str(saved_path),
                    )
                )

        logger.info("==================== [MULTI-FILE INGESTION FINISHED] ====================")
        return MultiFileUploadResponse(
            session_id=active_session_id,
            user_id=active_user_id,
            upload_date=date_str,
            total_files=len(processed_summaries),
            documents=processed_summaries,
        )

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
