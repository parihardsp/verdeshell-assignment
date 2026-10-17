"""Vector database abstractions and ChromaDB implementation.

Includes BaseVectorStore, IndexedChunk, and ChromaVectorStore with session scoping and dimension mismatch self-healing.
"""

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class IndexedChunk:
    """Vector index record with metadata attributes."""

    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    embedding: list[float]
    session_id: str = "default"
    user_id: str = "default_user"
    file_name: str = ""
    page_number: int = 1


class BaseVectorStore(ABC):
    """Contract for vector indexing and cosine similarity retrieval."""

    @abstractmethod
    async def add_chunks(self, chunks: list[IndexedChunk]) -> None:
        """Add or update chunks in the index."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.20,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        """Retrieve top-k most similar chunks meeting the relevance threshold, optionally filtered by session_id and user_id."""

    @abstractmethod
    async def remove_by_document_id(self, document_id: str) -> None:
        """Remove all indexed chunks belonging to a document."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed vectors."""

    @abstractmethod
    async def get_all_chunks(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[IndexedChunk]:
        """Fetch all indexed chunks within user and session scope."""


class ChromaVectorStore(BaseVectorStore):
    """Vector database backed by ChromaDB with metadata filtering by session_id and user_id."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "knowledge_chunks",
        client: chromadb.ClientAPI | None = None,
    ):
        self.collection_name = collection_name
        if client is not None:
            self._client = client
            self.persist_path = Path(persist_directory or "/tmp/chroma_ephemeral").resolve()
        else:
            dir_to_use = persist_directory or _settings.CHROMA_PERSIST_DIR
            self.persist_path = Path(dir_to_use).resolve()
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_path))

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Initialized ChromaVectorStore at '%s' with collection '%s' (%d items).",
            self.persist_path,
            self.collection_name,
            self._collection.count(),
        )

    def count(self) -> int:
        """Return total indexed vectors in collection."""
        return self._collection.count()

    def clear(self) -> None:
        """Clear all indexed documents from the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning("Error clearing Chroma collection: %s", exc)

    def _sync_add_chunks(self, chunks: list[IndexedChunk]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "session_id": c.session_id or "default",
                "user_id": c.user_id or "default_user",
                "document_id": c.document_id,
                "document_title": c.document_title,
                "file_name": c.file_name or "",
                "page_number": int(c.page_number or 1),
                "chunk_index": int(c.chunk_index),
            }
            for c in chunks
        ]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            if "dimension" in str(exc).lower() or "does not match" in str(exc).lower():
                logger.warning(
                    "ChromaDB dimension mismatch (%s). Re-creating collection '%s' with new vector dimension.",
                    exc,
                    self.collection_name,
                )
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                raise

    async def add_chunks(self, chunks: list[IndexedChunk]) -> None:
        """Add chunks to Chroma index offloading to thread pool."""
        await asyncio.to_thread(self._sync_add_chunks, chunks)

    def _sync_search(
        self,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        if self._collection.count() == 0:
            return []

        conds = []
        if user_id:
            conds.append({"user_id": {"$eq": user_id}})
        if session_id:
            conds.append({"session_id": {"$eq": session_id}})

        if len(conds) > 1:
            where_filter = {"$and": conds}
        elif len(conds) == 1:
            where_filter = conds[0]
        else:
            where_filter = None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower() or "does not match" in str(exc).lower():
                logger.warning(
                    "Query vector dimension mismatch with collection (%s). Returning empty.",
                    exc,
                )
                return []
            raise

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        matched: list[tuple[IndexedChunk, float]] = []
        for cid, dist, doc_text, meta in zip(ids, distances, documents, metadatas):
            similarity = round(max(0.0, 1.0 - float(dist)), 4)
            if similarity < threshold:
                continue

            chunk = IndexedChunk(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                document_title=meta.get("document_title", ""),
                chunk_index=meta.get("chunk_index", 0),
                content=doc_text,
                embedding=[],
                session_id=meta.get("session_id", "default"),
                user_id=meta.get("user_id", "default_user"),
                file_name=meta.get("file_name", ""),
                page_number=meta.get("page_number", 1),
            )
            matched.append((chunk, similarity))

        return matched

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.20,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        """Search top-k chunks, optionally scoped to a session_id and user_id."""
        return await asyncio.to_thread(
            self._sync_search, query_embedding, top_k, threshold, session_id, user_id
        )

    def _sync_get_all_chunks(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[IndexedChunk]:
        conds = []
        if session_id:
            conds.append({"session_id": session_id})
        if user_id:
            conds.append({"user_id": user_id})

        if len(conds) > 1:
            where_filter = {"$and": conds}
        elif len(conds) == 1:
            where_filter = conds[0]
        else:
            where_filter = None

        try:
            if where_filter:
                results = self._collection.get(
                    where=where_filter, include=["documents", "metadatas"]
                )
            else:
                results = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("Failed to fetch all chunks from ChromaDB (%s). Returning empty.", exc)
            return []

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        chunks: list[IndexedChunk] = []
        for cid, doc_text, meta in zip(ids, documents, metadatas):
            chunks.append(
                IndexedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", "") if meta else "",
                    document_title=meta.get("document_title", "Document") if meta else "Document",
                    chunk_index=meta.get("chunk_index", 0) if meta else 0,
                    content=doc_text or "",
                    embedding=[],
                    session_id=meta.get("session_id", "default") if meta else "default",
                    user_id=meta.get("user_id", "default_user") if meta else "default_user",
                    file_name=meta.get("file_name", "") if meta else "",
                    page_number=meta.get("page_number", 1) if meta else 1,
                )
            )
        return chunks

    async def get_all_chunks(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[IndexedChunk]:
        """Fetch all indexed chunks within user and session scope."""
        return await asyncio.to_thread(self._sync_get_all_chunks, session_id, user_id)

    def _sync_remove_by_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            logger.warning(
                "Failed to delete chunks for document %s from Chroma: %s",
                document_id,
                exc,
            )

    async def remove_by_document_id(self, document_id: str) -> None:
        """Remove document chunks from Chroma."""
        await asyncio.to_thread(self._sync_remove_by_document, document_id)

    def _sync_remove_by_session(self, session_id: str) -> None:
        try:
            self._collection.delete(where={"session_id": session_id})
        except Exception as exc:
            logger.warning(
                "Failed to delete chunks for session %s from Chroma: %s",
                session_id,
                exc,
            )

    async def remove_by_session_id(self, session_id: str) -> None:
        """Remove all chunks associated with a session."""
        await asyncio.to_thread(self._sync_remove_by_session, session_id)


# Backward compatibility aliases for deep imports
sys.modules["app.providers.vector_store.base"] = sys.modules[__name__]
sys.modules["app.providers.vector_store.chroma_store"] = sys.modules[__name__]
