"""Asynchronous Ingestion Job Manager with SSE and polling support.

Maintains in-memory job state with TTL eviction, fan-out event queues for
Server-Sent Events (SSE), and progress snapshots.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.logging import LogPhase

logger = logging.getLogger(__name__)


@dataclass
class IngestionJob:
    """In-memory representation of an asynchronous document ingestion job."""

    job_id: str
    session_id: str
    user_id: str
    status: str  # "queued", "processing", "completed", "failed", "completed_with_errors"
    total_files: int
    processed_files: int = 0
    current_file: str | None = None
    current_stage: str = "QUEUED"
    percent: int = 0
    message: str = "Job queued for indexing"
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IngestionJobManager:
    """Thread-safe manager for tracking asynchronous ingestion jobs and publishing SSE events."""

    def __init__(self, ttl_seconds: int = 1800):
        self._jobs: dict[str, IngestionJob] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self.ttl_seconds = ttl_seconds

    def create_job(
        self,
        session_id: str,
        user_id: str,
        total_files: int,
        file_names: list[str] | None = None,
    ) -> IngestionJob:
        """Create and register a new pending ingestion job."""
        self.cleanup_stale_jobs()
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = IngestionJob(
            job_id=job_id,
            session_id=session_id,
            user_id=user_id,
            status="queued",
            total_files=total_files,
            processed_files=0,
            current_stage="QUEUED",
            percent=0,
            message=f"Queued {total_files} file(s) for indexing",
        )
        self._jobs[job_id] = job
        logger.info(
            "Ingestion job created | job_id=%s files=%d session_id=%s",
            job_id,
            total_files,
            session_id,
            extra={"phase": LogPhase.INGESTION},
        )
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Retrieve the job instance by identifier."""
        return self._jobs.get(job_id)

    def get_job_dict(self, job_id: str) -> dict[str, Any] | None:
        """Return a serializable dictionary representation of the job state."""
        job = self.get_job(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "user_id": job.user_id,
            "status": job.status,
            "stage": job.current_stage,
            "percent": job.percent,
            "message": job.message,
            "current_file": job.current_file,
            "processed_files": job.processed_files,
            "total_files": job.total_files,
            "results": job.results,
            "errors": job.errors,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    async def update_progress(
        self,
        job_id: str,
        stage: str,
        percent: int,
        message: str,
        current_file: str | None = None,
        processed_files: int | None = None,
    ) -> None:
        """Update job progress and publish event to all connected SSE clients."""
        job = self.get_job(job_id)
        if not job:
            return

        job.status = "processing"
        job.current_stage = stage
        job.percent = min(100, max(0, percent))
        job.message = message
        if current_file is not None:
            job.current_file = current_file
        if processed_files is not None:
            job.processed_files = processed_files
        job.updated_at = time.time()

        payload = self.get_job_dict(job_id)
        await self._broadcast(job_id, payload)

    async def finish_job(
        self,
        job_id: str,
        results: list[dict],
        errors: list[dict] | None = None,
    ) -> None:
        """Mark job completed (or completed_with_errors) and publish terminal event."""
        job = self.get_job(job_id)
        if not job:
            return

        job.results = results
        job.errors = errors or []
        job.processed_files = len(results) + len(job.errors)
        job.percent = 100
        job.current_stage = "COMPLETED"
        job.updated_at = time.time()

        if job.errors and results:
            job.status = "completed_with_errors"
            job.message = f"Indexed {len(results)} file(s) with {len(job.errors)} failure(s)."
        elif job.errors and not results:
            job.status = "failed"
            job.message = f"All {len(job.errors)} file(s) failed indexing."
        else:
            job.status = "completed"
            job.message = f"Successfully indexed {len(results)} document(s)."

        logger.info(
            "Ingestion job completed | job_id=%s status=%s indexed=%d failed=%d",
            job_id,
            job.status,
            len(results),
            len(job.errors),
            extra={"phase": LogPhase.INGESTION},
        )

        payload = self.get_job_dict(job_id)
        await self._broadcast(job_id, payload)

    async def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job failed due to an unexpected system error."""
        job = self.get_job(job_id)
        if not job:
            return

        job.status = "failed"
        job.current_stage = "FAILED"
        job.percent = 100
        job.message = error_message
        job.updated_at = time.time()

        logger.error(
            "Ingestion job aborted | job_id=%s error=%s",
            job_id,
            error_message,
            extra={"phase": LogPhase.INGESTION},
        )

        payload = self.get_job_dict(job_id)
        await self._broadcast(job_id, payload)

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe an SSE listener to real-time events for the specified job."""
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers[job_id].append(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unregister an SSE listener queue."""
        async with self._lock:
            if job_id in self._subscribers and queue in self._subscribers[job_id]:
                self._subscribers[job_id].remove(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def _broadcast(self, job_id: str, payload: dict[str, Any] | None) -> None:
        """Broadcast payload to all active listener queues for the job."""
        if not payload:
            return
        async with self._lock:
            listeners = list(self._subscribers.get(job_id, []))
        for q in listeners:
            await q.put(payload)

    def cleanup_stale_jobs(self) -> int:
        """Purge jobs older than TTL. Returns number of purged entries."""
        now = time.time()
        stale_ids = [
            jid
            for jid, j in self._jobs.items()
            if (now - j.updated_at) > self.ttl_seconds
        ]
        for jid in stale_ids:
            self._jobs.pop(jid, None)
            self._subscribers.pop(jid, None)
        return len(stale_ids)


# Singleton manager instance
global_job_manager = IngestionJobManager()
