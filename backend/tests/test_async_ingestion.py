"""Unit and integration tests for asynchronous document ingestion (HTTP 202, SSE, polling, failure isolation)."""

import asyncio
import io
import json

import pytest
from httpx import AsyncClient

from app.services.job_manager import global_job_manager


@pytest.mark.asyncio
async def test_async_upload_returns_202_and_job_metadata(async_client: AsyncClient):
    """Test that POST /documents/upload?background=true immediately returns HTTP 202 Accepted."""
    files = [
        (
            "files",
            (
                "remote_work_policy.txt",
                io.BytesIO(b"Employees may work remotely up to two days per week upon manager approval."),
                "text/plain",
            ),
        ),
    ]

    res = await async_client.post("/api/v1/documents/upload?background=true", files=files)
    assert res.status_code == 202
    data = res.json()

    assert "job_id" in data
    assert data["status"] in ("queued", "processing", "completed")
    assert data["total_files"] == 1
    assert data["poll_url"].startswith("/api/v1/documents/jobs/")
    assert data["stream_url"].startswith("/api/v1/documents/jobs/")


@pytest.mark.asyncio
async def test_async_job_status_polling(async_client: AsyncClient):
    """Test that GET /documents/jobs/{job_id} returns snapshot and tracks progress to completion."""
    files = [
        (
            "files",
            (
                "compliance_guidelines.txt",
                io.BytesIO(b"Annual security compliance training is mandatory for all active contractors and employees."),
                "text/plain",
            ),
        ),
    ]

    upload_res = await async_client.post("/api/v1/documents/upload?background=true", files=files)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    # Poll until completed (timeout after 10s)
    max_retries = 20
    final_status = None
    for _ in range(max_retries):
        status_res = await async_client.get(f"/api/v1/documents/jobs/{job_id}")
        assert status_res.status_code == 200
        snap = status_res.json()
        assert snap["job_id"] == job_id
        if snap["status"] in ("completed", "failed", "completed_with_errors"):
            final_status = snap
            break
        await asyncio.sleep(0.2)

    assert final_status is not None
    assert final_status["status"] == "completed"
    assert final_status["percent"] == 100
    assert len(final_status["results"]) == 1
    assert final_status["results"][0]["status"] == "indexed"


@pytest.mark.asyncio
async def test_async_job_failure_isolation(async_client: AsyncClient):
    """Test individual file failure isolation: one valid file and one empty/unreadable file."""
    files = [
        (
            "files",
            (
                "valid_notes.txt",
                io.BytesIO(b"Company offsites are planned semiannually in June and November."),
                "text/plain",
            ),
        ),
        (
            "files",
            (
                "empty_corrupted.txt",
                io.BytesIO(b"   "),  # Pure whitespace, fails extraction/chunking validation
                "text/plain",
            ),
        ),
    ]

    upload_res = await async_client.post("/api/v1/documents/upload?background=true", files=files)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    # Poll until finished
    final_snap = None
    for _ in range(25):
        snap_res = await async_client.get(f"/api/v1/documents/jobs/{job_id}")
        assert snap_res.status_code == 200
        snap = snap_res.json()
        if snap["status"] in ("completed", "completed_with_errors", "failed"):
            final_snap = snap
            break
        await asyncio.sleep(0.2)

    assert final_snap is not None
    # Individual failure isolation: the valid file was indexed, while the bad file was recorded as an error
    assert final_snap["status"] == "completed_with_errors"
    assert len(final_snap["results"]) == 1
    assert final_snap["results"][0]["file_name"] == "valid_notes.txt"
    assert final_snap["results"][0]["status"] == "indexed"
    assert len(final_snap["errors"]) == 1
    assert final_snap["errors"][0]["file_name"] == "empty_corrupted.txt"
    assert final_snap["errors"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_async_job_sse_stream_initial_event(async_client: AsyncClient):
    """Test that SSE stream endpoint returns text/event-stream and initial JSON state."""
    job = global_job_manager.create_job(
        session_id="test_sess",
        user_id="test_user",
        total_files=2,
    )
    # Finish job so the generator yields initial terminal snapshot and cleanly finishes
    await global_job_manager.finish_job(job.job_id, results=[{"file_name": "test.txt", "status": "indexed"}])

    # Request SSE stream with stream=True
    async with async_client.stream("GET", f"/api/v1/documents/jobs/{job.job_id}/events") as sse_res:
        assert sse_res.status_code == 200
        assert "text/event-stream" in sse_res.headers.get("content-type", "")

        # Read the first event chunk
        first_chunk = ""
        async for line in sse_res.aiter_lines():
            if line.startswith("data:"):
                first_chunk = line
                break

        assert first_chunk.startswith("data:")
        raw_json = first_chunk.replace("data:", "").strip()
        parsed = json.loads(raw_json)
        assert parsed["job_id"] == job.job_id
        assert parsed["status"] == "completed"

