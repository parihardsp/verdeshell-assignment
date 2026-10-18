"""Integration tests for multi-file upload, PyMuPDF extraction, and session-scoped RAG."""

import io

import pymupdf
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_multi_upload_and_session_rag(async_client: AsyncClient):
    # 1. Create a dummy multi-page PDF using PyMuPDF in memory
    pdf_doc = pymupdf.open()
    page1 = pdf_doc.new_page()
    page1.insert_text(
        (50, 100),
        "IT Equipment Policy: Company laptops are refreshed every three years for full-time employees.",
    )
    page2 = pdf_doc.new_page()
    page2.insert_text(
        (50, 100),
        "Hardware Incident Protocol: Any hardware damage or theft must be reported within 24 hours to the IT security helpdesk.",
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    # 2. Prepare 3 files for upload
    files = [
        (
            "files",
            (
                "hr_policy.txt",
                io.BytesIO(
                    b"Employees receive 25 days of annual paid vacation leave. Requests require 14 days advance notice."
                ),
                "text/plain",
            ),
        ),
        (
            "files",
            (
                "security_policy.txt",
                io.BytesIO(
                    b"Mandatory 2FA authentication is enforced on all corporate accounts using hardware keys."
                ),
                "text/plain",
            ),
        ),
        (
            "files",
            ("it_guidelines.pdf", io.BytesIO(pdf_bytes), "application/pdf"),
        ),
    ]

    # 3. Perform upload
    upload_res = await async_client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    session_id = upload_data["session_id"]
    assert session_id.startswith("user_") or session_id.startswith("sess_")
    assert upload_data["total_files"] == 3

    # Check file persistence
    for doc in upload_data["documents"]:
        assert doc["status"] == "indexed"
        assert doc["chunk_count"] >= 1

    # 4. Ask session-scoped question about HR vacation days
    q1 = {
        "question": "How many days of paid vacation do employees receive?",
        "session_id": session_id,
    }
    ask_res1 = await async_client.post("/api/v1/ask", json=q1)
    assert ask_res1.status_code == 200
    data1 = ask_res1.json()
    assert len(data1["sources"]) > 0
    assert any(s["file_name"] == "hr_policy.txt" for s in data1["sources"])
    assert data1["session_id"] == session_id

    # 5. Ask question from Page 2 of the PDF
    q2 = {
        "question": "Within how many hours must hardware damage or theft be reported?",
        "session_id": session_id,
    }
    ask_res2 = await async_client.post("/api/v1/ask", json=q2)
    assert ask_res2.status_code == 200
    data2 = ask_res2.json()
    assert len(data2["sources"]) > 0
    # Verify page-aware attribution
    pdf_sources = [s for s in data2["sources"] if s["file_name"] == "it_guidelines.pdf"]
    assert len(pdf_sources) > 0
    assert any(s["page_number"] == 2 for s in pdf_sources)


@pytest.mark.asyncio
async def test_overlimit_file_upload_rejected(async_client: AsyncClient):
    # Attempt to upload 6 files (limit is 5)
    extra_files = [
        ("files", (f"file_{i}.txt", io.BytesIO(b"sample text"), "text/plain")) for i in range(6)
    ]
    overlimit_res = await async_client.post("/api/v1/documents/upload", files=extra_files)
    assert overlimit_res.status_code in [400, 422]
    error_msg = overlimit_res.json()["error"]["message"]
    assert "maximum file upload limit" in error_msg.lower() or "5 files" in error_msg.lower()


@pytest.mark.asyncio
async def test_session_isolation(async_client: AsyncClient):
    # Upload doc in session A
    files_a = [
        (
            "files",
            (
                "finance.txt",
                io.BytesIO(b"Project Alpha budget allocation is 500,000 USD."),
                "text/plain",
            ),
        )
    ]
    res_a = await async_client.post(
        "/api/v1/documents/upload",
        files=files_a,
        data={"session_id": "sess_finance_dept"},
    )
    assert res_a.status_code == 201

    # Query in session B (different session)
    q_b = {
        "question": "What is the budget allocation for Project Alpha?",
        "session_id": "sess_marketing_dept",
    }
    ask_b = await async_client.post("/api/v1/ask", json=q_b)
    assert ask_b.status_code == 200
    # Should NOT find anything because session B has no finance documents
    assert len(ask_b.json()["sources"]) == 0
    assert "does not contain" in ask_b.json()["answer"].lower()

    # Query in session A (matching session)
    q_a = {
        "question": "What is the budget allocation for Project Alpha?",
        "session_id": "sess_finance_dept",
    }
    ask_a = await async_client.post("/api/v1/ask", json=q_a)
    assert ask_a.status_code == 200
    assert len(ask_a.json()["sources"]) > 0


@pytest.mark.asyncio
async def test_batch_upload_with_unextractable_file_graceful_handling(async_client: AsyncClient):
    """Verify that an image-only/unextractable PDF does not abort the entire batch upload."""
    # 1. Create a dummy PDF with zero text (simulating scanned/image-only PDF)
    empty_doc = pymupdf.open()
    empty_doc.new_page()  # Page without text
    empty_pdf_bytes = empty_doc.tobytes()
    empty_doc.close()

    # 2. Batch with 1 valid file and 1 image-only PDF
    files = [
        (
            "files",
            (
                "engineering_guide.txt",
                io.BytesIO(b"Backend services use async SQLAlchemy with SQLite and FastAPI."),
                "text/plain",
            ),
        ),
        (
            "files",
            ("scanned_receipt.pdf", io.BytesIO(empty_pdf_bytes), "application/pdf"),
        ),
    ]

    res = await async_client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201
    data = res.json()

    assert data["total_files"] == 2
    assert data["indexed_count"] == 1
    assert data["failed_count"] == 1

    docs_by_name = {d["file_name"]: d for d in data["documents"]}
    assert docs_by_name["engineering_guide.txt"]["status"] == "indexed"
    assert docs_by_name["engineering_guide.txt"]["chunk_count"] >= 1

    assert docs_by_name["scanned_receipt.pdf"]["status"] == "failed"
    assert "no readable text" in docs_by_name["scanned_receipt.pdf"]["error_message"].lower()

    # 3. Verify that the indexed document can be queried via RAG
    ask_res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What backend framework and ORM are used?",
            "session_id": data["session_id"],
        },
    )
    assert ask_res.status_code == 200
    assert len(ask_res.json()["sources"]) > 0

