"""Integration tests for REST API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MockEmbeddingProvider" in data["embedding_provider"]


@pytest.mark.asyncio
async def test_create_and_list_document(async_client: AsyncClient):
    payload = {
        "title": "Cancellation Policy",
        "content": "Subscriptions can be cancelled at any time from your account settings. Upon cancellation, your access remains active until the end of the current billing cycle.",
    }
    # 1. Create
    res = await async_client.post("/documents", json=payload)
    assert res.status_code == 201
    doc_data = res.json()
    assert "document_id" in doc_data
    assert doc_data["status"] == "indexed"
    assert doc_data["title"] == "Cancellation Policy"
    assert doc_data["chunk_count"] >= 1

    doc_id = doc_data["document_id"]

    # 2. Get by ID
    get_res = await async_client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["document_id"] == doc_id

    # 3. List
    list_res = await async_client.get("/documents")
    assert list_res.status_code == 200
    items = list_res.json()["items"]
    assert any(item["document_id"] == doc_id for item in items)


@pytest.mark.asyncio
async def test_ask_grounded_question(async_client: AsyncClient):
    # Ingest source document
    doc_payload = {
        "title": "Warranty Policy",
        "content": "All hardware devices include a 2-year manufacturer warranty covering internal mechanical defects. Accidental water or drop damage is not covered.",
    }
    create_res = await async_client.post("/documents", json=doc_payload)
    assert create_res.status_code == 201

    # Ask relevant question
    ask_payload = {"question": "What is covered under the 2-year warranty?"}
    ask_res = await async_client.post("/ask", json=ask_payload)
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["title"] == "Warranty Policy"


@pytest.mark.asyncio
async def test_ask_irrelevant_question_returns_grounded_fallback(
    async_client: AsyncClient,
):
    ask_payload = {"question": "What is the capital of Antarctica?"}
    ask_res = await async_client.post("/ask", json=ask_payload)
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert "does not contain" in data["answer"].lower()
    assert len(data["sources"]) == 0


@pytest.mark.asyncio
async def test_api_v1_compatibility(async_client: AsyncClient):
    doc_payload = {
        "title": "Shipping FAQ",
        "content": "Standard domestic delivery takes 3 to 5 business days. Express overnight shipping is available for urgent orders.",
    }
    create_res = await async_client.post("/api/v1/documents", json=doc_payload)
    assert create_res.status_code == 201

    ask_res = await async_client.post(
        "/api/v1/ask", json={"question": "How long does standard delivery take?"}
    )
    assert ask_res.status_code == 200
    assert len(ask_res.json()["sources"]) > 0


@pytest.mark.asyncio
async def test_validation_errors(async_client: AsyncClient):
    # Empty title
    res_bad_title = await async_client.post(
        "/documents", json={"title": "   ", "content": "Valid content"}
    )
    assert res_bad_title.status_code in [400, 422]

    # Empty question
    res_bad_q = await async_client.post("/ask", json={"question": "  "})
    assert res_bad_q.status_code in [400, 422]


@pytest.mark.asyncio
async def test_question_history_api(async_client: AsyncClient):
    """Verify that question answering records audit logs queryable and purgable via /api/v1/history."""
    # 1. Seed document
    doc_payload = {
        "title": "Security FAQ",
        "content": "All databases are encrypted at rest with AES-256 keys. TLS 1.3 is enforced on all ingress endpoints.",
        "session_id": "sess_audit_1",
        "user_id": "usr_audit",
    }
    create_res = await async_client.post(
        "/api/v1/documents",
        json=doc_payload,
        headers={"X-User-ID": "usr_audit", "X-Session-ID": "sess_audit_1"},
    )
    assert create_res.status_code == 201

    # 2. Ask question
    ask_res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What encryption standard is used for database at-rest storage?",
            "session_id": "sess_audit_1",
            "user_id": "usr_audit",
        },
        headers={"X-User-ID": "usr_audit", "X-Session-ID": "sess_audit_1"},
    )
    assert ask_res.status_code == 200

    # 3. Retrieve history for this tenant session
    history_res = await async_client.get(
        "/api/v1/history",
        params={"session_id": "sess_audit_1", "user_id": "usr_audit"},
    )
    assert history_res.status_code == 200
    data = history_res.json()
    assert data["total"] >= 1
    recent = data["items"][0]
    assert "encryption standard" in recent["question"]
    assert recent["user_id"] == "usr_audit"
    assert recent["session_id"] == "sess_audit_1"
    assert len(recent["sources"]) > 0

    # 4. Verify tenant isolation (Different user sees 0 records)
    other_res = await async_client.get(
        "/api/v1/history",
        params={"user_id": "usr_completely_different"},
    )
    assert other_res.status_code == 200
    assert other_res.json()["total"] == 0

    # 5. Clear history
    del_res = await async_client.delete(
        "/api/v1/history",
        params={"session_id": "sess_audit_1", "user_id": "usr_audit"},
    )
    assert del_res.status_code == 204

    # 6. Verify cleared
    history_after = await async_client.get(
        "/api/v1/history",
        params={"session_id": "sess_audit_1", "user_id": "usr_audit"},
    )
    assert history_after.status_code == 200
    assert history_after.json()["total"] == 0

