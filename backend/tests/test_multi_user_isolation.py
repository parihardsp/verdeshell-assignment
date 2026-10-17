"""Unit and integration tests for Multi-Tenant User Isolation and Logging Scoping."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_multi_user_tenant_isolation(async_client: AsyncClient):
    """Verify that User A and User B documents are strictly isolated in ChromaDB and SQLite."""
    # 1. User Alice ingests a private document
    alice_doc = {
        "title": "Alice Secret Project",
        "content": "Project Nebula is a quantum cryptography initiative led exclusively by Alice.",
        "session_id": "sess_alice_1",
        "user_id": "usr_alice",
    }
    alice_res = await async_client.post(
        "/api/v1/documents",
        json=alice_doc,
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_alice_1"},
    )
    assert alice_res.status_code == 201
    assert alice_res.json()["user_id"] == "usr_alice"
    assert alice_res.headers.get("X-User-ID") == "usr_alice"

    # 2. User Bob ingests a private document
    bob_doc = {
        "title": "Bob Financial Report",
        "content": "Quarterly dividend allocation for Project Phoenix was exactly 4.2 million dollars.",
        "session_id": "sess_bob_1",
        "user_id": "usr_bob",
    }
    bob_res = await async_client.post(
        "/api/v1/documents",
        json=bob_doc,
        headers={"X-User-ID": "usr_bob", "X-Session-ID": "sess_bob_1"},
    )
    assert bob_res.status_code == 201
    assert bob_res.json()["user_id"] == "usr_bob"
    assert bob_res.headers.get("X-User-ID") == "usr_bob"

    # 3. User Alice queries for her own project
    alice_ask = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What is Project Nebula?",
            "session_id": "sess_alice_1",
            "user_id": "usr_alice",
        },
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_alice_1"},
    )
    assert alice_ask.status_code == 200
    alice_data = alice_ask.json()
    assert len(alice_data["sources"]) > 0
    assert any("Project Nebula" in s["snippet"] for s in alice_data["sources"])
    assert alice_data["user_id"] == "usr_alice"

    # 4. User Alice tries to query Bob's project (Must NOT return Bob's data!)
    alice_snoop = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What was the quarterly dividend for Project Phoenix?",
            "session_id": "sess_alice_1",
            "user_id": "usr_alice",
        },
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_alice_1"},
    )
    assert alice_snoop.status_code == 200
    snoop_data = alice_snoop.json()
    # Zero candidate chunks from Bob can leak into Alice's query results
    assert all(s["user_id"] == "usr_alice" for s in snoop_data["sources"])
    assert not any("Project Phoenix" in s["snippet"] for s in snoop_data["sources"])
    assert not any("Bob Financial Report" in s["title"] for s in snoop_data["sources"])

    # 4b. User Alice tries to query Bob's session directly with Alice's user_id
    alice_cross_session = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What was the quarterly dividend for Project Phoenix?",
            "session_id": "sess_bob_1",
            "user_id": "usr_alice",
        },
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_bob_1"},
    )
    assert alice_cross_session.status_code == 200
    cross_data = alice_cross_session.json()
    assert len(cross_data["sources"]) == 0
    assert "does not contain" in cross_data["answer"].lower()

    # 5. User Bob queries his own project
    bob_ask = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What was the quarterly dividend for Project Phoenix?",
            "session_id": "sess_bob_1",
            "user_id": "usr_bob",
        },
        headers={"X-User-ID": "usr_bob", "X-Session-ID": "sess_bob_1"},
    )
    assert bob_ask.status_code == 200
    bob_data = bob_ask.json()
    assert len(bob_data["sources"]) > 0
    assert any("Project Phoenix" in s["snippet"] for s in bob_data["sources"])
    assert bob_data["user_id"] == "usr_bob"


@pytest.mark.asyncio
async def test_list_documents_user_scoping(async_client: AsyncClient):
    """Verify that listing documents correctly filters by user_id."""
    res_alice = await async_client.get(
        "/api/v1/documents",
        params={"user_id": "usr_alice"},
        headers={"X-User-ID": "usr_alice"},
    )
    assert res_alice.status_code == 200
    items_alice = res_alice.json()["items"]
    for item in items_alice:
        assert item["user_id"] == "usr_alice"


@pytest.mark.asyncio
async def test_multi_user_deduplication_marks_document_for_new_user(
    async_client: AsyncClient,
):
    """Verify that when a duplicate file is ingested by a new user:
    1. The document is marked against the new user in SQLite with a distinct document_id.
    2. Chunks & precomputed embeddings are replicated in ChromaDB under the new user.
    3. The new user can retrieve and query their own copy.
    4. Deleting the new user's document does NOT impact the original user's copy.
    """
    shared_content = (
        "The annual company engineering summit will take place in Kyoto Japan from October 10 to 14. "
        "All engineers are eligible for travel sponsorship."
    )

    # 1. User Alice ingests the document
    alice_res = await async_client.post(
        "/api/v1/documents",
        json={
            "title": "Summit Guidelines",
            "content": shared_content,
            "session_id": "sess_alice_summit",
            "user_id": "usr_alice",
        },
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_alice_summit"},
    )
    assert alice_res.status_code == 201
    alice_doc = alice_res.json()
    assert alice_doc["user_id"] == "usr_alice"
    alice_doc_id = alice_doc["document_id"]

    # 2. User Bob ingests the EXACT same content (Triggering deduplication reuse!)
    bob_res = await async_client.post(
        "/api/v1/documents",
        json={
            "title": "Summit Guidelines",
            "content": shared_content,
            "session_id": "sess_bob_summit",
            "user_id": "usr_bob",
        },
        headers={"X-User-ID": "usr_bob", "X-Session-ID": "sess_bob_summit"},
    )
    assert bob_res.status_code == 201
    bob_doc = bob_res.json()
    bob_doc_id = bob_doc["document_id"]

    # Must be marked against Bob with a distinct document_id
    assert bob_doc["user_id"] == "usr_bob"
    assert bob_doc["session_id"] == "sess_bob_summit"
    assert bob_doc_id != alice_doc_id

    # 3. Verify Bob's document library contains his copy
    bob_list = await async_client.get(
        "/api/v1/documents",
        params={"user_id": "usr_bob"},
        headers={"X-User-ID": "usr_bob"},
    )
    assert bob_list.status_code == 200
    bob_items = bob_list.json()["items"]
    assert any(d["document_id"] == bob_doc_id for d in bob_items)
    assert all(d["user_id"] == "usr_bob" for d in bob_items)

    # 4. User Bob queries his document
    bob_ask = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "Where will the engineering summit take place?",
            "session_id": "sess_bob_summit",
            "user_id": "usr_bob",
        },
        headers={"X-User-ID": "usr_bob", "X-Session-ID": "sess_bob_summit"},
    )
    assert bob_ask.status_code == 200
    bob_data = bob_ask.json()
    assert len(bob_data["sources"]) > 0
    assert all(s["user_id"] == "usr_bob" for s in bob_data["sources"])
    assert any("Kyoto Japan" in s["snippet"] for s in bob_data["sources"])

    # 5. User Bob deletes his document
    del_res = await async_client.delete(f"/api/v1/documents/{bob_doc_id}")
    assert del_res.status_code == 204

    # 6. Verify Bob's document is gone
    bob_list_after = await async_client.get(
        "/api/v1/documents",
        params={"user_id": "usr_bob"},
        headers={"X-User-ID": "usr_bob"},
    )
    assert not any(d["document_id"] == bob_doc_id for d in bob_list_after.json()["items"])

    # 7. Verify Alice's document is STILL intact and queryable!
    alice_list_after = await async_client.get(
        "/api/v1/documents",
        params={"user_id": "usr_alice"},
        headers={"X-User-ID": "usr_alice"},
    )
    assert any(d["document_id"] == alice_doc_id for d in alice_list_after.json()["items"])

    alice_ask = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "Where will the engineering summit take place?",
            "session_id": "sess_alice_summit",
            "user_id": "usr_alice",
        },
        headers={"X-User-ID": "usr_alice", "X-Session-ID": "sess_alice_summit"},
    )
    assert alice_ask.status_code == 200
    alice_data = alice_ask.json()
    assert len(alice_data["sources"]) > 0
    assert all(s["user_id"] == "usr_alice" for s in alice_data["sources"])
    assert any("Kyoto Japan" in s["snippet"] for s in alice_data["sources"])
