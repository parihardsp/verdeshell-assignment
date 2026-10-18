"""Simplified Streamlit Reference Client for Enterprise Knowledge Assistant.

Clean, intuitive interface with:
- Automatic session identification per browser tab.
- Multi-file upload (.pdf, .txt, .md) with structured parsing and vector indexing.
- Direct question answering with grounded citations and latency tracking.
- Document management with live deletion.
"""

import re
import uuid

import httpx

# pyrefly: ignore[missing-import]
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .bot-summary-box {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .bot-summary-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .bot-summary-text {
        font-size: 1.05rem;
        line-height: 1.55;
        font-weight: 500;
    }
    .bot-breakdown-header {
        font-size: 0.84rem;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .bot-meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-top: 12px;
        margin-bottom: 8px;
        padding-top: 8px;
    }
    .bot-meta-pill {
        background: rgba(107, 114, 128, 0.12);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:4]}"

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:4]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_headers():
    """Return standard tenant identification headers."""
    return {
        "X-User-ID": st.session_state.user_id,
        "X-Session-ID": st.session_state.session_id,
    }


# ------------------------------------------------------------------------------
# Backend API Helpers
# ------------------------------------------------------------------------------
def get_health():
    """Check backend connectivity and configuration."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0, headers=get_headers())
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_session_docs(session_id: str, user_id: str):
    """Retrieve documents indexed under the current user and session."""
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/documents",
            params={"session_id": session_id, "user_id": user_id, "limit": 50},
            headers=get_headers(),
            timeout=5.0,
        )
        return resp.json().get("items", []) if resp.status_code == 200 else []
    except Exception:
        return []


def upload_files_to_backend(files, session_id: str, user_id: str):
    """Upload up to 5 files for PyMuPDF parsing and ChromaDB vector indexing."""
    multipart = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    return httpx.post(
        f"{BACKEND_URL}/api/v1/documents/upload",
        files=multipart,
        data={"session_id": session_id, "user_id": user_id},
        headers=get_headers(),
        timeout=60.0,
    )


def ingest_raw_text(title: str, content: str, session_id: str, user_id: str):
    """Ingest a direct text document into ChromaDB."""
    payload = {
        "title": title,
        "content": content,
        "session_id": session_id,
        "user_id": user_id,
    }
    return httpx.post(
        f"{BACKEND_URL}/api/v1/documents",
        json=payload,
        headers=get_headers(),
        timeout=30.0,
    )


def delete_doc_by_id(doc_id: str):
    """Delete a document and purge its vectors from ChromaDB."""
    try:
        resp = httpx.delete(
            f"{BACKEND_URL}/api/v1/documents/{doc_id}",
            headers=get_headers(),
            timeout=5.0,
        )
        return resp.status_code == 204
    except Exception:
        return False


def query_backend_ask(question: str, session_id: str, user_id: str):
    """Query the Grounded RAG endpoint scoped to user and session."""
    return httpx.post(
        f"{BACKEND_URL}/api/v1/ask",
        json={"question": question, "session_id": session_id, "user_id": user_id},
        headers=get_headers(),
        timeout=30.0,
    )


# ------------------------------------------------------------------------------
# Sidebar: System Status & Multi-User Identity Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Assistant Status")
    health = get_health()
    if health:
        st.success(f"● Connected (v{health.get('version', '0.1.0')})")
        llm_mode = str(health.get("llm_provider_mode", "gemini")).upper()
        emb_mode = str(health.get("embedding_provider_mode", "gemini")).upper()
        st.caption(f"**LLM:** `{health.get('llm_provider', 'gemini')}` ({llm_mode})")
        st.caption(
            f"**Embedding:** `{health.get('embedding_provider', 'GeminiEmbeddingProvider')}` ({emb_mode})"
        )
        st.caption(
            f"**Vector Store:** `{health.get('vector_store_provider', 'ChromaVectorStore')}` (ChromaDB)"
        )
    else:
        st.error("● Backend Offline")
        st.caption("Ensure `uvicorn app.main:app` is running on port 8000.")

    st.divider()
    st.markdown("### 👤 Session Identity")
    st.caption("Automatically assigned per browser tab:")
    st.markdown(f"**User:** `{st.session_state.user_id}`")
    st.markdown(f"**Session:** `{st.session_state.session_id}`")

    docs = get_session_docs(st.session_state.session_id, st.session_state.user_id)
    st.metric("Documents in Scope", len(docs))

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            "🔄 New Session",
            use_container_width=True,
            help="Start a new session in this browser tab",
        ):
            st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        if st.button(
            "🧹 Clear Chat", use_container_width=True, help="Clear active chat messages"
        ):
            st.session_state.messages = []
            st.rerun()


def render_source_card(s: dict, idx: int):
    page_str = f" • Page {s.get('page_number')}" if s.get("page_number") else ""
    file_str = f" ({s.get('file_name')})" if s.get("file_name") else ""
    channel = s.get("retrieval_channel", "dense")
    channel_badge = (
        "🟣 Hybrid (Vector + BM25)"
        if channel == "hybrid"
        else ("🔵 Dense Vector" if channel == "dense" else "🟢 BM25 Lexical")
    )
    reranker_score = s.get("reranker_score")
    if reranker_score is not None and reranker_score > 0:
        score_info = f"Groq Score: **{reranker_score:.1f}/10**"
    else:
        score_info = f"Score: **{s.get('similarity_score', 0):.4f}**"

    st.markdown(
        f"**[{idx}] {s.get('title')}{file_str}{page_str}**\n\n"
        f"`{channel_badge}` • {score_info}\n\n"
        f"> {s.get('snippet')}"
    )


def parse_bot_response(raw_text: str) -> tuple[str, str | None]:
    """Split response into a quick summary takeaway and a detailed points/steps breakdown."""
    text = raw_text.strip()
    if not text:
        return "", None

    # Check for double newline separator
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        summary = paragraphs[0]
        breakdown = "\n\n".join(paragraphs[1:]).strip()
        return summary, breakdown

    # Check single paragraph with embedded bullet points
    lines = text.splitlines()
    first_bullet_idx = next(
        (i for i, line_str in enumerate(lines) if line_str.strip().startswith(("-", "*", "1.", "2.", "3.", "•"))),
        None,
    )
    if first_bullet_idx and first_bullet_idx > 0:
        summary = " ".join(line_str.strip() for line_str in lines[:first_bullet_idx] if line_str.strip())
        breakdown = "\n".join(lines[first_bullet_idx:]).strip()
        return summary, breakdown

    return text, None


def render_assistant_response(
    answer: str,
    sources: list,
    latency_ms: int | None = None,
    confidence_score: float | None = None,
    is_expanded: bool = False,
):
    """Render a helpful, structured assistant response with summary, breakdown, metadata, and sources."""
    is_insufficient = (
        "does not contain sufficient information" in answer.lower()
        or "does not contain information" in answer.lower()
    )

    if is_insufficient:
        st.warning(f"⚠️ {answer}")
        st.caption(
            "💡 *Tip: Upload the relevant PDF, Markdown, or text documents in the 'Upload & Manage Documents' tab to ground this query.*"
        )
        if latency_ms:
            st.caption(f"⚡ Latency: `{latency_ms} ms`")
        return

    summary, breakdown = parse_bot_response(answer)

    # 1. Quick Takeaway / Summary Card
    if summary:
        clean_summary = re.sub(r"^\*\*(?:Quick\s*)?Summary:?\*\*\s*", "", summary, flags=re.IGNORECASE)
        st.markdown(
            f"""<div class="bot-summary-box">
                <div class="bot-summary-label">📌 Quick Summary</div>
                <div class="bot-summary-text">{clean_summary}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 2. Key Details & Step-by-Step Breakdown
    if breakdown:
        st.markdown(
            '<div class="bot-breakdown-header">📋 Key Details & Step-by-Step Breakdown</div>',
            unsafe_allow_html=True,
        )
        st.markdown(breakdown)

    # 3. Metadata Bar (Latency, Groundedness, Confidence, Sources)
    meta_pills = ['<span class="bot-meta-pill">🟢 Grounded in Session Docs</span>']
    if latency_ms:
        meta_pills.append(f'<span class="bot-meta-pill">⚡ {latency_ms} ms</span>')
    if confidence_score is not None and confidence_score > 0:
        conf_pct = int(confidence_score * 100)
        meta_pills.append(f'<span class="bot-meta-pill">🎯 {conf_pct}% Confidence</span>')
    if sources:
        meta_pills.append(f'<span class="bot-meta-pill">📑 {len(sources)} Sources Cited</span>')

    pills_html = " ".join(meta_pills)
    st.markdown(f'<div class="bot-meta-bar">{pills_html}</div>', unsafe_allow_html=True)

    # 4. Verified Sources Expander
    if sources:
        with st.expander(f"📑 Verified Sources ({len(sources)})", expanded=is_expanded):
            for idx, s in enumerate(sources, 1):
                render_source_card(s, idx)


# ------------------------------------------------------------------------------
# Main Application: 2 Simple Tabs
# ------------------------------------------------------------------------------
st.title("🌿 Mini Knowledge Assistant")

tab_chat, tab_docs = st.tabs(["💬 Ask Questions", "📁 Upload & Manage Documents"])

# ==============================================================================
# TAB 1: Grounded Q&A Chat
# ==============================================================================
with tab_chat:
    st.caption(
        f"Questions are answered strictly using documents in session `{st.session_state.session_id}`."
    )

    # Render Chat History
    if not st.session_state.messages:
        st.info(
            "💡 **Welcome!** Upload a document in the 'Upload & Manage Documents' tab, then ask questions here."
        )
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_assistant_response(
                        answer=msg["content"],
                        sources=msg.get("sources", []),
                        latency_ms=msg.get("latency_ms"),
                        confidence_score=msg.get("confidence_score"),
                        is_expanded=False,
                    )

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your knowledge articles...")
    if user_query and user_query.strip():
        q_text = user_query.strip()

        # Display user message
        st.session_state.messages.append({"role": "user", "content": q_text})
        with st.chat_message("user"):
            st.markdown(q_text)

        # Call RAG Backend
        with st.chat_message("assistant"):
            with st.spinner("Retrieving verified facts & synthesizing answer..."):
                try:
                    resp = query_backend_ask(
                        q_text, st.session_state.session_id, st.session_state.user_id
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        latency_ms = data.get("latency_ms", 0)
                        confidence_score = data.get("confidence_score")

                        render_assistant_response(
                            answer=answer,
                            sources=sources,
                            latency_ms=latency_ms,
                            confidence_score=confidence_score,
                            is_expanded=True,
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "latency_ms": latency_ms,
                                "confidence_score": confidence_score,
                            }
                        )

                    elif resp.status_code == 429:
                        st.error(
                            "⚠️ Rate limit reached (20 req/min). Please wait a moment before asking again."
                        )
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")


# ==============================================================================
# TAB 2: Upload & Manage Documents
# ==============================================================================
with tab_docs:
    st.subheader("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files to index (.pdf, .txt, .md - max 5 files, 5MB each):",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption(f"Selected **{len(uploaded_files)}** file(s).")
        if st.button("🚀 Upload & Index", type="primary"):
            with st.spinner(
                "Extracting with PyMuPDF, chunking, and indexing in ChromaDB..."
            ):
                try:
                    resp = upload_files_to_backend(
                        uploaded_files,
                        st.session_state.session_id,
                        st.session_state.user_id,
                    )
                    if resp.status_code == 201:
                        data = resp.json()
                        st.success(
                            f"✅ Successfully indexed {data.get('total_files')} file(s) into ChromaDB!"
                        )
                        st.rerun()
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    # Direct Text Option
    with st.expander("✍️ Or Paste Text Directly"):
        t_title = st.text_input("Article Title", placeholder="e.g. Remote Work Policy")
        t_body = st.text_area(
            "Article Content", placeholder="Paste body text here...", height=120
        )
        if st.button("Save Text Article"):
            if t_title.strip() and t_body.strip():
                try:
                    resp = ingest_raw_text(
                        t_title.strip(),
                        t_body.strip(),
                        st.session_state.session_id,
                        st.session_state.user_id,
                    )
                    if resp.status_code == 201:
                        st.success("✅ Text document indexed!")
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as exc:
                    st.error(f"Failed: {exc}")
            else:
                st.warning("Please provide both title and content.")

    st.divider()

    # Active Documents in Session
    st.subheader(
        f"📚 Documents in Scope (`{st.session_state.user_id}` / `{st.session_state.session_id}`)"
    )
    session_docs = get_session_docs(
        st.session_state.session_id, st.session_state.user_id
    )

    if not session_docs:
        st.info(
            "No documents uploaded yet in this session. Upload a file above to get started."
        )
    else:
        for d in session_docs:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                doc_title = d.get("title", "Untitled")
                doc_file = d.get("file_name") or "Direct Text"
                doc_chunks = d.get("chunk_count", 0)
                st.markdown(
                    f"**{doc_title}** (`{d.get('document_id')}`) — 📄 `{doc_file}` | 🧩 `{doc_chunks} chunks`"
                )
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{d.get('document_id')}"):
                    if delete_doc_by_id(d.get("document_id")):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
