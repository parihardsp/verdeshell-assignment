# Enterprise Knowledge Assistant — Backend Service

An asynchronous, production-grade Retrieval-Augmented Generation (RAG) backend service built with **Python 3.12**, **FastAPI**, **SQLAlchemy 2.0 (aiosqlite)**, **ChromaDB**, **PyMuPDF / python-docx**, and **Pydantic v2**.

The service ingests multi-format enterprise documentation, preserves structural and page-boundary metadata, indexes content into dual relational and vector stores, and answers user questions through a high-precision, 3-stage hybrid retrieval and grounded synthesis pipeline.

---

## Technical Architecture & Deep Dive

> [!NOTE]
> For comprehensive technical documentation, including end-to-end request–response sequence diagrams, phase-wise ingestion and retrieval flows, concurrency models, and an enterprise **100,000+ concurrent user scalability roadmap**, please refer to the dedicated architecture specification:
> 
> **[Backend Architecture & Technical Design Specification](ARCHITECTURE.md)**

---

## Key Capabilities & Highlights

- **Clean Architecture & SOLID Design**: Strict decoupling between presentation routers, application services, domain models, and external AI provider adapters (Ports & Adapters).
- **Multi-Format Enterprise Ingestion**: Cascading, non-blocking extraction engine supporting `.pdf`, `.docx`, `.doc`, `.txt`, `.md`, and `.csv` files with character-offset tracking and page-boundary preservation.
- **Asynchronous Decoupled Ingestion Pipeline (HTTP 202 Accepted)**:
  - Batch uploads immediately return `HTTP 202 Accepted` with a `job_id`, offloading extraction, chunking, and embedding to background worker tasks.
  - **Dual-Channel Telemetry**: Server-Sent Events (`GET /api/v1/documents/jobs/{job_id}/events`) stream real-time progress (`stage`, `percent`, `current_file`, `message`) with an automatic JSON snapshot fallback (`GET /api/v1/documents/jobs/{job_id}`).
  - **In-Memory Job Manager with TTL**: Thread-safe singleton managing active jobs with 30-minute sliding TTL eviction.
  - **Individual Failure Isolation**: Per-file `try...except` boundaries ensure that an unreadable or corrupt file does not abort the remaining valid documents in a batch.
- **3-Stage Hybrid Precision Funnel**:
  - **Stage 1 (Dual-Channel Hybrid Retrieval)**: Parallel Dense Vector Search (ChromaDB HNSW, 3072-d) and Lexical Keyword Search (BM25L) scoped strictly to `{user_id, session_id}` tenants.
  - **Stage 2A (Reciprocal Rank Fusion)**: Rank-based fusion using Cormack's RRF (`RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]`) to merge dense and lexical candidates into a balanced Top 15 pool.
  - **Stage 2B (Neural Cross-Attention Reranking)**: High-throughput Groq LLM cross-encoder scoring (0.0 to 10.0) with reasoning, pruning candidates to the Top 3 highest-precision chunks with automatic RRF fallback.
  - **Stage 3 (Grounded Synthesis)**: High-performance answer generation via Google Gemini Flash Lite or Groq with strict anti-hallucination prompt guardrails and source citations.
- **Asynchronous Concurrency & Scalability**:
  - SQLite Write-Ahead Logging (`WAL` mode) with `PRAGMA synchronous = NORMAL` and `busy_timeout = 5000` for lock-free concurrent reads during writes.
  - Multi-tenant composite B-Tree indexes on `documents`, `chunks`, and `question_history` tables.
  - Non-blocking ASGI event loop: compute-heavy document extraction and vector indexing are offloaded to background worker threads via `asyncio.to_thread()`.
  - In-memory singleton connection pooling and provider caching to reuse persistent HTTP/2 and TCP connections.
- **Enterprise Guardrails & Security**:
  - Sliding-window in-memory rate limiter on `POST /ask` (20 req / 60s per client IP) with automatic stale client memory eviction.
  - **Standardized Enterprise Logging Strategy**: Minimal, structured 9-column format (`Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata`), async `contextvars` request tracing, 17 standardized lifecycle phases, daily date-partitioned storage (`logs/YYYY-MM-DD/`) with automated 7-day retention, and zero-secret credential redaction.
  - SHA-256 content deduplication and strict file payload validation.
  - RFC 7807 problem details error shielding.
- **Rigorous Test Coverage**: 100% pass rate across **66 comprehensive unit, integration, and benchmark tests**.

---

## Core Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | 3.12+ | High-performance asynchronous execution runtime |
| **Package Manager** | `uv` (Astral) | Latest | Deterministic, high-speed virtual environments and lockfiles |
| **Web Framework** | FastAPI | 0.115+ | High-throughput ASGI web framework with OpenAPI validation |
| **ASGI Web Server** | Uvicorn | 0.34+ | Asynchronous server implementation |
| **Data Validation** | Pydantic | v2.10+ | Strict type validation, settings management, and DTO schemas |
| **Relational ORM** | SQLAlchemy | 2.0+ (aiosqlite) | Async relational persistence with SQLite WAL mode |
| **Vector Database** | ChromaDB | 0.6+ | Local and persistent HNSW dense vector indexing |
| **Document Parsing** | PyMuPDF / python-docx | 1.25+ / 1.1+ | Multi-tier extraction (PDF, DOCX, DOC, TXT, MD, CSV) |
| **AI Foundation Models** | Google Gemini / Groq | Latest | 3072-d embeddings, neural reranking, and grounded synthesis |
| **Evaluation Framework**| DeepEval | 2.5+ | Production RAG evaluation across recall, precision, and faithfulness |
| **Testing Suite** | Pytest / Pytest-Asyncio| 8.3+ | Automated async unit and integration verification |

---

## Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.12+**
- **uv** package manager ([Install uv](https://docs.astral.sh/uv/))

### 2. Environment Configuration
From the repository root, copy the environment template:
```bash
cp .env.example .env
```

Ensure active credentials and model choices are configured in `.env`:
```ini
# Embedding Configuration
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

# LLM Synthesis Configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-flash-lite-latest

# Reranker Configuration
RERANKER_TYPE=groq
GROQ_RERANKER_MODEL=openai/gpt-oss-120b
RETRIEVAL_TOP_K=15
FINAL_TOP_N=3

# External API Credentials
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# Distributed Observability (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://aws.api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Enterprise-RAG-Assistant
```

### 3. Run the Backend Service
From the `backend/` directory:
```bash
cd backend
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service will start on `http://localhost:8000`.

- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc API Documentation**: `http://localhost:8000/redoc`
- **Service Health Check**: `http://localhost:8000/api/v1/health`

### 4. Running the Automated Test Suite
Execute the complete test suite (62 unit and integration tests):
```bash
cd backend
uv run pytest
```

Expected output:
```text
======================= 62 passed, 8 warnings in 20.15s =======================
```

---

## Configuration Reference

The application is configured via environment variables mapped by Pydantic v2 `BaseSettings` in `app/core/config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `Mini Knowledge Assistant` | Name of the application service |
| `API_V1_STR` | `/api/v1` | URL prefix for REST API endpoints |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/knowledge_assistant.db` | SQLAlchemy async database connection URI |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | Persistent filesystem directory for ChromaDB vector indexes |
| `EMBEDDING_PROVIDER` | `gemini` | Active embedding provider (`gemini` or `mock`) |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-2` | Google Gemini dense vector model (3072 dimensions) |
| `LLM_PROVIDER` | `gemini` | Active LLM synthesis provider (`gemini`, `groq`, or `mock`) |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Google Gemini synthesis model |
| `RERANKER_TYPE` | `groq` | Active reranker provider (`groq`, `rrf`, or `mock`) |
| `GROQ_RERANKER_MODEL` | `openai/gpt-oss-120b` | High-throughput Groq model for neural cross-attention |
| `RETRIEVAL_TOP_K` | `15` | Number of candidate chunks retrieved per channel in Stage 1 |
| `FINAL_TOP_N` | `3` | Number of high-precision chunks passed to Stage 3 synthesis |
| `CHUNK_SIZE` | `800` | Target chunk character count |
| `CHUNK_OVERLAP` | `150` | Overlap character count between consecutive chunks |
| `RATE_LIMIT_REQUESTS` | `20` | Maximum allowed requests per client IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window duration in seconds |

---

## REST API Reference

All API routes require or support multi-tenant session scoping via the `X-Session-ID` and `X-User-ID` headers (or cookies/query parameters).

| Method | Endpoint | Description | Request Body / Params | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, active AI provider inspection, and index statistics | None | `HealthResponse` (200) |
| `POST` | `/api/v1/documents/upload` | **Synchronous Batch Ingestion**: Ingest up to 5 files (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.csv`, max 5MB each) | `multipart/form-data` (`files`) | `MultiFileUploadResponse` (201) |
| `POST` | `/api/v1/documents/upload?background=true` | **Asynchronous Decoupled Ingestion**: Enqueues indexing into background worker, returns job tracker URLs | `multipart/form-data` (`files`), Query: `background=true` | `AsyncIngestionJobResponse` (202 Accepted) |
| `GET` | `/api/v1/documents/jobs/{job_id}/events` | **Real-Time Telemetry Stream**: Live SSE event stream emitting file progress, stage, and completion status | Path: `job_id` | `text/event-stream` (SSE Events) |
| `GET` | `/api/v1/documents/jobs/{job_id}` | **Job Status Snapshot**: Poll current ingestion status, processed file count, and errors | Path: `job_id` | `JobStatusResponse` (200) |
| `POST` | `/api/v1/documents` | Create raw text/markdown document directly in knowledge base | JSON (`title`, `content`, `session_id`, `user_id`) | `DocumentResponse` (201) |
| `GET` | `/api/v1/documents` | List indexed documents for the active tenant session | Query: `limit`, `offset`, `session_id`, `user_id`, `as_list` | `DocumentListResponse` (200) |
| `GET` | `/api/v1/documents/{document_id}` | Inspect document details, total pages, and chunk breakdown | Path: `document_id` | `DocumentResponse` (200) |
| `DELETE` | `/api/v1/documents/{document_id}` | Cascade purge: deletes document record, chunk metadata, and ChromaDB vectors | Path: `document_id` | Status 204 No Content |
| `POST` | `/api/v1/ask` | 3-stage grounded question answering with reranking and citations | JSON (`question`, `session_id`, `user_id`) | `AskResponse` (200) |
| `GET` | `/api/v1/history` | Retrieve persistent Q&A and citation history for the active session and user | Query: `session_id`, `user_id`, `limit` | `QuestionHistoryResponse` (200) |
| `DELETE` | `/api/v1/history` | Clear Q&A conversation history for a given tenant session or user | Query: `session_id`, `user_id` | Status 204 No Content |

---

### Exercising the Main Endpoints via cURL

#### 1. System Health Check
```bash
curl -X GET "http://localhost:8000/api/v1/health" -H "Accept: application/json"
```

#### 2. Asynchronous Batch Document Upload (Background Mode)
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload?background=true" \
  -H "X-Session-ID: sess_demo_123" \
  -H "X-User-ID: usr_tester" \
  -F "files=@sample.pdf" \
  -F "files=@notes.docx"
```
*Returns HTTP 202 Accepted:*
```json
{
  "job_id": "job_a1b2c3d4",
  "session_id": "sess_demo_123",
  "user_id": "usr_tester",
  "status": "queued",
  "total_files": 2,
  "message": "Indexing job queued with 2 file(s).",
  "poll_url": "/api/v1/documents/jobs/job_a1b2c3d4",
  "stream_url": "/api/v1/documents/jobs/job_a1b2c3d4/events"
}
```

#### 3. Stream Real-Time Ingestion Progress (SSE)
```bash
curl -N "http://localhost:8000/api/v1/documents/jobs/job_a1b2c3d4/events"
```
*Emits live telemetry data:*
```text
data: {"job_id":"job_a1b2c3d4","status":"processing","stage":"chunking","percent":45,"current_file":"sample.pdf"}
...
data: {"job_id":"job_a1b2c3d4","status":"completed","stage":"completed","percent":100,"processed_files":2,"failed_files":0}
```

#### 4. Poll Ingestion Job Status (Fallback Snapshot)
```bash
curl -X GET "http://localhost:8000/api/v1/documents/jobs/job_a1b2c3d4"
```

#### 5. Ask Grounded Question with Citations
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess_demo_123" \
  -d '{
    "question": "What is the primary conclusion regarding quarterly revenue growth?",
    "session_id": "sess_demo_123"
  }'
```

#### 6. Retrieve Q&A Conversation & Citation History
```bash
curl -X GET "http://localhost:8000/api/v1/history?session_id=sess_demo_123" \
  -H "Accept: application/json"
```

#### 7. Clear Session Conversation History
```bash
curl -X DELETE "http://localhost:8000/api/v1/history?session_id=sess_demo_123"
```

---

## Standardized Enterprise Logging & Observability

The backend incorporates a unified, non-intrusive logging architecture:

- **Standard 9-Column Format**:
  ```text
  Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata
  ```
  *Example:*
  ```text
  2026-10-18 16:35:54.723 | INFO | RETRIEVAL | req_f293070c | usr_alice | sess_summit | app.rag.retriever | Hybrid retrieval completed | dense=1 bm25=1 merged=1 overlap=1 latency_ms=5.8
  ```
- **Context Injection**: Non-blocking `contextvars` automatically inject `request_id`, `user_id`, and `session_id` into every coroutine. Incoming `X-Request-ID` headers are propagated, or auto-generated `req_*` IDs are assigned.
- **17 Standard Phases**: `STARTUP`, `REQUEST`, `AUTH`, `VALIDATION`, `INGESTION`, `EXTRACTION`, `CHUNKING`, `EMBEDDING`, `RETRIEVAL`, `RERANKING`, `GENERATION`, `DATABASE`, `VECTOR_STORE`, `PROVIDER`, `RESPONSE`, `BACKGROUND`, `SHUTDOWN`.
- **Date-Partitioned Daily Storage with 7-Day Retention**:
  ```text
  logs/
  └── YYYY-MM-DD/
      ├── application.log   # General application lifecycle events (INFO+)
      ├── error.log         # Recoverable warnings and exceptions (WARNING+)
      └── access.log        # HTTP request metrics, latency, and status codes
  ```
  Older daily folders exceeding 7 days are automatically purged on startup and rollover.
- **Zero-Secret & Zero-Snippet Redaction**: `SecretMaskingFilter` scrubs API keys (Google, Groq, OpenAI, Anthropic, LangSmith) and bearer tokens. Raw document content is strictly barred from log statements.

---

## High-Level Directory Structure

```text
backend/
├── app/
│   ├── api/             # REST API routers, endpoints, and dependency injection factories
│   ├── core/            # Config, 9-column logging, exception handlers, rate limiting
│   ├── db/              # SQLAlchemy models, SQLite WAL database, and repositories
│   ├── ingestion/       # DocumentExtractor, file validation, and partitioned storage
│   ├── providers/       # Gemini, Groq, ChromaDB, and Mock adapter implementations
│   ├── rag/             # Chunker, HybridRetriever, 2-Stage Rerankers, and Synthesizer
│   ├── schemas/         # Pydantic v2 schemas for requests, responses, jobs, and citations
│   ├── services/        # High-level domain facades (DocumentService, ChatService, IngestionJobManager)
│   └── main.py          # ASGI application entry point and lifespan event handlers
│
├── data/                # Local persistent data directory (SQLite WAL and ChromaDB)
├── logs/                # Date-partitioned persistent log storage (logs/YYYY-MM-DD/)
├── tests/               # 66 automated unit, integration, and benchmark tests
├── streamlit_app.py     # Python Streamlit reference UI client
├── pyproject.toml       # Project dependencies and tool configurations
└── uv.lock              # Deterministic package lockfile
```

---

## Documentation Links

For deeper architectural inquiries and engineering blueprints:
- **[Backend Architecture Specification](ARCHITECTURE.md)**: Exhaustive technical analysis of request lifecycles, concurrency patterns, and the 100K+ concurrent user scaling roadmap.
- **[System Architecture (Root)](../ARCHITECTURE.md)**: Global platform architecture covering both React frontend and FastAPI backend.
- **[Project README (Root)](../README.md)**: Global repository overview, installation, and deployment instructions.
- **[AI Usage Disclosure (Root)](../AI_USAGE.md)**: Disclosure of AI tooling, design influence, and engineering ownership.
