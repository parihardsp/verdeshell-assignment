# Enterprise Backend Architecture & Engineering Design Specification

This document delivers an in-depth architectural and technical design specification for the **Mini Knowledge Assistant Backend Service**. It covers system architecture, component responsibilities, end-to-end request–response lifecycles, implemented concurrency and scalability patterns, and an enterprise scalability roadmap engineered to support **100,000+ concurrent active users**.

---

## 1. High-Level System Architecture

The backend is engineered around the principles of **Clean Architecture** and **Ports & Adapters (Hexagonal Architecture)**. Domain business logic, document processing pipelines, and retrieval algorithms remain strictly decoupled from HTTP transport frameworks, persistence engines, and external AI foundation model providers.

```text
+-------------------------------------------------------------------------------------------------------------------+
|                                                CLIENT ACCESS TIER                                                 |
|                                    React 18 + Vite Web Application / External API Clients                         |
|                             Headers: X-User-ID, X-Session-ID, Content-Type, Authorization                         |
+---------------------------------------------------------+---------------------------------------------------------+
                                                          | HTTP / REST (Port 8000)
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|                                        FASTAPI API GATEWAY & SECURITY TIER                                        |
|  ┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌─────────────────────────────────────────┐  |
|  │ Context Logging Middleware    │ │ SlidingWindowRateLimiter      │ │ RFC 7807 Error Shield                   │  |
|  │ - ContextVar [user:session]   │ │ - In-Memory Sliding Window    │ │ - Sanitized ProblemDetails JSON         │  |
|  │ - SecretMaskingFilter Redact  │ │ - 20 req / 60s per Client IP  │ │ - Stack Trace & Internal Key Shielding  │  |
|  │ - Microsecond Request Latency │ │ - Auto Stale Client Eviction  │ │ - Global Exception Handlers             │  |
|  └───────────────────────────────┘ └───────────────────────────────┘ └─────────────────────────────────────────┘  |
|                                                         |                                                         |
|                                                         v                                                         |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │                                      API ROUTER LAYER (app/api/routes.py)                                   │  |
|  │  - POST /api/v1/documents/upload  (Batch multi-format ingestion: sync 201 or async ?background=true 202)    │  |
|  │  - GET  /api/v1/documents/jobs/{job_id}/events (Real-time SSE telemetry stream)                             │  |
|  │  - GET  /api/v1/documents/jobs/{job_id}        (Job status snapshot / polling fallback)                     │  |
|  │  - POST /api/v1/documents         (Raw markdown/text article creation)                                      │  |
|  │  - GET  /api/v1/documents         (Paginated session document inventory)                                    │  |
|  │  - GET  /api/v1/documents/{id}    (Document inspection & chunk inspection)                                  │  |
|  │  - DELETE /api/v1/documents/{id}  (Cascade purge: relational metadata + ChromaDB vector index)              │  |
|  │  - POST /api/v1/ask               (3-Stage hybrid RAG pipeline with grounded citations)                     │  |
|  │  - GET  /api/v1/history            (Paginated Q&A conversation & citation history)                          │  |
|  │  - DELETE /api/v1/history         (Purge Q&A history for active session / user)                             │  |
|  │  - GET  /api/v1/health            (Service health, active AI provider inspect, index counts)                │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  |
+------------------------------------+-----------------------------------------------------+------------------------+
                                     |                                                     |
                                     v                                                     v
+---------------------------------------------------------+   +-----------------------------------------------------+
|              DOCUMENT ORCHESTRATION SERVICE             |   |                 CHAT SERVICE FACADE                 |
|             (app/services/document_service.py)          |   |          (app/services/chat_service.py)             |
|  - Batch ingestion validation & MIME / magic inspection |   |  - High-level Q&A orchestration facade               |
|  - Session-partitioned disk persistence                 |   |  - Session & user tenancy boundary enforcement      |
|  - Non-blocking extraction offload (asyncio.to_thread)  |   |  - Decoupled RAG pipeline stage execution           |
|  - Structural recursive text chunking                   |   |  - Source citation mapping & latency tracking       |
|  - SHA-256 deduplication & dual-store transactional sync|   |  - Exponential backoff retry on transient 429 limits|
+----------------------------+----------------------------+   +--------------------------+--------------------------+
                             |                                                           |
                             v                                                           v
+---------------------------------------------------------+   +-----------------------------------------------------+
|                DOCUMENT INGESTION ENGINE                |   |                DECOUPLED RAG PIPELINE               |
|                   (app/ingestion/)                      |   |                      (app/rag/)                     |
|  ┌───────────────────────────────────────────────────┐  |   |  ┌───────────────────────────────────────────────┐  |
|  │ FileValidationService                             │  |   |  │ Stage 1: HybridRetriever                      │  |
|  │ - 5MB per-file size enforcement                   │  |   |  │ - Dense Vector Search (ChromaDB HNSW, 3072-d) │  |
|  │ - File extension & magic byte gating              │  |   |  │ - Lexical Keyword Search (BM25L Index)        │  |
|  │ - Content SHA-256 checksum computation            │  |   |  └───────────────────────┬───────────────────────┘  |
|  └─────────────────────────┬─────────────────────────┘  |   |                          v                          |
|                            v                            |   |  ┌───────────────────────────────────────────────┐  |
|  ┌───────────────────────────────────────────────────┐  |   |  │ Stage 2A: ReciprocalRankReranker              │  |
|  │ SessionFileStorageService                         │  |   |  │ - Cormack RRF (k=60) Coarse Cut -> Top 15     │  |
|  │ - data/uploads/{user_id}_{session_id}_{date}/     │  |   |  └───────────────────────┬───────────────────────┘  |
|  └─────────────────────────┬─────────────────────────┘  |   |                          v                          |
|                            v                            |   |  ┌───────────────────────────────────────────────┐  |
|  ┌───────────────────────────────────────────────────┐  |   |  │ Stage 2B: GroqLLMReranker                     │  |
|  │ DocumentExtractor (Multi-Tier Parsing Engine)     │  |   |  │ - High-Throughput Neural Cross-Attention      │  |
|  │ - PDF: PyMuPDF4LLM -> Native Block Coordinate Sort│  |   |  │ - Joint Candidate Evaluation (0-10) -> Top 3  │  |
|  │ - DOCX: PyMuPDF4LLM -> python-docx -> MD Tables   │  |   |  └───────────────────────┬───────────────────────┘  |
|  │ - DOC: PyMuPDF -> macOS textutil -> OLE2 Stream   │  |   |                          v                          |
|  │ - TXT/MD: UTF-8 / CP1252 + Null-Byte Binary Guard │  |   |  ┌───────────────────────────────────────────────┐  |
|  │ - CSV: Structured Markdown Table Matrix           │  |   |  │ Stage 3: GroundedSynthesizer                  │  |
|  └───────────────────────────────────────────────────┘  |   |  │ - Anti-Hallucination Prompt Template          │  |
|                                                         |   |  │ - Gemini Flash Lite / Groq OSS LLM            │  |
|                                                         |   |  │ - Markdown Citation Footnotes & Confidence    │  |
|                                                         |   |  │ - LangSmith Distributed Tracing V2            │  |
|                                                         |   |  └───────────────────────────────────────────────┘  |
+----------------------------+----------------------------+   +--------------------------+--------------------------+
                             |                                                           |
                             +-----------------------------+-----------------------------+
                                                           |
                                                           v
+-------------------------------------------------------------------------------------------------------------------+
|                                         PERSISTENCE & PROVIDER ADAPTERS TIER                                      |
|  ┌───────────────────────────────────────────────────────┐ ┌────────────────────────────────────────────────────┐  |
|  │ Relational Persistence Layer (app/db/)                │ │ Vector Store Adapter (app/providers/vector_store.py)│  |
|  │ - SQLAlchemy 2.0 Async Engine (aiosqlite)             │ │ - ChromaVectorStore (Persistent ChromaDB HNSW)     │  |
|  │ - SQLite WAL Mode (PRAGMA journal_mode=WAL)           │ │ - Tenant Filtering: {$and: [{user_id}, {sess_id}]} │  |
|  │ - Multi-Tenant Composite B-Tree Indexes               │ │ - Self-Healing Vector Dimension Recovery           │  |
|  │ - Document, Chunk, and QuestionHistory Repositories   │ │ - Cosine Distance Metric Space                     │  |
|  └───────────────────────────────────────────────────────┘ └────────────────────────────────────────────────────┘  |
|  ┌───────────────────────────────────────────────────────┐ ┌────────────────────────────────────────────────────┐  |
|  │ Embedding Provider Port (app/providers/embeddings.py) │ │ LLM Synthesis Provider Port (app/providers/llm.py) │  |
|  │ - GeminiEmbeddingProvider (models/gemini-embedding-2) │ │ - LangChainLLMProvider (Google Gemini Flash Lite)  │  |
|  │ - MockEmbeddingProvider (Offline Deterministic 384-d) │ │ - GroqProvider (openai/gpt-oss-120b)               │  |
|  │ - Singleton Connection Pooling & TCP Socket Reuse     │ │ - MockLLMProvider (Offline Deterministic Grounding)│  |
|  └───────────────────────────────────────────────────────┘ └────────────────────────────────────────────────────┘  |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Core Components & Responsibilities Matrix

| Component | Module Path | Layer | Key Responsibilities | Design Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **API Router** | `app/api/routes.py` | Presentation / HTTP | Exposes REST endpoints (/health, /documents, /upload, /jobs, /ask), validates parameters, supports sync & async SSE streaming, and maps domain models to Pydantic v2 schemas. | Controller |
| **API Dependencies** | `app/api/dependencies.py` | Presentation / DI | Inversion-of-Control factory assembling database repositories, providers, and services. Implements singleton connection pooling. | Dependency Injection / Factory |
| **Rate Limiter** | `app/core/rate_limiter.py` | Infrastructure / Security | Enforces client IP sliding-window rate limits (20 req / 60s), generates RFC-standard `Retry-After` headers, and automatically purges stale clients. | Sliding Window / Token Bucket |
| **Enterprise Logging** | `app/core/logging.py` | Cross-Cutting | Injects `request_id`, `user_id`, and `session_id` into async `ContextVar` scopes; formats 9-column structured logs, manages daily date-partitioned log directories with automated 7-day retention, and scrubs secrets. | Interceptor / Filter Pipeline |
| **Error Handling** | `app/core/exceptions.py` | Presentation / Error | Intercepts domain and framework exceptions, returning RFC 7807 problem details while shielding internal stack traces. | Exception Shielding |
| **Document Service** | `app/services/document_service.py` | Application Service | Orchestrates batch file ingestion, storage persistence, extraction, chunking, and dual-store commit (SQLite + ChromaDB). | Facade / Orchestrator |
| **Job Manager** | `app/services/job_manager.py` | Application Service / Telemetry | In-memory asynchronous ingestion job registry with 30-min sliding TTL eviction and non-blocking `asyncio.Queue` fan-out for real-time SSE progress events. | Publisher-Subscriber / Observer |
| **Chat Service (RAG)** | `app/services/chat_service.py` | Application Service | High-level facade orchestrating multi-stage retrieval, reranking, synthesis, citation assembly, and audit logging. | Facade |
| **File Validator** | `app/ingestion/validator.py` | Domain / Ingestion | Validates batch counts, file sizes, MIME types, magic headers, and computes SHA-256 digests. | Specification |
| **File Storage** | `app/ingestion/storage.py` | Infrastructure / Storage | Manages sanitized disk storage partitioned by tenant ID, session ID, and ingestion date. | Gateway / DAO |
| **Document Extractor** | `app/ingestion/extractor.py` | Domain / Ingestion | Multi-tier cascading extraction engine parsing PDF, DOCX, DOC, TXT, MD, and CSV with page-boundary metadata preservation. | Strategy / Chain of Responsibility |
| **Recursive Chunker** | `app/rag/chunker.py` | Domain / RAG | Splits extracted text into fixed-size chunks (800 chars, 150 overlap) respecting paragraph boundaries and tracking exact character offsets and page indices. | Splitter Strategy |
| **Hybrid Retriever** | `app/rag/retriever.py` | Domain / RAG | Executes parallel dense vector search (ChromaDB HNSW) and sparse lexical search (BM25L) with compound tenant scoping. | Composite / Strategy |
| **Reranker Engine** | `app/rag/reranker.py` | Domain / RAG | Two-stage reranking: Cormack RRF (k=60) coarse filtering followed by Groq LLM cross-attention precision scoring (0–10). | Pipeline / Fallback Strategy |
| **Grounded Synthesizer** | `app/rag/synthesizer.py` | Domain / RAG | Injects reranked chunks into anti-hallucination prompt templates, executes LLM generation, formats citations, and emits LangSmith traces. | Template Method |
| **Database Repositories** | `app/db/repositories.py` | Persistence | Encapsulates async SQLAlchemy 2.0 CRUD operations for `Document`, `Chunk`, and `QuestionHistory` entities. | Repository Pattern |
| **Chroma Vector Store** | `app/providers/vector_store.py` | Persistence / Adapter | Manages persistent ChromaDB vector collections, tenant metadata filtering, and self-healing vector dimension recovery. | Adapter |
| **AI Providers** | `app/providers/` | External Ports / Adapters | Pluggable implementations for Google Gemini, Groq, and Mock foundation models and embeddings. | Ports & Adapters (Hexagonal) |

---

## 3. Project Directory & Module Topology

```text
backend/
├── app/
│   ├── api/                         # Presentation Layer (FastAPI Routers & DI)
│   │   ├── __init__.py              # Exports consolidated api_router
│   │   ├── dependencies.py          # Dependency injection container & provider cache singletons
│   │   └── routes.py                # REST API controllers (/health, /documents, /upload, /jobs, /ask, /history)
│   │
│   ├── core/                        # System Infrastructure & Cross-Cutting Concerns
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic v2 BaseSettings configuration schema
│   │   ├── exceptions.py            # RFC 7807 exception taxonomy & exception handlers
│   │   ├── logging.py               # Standard 9-column logging, contextvars, daily rotation & retention
│   │   └── rate_limiter.py          # Sliding window rate limiter with auto stale client eviction
│   │
│   ├── db/                          # Relational Persistence Layer (SQLAlchemy 2.0 Async)
│   │   ├── __init__.py              # Package exports (engine, session, repositories)
│   │   ├── database.py              # Engine lifecycle, WAL PRAGMA config, composite indexes
│   │   ├── models.py                # Declarative models: Document, Chunk, QuestionHistory
│   │   └── repositories.py          # BaseRepository and domain-specific repository classes
│   │
│   ├── ingestion/                   # Document Extraction & Storage Pipeline
│   │   ├── __init__.py
│   │   ├── extractor.py             # DocumentExtractor (PyMuPDF, python-docx, textutil, OLE2)
│   │   ├── storage.py               # SessionFileStorageService (tenant/date disk layout)
│   │   └── validator.py             # FileValidationService (SHA-256 hashing, size & MIME gates)
│   │
│   ├── providers/                   # External AI Adapters (Ports & Adapters)
│   │   ├── __init__.py
│   │   ├── embeddings.py            # GeminiEmbeddingProvider (3072-d) & MockEmbeddingProvider
│   │   ├── llm.py                   # LangChainLLMProvider, GroqProvider, MockLLMProvider
│   │   └── vector_store.py          # ChromaVectorStore (Persistent HNSW Cosine Index)
│   │
│   ├── rag/                         # RAG Pipeline Domain Logic
│   │   ├── __init__.py
│   │   ├── chunker.py               # RecursiveTextChunker (character & page-aware windowing)
│   │   ├── embedder.py              # Batch vectorization orchestration helper
│   │   ├── evaluation.py            # DeepEval 2-Pillar RAG benchmark evaluation suite
│   │   ├── pipeline.py              # Decoupled RAGPipeline stage runner
│   │   ├── prompts/
│   │   │   └── synthesis_prompts.py # Anti-hallucination prompts & Pydantic response schemas
│   │   ├── reranker.py              # ReciprocalRankReranker (RRF) & GroqLLMReranker
│   │   ├── retriever.py             # VectorRetriever, BM25Retriever, and HybridRetriever
│   │   └── synthesizer.py           # GroundedSynthesizer with LangSmith v2 tracing
│   │
│   ├── schemas/                     # Data Transfer Objects (Pydantic v2 Models)
│   │   ├── __init__.py              # Unified schema exports
│   │   ├── common.py                # HealthResponse, ErrorResponse, PaginationMeta
│   │   ├── document.py              # DocumentCreate, DocumentResponse, MultiFileUploadResponse, AsyncIngestionJobResponse, JobStatusResponse
│   │   └── rag.py                   # AskRequest, AskResponse, QuestionHistoryResponse, SourceCitation
│   │
│   ├── services/                    # Application Domain Services
│   │   ├── __init__.py
│   │   ├── chat_service.py          # RAGService high-level Q&A facade
│   │   ├── document_service.py      # DocumentService ingestion & lifecycle manager
│   │   ├── job_manager.py           # IngestionJobManager (async background job registry & SSE event bus)
│   │   └── storage_service.py       # FileStorageService compatibility alias
│   │
│   └── main.py                      # ASGI entrypoint, lifespan startup/shutdown, CORS
│
├── data/                            # Persistent Application State
│   ├── chroma_db/                   # ChromaDB binary vector indexes (HNSW)
│   ├── uploads/                     # Partitioned raw document storage
│   └── knowledge_assistant.db       # SQLite WAL database file
│
├── logs/                            # Daily date-partitioned log directories (logs/YYYY-MM-DD/)
│   └── YYYY-MM-DD/                  # application.log, error.log, access.log (7-day retention)
│
├── tests/                           # Automated Verification Suite (67 pytest tests)
│   ├── conftest.py                  # Pytest async fixtures & mock provider stubs
│   ├── test_api_endpoints.py        # REST API endpoint integration tests
│   ├── test_async_ingestion.py      # Decoupled background ingestion & SSE streaming tests
│   ├── test_chunker.py              # RecursiveTextChunker windowing and offset tests
│   ├── test_deepeval_framework.py   # DeepEval 2-Pillar RAG evaluation metrics
│   ├── test_document_extractor.py   # Multi-format document parsing & extraction tests
│   ├── test_file_validator.py       # File size, magic bytes, and count validation tests
│   ├── test_multi_upload_session.py # Multi-file session persistence tests
│   ├── test_multi_user_isolation.py # Tenant isolation across SQLite & ChromaDB
│   ├── test_rag_pipeline_stages.py  # Hybrid retriever, RRF, Groq reranker, and synthesis tests
│   ├── test_rate_limiter.py         # Sliding window rate limiter tests
│   └── test_retry_and_secret_masking.py # Exponential backoff retry & secret masking tests
│
├── streamlit_app.py                 # Python Streamlit reference UI client
├── pyproject.toml                   # Project metadata and locked dependency specifications
└── uv.lock                          # Deterministic dependency resolution lockfile
```

---

## 4. End-to-End Data Flow & Request–Response Lifecycles

### 4.1 Write Path: Document Ingestion Lifecycle (`POST /api/v1/documents/upload`)

Document extraction, chunking, and vector embedding computation happen strictly at **Upload Time**. To eliminate long-lived HTTP socket vulnerability (which causes proxy, reverse proxy, ALB, and mobile client dropouts during 10–30s ingestion workloads), document ingestion is **decoupled** into an asynchronous background pipeline with dual-channel telemetry (real-time Server-Sent Events with JSON polling fallback).

```text
CLIENT                     GATEWAY                VALIDATOR / STORAGE         JOB MANAGER / SSE           BACKGROUND INGESTION WORKER
  │                           │                            │                          │                            │
  │ 1. POST /upload (files)   │                            │                          │                            │
  │    ?background=true       │                            │                          │                            │
  │──────────────────────────>│                            │                          │                            │
  │                           │ 2. ContextVar Scope Inject │                          │                            │
  │                           │    & Sliding Window Check  │                          │                            │
  │                           │───────────────────────────>│                          │                            │
  │                           │                            │ 3. Validate Constraints  │                            │
  │                           │                            │    (Count <= 5, <= 5MB)  │                            │
  │                           │                            │ 4. Fast Disk Persist     │                            │
  │                           │                            │    data/uploads/{tenant} │                            │
  │                           │                            │─────────────────────────>│                            │
  │                           │                            │                          │ 5. Create IngestionJob     │
  │                           │                            │                          │    (30-min TTL Eviction)   │
  │                           │                            │                          │ 6. asyncio.create_task()   │
  │                           │                            │                          │───────────────────────────>│
  │                           │<──────────────────────────────────────────────────────│                            │
  │ 7. Return HTTP 202        │                            │                          │                            │
  │    (job_id, stream_url)   │                            │                          │                            │
  │<──────────────────────────│                            │                          │                            │
  │                                                                                   │                            │
  │ 8. Connect SSE Stream: GET /documents/jobs/{job_id}/events                        │                            │
  │──────────────────────────────────────────────────────────────────────────────────>│                            │
  │                                                                                   │                            │
  │                                                                                   │ 9. Sequential File Loop:   │
  │                                                                                   │    - PyMuPDF Extraction    │
  │                                                                                   │    - SHA-256 Deduplication │
  │                                                                                   │    - 800-char Chunking     │
  │                                                                                   │    - Gemini 3072-d Vectors │
  │                                                                                   │    - ChromaDB HNSW Index   │
  │                                                                                   │    - SQLite DB Record      │
  │                                                                                   │<───────────────────────────│
  │ 10. SSE Progress Broadcast: { stage, percent, current_file, message }             │                            │
  │<──────────────────────────────────────────────────────────────────────────────────│                            │
  │                                                                                   │                            │
  │ 11. SSE Terminal Event: { status: 'completed' | 'completed_with_errors' }         │                            │
  │<──────────────────────────────────────────────────────────────────────────────────│                            │
```

#### Architectural Trade-Offs & V1 Design Rationale (Option A)

The V1 asynchronous ingestion implementation incorporates four key strategic trade-offs:

1. **In-Memory Job Manager with TTL (Option A) vs. Distributed Broker (Celery / RabbitMQ / Redis Streams)**:
   - *Design Decision*: Uses `IngestionJobManager` singleton in `app/services/job_manager.py` with sliding 30-minute TTL cleanup and in-memory `asyncio.Queue` fan-out.
   - *Rationale*: Speed, simplicity, and zero-dependency local execution. The entire application runs out-of-the-box without requiring Docker containers or external broker daemon processes.
   - *Future Evolution*: In-memory state is bound to the single process. In the distributed 100K+ concurrent architecture (Section 5.2), this will evolve to Redis Streams / Celery + S3.

2. **Sequential File Processing vs. Unconstrained Parallelism**:
   - *Design Decision*: Multi-file upload batches are processed sequentially (`1..N`) inside the background task rather than launching parallel worker coroutines with `asyncio.gather()`.
   - *Rationale*: Eliminates Google Gemini embedding API quota exhaustion (`429 Resource Exhausted`) and avoids SQLite write-lock contention (`PRAGMA busy_timeout=5000`) and ChromaDB index concurrency locks.
   - *Future Evolution*: In production clusters with PostgreSQL and dedicated enterprise embedding endpoints, concurrency can be scaled via `asyncio.Semaphore(2)` or distributed Celery task pools.

3. **Dual Telemetry Endpoints: Real-Time SSE Stream + JSON Polling Fallback**:
   - *Design Decision*: Dual endpoints: `GET /documents/jobs/{job_id}/events` (SSE stream) + `GET /documents/jobs/{job_id}` (JSON polling snapshot).
   - *Rationale*: Maximum resilience. SSE gives sub-second live progress telemetry without client polling overhead, while the snapshot endpoint guarantees continuous fallback operation in restrictive enterprise network topologies (e.g. corporate proxy buffering, mobile background network throttling).

4. **Individual Document Failure Isolation with Granular Error Reporting**:
   - *Design Decision*: Each file in a batch is isolated in a `try...except` boundary during extraction, chunking, and indexing.
   - *Rationale*: Prevents a single unreadable, scanned, or corrupted file in a batch upload from aborting the entire ingestion batch. Successful documents are indexed and immediately available for grounded Q&A, while failed files are clearly tagged with individual error messages and returned with status `completed_with_errors`.

#### Step-by-Step Processing Phases

1. **Phase 1: Ingestion Gateway & Security Gating**
   - The client sends a multipart request containing up to 5 files accompanied by `X-Session-ID` and `X-User-ID` headers.
   - `ContextLoggingMiddleware` captures or assigns the session and tenant IDs, injecting them into the current thread/task's `ContextVar` scope.
   - `FileValidationService` verifies that the file count is `<= 5`, individual file sizes are `<= 5MB`, and file extensions match supported types (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.csv`).
   - A cryptographic SHA-256 digest of the raw byte stream is calculated.

2. **Phase 2: Partitioned File System Persistence**
   - Raw files are committed to disk under an isolated directory structure: `data/uploads/{user_id}_{session_id}_{YYYY-MM-DD}/{sanitized_filename}`.
   - This ensures traceability, prevents cross-tenant file collisions, and simplifies archival workflows.

3. **Phase 3: Non-Blocking Multi-Tier Document Extraction**
   - Extraction is offloaded to a worker thread pool via `await asyncio.to_thread()` to prevent blocking FastAPI's ASGI event loop during compute-intensive PDF rendering or DOCX parsing.
   - The cascading extraction hierarchy selects the optimal parsing strategy:
     - **PDF**: PyMuPDF4LLM renders structured Markdown page chunks; falls back to native block coordinate sorting.
     - **DOCX**: PyMuPDF4LLM page chunks; falls back to `python-docx` for structured paragraph and table extraction.
     - **DOC**: PyMuPDF direct read; falls back to macOS `/usr/bin/textutil` conversion, then to OLE2 stream extraction.
     - **TXT / MD**: Multi-encoding decoder (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`) with a Git-standard null-byte check (`\x00` in the first 4KB) to prevent binary archive corruption.
     - **CSV**: Converted into structured Markdown table syntax (`| Col 1 | Col 2 |`).

4. **Phase 4: Structural & Page-Aware Recursive Chunking**
   - `RecursiveTextChunker` splits extracted content into windows of 800 characters with a 150-character overlap.
   - Chunker boundaries prioritize double line breaks (`\n\n`), single line breaks (`\n`), sentence terminators (`. `, `? `, `! `), and spaces (` `).
   - Exact character offsets (`char_start`, `char_end`) and original page numbers (`page_number`) are stamped onto every chunk.

5. **Phase 5: Dense Vector Computation & Dual-Store Commit**
   - `GeminiEmbeddingProvider` vectorizes chunks in batches (up to 100 per call) into 3072-dimensional dense float arrays.
   - An atomic SQLite transaction writes:
     - A row to `documents` containing metadata, page count, and SHA-256 hash.
     - Multiple rows to `chunks` containing chunk text, offsets, and page coordinates.
   - The embeddings are committed to ChromaDB with compound metadata filters:
     ```json
     {
       "document_id": "doc_...",
       "session_id": "sess_...",
       "user_id": "usr_...",
       "page_number": 1,
       "chunk_index": 0
     }
     ```

---

### 4.2 Read Path: Grounded Question-Answering Lifecycle (`POST /api/v1/ask`)

The retrieval pipeline implements a **3-Stage Hybrid Precision Funnel** engineered to eliminate hallucinations while preserving low latency.

```text
CLIENT                     GATEWAY                HYBRID RETRIEVER              2-STAGE RERANKER             LLM SYNTHESIZER
  │                           │                            │                            │                            │
  │ 1. POST /ask (question)   │                            │                            │                            │
  │──────────────────────────>│                            │                            │                            │
  │                           │ 2. Sliding Window Check    │                            │                            │
  │                           │    (20 req / 60s per IP)   │                            │                            │
  │                           │ 3. ContextVar Scope Inject │                            │                            │
  │                           │───────────────────────────>│                            │                            │
  │                           │                            │ 4. STAGE 1: DUAL RETRIEVAL │                            │
  │                           │                            │    ├─ ChromaDB HNSW (Dense)│                            │
  │                           │                            │    └─ BM25L Index (Lexical)│                            │
  │                           │                            │    Scoping: {user, session}│                            │
  │                           │                            │───────────────────────────>│                            │
  │                           │                            │                            │ 5. STAGE 2A: RRF FUSION    │
  │                           │                            │                            │    RRF Score (k=60)        │
  │                           │                            │                            │    Merge -> Top 15 Pool    │
  │                           │                            │                            │ 6. STAGE 2B: NEURAL RERANK │
  │                           │                            │                            │    Groq Joint Attention    │
  │                           │                            │                            │    Score (0-10) -> Top 3   │
  │                           │                            │                            │───────────────────────────>│
  │                           │                            │                            │                            │ 7. STAGE 3: SYNTHESIS
  │                           │                            │                            │                            │    - Prompt Template Guard
  │                           │                            │                            │                            │    - Strict Anti-Hallucination
  │                           │                            │                            │                            │    - Gemini Flash Lite / Groq
  │                           │                            │                            │                            │    - Map Grounded Citations
  │                           │                            │                            │                            │ 8. Audit History Log
  │                           │                            │                            │<───────────────────────────│
  │                           │<────────────────────────────────────────────────────────│                            │
  │ 9. Return AskResponse     │                            │                            │                            │
  │    (Answer, Sources, Conf)│                            │                            │                            │
  │<──────────────────────────│                            │                            │                            │
```

#### Step-by-Step Processing Phases

1. **Phase 1: Ingestion & Rate Limit Validation**
   - The incoming request is intercepted by `SlidingWindowRateLimiter`. If the client IP has exceeded 20 requests within the past 60 seconds, an HTTP 429 response is returned immediately with a computed `Retry-After` header.
   - `X-Session-ID` and `X-User-ID` are validated and bound to the thread-local async context.

2. **Phase 2: Stage 1 Dual-Channel Hybrid Retrieval**
   - The query is dispatched concurrently to two retrieval channels scoped strictly to `{user_id, session_id}`:
     - **Dense Vector Search**: The query is embedded via `GeminiEmbeddingProvider` and queried against ChromaDB's HNSW index using Cosine Similarity, retrieving the Top `K` candidates (default: 15).
     - **Lexical Keyword Search**: The query is tokenized and evaluated against an in-memory BM25L inverted index built from all chunks belonging to the active session, retrieving the Top `K` candidates.

3. **Phase 3: Stage 2A Reciprocal Rank Fusion (Coarse Cut)**
   - The dense and lexical ranked lists are merged using Cormack's Reciprocal Rank Fusion formula:
     ```text
     RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]
     ```
   - This algorithm combines rank positions rather than raw scores, preventing dense vector score distributions from dominating keyword matches. The candidate set is pruned to the Top 15 entries.

4. **Phase 4: Stage 2B Neural Precision Reranking**
   - The Top 15 candidate chunks and the user's question are formatted into a single prompt dispatched to a high-throughput Groq LLM (`openai/gpt-oss-120b`).
   - The model evaluates cross-attention relevance between each candidate chunk and the query, returning a JSON array of scores (0.0 to 10.0) accompanied by brief reasoning tokens.
   - The candidate pool is sorted by reranker score and pruned to the Top `N` (default: 3) highest-precision chunks. If the reranker call fails or times out, the pipeline falls back gracefully to the RRF rank order.

5. **Phase 5: Stage 3 Grounded LLM Synthesis & Hallucination Guardrails**
   - The Top 3 chunks are formatted into the anti-hallucination prompt template.
   - Each chunk is labeled with document title, chunk index, and page number.
   - System instructions explicitly forbid generating information outside the supplied context:
     ```text
     "Base your answer ONLY on the provided context. If the context does not contain sufficient
      information to answer the question, state clearly: 'I cannot answer this question based on the provided context.'"
     ```
   - Google Gemini Flash Lite generates the final synthesized response with embedded Markdown citation anchors (`[^1]`).
   - The synthesis execution is traced end-to-end via LangSmith V2.

6. **Phase 6: Citation Mapping & Latency Audit Logging**
   - Each cited chunk is mapped to a `SourceCitation` object containing:
     - `document_id`, `title`, `chunk_index`, `page_number`
     - `snippet` (context snippet)
     - `similarity_score` (cosine similarity)
     - `reranker_score` (cross-attention score)
     - `retrieval_channel` (`vector`, `keyword`, or `hybrid`)
   - An audit record is persisted asynchronously to the `question_history` table recording the question, answer, latency in milliseconds, source metadata, and client IP.

---

## 5. Engineering Decisions & Design Patterns

### 5.1 Clean Architecture & Hexagonal Ports & Adapters
The application core contains zero references to HTTP frameworks, database ORMs, or third-party AI SDKs. All external integrations are mediated via abstract base classes (Ports) and concrete implementations (Adapters):
- `BaseEmbeddingProvider` -> `GeminiEmbeddingProvider`, `MockEmbeddingProvider`
- `BaseLLMProvider` -> `LangChainLLMProvider`, `GroqProvider`, `MockLLMProvider`
- `BaseVectorStore` -> `ChromaVectorStore`

This guarantees that external providers can be swapped (e.g., migrating from Google Gemini to AWS Bedrock or self-hosted vLLM) by modifying environment variables without altering domain logic.

### 5.2 Facade Pattern
Both `DocumentService` and `RAGService` act as high-level facades that hide complex multi-stage subsystems behind simple, unified method contracts (`ingest_documents`, `answer_question`). Callers at the API layer remain decoupled from chunking formulas, embedding batching, and reranking heuristics.

### 5.3 Strategy & Chain of Responsibility Patterns
Document extraction relies on `DocumentExtractor`, which implements a cascading fallback strategy. If modern structured parsing fails, control flows down the chain to lower-level coordinate parsers, system utilities, and binary stream extractors, ensuring maximum file compatibility.

### 5.4 Self-Healing Vector Dimension Recovery
When changing embedding providers (e.g., switching from 384-dimensional offline mock embeddings to 3072-dimensional Gemini embeddings), ChromaDB collections with mismatched dimensions typically crash. `ChromaVectorStore` detects dimension mismatch errors on startup, logs a recovery notice, wipes the stale collection, and automatically re-indexes existing chunks from SQLite, restoring consistency without manual intervention.

### 5.5 Comprehensive System-Wide Architectural Trade-Offs & Design Rationale

The table below catalogs every strategic architectural trade-off taken across the entire backend architecture, contrasting the V1 design decision against alternatives, explaining the engineering rationale, and identifying the production evolution path:

| Domain / Subsystem | Architectural Decision (V1) | Alternative Evaluated | Engineering Rationale | Known Trade-Off / Limitation | Production Evolution Path (100K+ Scale) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion Concurrency** | **In-Memory Job Manager with TTL (Option A)** | Distributed Broker (Celery + Redis / RabbitMQ) | Zero external infrastructure dependencies; single-command developer boot (`uv run uvicorn`); sub-millisecond task dispatch. | Job state is bound to ASGI memory and lost on container restart; limited to single-node deployments. | Migrate to Redis Streams or Celery with worker fleets and state stored in PostgreSQL (Section 7.4). |
| **Batch Concurrency** | **Sequential Document Processing (`1..N`)** | Unconstrained Parallelism (`asyncio.gather`) | Protects against Google Gemini embedding API `429 Too Many Requests` limits; prevents SQLite write locks and ChromaDB HNSW contention. | Batch processing latency scales linearly with file count (~2s per document). | Concurrency throttled via distributed `asyncio.Semaphore(2)` or worker task pools with token-bucket rate limiters. |
| **Ingestion Telemetry** | **Dual-Channel: Server-Sent Events (SSE) + JSON Polling Fallback** | Full-Duplex WebSockets | SSE operates over standard HTTP/2, auto-reconnects, and requires no protocol upgrade; JSON polling guarantees operation behind corporate proxies. | SSE is unidirectional (client cannot send control commands over the stream without a separate POST request). | WebSockets for collaborative multi-user sessions; SSE retained for unidirectional telemetry. |
| **Relational Storage** | **SQLite WAL Mode (`busy_timeout=5000`)** | External Managed RDBMS (PostgreSQL / MySQL) | Embedded zero-network-latency queries; ACID transactional safety; zero cloud hosting costs for local evaluation and edge PoCs. | Single active writer constraint across processes; cannot scale horizontally across multi-pod clusters. | Aurora PostgreSQL Multi-AZ with PgBouncer connection pooling and monthly table range partitioning (Section 7.2). |
| **Vector Database** | **Embedded ChromaDB HNSW (Persistent)** | Distributed Vector Engine (Qdrant / Milvus) | Embedded zero-network persistence in `data/chroma_db/`; native cosine distance metric; zero external daemon dependencies. | High RAM consumption for massive corpora (> 100K chunks); lacks distributed multi-node sharding. | Distributed Qdrant cluster with scalar quantization (SQ8) and tenant-based shard keys (Section 7.3). |
| **Hybrid Retrieval** | **Dual-Channel Dense (3072-d) + BM25L with RRF (k=60)** | Dense-Only or Elasticsearch Sparse-Only | Eliminates semantic blindness to exact part numbers, error codes, and alphanumeric IDs while maintaining semantic generalization. | In-memory BM25 index is constructed from session SQLite chunks per query; memory bounded per session. | Persistent Elasticsearch / OpenSearch inverted index or pgvector hybrid with precomputed inverted lists. |
| **Neural Reranker** | **2-Stage: RRF Coarse Cut (Top 15) -> Groq LLM Cross-Attention (Top 3)** | Heavy Cross-Encoder Model (BGE / Cohere) | Groq LPU delivers deep cross-attention reasoning tokens in < 300ms without GPU hardware overhead; automatic RRF fallback on failure. | External API call dependency; network hop adds 200-400ms to total Q&A response lifecycle. | Self-hosted cross-encoder (e.g. `bge-reranker-large`) on private vLLM GPU inference clusters (Section 7.6). |
| **Rate Limiter** | **In-Memory Sliding Window per IP with Lazy Eviction** | Distributed Redis Token Bucket | Zero latency in-process check (< 0.1ms); automatic eviction of IPs idle for > 5 minutes prevents memory leaks. | State is local to process memory; cannot enforce shared rate limits across multiple container instances. | Distributed Redis token bucket using atomic Lua scripts and tiered tenant API key limits. |
| **Enterprise Logging** | **Structured 9-Column Async Pipe Format (`ContextVar`)** | Arbitrary Unstructured JSON Log Blobs | High human scannability during live terminal debugging; zero secret exposure via `SecretMaskingFilter`; daily date-partitioned storage. | Requires structured log collector (Fluentbit / Logstash) to parse pipe delimiters for Datadog or CloudWatch. | Dual-mode logging handler: human-readable ANSI pipe format for local dev; structured JSON format for K8s stdout. |

---

## 6. Implemented Scalability & Engineering Best Practices

The backend incorporates multiple production-grade engineering optimizations designed to maximize throughput, minimize latency, and maintain strict data integrity under concurrent load.

### 6.1 SQLite Write-Ahead Logging (WAL) Concurrency Architecture

Traditional SQLite instances lock the entire database file during writes, blocking concurrent readers. The backend enables Write-Ahead Logging (`WAL` mode) during application startup:

```text
CONCURRENT CLIENT READERS                     SQLITE ENGINE                      PERSISTENCE ON DISK
  ┌───────────────────────┐                                                     ┌──────────────────────┐
  │ Reader Task 1 (Async) │──┐                                                  │ Base Database File   │
  └───────────────────────┘  │                                                  │ knowledge_...db      │
  ┌───────────────────────┐  │    Lock-Free Concurrent Reads                    └──────────▲───────────┘
  │ Reader Task 2 (Async) │──┼─────────────────────────────────┐                           │ Checkpoint
  └───────────────────────┘  │                                 │                           │ Sync
  ┌───────────────────────┐  │                                 ▼                           │
  │ Reader Task 3 (Async) │──┘                    ┌─────────────────────────┐   ┌──────────┴───────────┐
  └───────────────────────┘                       │ Shared Memory Index     │   │ Write-Ahead Log      │
                                                  │ knowledge_...db-shm     │   │ knowledge_...db-wal  │
  ┌───────────────────────┐                       └─────────────────────────┘   └──────────▲───────────┘
  │ Writer Task (Async)   │────────────────────────────────────────────────────────────────┘
  └───────────────────────┘                        Serialized Append-Only Write
```

- **Lock-Free Concurrent Reads**: Readers access snapshots via the shared memory index (`.shm`) and base file without acquiring database-level locks, allowing unlimited concurrent read operations.
- **Serialized Append-Only Writes**: Incoming write transactions append records to the write-ahead log (`.wal`) sequentially without blocking active readers.
- **Optimized PRAGMA Configuration**:
  - `PRAGMA journal_mode=WAL;`: Activates Write-Ahead Logging.
  - `PRAGMA synchronous=NORMAL;`: Reduces filesystem sync operations while preserving durability.
  - `PRAGMA busy_timeout=5000;`: Waits up to 5,000ms on lock contention before throwing an error, eliminating transient lock failures under burst traffic.

### 6.2 Multi-Tenant Composite B-Tree Indexing

To prevent full table scans and guarantee sub-millisecond query execution as relational tables grow to millions of rows, composite B-Tree indexes are defined across all key entities:

```text
Index Lookup Path:
Query: WHERE session_id = 'sess_123' AND user_id = 'usr_456' ORDER BY created_at DESC

[ B-Tree Root: ix_documents_tenant_session ]
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
['sess_000'..'sess_199']   ['sess_200'..'sess_999']
       │
       ▼
['sess_123' / 'usr_456'] ──> Direct Leaf Node Match (Index Seek O(log N))
                             Zero Full Table Scans
```

- `ix_documents_tenant_session`: Composite index on `documents (session_id, user_id, created_at)`.
- `ix_chunks_tenant_session`: Composite index on `chunks (session_id, user_id)`.
- `ix_chunks_doc_index`: Composite index on `chunks (document_id, chunk_index)`.
- `ix_query_history_tenant_session`: Composite index on `question_history (session_id, user_id, created_at)`.

### 6.3 Non-Blocking Event Loop Architecture (`asyncio.to_thread`)

FastAPI operates on a single-threaded asynchronous event loop. Executing CPU-intensive operations (such as PyMuPDF document rendering, DOCX XML parsing, or ChromaDB vector distance math) on the main thread blocks all concurrent HTTP requests.

The backend offloads all CPU-heavy parsing and vector store interactions to Python's background `ThreadPoolExecutor`:

```text
INCOMING REQUESTS                ASGI ASYNC EVENT LOOP               BACKGROUND THREAD POOL
        │                                  │                                    │
        │ 1. POST /documents/upload        │                                    │
        ├─────────────────────────────────>│                                    │
        │                                  │ 2. await asyncio.to_thread(        │
        │                                  │        extractor.extract, path)    │
        │                                  │───────────────────────────────────>│ (CPU Worker Thread)
        │                                  │                                    │ - PyMuPDF Parsing
        │ 3. GET /health (Concurrent)      │ [Event Loop Immediately Free]      │ - python-docx XML Parse
        ├─────────────────────────────────>│                                    │ - Table Formatting
        │<─────────────────────────────────│ 4. Fast JSON Response (< 2ms)      │
        │                                  │                                    │
        │                                  │ 5. Extraction Complete             │
        │                                  │<───────────────────────────────────│
        │                                  │                                    │
        │                                  │ 6. Resume Ingestion Pipeline       │
        │<─────────────────────────────────│                                    │
```

- Document extraction (`DocumentExtractor.extract_from_path`) runs inside `asyncio.to_thread()`.
- Vector store synchronization runs inside `asyncio.to_thread()`.
- The ASGI event loop remains completely unblocked, allowing concurrent health checks, reads, and queries to process with sub-millisecond dispatch times.

### 6.4 Singleton Connection Pooling & Provider Caching

Instantiating new HTTP client sessions, SSL/TLS contexts, and SDK provider objects on every request introduces significant TCP handshake and cryptographic negotiation overhead.

The dependency injection layer implements an in-memory singleton cache for AI providers keyed by configuration signature:

```text
Incoming Request ──> get_chat_service() ──> Provider Factory
                                                   │
                                                   ▼
                                       ┌───────────────────────┐
                                       │ Provider Cache Map    │
                                       │ Key: (provider, model)│
                                       └───────────┬───────────┘
                                                   │
                       ┌───────────────────────────┴───────────────────────────┐
                       ▼ Cache Hit                                             ▼ Cache Miss
             [Return Cached Singleton]                                [Instantiate New Provider]
             - Reuses persistent TCP sockets                          - Negotiates TLS handshake
             - Preserves HTTP/2 connection pool                       - Stores in cache dictionary
             - Zero instantiation latency                             - Returns instance
```

- `_embedding_cache`: Reuses `GeminiEmbeddingProvider` instances and HTTP/2 connection pools.
- `_llm_cache`: Reuses `LangChainLLMProvider` and `GroqProvider` client sessions.
- `_reranker_cache`: Reuses `GroqLLMReranker` instances.

### 6.5 Sliding-Window Rate Limiting with Stale Client Eviction

The in-memory rate limiter tracks request timestamps per client IP. To prevent memory leaks caused by high-cardinality, transient client IPs, the rate limiter incorporates automatic cleanup logic:

- Every incoming request purges timestamps older than 60 seconds for the calling IP.
- Whenever total tracked client entries exceed 500, a sweep removes all IP records whose most recent activity is older than the rate limit window.
- Guarantees bounded memory consumption regardless of traffic duration.

### 6.6 Zero-Secret Redaction Pipeline & RFC 7807 Shielding

To satisfy SOC 2 and ISO 27001 enterprise security standards:
- **`SecretMaskingFilter`**: Automatically intercepts log records and redacts Anthropic (`sk-ant-***`), Groq (`gsk_***`), Google Gemini (`AIzaSy***`), OpenAI (`sk-***`), and HTTP Authorization Bearer tokens.
- **RFC 7807 Error Shielding**: Uncaught database exceptions, foreign key violations, and third-party API error traces are sanitized into standardized problem details responses, preventing internal schema or credential leakage to end users.

### 6.7 Standardized Enterprise Logging & Observability Architecture

The backend standardizes on a minimal, actionable, and structured logging strategy designed for enterprise traceability:

```text
Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata
```

*Production Log Example:*
```text
2026-10-18 16:35:54.723 | INFO | RETRIEVAL | req_f293070c | usr_alice | sess_summit | app.rag.retriever | Hybrid retrieval completed | dense=1 bm25=1 merged=1 overlap=1 latency_ms=5.8
```

1. **Async Context Variable Injection**: Uses non-blocking Python `contextvars` (`request_id_context`, `user_context`, `session_context`, `phase_context`). The ASGI middleware extracts or generates a unique `request_id` (`req_*`), binding it alongside tenant IDs to all asynchronous coroutines.
2. **Lifecycle Phase Categorization**: Every event is explicitly classified into one of 17 standard phases (`STARTUP`, `REQUEST`, `AUTH`, `VALIDATION`, `INGESTION`, `EXTRACTION`, `CHUNKING`, `EMBEDDING`, `RETRIEVAL`, `RERANKING`, `GENERATION`, `DATABASE`, `VECTOR_STORE`, `PROVIDER`, `RESPONSE`, `BACKGROUND`, `SHUTDOWN`).
3. **Daily Date-Partitioned Storage & 7-Day Retention**:
   - Log files are persisted under `logs/YYYY-MM-DD/` partitioned into `application.log` (INFO+), `error.log` (WARNING+), and `access.log` (HTTP access metrics).
   - Thread-safe `DailyDatedFileHandler` manages rollover at midnight UTC.
   - Built-in retention cleaner automatically removes log directories older than 7 days on server startup and rollover.
4. **Information Security**: Zero snippet leaks in reranking/evaluation and comprehensive regex masking across all providers.

---

## 7. Future Scalability Roadmap: Scaling to 100,000+ Concurrent Users

To transition the backend from a single-node deployment to an enterprise distributed architecture capable of handling **100,000+ concurrent active users** (generating an estimated 25,000 to 50,000 QPS at peak), the following architectural evolution plan is established.

### 7.1 C100K Target Architecture Blueprint

```text
                                       [ CLIENT TRAFFIC: 100,000+ CONCURRENT USERS ]
                                                           │
                                                           ▼
                                            ┌─────────────────────────────┐
                                            │ Cloudflare Global Anycast   │
                                            │ - DDoS Mitigation & WAF     │
                                            │ - Edge Static Asset Caching │
                                            │ - TLS 1.3 Termination       │
                                            └──────────────┬──────────────┘
                                                           │
                                                           ▼
                                            ┌─────────────────────────────┐
                                            │ AWS Application Load        │
                                            │ Balancer (ALB) / NGINX Plus │
                                            │ - Least Outstanding Requests│
                                            │ - Sticky Session Routing    │
                                            └──────────────┬──────────────┘
                                                           │
                      ┌────────────────────────────────────┼────────────────────────────────────┐
                      ▼                                    ▼                                    ▼
       ┌─────────────────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────────┐
       │ Kubernetes Cluster (EKS/GKE)│      │ Kubernetes Cluster (EKS/GKE)│      │ Kubernetes Cluster (EKS/GKE)│
       │ FastAPI Pod Replica 1       │      │ FastAPI Pod Replica 2       │      │ FastAPI Pod Replica N (HPA) │
       │ (Stateless ASGI Workers)    │      │ (Stateless ASGI Workers)    │      │ (Autoscale: 50 to 500 Pods) │
       └──────────────┬──────────────┘      └──────────────┬──────────────┘      └──────────────┬──────────────┘
                      │                                    │                                    │
                      └────────────────────────────────────┼────────────────────────────────────┘
                                                           │
       ┌───────────────────────────────────────────────────┼───────────────────────────────────────────────────┐
       ▼                                                   ▼                                                   ▼
┌──────────────────────────────┐        ┌──────────────────────────────────────┐        ┌──────────────────────────────┐
│ REDIS CLUSTER TIER           │        │ ASYNCHRONOUS INGESTION FLEET         │        │ DISTRIBUTED VECTOR CLUSTER   │
│ - Redis Sentinel / Cluster   │        │ - RabbitMQ / Apache Kafka Message Bus│        │ - Qdrant / Milvus Cluster    │
│ - Semantic Vector Query Cache│        │ - Celery / Temporal Worker Fleet     │        │ - Multi-Node HNSW Shards     │
│   (Cosine Sim >= 0.98 -> <5ms)│       │ - Direct S3 Presigned Uploads        │        │ - Raft Replicated Sharding   │
│ - Distributed Rate Limiter   │        │ - Background OCR & Ingestion Pipeline│        │ - 100M+ Vectors Capacity     │
│ - Session Metadata Cache     │        │ - SSE / WebSocket Progress Pushes    │        │ - Sub-10ms ANN Query Latency │
└──────────────┬───────────────┘        └──────────────────┬───────────────────┘        └──────────────┬───────────────┘
               │                                           │                                           │
               └───────────────────────────────────────────┼───────────────────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────────────┐
                                        │ MANAGED RELATIONAL DATA TIER         │
                                        │ - AWS Aurora PostgreSQL Multi-AZ     │
                                        │ - PgBouncer Connection Pooler        │
                                        │ - 1 Primary Writer + 5 Read Replicas │
                                        │ - Declarative Tenant Partitioning    │
                                        └──────────────────┬───────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────────────┐
                                        │ ENTERPRISE OBSERVABILITY & SRE       │
                                        │ - OpenTelemetry Distributed Traces   │
                                        │ - Prometheus Metrics & Grafana Alert │
                                        │ - Datadog / ELK Centralized Logs     │
                                        │ - LangSmith LLM Quality Monitoring   │
                                        └──────────────────────────────────────┘
```

---

### 7.2 Detailed Tier-by-Tier Scaling Strategy

#### 1. Compute Tier: Stateless Kubernetes Pods & Autoscaling
- **Horizontal Pod Autoscaler (HPA)**: Deploy FastAPI containers on AWS EKS or Google GKE. Configure HPA to scale pod replicas dynamically between 50 and 500 instances based on CPU utilization (> 65%) and HTTP request latency (P95 > 250ms).
- **Stateless Design**: All local state (in-memory rate limiters, local SQLite files) is moved to external distributed tiers. Containers remain 100% stateless, allowing instant termination, replacement, and zero-downtime rolling upgrades.
- **Worker Configuration**: Run Uvicorn workers behind a Gunicorn master process configured for containerized memory limits:
  ```bash
  gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000 --max-requests 10000 --max-requests-jitter 1000
  ```

#### 2. Relational Persistence: Migration from SQLite WAL to Aurora PostgreSQL
- **Managed PostgreSQL (AWS Aurora Multi-AZ)**:
  - Migrate from SQLite to PostgreSQL with an automated primary writer and horizontally scaled read replicas.
  - Read queries (listing documents, inspecting history) are routed automatically to read replicas.
  - Write queries (document creation, status updates) execute on the primary writer.
- **PgBouncer Connection Pooling**: Deploy PgBouncer sidecars or dedicated pooling clusters in transaction-pooling mode, supporting up to 50,000 incoming client connections while maintaining a lean pool of 200 server connections to PostgreSQL.
- **Declarative Partitioning**: Partition the `documents`, `chunks`, and `question_history` tables by `tenant_id` and range-partition by `created_at` (monthly partitions), ensuring index lookups remain bounded regardless of historical table growth.

#### 3. Vector Tier: Transition from Embedded ChromaDB to Distributed Qdrant / Milvus
- Replace single-node embedded ChromaDB with a distributed, clustered vector engine such as **Qdrant** or **Milvus Distributed**:
  - **Horizontal Sharding**: Partition vector indexes across multiple dedicated worker nodes using tenant hash keys (`user_id`, `session_id`).
  - **In-Memory HNSW with Disk Quantization**: Leverage scalar quantization (SQ8) or product quantization (PQ) to reduce RAM usage by up to 75% while maintaining > 98% recall precision.
  - **Raft Consensus & High Availability**: Multi-node replication ensures that node failures do not interrupt retrieval operations or cause data loss.

#### 4. Ingestion Pipeline: Asynchronous Decoupling via Queues & Object Storage
- **Direct S3 Presigned Uploads**: Eliminate backend proxying of large files. The client requests a presigned S3/GCS PUT URL from FastAPI and uploads files directly to cloud object storage.
- **Event-Driven Workers**: An S3 upload event publishes a message to a **RabbitMQ** or **Apache Kafka** topic.
- **Asynchronous Worker Fleet**: A fleet of dedicated **Celery** or **Temporal** workers consumes the ingestion queue:
  - Performs OCR and structured extraction asynchronously without impacting web traffic.
  - Computes embeddings via batched inference pipelines.
  - Updates document status in PostgreSQL and ChromaDB/Qdrant.
- **Real-Time Client Updates**: The frontend receives real-time extraction progress updates via WebSockets or Server-Sent Events (SSE).

#### 5. Multi-Tier Distributed Caching Strategy
- **Semantic Vector Query Caching (Redis Cluster)**:
  - Before invoking the 3-stage retrieval pipeline, compute the incoming question's vector embedding and query a Redis Semantic Cache.
  - If a cached query within the same session has a cosine similarity score `>= 0.98`, return the cached response immediately.
  - Reduces LLM provider latency from ~1,200ms to **under 5ms** and slashes foundation model API costs by up to 60%.
- **Session & Document Metadata Cache**: Cache session validation checks and document inventory listings in Redis with a 5-minute TTL, invalidating on document write/delete events.
- **Distributed Rate Limiting**: Replace the single-node in-memory sliding window rate limiter with a distributed Redis token-bucket implementation using atomic Lua scripts.

#### 6. Private Dedicated LLM Inference Tier
- For enterprise deployments requiring strict data residency, zero third-party data egress, or predictable cost ceilings:
  - Deploy self-hosted open-weights LLMs (such as Llama 3 70B or Mistral Large) on private GPU clusters using **vLLM** or **NVIDIA Triton Inference Server**.
  - Enable continuous batching, PagedAttention, and FP8 quantization to achieve up to 4x higher token generation throughput.
  - Implement dynamic fallback routing: prioritize private vLLM clusters during normal loads, bursting to Google Gemini or Groq APIs during traffic spikes.

#### 7. Enterprise Observability & Site Reliability Engineering (SRE)
- **OpenTelemetry Instrumentation**: Trace every request end-to-end from the React frontend, through the API gateway, hybrid retrievers, vector databases, and LLM providers.
- **Prometheus Metrics**: Expose golden operational signals:
  - `http_requests_total` (by route, status code, tenant)
  - `http_request_duration_seconds` (P50, P90, P99 histograms)
  - `rag_retrieval_latency_seconds` (vector vs BM25)
  - `llm_token_generation_seconds` & token counts
  - `active_db_connections` & connection pool wait times
- **Automated Health Probes & Circuit Breakers**:
  - Kubernetes liveness (`/health/live`) and readiness (`/health/ready`) probes.
  - Resilience4j / Tenacity circuit breakers protecting against external provider degradation, automatically isolating failing models and routing queries to fallback providers.

---

## 8. Summary of Engineering Artifacts

| Document | Primary Audience | Scope |
| :--- | :--- | :--- |
| **`backend/README.md`** | Developers, DevOps, QA | High-level system overview, capabilities, setup instructions, configuration reference, and quickstart commands. |
| **`backend/ARCHITECTURE.md`** (This Document) | Enterprise Architects, Tech Leads, Senior Engineers | Comprehensive technical specification, lifecycles, data flow diagrams, implemented concurrency patterns, and 100K+ scalability roadmap. |
| **`ARCHITECTURE.md`** (Root) | Cross-Functional Stakeholders | Full-stack platform architecture bridging React 18 client and FastAPI backend services. |
