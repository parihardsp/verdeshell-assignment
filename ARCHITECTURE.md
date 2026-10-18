# Architecture & System Design Document

This document details the architectural decisions, component contracts, async concurrency model, end-to-end request-response flows, operational failure modes, security guardrails, and a detailed scalability roadmap (10K, 100K, 1M concurrent requests) for the **Enterprise Grounded Knowledge Assistant**.

---

## 1. High-Level System Architecture

The Mini Knowledge Assistant is structured according to **Clean Architecture / Ports & Adapters (Hexagonal Architecture)**. Core domain business logic, data models, and RAG pipeline stages are strictly decoupled from external web frameworks, database engines, and third-party AI provider APIs.

```text
+-------------------------------------------------------------------------------------------------------+
|                                             CLIENT LAYER                                              |
|                                    React 18 + Vite (Port 5173)                                        |
|   ┌────────────────────────────────────────────────────────┐ ┌────────────────────────────────────┐   |
|   │               Desktop Split Workspace                  │ │     Mobile Viewports (iOS & Android) │   |
|   │ - 2-Column: Center Workspace + Pinned Session Sidebar  │ │ - Slide-Over Fixed Drawer (88vw)   │   |
|   │ - Expanded Header, Full Tab & Status Labels            │ │ - Top ✕ Close & Backdrop Dismissal │   |
|   │ - In-Table Real-Time Ingestion Status (Zero Overlays)  │ │ - 16px Inputs (No Auto-Zoom)       │   |
|   │   (Optimistic Queued, Live Amber Beacon, Micro-Bar)    │ │ - Table Horizontal Scroll (620px)  │   |
|   └────────────────────────────────────────────────────────┘ └────────────────────────────────────┘   |
|       - Dual Tabs: 💬 AI Chatbot (Dynamic Thought Indicator) | 📁 Document Hub (Dual Ingestion)       |
|       - Dual Telemetry Channels: Server-Sent Events (SSE) Stream + Fallback JSON Polling Snapshot     |
|       - Tab-Scoped sessionStorage Multi-Tenant Isolation (X-User-ID, X-Session-ID)                    |
|       - In-App Confirmation Modal System (Zero native browser window.confirm popups)                  |
|       - Optional Reference Client: Streamlit Web UI (backend/streamlit_app.py, Port 8501)             |
+---------------------------------------------------+---------------------------------------------------+
                                                    | HTTP REST API (port 8000)
                                                    | Headers: X-User-ID, X-Session-ID
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                      FASTAPI API GATEWAY / CORE                                       |
|   +-------------------------------+  +-------------------------------+  +-------------------------+   |
|   | Enterprise Logging Middleware |  | Sliding Window Rate Limiter   |  | RFC 7807 Error Shield   |   |
|   | (ContextVar [req:user:sess])  |  | (20 req / 60s per Client IP)  |  | (Zero-Secret Masking)   |   |
|   +---------------+---------------+  +---------------+---------------+  +------------+------------+   |
|                   |                                  |                               |                |
|                   v                                  v                               v                |
|   +-----------------------------------------------------------------------------------------------+   |
|   |                                 API ROUTER (app/api/routes.py)                                |   |
|   |   - POST /api/v1/documents/upload?background=true ──► HTTP 202 Accepted {job_id}              |   |
|   |   - GET  /api/v1/documents/jobs/{job_id}/events    ──► text/event-stream (SSE live telemetry) |   |
|   |   - GET  /api/v1/documents/jobs/{job_id}           ──► JSON snapshot (polling fallback)       |   |
|   |   - POST /api/v1/documents     - GET  /api/v1/documents     - DELETE /api/v1/documents/{id}   |   |
|   |   - GET  /api/v1/documents/{id}- POST /api/v1/ask           - GET  /api/v1/health             |   |
|   +-----------------------+---------------------------------------+-------------------------------+   |
+---------------------------|---------------------------------------|-----------------------------------+
                            |                                       |
                            v                                       v
+-------------------------------------------+   +-------------------------------------------------------+
|             DOCUMENT SERVICE              |   |                     CHAT SERVICE                      |
|      (app/services/document_service)      |   |          (app/services/chat_service - RAGService)     |
| - Gating: Batch size <= 5, size <= 5MB    |   | - High-level Q&A facade with tenant scoping           |
| - Storage: Partitioned disk persistence   |   | - Wires decoupled RAGPipeline orchestrator            |
| - Multi-Tier Document Extractor:          |   | - Latency audit logging & source citation formatting  |
|   * PDF: PyMuPDF4LLM + native blocks      |   | - Rate limit exponential backoff retry                |
|   * DOCX: PyMuPDF4LLM + python-docx       |   +---------------------------+---------------------------+
|   * DOC: PyMuPDF + macOS textutil / OLE2  |                               |
|   * TXT/MD: UTF-8 + null-byte binary guard|                               v
|   * CSV: Direct Markdown table conversion |   +-------------------------------------------------------+
| - Chunker: Page-preserving character runs |   |                 DECOUPLED RAG PIPELINE                |
| - Deduplication: SHA-256 tenant check     |   |                       (app/rag/)                      |
+---------------------+---------------------+   |  Stage 1: HybridRetriever                             |
                      |                         |    ├─ Dense Vector Search (ChromaDB HNSW, 3072-d)     |
                      |                         |    └─ Lexical Search (BM25L Inverted Index)           |
                      |                         |  Stage 2A: ReciprocalRankReranker (RRF k=60 -> Top 15)|
                      |                         |  Stage 2B: GroqLLMReranker (Joint Cross-Attention)    |
                      |                         |  Stage 3: GroundedSynthesizer (Gemini Flash Lite)     |
                      |                         |  Benchmarking: DeepEval 2-Pillar Metric Suite         |
                      |                         +---------------------------+---------------------------+
                      v                                                     |
+-------------------------------------------+                               v
|              DATA PERSISTENCE             |   +-------------------------------------------------------+
|                 (app/db/)                 |   |                   PROVIDER ADAPTERS                   |
| - SQLite Async Engine (aiosqlite)         |   |                   (app/providers/)                    |
| - SQLAlchemy 2.0 Declarative Models:      |   | - Embedding Providers:                                |
|   * Document, Chunk, QuestionHistory      |   |   * Google Gemini (models/gemini-embedding-2, 3072-d) |
| - Persistent Repositories:                |   |   * Deterministic Offline Mock (384-d)                |
|   * Document, Chunk, QuestionHistory repos|   | - LLM Generation Providers:                           |
| - ChromaDB Persistent Vector Store        |   |   * Google Gemini (gemini-flash-lite-latest)          |
|   (HNSW Cosine under data/chroma_db/)     |   |   * Groq LLM (openai/gpt-oss-120b)                    |
|   Compound filters: {user_id, session_id} |   |   * Deterministic Offline Grounded Mock               |
+-------------------------------------------+   +-------------------------------------------------------+
```

---

## 2. End-to-End Request-Response Flows

### 2.1 Complete Document Ingestion Flow (Write Path)

Document extraction, chunking, and vector embedding computation happen strictly at **Upload Time** to guarantee sub-second Q&A generation latencies. To prevent HTTP connection timeouts across reverse proxies, load balancers, and mobile networks, document ingestion is **decoupled** from the HTTP request-response cycle into an asynchronous background pipeline with real-time SSE progress streaming.

```text
Sequence of Asynchronous Decoupled Ingestion:

[Client / React FileDropzone]
        │
        │ 1. POST /api/v1/documents/upload?background=true (Multipart: files <= 5, session_id, user_id)
        ▼
[API Gateway: Context Logging Middleware]
        │
        │ 2. Extract or auto-assign session_id & user_id; bind to ContextVar logging scope
        ▼
[FileValidationService (app/ingestion/validator.py)]
        │
        │ 3. Fast validation: file count (<= 5), size (<= 5MB), and allowed extensions (.pdf, .docx, .doc, .txt, .md, .csv)
        ▼
[SessionFileStorageService (app/ingestion/storage.py)]
        │
        │ 4. Fast disk persistence: data/uploads/{user_id}_{session_id}_{date}/{filename}
        ▼
[IngestionJobManager (app/services/job_manager.py)]
        │
        │ 5. Register in-memory job (status='queued', total_files=N) with 30-min sliding TTL
        │ 6. Dispatch background task via asyncio.create_task() with dedicated DB session
        ▼
[Immediate Client Response]
        ◀── HTTP 202 Accepted: AsyncIngestionJobResponse (job_id, status='queued', poll_url, stream_url)

────────────────────────────────────────────────────────────────────────────────────────────────
BACKGROUND PROCESSING & REAL-TIME DUAL-CHANNEL TELEMETRY:

[Background Ingestion Worker]
        │
        │ 7. Connects to SSE Channel: GET /api/v1/documents/jobs/{job_id}/events
        │    (Fallback: Client polls GET /api/v1/documents/jobs/{job_id} every 1.2s)
        │
        │ 8. Sequential File Processing Loop (1..N):
        │    ├─ Stage: EXTRACTION ──► PyMuPDF / python-docx / macOS textutil offloaded via asyncio.to_thread()
        │    │                        Computes SHA-256 hash for tenant deduplication check
        │    ├─ Stage: CHUNKING   ──► RecursiveTextChunker creates 800-char passages with page awareness
        │    ├─ Stage: EMBEDDING  ──► GeminiEmbeddingProvider computes 3072-d vectors (with 429 exponential backoff)
        │    ├─ Stage: VECTOR_DB  ──► ChromaDB upserts chunk vectors with compound tenant metadata
        │    └─ Stage: PERSIST    ──► SQLAlchemy commits Chunk records and marks Document status='indexed'
        │
        │ 9. Real-Time Telemetry Broadcast:
        │    Emits SSE events: { job_id, stage, percent, current_file, processed_files, total_files, message }
        │
        │ 10. Job Finalization:
        │     - If all files succeed ──► status='completed' (percent=100)
        │     - If partial failure   ──► status='completed_with_errors' (isolated failure reporting)
        │     - Closes SSE stream cleanly; job entry expires automatically after 30-min TTL
```

#### Architectural Trade-Offs & V1 Design Rationale (Option A)

Due to V1 architectural goals and operational constraints, the ingestion subsystem implements the following strategic engineering trade-offs:

1. **Option A: In-Memory Job Manager with TTL vs. Distributed Task Queue (Redis / Celery)**
   - *Design Choice*: The backend utilizes `IngestionJobManager`, an in-process thread-safe singleton maintaining job state in memory with 30-minute sliding TTL eviction and `asyncio.Queue` event broadcasting.
   - *Rationale*: Speed and zero-dependency operational simplicity. It allows developers and evaluators to run the entire backend with a single `uv run uvicorn` command without mandating external Redis, RabbitMQ, or Celery worker daemons.
   - *Trade-off & Evolution Path*: In-memory state does not survive ASGI process restarts and is bound to a single container instance. When scaling to multi-pod horizontal clusters (Section 3.2), this layer seamlessly transitions to Redis Streams / Celery workers with task state in PostgreSQL / Redis.

2. **Sequential File Processing vs. Unconstrained Parallelism**
   - *Design Choice*: Multi-file upload batches are processed sequentially (`1..N`) inside the background task rather than launching parallel worker coroutines with `asyncio.gather()`.
   - *Rationale*: Protects against Google Gemini embedding API quota exhaustion (`429 Too Many Requests / Resource Exhausted`) and eliminates SQLite write-lock contention (`PRAGMA busy_timeout=5000`) and ChromaDB index concurrency locks.
   - *Trade-off & Evolution Path*: Slightly higher batch processing duration in exchange for 100% indexing reliability. In enterprise production with PostgreSQL and dedicated enterprise embedding endpoints, concurrency can be scaled via `asyncio.Semaphore(2)` or distributed Celery task pools.

3. **Dual Telemetry Endpoints: Real-Time SSE Stream + JSON Polling Fallback**
   - *Design Choice*: The backend provides both `GET /api/v1/documents/jobs/{job_id}/events` (`text/event-stream`) and `GET /api/v1/documents/jobs/{job_id}` (JSON snapshot).
   - *Rationale*: Server-Sent Events (SSE) deliver real-time sub-second visual feedback (stage, percentage, current filename) to the frontend with minimal network overhead. However, certain corporate proxies, VPNs, and mobile browsers buffer or terminate long-lived HTTP streams; the snapshot polling endpoint ensures an automatic, zero-friction fallback mechanism.

4. **Individual Document Failure Isolation with Granular Error Reporting**
   - *Design Choice*: Each file in a batch is isolated in a `try...except` boundary during extraction, chunking, and indexing.
   - *Rationale*: In multi-file batch uploads (e.g. 3 files), an unreadable, password-protected, or scanned image-only PDF does not fail the entire batch. Valid documents are indexed and immediately available for grounded Q&A, while the failed file is clearly reported with its individual filename and error cause under status `completed_with_errors`.

5. **In-Table Real-Time Ingestion Status (Zero Disruptive Overlays)**
   - *Design Choice*: Rather than occupying screen space with separate non-collapsed banners or blocking modal dialogs, real-time background indexing telemetry connects directly to the **Session Knowledge Inventory** table.
   - *User Experience*: When files are dropped, optimistic table rows appear immediately (`○ Queued`). The active file transitions dynamically to `● Processing {percent}%` with a pulsing amber beacon, stage moniker (`Stage: EXTRACTION / CHUNKING / EMBEDDING / VECTOR_DB`), and a micro-progress bar. Failed items display red `● Failed / Skipped` with specific error tooltips, while completed documents seamlessly switch to `● Indexed` with chunk metrics.

---

### 2.2 Complete Grounded Q&A Flow (Read Path)

```text
Sequence of Question Answering:

[Client / React ChatView]
        │
        │ 1. POST /api/v1/ask { "question": "...", "session_id": "...", "user_id": "..." }
        ▼
[RateLimiterMiddleware (app/core/rate_limiter.py)]
        │
        │ 2. Check client IP sliding window rate limit (20 req / 60s)
        │    - If exceeded: Return HTTP 429 Too Many Requests with Retry-After: <sec> header
        ▼
[RAGPipeline Orchestrator (app/rag/pipeline.py)]
        │
        ├─► [STAGE 1: DUAL-CHANNEL HYBRID RETRIEVAL] (HybridRetriever)
        │       Executes two retrieval channels in parallel via asyncio.gather:
        │
        │       CHANNEL 1: Dense Vector Retrieval (VectorRetriever)
        │       - Embed question via GeminiEmbeddingProvider (3072-dim)
        │       - Query ChromaDB with compound filter: {"$and": [{"user_id": user_id}, {"session_id": session_id}]}
        │       - Fetch Top 15 semantic candidate chunks (cosine threshold >= 0.20)
        │
        │       CHANNEL 2: Lexical Keyword Retrieval (BM25Retriever)
        │       - Query corpus chunks matching user_id & session_id
        │       - Tokenize query with stopword filtering
        │       - Compute BM25L scoring across chunk vocabulary
        │       - Fetch Top 15 exact-term candidate chunks
        │
        │       Candidate Pool Merger:
        │       - Merges candidates by chunk_id, annotating channel origin ("dense", "bm25", or "hybrid")
        │       - Yields up to 30 unique candidate chunks
        │
        ├─► [STAGE 2A: RECIPROCAL RANK FUSION (RRF COARSE CUT)] (ReciprocalRankReranker)
        │       - Computes rank positions for Dense and Lexical channels
        │       - Calculates RRF Score: RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]
        │       - Normalizes score distributions into a Top 15 candidate pool
        │
        ├─► [STAGE 2B: GROQ NEURAL PRECISION RERANKER] (GroqLLMReranker)
        │       - Calls Groq LLM (openai/gpt-oss-120b) with zero-temperature cross-attention prompt
        │       - Groq evaluates joint query-passage relevance, rating each chunk 0.0 to 10.0 with rationale
        │       - Selects the final Top 3 precision chunks for synthesis
        │       - Graceful Fallback: Seamlessly falls back to RRF Top 3 if Groq times out or rate limits
        │
        ├─► [STAGE 3: GROUNDED SYNTHESIS] (GroundedSynthesizer)
        │       - Builds prompt templates enforcing Pydantic structured output schema (GroundedSynthesisOutput):
        │           • answer: Direct, concise answer without conversational filler
        │           • has_sufficient_context: Strict boolean commitment
        │           • cited_chunk_indices: Exact candidate chunk numbers used as evidence
        │           • confidence_score: Grounded confidence rating (0.0 to 1.0)
        │       - Zero-Hallucination Gating: If has_sufficient_context=False, returns polite refusal
        │       - Attaches SourceCitation objects exclusively for cited chunks with page numbers
        │       - LangSmith automatically traces full prompt, output schemas, and token usage
        │
        └─► [AUDIT LOGGING] (QuestionHistoryRepository)
                - Asynchronously records question, answer, sources, latency, and client IP in SQLite
        ▼
[Client Response]
        ◀── HTTP 200 OK: AskResponse (answer, sources with page numbers & scores, confidence, latency_ms)
```

---

## 3. End-to-End System Architectural Trade-Offs & Design Rationale

The table below catalogs every strategic engineering trade-off taken across the client and server platform, contrasting the V1 design decision against alternatives, explaining the engineering rationale, and identifying the production evolution path:

| Domain / Subsystem | Architectural Decision (V1) | Alternative Evaluated | Engineering Rationale | Known Trade-Off / Limitation | Production Evolution Path (100K+ Scale) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion Concurrency** | **In-Memory Job Manager with TTL (Option A)** | Distributed Broker (Celery + Redis / RabbitMQ) | Zero external infrastructure dependencies; single-command developer boot (`uv run uvicorn`); sub-millisecond task dispatch. | Job state is bound to ASGI memory and lost on container restart; limited to single-node deployments. | Migrate to Redis Streams or Celery with worker fleets and state stored in PostgreSQL (Section 4.2). |
| **Batch Concurrency** | **Sequential Document Processing (`1..N`)** | Unconstrained Parallelism (`asyncio.gather`) | Protects against Google Gemini embedding API `429 Too Many Requests` limits; prevents SQLite write locks and ChromaDB HNSW contention. | Batch processing latency scales linearly with file count (~2s per document). | Concurrency throttled via distributed `asyncio.Semaphore(2)` or worker task pools with token-bucket rate limiters. |
| **Ingestion Telemetry** | **Dual-Channel: Server-Sent Events (SSE) + JSON Polling Fallback** | Full-Duplex WebSockets | SSE operates over standard HTTP/2, auto-reconnects, and requires no protocol upgrade; JSON polling guarantees operation behind corporate proxies. | SSE is unidirectional (client cannot send control commands over the stream without a separate POST request). | WebSockets for collaborative multi-user sessions; SSE retained for unidirectional telemetry. |
| **Client Ingestion UX** | **In-Table Inline Status Tracking (Zero Overlays)** | Blocking Modal / Non-Collapsed Banner Overlay | Eliminates UI disruption. Users can browse existing documents, read previews, or search while files index in the background. | Table rows must manage optimistic temporary state before backend persistence is confirmed. | Optimistic state manager with WebSocket reconnect and persistent job history across page reloads. |
| **Relational Storage** | **SQLite WAL Mode (`busy_timeout=5000`)** | External Managed RDBMS (PostgreSQL / MySQL) | Embedded zero-network-latency queries; ACID transactional safety; zero cloud hosting costs for local evaluation and edge PoCs. | Single active writer constraint across processes; cannot scale horizontally across multi-pod clusters. | Aurora PostgreSQL Multi-AZ with PgBouncer connection pooling and monthly table range partitioning (Section 4.2). |
| **Vector Database** | **Embedded ChromaDB HNSW (Persistent)** | Distributed Vector Engine (Qdrant / Milvus) | Embedded zero-network persistence in `data/chroma_db/`; native cosine distance metric; zero external daemon dependencies. | High RAM consumption for massive corpora (> 100K chunks); lacks distributed multi-node sharding. | Distributed Qdrant cluster with scalar quantization (SQ8) and tenant-based shard keys (Section 4.2). |
| **Hybrid Retrieval** | **Dual-Channel Dense (3072-d) + BM25L with RRF (k=60)** | Dense-Only or Elasticsearch Sparse-Only | Eliminates semantic blindness to exact part numbers, error codes, and alphanumeric IDs while maintaining semantic generalization. | In-memory BM25 index is constructed from session SQLite chunks per query; memory bounded per session. | Persistent Elasticsearch / OpenSearch inverted index or pgvector hybrid with precomputed inverted lists. |
| **Neural Reranker** | **2-Stage: RRF Coarse Cut (Top 15) -> Groq LLM Cross-Attention (Top 3)** | Heavy Cross-Encoder Model (BGE / Cohere) | Groq LPU delivers deep cross-attention reasoning tokens in < 300ms without GPU hardware overhead; automatic RRF fallback on failure. | External API call dependency; network hop adds 200-400ms to total Q&A response lifecycle. | Self-hosted cross-encoder (e.g. `bge-reranker-large`) on private vLLM GPU inference clusters (Section 4.3). |
| **Rate Limiter** | **In-Memory Sliding Window per IP with Lazy Eviction** | Distributed Redis Token Bucket | Zero latency in-process check (< 0.1ms); automatic eviction of IPs idle for > 5 minutes prevents memory leaks. | State is local to process memory; cannot enforce shared rate limits across multiple container instances. | Distributed Redis token bucket using atomic Lua scripts and tiered tenant API key limits. |
| **Enterprise Logging** | **Structured 9-Column Async Pipe Format (`ContextVar`)** | Arbitrary Unstructured JSON Log Blobs | High human scannability during live terminal debugging; zero secret exposure via `SecretMaskingFilter`; daily date-partitioned storage. | Requires structured log collector (Fluentbit / Logstash) to parse pipe delimiters for Datadog or CloudWatch. | Dual-mode logging handler: human-readable ANSI pipe format for local dev; structured JSON format for K8s stdout. |

---

## 4. Scalability Engineering Roadmap (10K, 100K, 1M Concurrent Requests)

### 4.1 Scaling to 10,000 Concurrent Requests (Single Node / Small Cluster)

At **10K concurrent requests**, the primary bottlenecks are CPU starvation on document parsing, thread pool exhaustion, SQLite write locks, and downstream LLM API connection limits.

#### Architectural Solutions:
1. **Async Concurrency & Worker Model**:
   - Run **Gunicorn** managing **Uvicorn worker processes** (`workers = (2 * CPU_CORES) + 1`).
   - Use `uvloop` for high-performance event loop execution.
   - Offload CPU-bound tasks (PyMuPDF document extraction, PyPDF parsing) and sync SDK operations to thread pools via `asyncio.to_thread()`, preventing event loop blocking.
2. **Database Connection Pooling**:
   - Transition from SQLite (`aiosqlite`) to **PostgreSQL** with `pgbouncer` or `asyncpg` connection pooling (`pool_size=50`, `max_overflow=20`).
3. **HTTP Client Connection Reuse**:
   - Reuse long-lived persistent `httpx.AsyncClient` instances with HTTP/2 keep-alive for Groq, Gemini, and LangSmith API calls.
4. **Client IP Rate Limiting**:
   - In-memory sliding-window limiter prevents single-client abuse from degrading cluster availability.

---

### 4.2 Scaling to 100,000 Concurrent Requests (Distributed Cluster)

At **100K concurrent requests**, a single node or database cannot handle the traffic. Bottlenecks shift to shared state, vector indexing throughput, and external LLM rate limits.

#### Architectural Solutions:
1. **Horizontal Scaling Behind Load Balancer**:
   - Deploy stateless FastAPI pods on **Kubernetes (EKS/GKE)** behind an **AWS Application Load Balancer (ALB)** or **NGINX Reverse Proxy** with Round-Robin / Least-Connections routing.
   - Autoscaling based on CPU utilization and request latency (HPA target: 70% CPU).
2. **Distributed Semantic & Response Caching**:
   - Introduce **Redis Cluster**:
     - **Exact Match Cache**: Cache identical `(question, session_id)` queries with 1-hour TTL.
     - **Semantic Cache**: Store query embeddings in Redis to match semantically equivalent queries (cosine similarity >= 0.98), bypassing LLM synthesis for common questions.
     - **Distributed Rate Limiting**: Migrate the in-memory sliding window to Redis token buckets (`redis-py` Lua scripts) shared across all API pods.
3. **Asynchronous Ingestion Queueing**:
   - Decouple synchronous file uploads from vector processing using **Celery with Redis** or **RabbitMQ**:
     - Client uploads file → API saves to S3/GCS → pushes message to `document-ingest` queue → returns HTTP 202 Accepted with task ID.
     - Background worker pool parses, chunks, and embeds documents asynchronously, updating status to `indexed`.
4. **Dedicated Vector Store Infrastructure**:
   - Migrate from embedded ChromaDB to a distributed vector database: **Chroma Cloud**, **Qdrant Distributed**, or **Milvus Cluster** with read replicas for horizontal retrieval scaling.

---

### 4.3 Scaling to 1,000,000 Concurrent Requests (Enterprise Global Scale)

At **1M concurrent requests** (~1,000,000 req / minute ≈ 16,666 req / second), synchronous generation through public LLM endpoints faces severe rate limiting and global network latency. The architecture must become fully event-driven, decoupled, and multi-region.

#### Architectural Solutions:
1. **CQRS & Complete Read/Write Separation**:
   - **Write Path (Document Ingestion)**:
     - Files uploaded directly to object storage via **S3 Presigned URLs** (bypassing API servers).
     - Storage notifications trigger **Apache Kafka** event streams (`doc.uploaded`).
     - Scalable consumer microservices on dedicated GPU nodes compute embeddings in batch.
   - **Read Path (Q&A Generation)**:
     - Highly optimized stateless read workers with multi-tier edge caching (**Cloudflare Workers / Fastly**).
2. **Dedicated Private LLM Inference Clusters**:
   - Replace public shared API endpoints with dedicated private LLM infrastructure:
     - **vLLM / TensorRT-LLM** on AWS Inf2 / H100 clusters serving quantized models (`Llama 3.3 70B`, `Mixtral`).
     - Dynamic batching, PagedAttention, and continuous KV-cache reuse.
     - Multi-provider fallback routing (AWS Bedrock → Groq → Google Cloud Vertex AI → Azure OpenAI).
3. **Multi-Region Vector Partitioning**:
   - Multi-region vector storage deployment (e.g. US-East, EU-Central, AP-South) with geo-DNS routing.
   - Vector indexing partitioned by `tenant_id` / `organization_id` (sharded HNSW indexes), ensuring queries only scan their isolated shard.
4. **Graceful Degradation & Backpressure**:
   - Circuit breakers (Netflix Hystrix pattern) automatically shed load when downstream latency exceeds SLA targets.
   - Under extreme load spikes:
     - Tier 1: Bypass neural reranker and serve top chunks directly from RRF.
     - Tier 2: Return precomputed extractive summaries if generative LLMs exceed capacity.

---

## 5. Multi-Tenant Data Isolation Strategy

1. **Relational Isolation (SQLite / PostgreSQL)**:
   - All `documents`, `chunks`, and `question_history` records contain `user_id` and `session_id` columns with indexed foreign keys.
   - Repositories enforce tenant-scoped queries:
     ```sql
     SELECT * FROM documents WHERE user_id = :user_id AND session_id = :session_id;
     ```
2. **Vector Space Isolation (ChromaDB)**:
   - ChromaDB collections use compound metadata filtering on every search:
     ```json
     {
       "$and": [
         { "user_id": { "$eq": "usr_abc" } },
         { "session_id": { "$eq": "sess_xyz" } }
       ]
     }
     ```
   - Chunks from Session A are mathematically unreachable by queries in Session B.
3. **Filesystem Partitioning**:
   - Uploaded files are persisted under isolated directory trees:
     `data/uploads/{user_id}_{session_id}_{date}/{filename}`.

---

## 6. DeepEval 2-Pillar Evaluation Benchmark

Implemented in `backend/app/rag/evaluation.py` and tested via `backend/tests/test_deepeval_framework.py`:

```text
+---------------------------------------------------------------------------------------+
|                    THE 2-PILLAR PRODUCTION RAG EVALUATION SUITE                       |
+---------------------------------------------------------------------------------------+
| PILLAR 1: RETRIEVAL QUALITY (DeepEval Metrics)                                        |
| - ContextualRecallMetric: Did the hybrid retriever capture all ground-truth facts?    |
| - ContextualPrecisionMetric: Are the most relevant chunks prioritized at the top?    |
| - ContextualRelevancyMetric: What proportion of retrieved chunk text is relevant?     |
+---------------------------------------------------------------------------------------+
| PILLAR 2: GENERATION QUALITY (DeepEval Metrics)                                       |
| - FaithfulnessMetric: Is the answer 100% grounded in retrieved text?                  |
| - HallucinationMetric: Did the synthesizer introduce any ungrounded external claims?  |
| - AnswerRelevancyMetric: Does the answer directly and completely address the query?   |
+---------------------------------------------------------------------------------------+
```

---

## 7. Security, Zero-Secret Masking & Enterprise Logging

1. **Standardized Enterprise Logging (`backend/app/core/logging.py`)**:
   - Standard 9-column schema: `Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata`.
   - Automatic `contextvars` injection binding `request_id`, `user_id`, and `session_id` across async request execution.
   - 17 Standard lifecycle phases (`STARTUP`, `REQUEST`, `AUTH`, `VALIDATION`, `INGESTION`, `EXTRACTION`, `CHUNKING`, `EMBEDDING`, `RETRIEVAL`, `RERANKING`, `GENERATION`, `DATABASE`, `VECTOR_STORE`, `PROVIDER`, `RESPONSE`, `BACKGROUND`, `SHUTDOWN`).
   - Daily date-partitioned storage: `logs/YYYY-MM-DD/{application.log, error.log, access.log}` with automated 7-day retention cleanup.
   - Custom `SecretMaskingFilter` intercepts and redacts credentials (Anthropic `sk-ant-*`, Groq `gsk_*`, Gemini `AIza*`, OpenAI `sk-*`, Bearer tokens) and bars raw document content from log statements.
2. **Sliding-Window Rate Limiter (`backend/app/core/rate_limiter.py`)**:
   - Limits queries on `POST /ask` to 20 requests per 60 seconds per client IP.
   - Emits `HTTP 429 Too Many Requests` with standard `Retry-After: <seconds>` headers.
3. **Sanitized Error Shielding (`backend/app/core/exceptions.py`)**:
   - RFC 7807 Problem Details format masks internal database credentials and tracebacks.

