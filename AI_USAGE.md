# AI Coding Tools Disclosure

This document discloses the utilization of AI coding assistants during the design, implementation, testing, and refinement of the **Mini Knowledge Assistant** platform.

---

## 1. Tools Utilized & Scope of Tasks

| Tool                                        | Environment / Interface                  | Tasks Performed                                                                                                                                                               |
| :------------------------------------------ | :--------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Antigravity AI Assistant**                | Google Antigravity IDE                   | Interactive pair programming, code scaffolding (Pydantic v2, SQLAlchemy 2.0), automated test fixture authoring, CSS styling token implementation, and documentation drafting. |
| **Google Gemini Flash / Flash Lite Models** | Backend Runtime LLM Provider & LangChain | Dynamic runtime RAG synthesis, contextual grounding, anti-hallucination constraint enforcement, and source citation attribution.                                              |
| **Google Gemini Embedding-2**               | Backend Runtime Vector Provider          | High-dimensional dense text embedding (3,072 dimensions) for indexing and vector similarity search.                                                                           |
| **Groq Llama 3 / GPT-OSS-120B**             | Backend Runtime Cross-Encoder Reranker   | Neural cross-attention candidate reranking and joint precision scoring (0–10) over retrieved candidate passages.                                                              |

---

## 2. Division of Architectural Responsibility & Material AI Influence

To maintain technical transparency, engineering credit and architectural decisions were divided between my direction as the lead engineer and the AI assistant as an accelerator:

### Core Architectural Concepts Directed by Me

1. **3-Stage Hybrid RAG Funnel Architecture**:
   - Rather than accepting standard, naive single-vector search or basic LangChain boilerplate, I conceived and directed the multi-stage precision funnel:
     - **Stage 1 (Dual Retrieval)**: Parallel dense semantic search (ChromaDB HNSW, 3072-d) and lexical keyword search (BM25L) across session and user scopes.
     - **Stage 2A (Reciprocal Rank Fusion)**: Pure Cormack RRF with smoothing constant `k = 60` for fair rank ordering across disparate retrieval distributions.
     - **Stage 2B (Groq Neural Cross-Attention)**: Dedicated cross-encoder neural evaluation selecting the top 3 most relevant passages.
     - **Stage 3 (Grounded Synthesis)**: Strict anti-hallucination prompt schema enforcing citation attribution and zero outside-knowledge leakage.

2. **Decoupled Asynchronous Ingestion with Dual-Channel Telemetry**:
   - Recognizing that real-world multi-document PDF/DOCX extraction and vectorization easily exceeds typical 30-second client HTTP socket timeouts, I formulated the decoupled background ingestion architecture:
     - Immediate disk write and validation under 50ms, returning `HTTP 202 Accepted` with an `AsyncIngestionJobResponse`.
     - Background worker execution spawned via `asyncio.create_task` with isolated database session management.
     - Dual-channel telemetry providing real-time Server-Sent Events (SSE) streaming (`/documents/jobs/{job_id}/events`) paired with a robust JSON polling fallback (`/documents/jobs/{job_id}`).

### Concrete Areas Where the AI Assistant Materially Accelerated Implementation

1. **Type-Safe Contract & Boilerplate Scaffolding**:
   - The AI assistant rapidly scaffolded Pydantic v2 request/response schemas, SQLAlchemy 2.0 asynchronous repository classes, and RFC 7807 Problem Details exception handlers based on my architectural specifications, saving significant development time.

2. **Comprehensive Test Suite & Benchmark Authoring**:
   - The AI assisted in scaffolding the extensive pytest suite (**67 passing tests**), including ephemeral ChromaDB fixtures, simulated rate limiter token buckets, tenant isolation assertions, and DeepEval evaluation metric harnesses (faithfulness, answer relevancy, and hallucination).

3. **9-Column Context Logging & Privacy Shielding**:
   - The AI helped implement my structured logging specification:
     ```text
     Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata
     ```
     including `contextvars` context propagation across asynchronous task boundaries, 7-day disk rollover retention, and regex-based `SecretMaskingFilter` to sanitize API keys from stdout and logs.

4. **Frontend Component Ergonomics & CSS Token Design**:
   - The AI assisted in translating my UX requirements into Vanilla CSS custom property design systems, implementing glassmorphic cards, responsive sidebar drawers, auto-expanding multi-line textareas, and optimistic table state updates.

---

## 3. Concrete Examples of Reviewed, Changed, or Rejected AI Suggestions

### Example 1: Rejected — Concurrent `asyncio.gather()` File Ingestion

- **Initial AI Suggestion**: When batch uploading up to 5 files, the AI proposed parallelizing the extraction, chunking, and embedding generation for all files simultaneously using `asyncio.gather(*[process_file(f) for f in files])`.
- **Reason for Rejection**:
  1. _API Quota Limits_: Concurrently vectorizing dozens of chunks across 5 large PDF files simultaneously triggered immediate HTTP 429 Too Many Requests rate limit exceptions from the Google Gemini Embedding API.
  2. _Database Concurrency Contention_: Simultaneous asynchronous writes to SQLite across multiple coroutines produced `sqlite3.OperationalError: database is locked` errors, even with WAL mode enabled.
  3. _Error Isolation & Telemetry_: Concurrent execution made sequential per-file progress reporting and partial failure isolation significantly harder to convey over Server-Sent Events.
- **Architectural Adjustment**: I rejected `asyncio.gather` and implemented sequential per-file execution within the background worker. Each file is processed, chunked, embedded, and committed in order, providing clean, incremental SSE progress events (e.g., `Processing file 2 of 4`) and allowing the batch to complete with status `completed_with_errors` if one file is corrupt, while successfully indexing all valid files.

---

### Example 2: Changed — Non-Intrusive In-Table Ingestion Status vs. Blocking Modal

- **Initial AI Suggestion**: For the React frontend, the AI initially proposed a full-screen or floating modal overlay that rendered whenever a document upload was queued, blocking user navigation until the background job reached a terminal state.
- **Reason for Change**:
  - Modal overlays interrupt user workflows. If an indexing job takes 10–20 seconds (e.g., for multi-page research PDFs), the user should still be able to ask questions about existing documents, review previous citations, or switch tabs without an intrusive overlay locking their screen.
- **Architectural Adjustment**: I redesigned the telemetry UX to be completely non-intrusive and embedded directly into `DocumentTable.jsx`:
  - When files are dropped, optimistic placeholder rows are injected immediately into the document inventory table with an animated amber beacon (`status-indexing`) and a subtle progress bar showing the live percentage.
  - The user remains completely free to navigate tabs and chat with the assistant while background indexing proceeds smoothly.
  - Once the terminal SSE event is received, the table row transitions seamlessly to a green `indexed` badge with exact page and chunk counts.

---

### Example 3: Rejected — Dynamic Min-Max Normalization on Reciprocal Rank Fusion (RRF)

- **Initial AI Suggestion**: When integrating Cormack Reciprocal Rank Fusion for hybrid retrieval:
  ```text
  RRF_Score(d) = Σ_m [ 1 / (k + rank_m(d)) ]
  ```
  with `k = 60`, individual document scores typically range from `~0.01` to `~0.03`. The AI proposed applying per-query dynamic min-max normalization:
  ```text
  Normalized_Score(d) = (RRF_Score(d) - Min_Score) / (Max_Score - Min_Score + 1e-9)
  ```
  to rescale the candidate batch into `[0.0, 1.0]` for downstream thresholding and confidence scoring.
- **Reason for Rejection**:
  1. _False Confidence on Irrelevant Queries_: For queries with zero relevant matches, dynamic min-max artificially inflates the least-irrelevant passage to `1.0` (100% confidence).
  2. _Artificial Deflation of High-Quality Runner-Ups_: If multiple candidates are near-identical top matches (e.g., RRF scores `0.0328`, `0.0325`, `0.0321`), min-max punishes the 3rd or 4th candidate down toward `0.0`, falsely signaling low relevance.
  3. \_Violating RRF’s Core Purpose\*: Cormack RRF is mathematically designed **strictly for monotonic rank ordering**, bypassing the unbounded score scaling and vocabulary size dependencies of lexical search (BM25). Forcing dynamic min-max onto rank fusion breaks score comparability across queries.
- **Architectural Adjustment**: I rejected dynamic min-max normalization on RRF and implemented an enterprise-grade **Two-Stage Reranking Pipeline**:
  - **Stage 2A — Coarse Rank Ordering (RRF)**: Reciprocal Rank Fusion is used strictly to establish relative candidate order and prune hybrid (Dense + BM25) candidate pools down to top candidates (e.g., 15 chunks).
  - **Stage 2B — Deep Precision Scoring (Groq LLM Cross-Encoder)**: The `GroqLLMReranker` evaluates query-passage pairs using full cross-attention, generating calibrated precision scores scaled to `[0.0, 1.0]` (`groq_score / 10.0`).
  - **Fallback Bounding**: If external LLM calls encounter rate limits, the system falls back to RRF rank order bounded by the static theoretical maximum `Max_RRF = M / (k + 1)` rather than dynamic batch min-max.

---

## 4. Engineering Ownership & Responsibility Affirmation

As the engineer submitting this assessment:

- **Full Technical Accountability:** I remain responsible for the overall design, implementation, architectural decisions, security considerations, maintainability, and quality of the code submitted in this repository.
- **Rigorous Code Verification:** All AI-assisted code was manually reviewed, refactored, integrated, and validated against the project's engineering standards and established best practices, including RFC 7807-compliant error responses, asynchronous SQLAlchemy 2.0 transaction management, secret masking in logs, and responsive frontend design, to the best of my ability.
- **Automated Verification:** The implementation is supported by automated unit, integration, rate-limiting, and RAG evaluation tests, all passing successfully at the time of submission. The frontend also produces a successful optimized production build using Vite.
