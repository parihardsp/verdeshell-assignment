# Mini Knowledge Assistant — Backend Service

A production-grade, asynchronous Retrieval-Augmented Generation (RAG) backend service built with **Python 3.12**, **FastAPI**, **SQLAlchemy 2.0 (aiosqlite)**, **ChromaDB**, and **Pydantic v2**.

---

## 1. System Overview

The **Mini Knowledge Assistant Backend** ingests multi-format operational documents (`.pdf`, `.txt`, `.md`), segments them with page-level structural preservation, computes dense vector embeddings, and answers complex user questions via a 3-stage hybrid precision pipeline with grounded citations and zero hallucinations.

### Key Highlights
- **Clean Architecture & SOLID**: Strict separation between API controllers, domain services, persistence repositories, and external AI providers.
- **3-Stage Hybrid RAG Funnel**:
  - **Stage 1 (Dual-Channel Hybrid Retrieval)**: Parallel Dense Vector Search (ChromaDB HNSW cosine similarity) + Lexical Keyword Search (BM25L) with compound tenant scoping (`user_id` and `session_id`).
  - **Stage 2A (Reciprocal Rank Fusion Coarse Cut)**: Cormack's Reciprocal Rank Fusion (`RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]`) merging dense and BM25 rank lists into a Top 15 candidate pool.
  - **Stage 2B (Groq Neural Precision Reranker)**: High-throughput Groq LLM (`openai/gpt-oss-120b`) cross-attention scoring (0.0 to 10.0) with reasoning, pruning to the Top 3 precision chunks with graceful fallback to RRF.
  - **Stage 3 (Grounded Synthesis)**: High-performance generation via **Google Gemini Flash Lite** (`gemini-flash-lite-latest`), **Groq** (`openai/gpt-oss-120b`), or offline **Mock LLM** with native LangSmith tracing and strict anti-hallucination prompt guardrails.
- **Standardized Model Selection via `.env`**:
  - **Embedding Providers (2 choices)**: `gemini` (Google Gemini 3072-dim vectors, default) or `mock` (offline 384-dim deterministic vectors).
  - **LLM Generation Providers (3 choices)**: `gemini` (Google Gemini, default), `groq` (Groq ultra-fast Llama), or `mock` (offline synthesizer).
  - **Reranker Providers (3 choices)**: `groq` (Fast Groq LLM reranker, default), `rrf` (Pure Reciprocal Rank Fusion), or `mock`.
- **PyMuPDF Document Ingestion**: Multi-file batch upload (up to 5 files, 5MB each) with page-preserving text extraction and disk persistence under `data/uploads/{user_id}_{session_id}_{date}/`.
- **DeepEval 2-Pillar Evaluation Framework**: Production quality test suite assessing Retrieval Recall, Retrieval Precision, Retrieval Relevancy, Generation Faithfulness, Hallucination, and Answer Relevancy powered by Groq.
- **Enterprise Guardrails**:
  - Sliding-window in-memory rate limiter on `POST /ask` (20 req / 60s per client IP) returning HTTP 429 with standard `Retry-After` headers.
  - SHA-256 duplicate detection and payload validation.
  - RFC 7807 compliant sanitized error shielding.

---

## 2. Project Directory Structure

```text
backend/app/
├── api/
│   ├── __init__.py              # Exports consolidated api_router
│   ├── dependencies.py          # FastAPI Depends factories & startup vector hydration
│   └── routes.py                # Consolidated REST endpoints (/health, /documents, /ask)
│
├── core/
│   ├── __init__.py
│   ├── config.py                # Pydantic v2 BaseSettings with active provider configs
│   ├── exceptions.py            # RFC 7807 domain exception classes & global handlers
│   ├── logging.py               # ContextVar session-scoped structured logging
│   └── rate_limiter.py          # Sliding window token rate limiter with Retry-After
│
├── db/
│   ├── __init__.py              # Unified database package exports
│   ├── database.py              # Async engine, AsyncSessionLocal, and schema auto-migrations
│   ├── models.py                # Declarative Base, Document, Chunk, QuestionHistory models
│   └── repositories.py          # BaseRepository, Document, Chunk, and QuestionHistory repos
│
├── ingestion/
│   ├── __init__.py              # Ingestion package exports
│   ├── extractor.py             # DocumentExtractor (PyMuPDF parser & text decoders)
│   ├── storage.py               # SessionFileStorageService (session/date disk partitioning)
│   └── validator.py             # FileValidationService (SHA-256 deduplication, 5MB limits)
│
├── providers/
│   ├── __init__.py              # Provider adapters package exports
│   ├── embeddings.py            # GeminiEmbeddingProvider (3072-dim) & MockEmbeddingProvider
│   ├── llm.py                   # LangChainLLMProvider, GroqLLMProvider, GeminiLLMProvider
│   └── vector_store.py          # ChromaVectorStore with compound session & user scoping
│
├── rag/
│   ├── __init__.py              # RAG pipeline package exports
│   ├── chunker.py               # RecursiveTextChunker (paragraph/sentence/word splits)
│   ├── embedder.py              # RAGEmbedder batch coordination stage
│   ├── evaluation.py            # DeepEval 4-pillar quality benchmark framework (Groq)
│   ├── pipeline.py              # Decoupled RAGPipeline orchestrator
│   ├── reranker.py              # GroqLLMReranker, ReciprocalRankReranker, MockReranker
│   ├── retriever.py             # HybridRetriever (Dense Vector + BM25L parallel retrieval)
│   └── synthesizer.py          # GroundedSynthesizer with source citation assembly
│
├── schemas/
│   ├── __init__.py              # Pydantic v2 schemas exports
│   ├── common.py                # HealthResponse, ErrorDetail
│   ├── document.py              # DocumentCreate, DocumentResponse, MultiFileUploadResponse
│   └── rag.py                   # AskRequest, AskResponse, SourceCitation
│
├── services/
│   ├── __init__.py              # Domain services package exports
│   ├── chat_service.py          # High-level Q&A facade (aliased to RAGService)
│   └── document_service.py      # Ingestion, validation, chunking, and persistence orchestration
│
└── main.py                      # FastAPI lifespan startup, CORS, middleware, router mount
```

---

## 3. End-to-End System Flows

### 3.1 Document Ingestion Flow (Write Path)

Document chunking and vector embedding computation occur strictly at **Upload Time** to guarantee sub-second read queries:

```text
[Client / Streamlit]
       │  POST /documents/upload (Multipart: up to 5 files, session_id, user_id)
       ▼
[validator.py: FileValidationService]
       │  Enforce batch limit (<= 5), size limit (<= 5MB), allowed extensions (.pdf/.txt/.md)
       │  Calculate SHA-256 content hash for deduplication
       ▼
[storage.py: SessionFileStorageService]
       │  Persist sanitized raw files to disk: data/uploads/{user_id}_{session_id}_{date}/
       ▼
[extractor.py: DocumentExtractor]
       │  PyMuPDF extracts page-by-page text preserving ExtractedPage(page_number, text)
       ▼
[chunker.py: RecursiveTextChunker]
       │  Recursive fallback splitting (\n\n -> \n -> sentence -> word)
       │  Output: List[TextChunk] with char offsets and page numbers
       ▼
[embeddings.py: GeminiEmbeddingProvider]
       │  Batch vectorization via Google Gemini models/gemini-embedding-2 (dim: 3072)
       ▼
[database.py & repositories.py] ───► SQLite (data/knowledge_base.db):
       │                             - Persist Document row
       │                             - Bulk insert Chunk rows with JSON embeddings
       ▼
[vector_store.py: ChromaVectorStore]
       │  Upsert IndexedChunk records into ChromaDB HNSW collection (data/chroma_db/)
       ▼
[Response] ──► MultiFileUploadResponse (session_id, user_id, processed files, page & chunk counts)
```

---

### 3.2 RAG Q&A Execution Flow (Read Path)

```text
[Client / Streamlit]
       │  POST /ask {"question": "...", "session_id": "sess_1", "user_id": "usr_1"}
       │  Headers: X-User-ID: usr_1, X-Session-ID: sess_1
       ▼
[Middleware: user_session_context_middleware]
       │  Extract user_id & session_id, set ContextVars for structured logging: [user_id:session_id]
       ▼
[rate_limiter.py: SlidingWindowRateLimiter]
       │  Check client IP: 20 requests per 60s window (returns 429 Retry-After if exceeded)
       ▼
[pipeline.py: RAGPipeline]
       │
       ├─► [STAGE 1: DUAL-CHANNEL HYBRID RETRIEVAL] (retriever.py: HybridRetriever)
       │      │  Channel 1: Dense Vector Retrieval (ChromaDB HNSW, Top 15, threshold >= 0.20)
       │      │  Channel 2: Lexical Keyword Retrieval (BM25L Inverted Index, Top 15)
       │      │  Parallel execution via asyncio.gather -> Merged Pool (up to 30 chunks)
       │      ▼  Tags each candidate with channel origin: "dense", "bm25", or "hybrid"
       │
       ├─► [STAGE 2A: RRF COARSE CUT] (reranker.py: ReciprocalRankReranker)
       │      │  Dense rank + Lexical rank fusion:
       │      │  RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]
       │      ▼  Selects Top 15 candidates for neural evaluation
       │
       ├─► [STAGE 2B: GROQ NEURAL PRECISION RERANKER] (reranker.py: GroqLLMReranker)
       │      │  Zero-temperature joint cross-attention via Groq LLM (openai/gpt-oss-120b)
       │      │  Assigns calibrated relevance score (0.0 to 10.0) with reasoning
       │      │  Selects final Top 3 precision chunks (with automatic fallback to RRF)
       │      ▼
       │
       ├─► [STAGE 3: GROUNDED SYNTHESIS] (synthesizer.py: GroundedSynthesizer)
       │      │  Modular prompt assembly via app.rag.prompts.synthesis_prompts
       │      │  Enforce Pydantic schema: GroundedSynthesisOutput (answer, has_sufficient_context, cited_chunk_indices)
       │      │  Zero-hallucination gating: Returns clean refusal when has_sufficient_context=False
       │      │  Precision citations: Only cited_chunk_indices are attached to SourceCitation objects
       │      │  Invoke Gemini Flash Lite (gemini-flash-lite-latest) with LangSmith tracing
       │      ▼  Attach SourceCitation objects (doc_id, title, page_number, channel, reranker_score)
       │
       └─► [AUDIT LOGGING] (repositories.py: QuestionHistoryRepository)
              │  Record question, answer, citations, latency_ms, client_ip in SQLite
              ▼
[Response] ──► AskResponse (answer, sources with citations, session_id, latency_ms)
```

---

## 4. DeepEval 2-Pillar Evaluation Framework & Golden Dataset

Located in `app/rag/evaluation.py` and `scripts/run_rag_evaluation.py`, the evaluation harness verifies both sides of the RAG pipeline using DeepEval's RAG Triad metrics:

```text
+------------------------------------------------------------------------+
|             THE 2-PILLAR PRODUCTION RAG EVALUATION FRAMEWORK           |
+------------------------------------------------------------------------+
| PILLAR 1: RETRIEVAL QUALITY (DeepEval)                                 |
| - ContextualRecallMetric: Did we retrieve all ground truth facts?      |
| - ContextualPrecisionMetric: Are the most relevant chunks at the top?  |
| - ContextualRelevancyMetric: Signal-to-noise ratio in retrieved text   |
+------------------------------------------------------------------------+
| PILLAR 2: GENERATION QUALITY (DeepEval)                                |
| - FaithfulnessMetric: Is the answer 100% grounded in retrieved text?   |
| - HallucinationMetric: Did the LLM inject external ungrounded facts?   |
| - AnswerRelevancyMetric: Does the answer directly address the user?    |
+------------------------------------------------------------------------+
```

### 4.1 Canonical Golden Dataset
The evaluation harness includes a curated `GOLDEN_DATASET` (encapsulated as a DeepEval `EvaluationDataset`) covering 4 critical enterprise RAG scenarios:

| # | Scenario | Query | Ground Truth Focus |
|---|---|---|---|
| **1** | `policy_clause_lookup` | Uptime & penalty-free contract termination | Clause 9.1: 45 days written notice if uptime < 99.5% for 2 consecutive quarters |
| **2** | `multi_constraint_temporal` | Scheduled maintenance & patch notice | 2nd Saturday 02:00-06:00 UTC, 2-hour emergency patch notification |
| **3** | `compliance_governance` | Data residency & certification | EU-West (Frankfurt) & US-East (Virginia), ISO-27001 certified |
| **4** | `zero_hallucination_refusal` | Support ticket 15-minute response fee | Refusal: Knowledge base does not contain monetary penalty fees for response times |

### 4.2 How to Configure & Run the Evaluation Benchmark

The evaluation benchmark mode and model selection are streamlined **directly in `.env`** and can be customized or overridden via CLI flags:

```env
# In .env:
# Switch between 'live' (real Groq LLM & Gemini embeddings) or 'mock' (offline deterministic, 0 tokens)
EVALUATION_MODE=live

# Unified model used across the evaluation suite for both extractor synthesis and DeepEval judging
DEEPEVAL_MODEL=qwen/qwen3.8-27b
```

Running the benchmark runner (ensure you are in the `backend` directory):

```bash
cd backend

# 1. Run using .env configuration (defaults to EVALUATION_MODE=live and DEEPEVAL_MODEL=qwen/qwen3.8-27b):
uv run python scripts/run_rag_evaluation.py

# 2. Explicitly force offline mock mode (no external API calls, instantaneous):
uv run python scripts/run_rag_evaluation.py --mock

# 3. Explicitly force live benchmark against Groq & Gemini:
uv run python scripts/run_rag_evaluation.py --live

# 4. Override the unified extractor/judge model on-the-fly:
uv run python scripts/run_rag_evaluation.py --live --model qwen/qwen3.8-27b

# 5. Filter by metric subset (generation or retrieval) and export report to JSON:
uv run python scripts/run_rag_evaluation.py --metrics generation --output eval_report.json

# 6. Run automated test suite verifying evaluation framework (all 48 tests pass):
uv run pytest tests/test_deepeval_framework.py
```

---

## 5. Quickstart & Installation

### 5.1 Prerequisites
- Python 3.12+
- `uv` package manager ([Install uv](https://docs.astral.sh/uv/))

### 5.2 Environment Configuration
Copy `.env.example` to `.env` in the repository root:

```bash
cp ../.env.example ../.env
```

Ensure active providers in `.env`:
```ini
# Standardized Model Selection
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-flash-lite-latest

RERANKER_TYPE=groq
GROQ_RERANKER_MODEL=openai/gpt-oss-120b
RETRIEVAL_TOP_K=15
FINAL_TOP_N=3

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Vector Store & Paths
DATABASE_URL=sqlite+aiosqlite:///data/knowledge_base.db
CHROMA_PERSIST_DIR=data/chroma_db
UPLOAD_DIR=data/uploads
```

### 5.3 Install Dependencies
```bash
cd backend
uv sync
```

### 5.4 Run the Backend Service
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 6. Testing

Run the automated test suite covering unit tests, integration tests, rate limiting, and DeepEval evaluation metrics:

```bash
uv run pytest
```

Output:
```text
======================== 38 passed, 8 warnings in 3.11s ========================
```

---

## 7. API Reference (cURL Examples)

### Health Check
```bash
curl -s http://localhost:8000/health | jq
```

### Direct Article Ingestion (JSON)
```bash
curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -H "X-User-ID: usr_1001" \
  -H "X-Session-ID: sess_prod_01" \
  -d '{
    "title": "Corporate Leave Policy",
    "content": "Full-time employees receive 25 days of annual paid leave.",
    "session_id": "sess_prod_01",
    "user_id": "usr_1001"
  }' | jq
```

### Multi-File Upload
```bash
curl -s -X POST http://localhost:8000/documents/upload \
  -H "X-User-ID: usr_1001" \
  -H "X-Session-ID: sess_prod_01" \
  -F "files=@/path/to/handbook.pdf" \
  -F "files=@/path/to/faq.txt" \
  -F "session_id=sess_prod_01" \
  -F "user_id=usr_1001" | jq
```

### Ask Grounded Question
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-User-ID: usr_1001" \
  -H "X-Session-ID: sess_prod_01" \
  -d '{
    "question": "How many days of paid leave do employees receive?",
    "session_id": "sess_prod_01",
    "user_id": "usr_1001"
  }' | jq
```
