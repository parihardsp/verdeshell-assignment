# Backend Architecture & Technical Design Document

This document provides a comprehensive technical overview of the **Mini Knowledge Assistant Backend**, detailing its architectural layers, component responsibilities, asynchronous concurrency model, persistence strategy, and stage-by-stage request lifecycles.

---

## 1. Architectural Philosophy: Clean Architecture & SOLID

The backend adheres strictly to **Clean Architecture / Ports & Adapters (Hexagonal Architecture)**. Domain business logic is completely isolated from HTTP transport frameworks, database engines, and external AI providers.

```text
                               +----------------------------+
                               |     HTTP / Client Tier     |
                               |    (Streamlit / cURL / UI) |
                               +--------------+-------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                               FASTAPI APPLICATION LAYER                                 |
|  - Lifespan Startup: SQLite schema auto-migration & ChromaDB vector index pre-hydration |
|  - Security & Guardrails: SlidingWindowRateLimiter (20 req/60s) & FileValidationService |
|  - ContextVar Scope: [user_id:session_id] injection across all downstream log messages   |
|  - Exception Handlers: RFC 7807 problem details shielding internal stack traces & keys  |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                                    API ROUTER LAYER                                     |
|                               (app/api/routes.py)                                       |
|  - POST /documents/upload   (Multi-file upload up to 5 files, session/user partitioned) |
|  - POST /documents          (Single document article creation)                          |
|  - GET  /documents          (List indexed documents with pagination and user scoping)   |
|  - POST /ask                (Multi-stage grounded Q&A with hybrid retrieval & rerank)    |
|  - GET  /health             (Service health, provider inspect & vector count)           |
+----------------------+--------------------------------------+---------------------------+
                       |                                      |
                       v                                      v
+------------------------------------+   +------------------------------------------------+
|          DOCUMENT SERVICE          |   |                  CHAT SERVICE                  |
|    (app/services/document_service) |   |      (app/services/chat_service - RAGService)  |
|  - Orchestrates multi-file ingest  |   |  - High-level Q&A facade with tenant scoping   |
|  - File size gating & validation   |   |  - Wires decoupled RAG pipeline stages         |
|  - Session folder assignment       |   |  - Enforces session & user isolation           |
|  - Page-preserving PyMuPDF parsing |   |  - Formats grounded source citations & latency |
|  - Context-preserving chunking     |   +--------------------+---------------------------+
|  - Batch vectorization & indexing  |                        |
+------------------+-----------------+                        |
                   |                                          |
                   v                                          v
+------------------------------------+   +------------------------------------------------+
|         INGESTION LAYER            |   |                  RAG PIPELINE                  |
|       (app/ingestion/)             |   |                 (app/rag/)                     |
|  - validator.py: Guardrails & SHA  |   |  - retriever.py: True Dual-Channel Retrieval   |
|  - storage.py: Session disk persist|   |    * Dense Vector Search (ChromaDB HNSW)       |
|  - extractor.py: PyMuPDF parser    |   |    * Lexical Keyword Search (BM25L)            |
+------------------+-----------------+   |  - reranker.py: 2-Stage Precision Reranking    |
                   |                     |    * Stage 2A: RRF Fusion (k=60 -> Top 15)     |
                   v                     |    * Stage 2B: Groq Neural Cross-Encoder (-> 3)|
+------------------------------------+   |  - synthesizer.py: Grounded synthesis & cite   |
|          DATABASE LAYER            |   |  - chunker.py: Recursive text chunker          |
|            (app/db/)               |   |  - evaluation.py: DeepEval 4-pillar benchmark  |
|  - database.py: Async SQLite engine|   +--------------------+---------------------------+
|  - models.py: Declarative models   |                        |
|  - repositories.py: Repositories   |                        v
+------------------+-----------------+   +------------------------------------------------+
                   |                     |               PROVIDER ADAPTERS                |
                   v                     |               (app/providers/)                 |
+------------------------------------+   |  - Embedding Providers:                        |
|        CHROMA VECTOR STORE         |   |    * Gemini (models/gemini-embedding-2, 3072-d)|
|    (app/providers/vector_store.py) |   |    * Mock Deterministic (384-dim)              |
|  - Persistent HNSW Cosine store    |   |  - LLM Providers:                              |
|  - Compound tenant scoping filters |   |    * Gemini (gemini-flash-lite-latest)         |
+------------------------------------+   |    * Groq (openai/gpt-oss-120b)                |
                                         |    * Mock Deterministic Synthesizer            |
                                         +------------------------------------------------+
```

---

## 2. End-to-End Request-Response Lifecycles

### 2.1 Complete Document Ingestion Lifecycle (Write Path)

```text
Sequence of Ingestion:

[HTTP Client] ──► POST /documents/upload (Multipart/form-data)
    │
    ▼
[FastAPI Middleware & Dependency Injection]
    │  Injects DocumentService, SessionFileStorageService, GeminiEmbeddingProvider, ChromaVectorStore
    │  Extracts user_id and session_id from headers or form-data
    ▼
[DocumentService: Multi-File Batch Loop]
    │
    ├─► Step 1: Validation (FileValidationService)
    │      Checks file count (<= 5), size (<= 5MB), and MIME/extension (.pdf, .md, .txt)
    │      Computes SHA-256 hash for document deduplication
    │
    ├─► Step 2: Storage (SessionFileStorageService)
    │      Persists raw bytes to: data/uploads/{user_id}_{session_id}_{date}/{file_name}
    │
    ├─► Step 3: Text & Page Extraction (DocumentExtractor)
    │      PyMuPDF parses PDF pages or text decoders parse markdown/plaintext
    │      Returns List[ExtractedPage(page_number, text)]
    │
    ├─► Step 4: Recursive Semantic Chunking (RecursiveTextChunker)
    │      Splits text into chunks (CHUNK_SIZE=1000, CHUNK_OVERLAP=150)
    │      Preserves page numbers, character start/end offsets, and sentence integrity
    │
    ├─► Step 5: Dense Vector Embedding (GeminiEmbeddingProvider)
    │      Computes batch embeddings via Google Gemini models/gemini-embedding-2 (3072 dimensions)
    │
    ├─► Step 6: Database & Vector Index Persistence
    │      - Inserts Document row and bulk-inserts Chunk rows into SQLite
    │      - Upserts IndexedChunk objects into ChromaDB HNSW collection with compound metadata:
    │        {"document_id": id, "session_id": session_id, "user_id": user_id, "file_name": name}
    ▼
[Response] ──► HTTP 201 Created: MultiFileUploadResponse
```

---

### 2.2 Complete Grounded Q&A Lifecycle (Read Path)

```text
Sequence of Q&A:

[HTTP Client] ──► POST /ask {"question": "...", "session_id": "...", "user_id": "..."}
    │
    ▼
[FastAPI Gateway: Rate Limiter & Context Middleware]
    │  - SlidingWindowRateLimiter checks IP sliding window: 20 req / 60s
    │  - Binds user_id and session_id to ContextVar structured logger: [user_id:session_id]
    ▼
[RAGPipeline Orchestrator (app/rag/pipeline.py)]
    │
    ├─► STAGE 1: DUAL-CHANNEL HYBRID RETRIEVAL (app/rag/retriever.py: HybridRetriever)
    │      Runs Dense Vector and BM25 Lexical search in parallel via asyncio.gather:
    │
    │      Channel 1: VectorRetriever (Dense Vector Search)
    │      - Generates query embedding via GeminiEmbeddingProvider (3072 dimensions)
    │      - Queries ChromaDB with compound filter:
    │        {"$and": [{"user_id": user_id}, {"session_id": session_id}]}
    │      - Retrieves Top 15 semantic chunks (cosine similarity >= 0.20)
    │
    │      Channel 2: BM25Retriever (Lexical Keyword Search)
    │      - Loads corpus chunks matching user_id & session_id
    │      - Tokenizes query with stopword removal
    │      - Computes BM25L scores across chunk vocabulary
    │      - Retrieves Top 15 keyword chunks
    │
    │      Candidate Pool Merge:
    │      - Merges candidates by chunk_id
    │      - Annotates each chunk with channel: "dense", "bm25", or "hybrid"
    │      - Yields a unified pool of up to 30 unique candidate chunks
    │
    ├─► STAGE 2A: RECIPROCAL RANK FUSION (RRF COARSE CUT) (app/rag/reranker.py)
    │      - Orders candidates by dense rank and lexical rank
    │      - Calculates Cormack's Reciprocal Rank Fusion score:
    │        RRF_Score(d) = Σ [ 1 / (60 + rank_m(d)) ]
    │      - Ranks candidates and prunes to Top 15 candidates for neural evaluation
    │
    ├─► STAGE 2B: GROQ NEURAL PRECISION RERANKER (app/rag/reranker.py: GroqLLMReranker)
    │      - Prepares the 15 candidate passages with their indices
    │      - Invokes Groq LLM (openai/gpt-oss-120b) with a strict zero-temperature JSON prompt
    │      - Evaluates joint cross-attention between question and candidate text
    │      - Assigns calibrated relevance score (0.0 to 10.0) and explanation
    │      - Selects the final Top 3 precision chunks
    │      - Graceful Fallback: If Groq hits any rate limit or network error, transparently
    │        falls back to RRF rank order without failing the request
    │
    ├─► STAGE 3: GROUNDED SYNTHESIS (app/rag/synthesizer.py: GroundedSynthesizer)
    │      - Assembles grounded context with the Top 3 precision chunks
    │      - Enforces strict anti-hallucination prompt guardrails
    │      - Invokes Google Gemini Flash Lite (gemini-flash-lite-latest) via LangChain
    │      - Traces prompt, completion tokens, and latency in LangSmith
    │      - Builds SourceCitation objects including snippet, retrieval_channel, and reranker_score
    │
    └─► AUDIT LOGGING (app/db/repositories.py: QuestionHistoryRepository)
           - Asynchronously logs question, answer, citations, latency_ms, client_ip into SQLite
    ▼
[Response] ──► HTTP 200 OK: AskResponse
```

---

## 3. Concurrency & Asynchronous Execution Model

- **FastAPI / AnyIO**: Fully asynchronous execution loop running on Uvicorn.
- **Parallel Retrieval (`asyncio.gather`)**: Stage 1 executes Dense Vector search and BM25 Lexical search concurrently, cutting retrieval latency by half.
- **Async Relational Persistence (`aiosqlite`)**: AsyncSession connections yield execution during disk I/O, preventing event-loop blocking.
- **LangSmith Non-Blocking Tracing**: LangSmith telemetry runs asynchronously in background worker queues without degrading synthesis response times.
