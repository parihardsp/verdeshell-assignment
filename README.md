# Enterprise Grounded Knowledge Assistant

A production-grade, enterprise grounded **Retrieval-Augmented Generation (RAG)** platform designed for internal support and knowledge operations. Built with a high-performance **FastAPI backend** (Python 3.12, UV) and a sleek, modern **React 18 + Vite frontend** with responsive mobile and desktop interfaces.

---

## 🌟 Key Highlights

- **Multi-Format Document Ingestion**: Ingests **PDF**, **DOCX**, **DOC**, **TXT**, **MD**, and **CSV** files with page-level structural preservation and automatic table-to-markdown formatting.
- **Asynchronous Decoupled Background Ingestion**: Immediate `HTTP 202 Accepted` response for batch uploads, offloading parsing, chunking, and vector embedding to background tasks with zero connection timeouts.
- **Dual-Channel Telemetry & In-Table Status**: Live Server-Sent Events (`text/event-stream`) with automatic JSON polling fallback. Telemetry renders directly in the **Session Knowledge Inventory** table rows (`● Processing {percent}%` with pulsing amber beacon, stage moniker, and micro-progress bar; `○ Queued`; `● Failed / Skipped` with error isolation; `● Indexed`).
- **3-Stage Hybrid RAG Precision Funnel**:
  1. **Dual-Channel High-Recall Retrieval**: Dense vector search (ChromaDB HNSW, cosine similarity) + Lexical keyword search (BM25L) retrieving Top 15 candidate chunks per channel.
  2. **Reciprocal Rank Fusion (RRF k=60)**: Coarse-cut rank aggregation normalizing scores into a unified Top 15 candidate pool.
  3. **Groq Neural Cross-Attention Reranker (`openai/gpt-oss-120b`)**: Deep cross-attention relevance scoring (0.0 to 10.0) selecting the Top 3 precision chunks with automatic RRF fallback.
  4. **Grounded Synthesis (Gemini Flash Lite)**: Zero-hallucination generation with LangSmith observability and verified source citations.
- **Strict Multi-Tenant & Session Isolation**: Complete isolation between users and sessions across database metadata, disk persistence, and vector store compound filters (`user_id` and `session_id`).
- **Dynamic Thought & Retrieval Phase Indicator**: Animated rotating star (`✦ Searching... Querying... Synthesizing...`) providing real-time feedback during generation.
- **Desktop & Mobile Responsive Experience**:
  - **Desktop Viewports**: Pinned 2-column layout with center workspace and permanent right Session Hub sidebar.
  - **Mobile Viewports (iOS & Android)**: Collapsible slide-over drawer with high-contrast close button, outside backdrop dismissal, condensed responsive header, and 16px input scaling to prevent mobile browser auto-zoom.
- **DeepEval Evaluation Benchmarks**: Integrated evaluation suite testing Retrieval Recall, Precision, Faithfulness, Hallucination, and Answer Relevancy.
- **Enterprise Security & Guardrails**:
  - Sliding-window rate limiting on `/ask` (20 req / 60s per client IP) with standard `Retry-After` headers and UI cooldown countdown.
  - Standardized enterprise 9-column logging (`Timestamp | Level | Phase | request_id | user_id | session_id | Module | Message | Metadata`), daily date-partitioned storage (`logs/YYYY-MM-DD/`) with 7-day retention, and zero-secret credential masking.
  - SHA-256 deduplication and RFC 7807 sanitized error shielding.
  - 100% test pass rate across **67 comprehensive unit, integration, and benchmark tests** (67 passed, 1 skipped).

---

## 🏗️ Architecture Overview

```text
                               +-----------------------------+
                               |     REACT 18 + VITE CLIENT  |
                               |    (Desktop Split / Mobile) |
                               |  - Grounded Chat & Search   |
                               |  - Document Hub Ingestion   |
                               |  - Session Intelligence Hub |
                               +--------------+--------------+
                                              |
                                              v HTTP REST (port 8000)
                                              Headers: X-User-ID, X-Session-ID
+-----------------------------------------------------------------------------------------------+
|                                      FASTAPI BACKEND                                          |
|                                                                                               |
|  [app/core/] ────────► config.py | logging.py (9-Column Tracing & Masking) | rate_limiter.py  |
|                                                                                               |
|  [app/api/] ─────────► routes.py | dependencies.py                                            |
|                                                                                               |
|  [app/services/] ────► document_service.py | chat_service.py | job_manager.py (Async Ingestion)│
|                                                                                               |
|  [app/ingestion/] ───► validator.py | storage.py | extractor.py (PyMuPDF + python-docx)       |
|                                                                                               |
|  [app/rag/] ─────────► chunker.py | embedder.py | retriever.py | reranker.py | synthesizer.py |
|                                                                                               |
|  [app/providers/] ───► llm.py (Gemini/Groq) | embeddings.py (Gemini 3072-d) | vector_store.py  |
|                                                                                               |
|  [app/db/] ──────────► models.py | database.py (aiosqlite) | repositories.py                  |
|                                                                                               |
|  [data/] ────────────► knowledge_base.db | chroma_db/ | uploads/{user}_{session}_{date}/      |
|                                                                                               |
|  [logs/] ────────────► logs/YYYY-MM-DD/{application.log, error.log, access.log} (7d retention)|
+-----------------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
enterprise-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── api/                 # REST routes (/health, /documents, /documents/upload, /ask)
│   │   ├── core/                # config.py, logging.py, exceptions.py, rate_limiter.py
│   │   ├── db/                  # database.py, models.py, repositories.py
│   │   ├── ingestion/           # validator.py, storage.py, extractor.py (PyMuPDF + python-docx)
│   │   ├── providers/           # llm.py, embeddings.py, vector_store.py (ChromaDB)
│   │   ├── rag/                 # chunker.py, retriever.py, reranker.py, synthesizer.py, evaluation.py
│   │   ├── schemas/             # Pydantic v2 schemas: common.py, document.py, rag.py
│   │   ├── services/            # document_service.py, chat_service.py, job_manager.py
│   │   └── main.py              # Application entrypoint, CORS, lifespan, middleware
│   ├── logs/                    # Daily date-partitioned logs (logs/YYYY-MM-DD/, 7-day retention)
│   ├── tests/                   # Pytest test suite (67 passed, 1 skipped unit & integration tests)
│   ├── streamlit_app.py         # Lightweight Streamlit reference client
│   ├── ARCHITECTURE.md          # Backend technical architecture specification
│   └── README.md                # Backend service setup and API reference
│
├── frontend/
│   ├── src/
│   │   ├── api/                 # Centralized API client & typed endpoints
│   │   ├── components/
│   │   │   ├── chat/            # ChatView, ChatMessage, QueryInput, RateLimitCountdown
│   │   │   ├── documents/       # DocumentsView, FileDropzone, DocumentTable, DocumentPreviewModal
│   │   │   ├── Header.jsx       # Responsive brand header, status pill & drawer toggle
│   │   │   ├── UserSessionSidebar.jsx  # Pinned desktop sidebar & mobile slide-out drawer
│   │   │   ├── ConfirmModal.jsx # In-app confirmation dialog
│   │   │   └── Toast.jsx        # Non-intrusive notification banners
│   │   ├── utils/               # Session ID generation, formatters
│   │   ├── App.jsx              # App shell, tab routing, drawer state
│   │   ├── App.css              # Custom properties, glassmorphism, responsive styles
│   │   └── main.jsx             # React DOM entrypoint
│   ├── index.html               # HTML5 entrypoint with viewport-fit=cover
│   ├── vite.config.js           # Vite dev proxy configuration
│   ├── ARCHITECTURE.md          # Frontend architecture specification
│   └── README.md                # Frontend setup and component documentation
│
├── data/                        # Persistent root data directory
│   ├── knowledge_base.db        # SQLite relational database (documents, chunks, question history)
│   ├── chroma_db/               # ChromaDB persistent vector collection
│   └── uploads/                 # Session-partitioned document storage
│
├── AI_USAGE.md                  # AI coding tools disclosure, design influences & rejected suggestions
├── ARCHITECTURE.md              # System architecture, data flow & 10K/100K/1M scalability note
├── README.md                    # Root project overview & run guide (this document)
└── .env.example                 # Standardized environment configuration template
```

---

## 🚀 Quickstart & Setup Guide

Follow these steps to set up and run the entire platform locally from scratch.

---

### Step 1: Prerequisites & Tooling Setup

Ensure you have the required runtimes installed:
- **Python 3.12+**
- **Node.js 18+** & **npm**

#### Install `uv` (Fast Python Package & Virtual Environment Manager)
If `uv` is not already installed on your machine, install it via the official installer:

- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  *(Or via Homebrew: `brew install uv`)*

- **Windows**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **Via Pip**:
  ```bash
  pip install uv
  ```

Verify your installation:
```bash
uv --version
```

---

### Step 2: Configure Environment Variables

From the repository root, copy the environment template to create your `.env` file:

```bash
cp .env.example .env
```

Open `.env` and fill in your active API credentials:

```ini
# Standardized AI Model Configuration
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-flash-lite-latest

# Groq Neural Cross-Attention Reranker
RERANKER_TYPE=groq
GROQ_RERANKER_MODEL=openai/gpt-oss-120b
RETRIEVAL_TOP_K=15
FINAL_TOP_N=3

# External API Keys (Required)
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# LangSmith Observability & Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://aws.api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Enterprise-Knowledge-Assistant
```

---

### Step 3: Setup & Start the Backend Service

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment & install all dependencies**:
   Run `uv sync` to automatically create an isolated Python 3.12 `.venv` and install all required packages from `pyproject.toml` and `uv.lock`:
   ```bash
   uv sync
   ```

   *(Optional) If you prefer activating the virtual environment manually:*
   - macOS / Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`

3. **Launch the FastAPI backend server**:
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Verify Backend Health**:
   - **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc API Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - **Health Check API**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

### Step 4: Setup & Start the React Frontend

In a **second terminal window**, start the React single-page application:

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev
   ```

4. **Access the Application**:
   Open **[http://localhost:5173](http://localhost:5173)** in your browser (or your local IP address for mobile viewports).

   > The built-in Vite reverse proxy in `vite.config.js` automatically forwards `/api` and `/health` requests to `http://localhost:8000`.

---

### Step 5: (Alternative) Run Streamlit Reference Client

If you prefer running the lightweight Python Streamlit interface instead of the React client:

1. In a terminal window from the `backend/` directory:
   ```bash
   cd backend
   uv run streamlit run streamlit_app.py
   ```

2. Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Running Automated Tests

The backend test suite contains **68 comprehensive unit, integration, and benchmark tests** (67 passed, 1 skipped) covering file validation, PyMuPDF / python-docx extraction, rate limiting, session isolation, RRF ranking, Groq reranker, asynchronous background ingestion (HTTP 202), SSE streaming, and DeepEval metrics:

```bash
cd backend
uv run pytest
```

Output:
```text
================== 67 passed, 1 skipped, 8 warnings in 13.39s ==================
```

To build and validate the production frontend bundle:
```bash
cd frontend
npm run build
```

---

## 🔌 REST API Reference & Exercising Endpoints

The FastAPI backend exposes versioned REST endpoints (`/api/v1/`) with multi-tenant session isolation:

| Method | Endpoint | Description | Request Body / Params | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, active AI provider inspect, index statistics | None | `HealthResponse` (200) |
| `POST` | `/api/v1/documents/upload` | **Synchronous Batch Ingestion**: Ingest up to 5 files (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.csv`, max 5MB each) | `multipart/form-data` (`files`) | `MultiFileUploadResponse` (201) |
| `POST` | `/api/v1/documents/upload?background=true` | **Asynchronous Decoupled Ingestion**: Enqueues indexing into background worker, returns job tracker URLs | `multipart/form-data` (`files`), Query: `background=true` | `AsyncIngestionJobResponse` (202 Accepted) |
| `GET` | `/api/v1/documents/jobs/{job_id}/events` | **Real-Time Telemetry Stream**: Live SSE event stream emitting file progress, stage, and completion status | Path: `job_id` | `text/event-stream` (SSE Events) |
| `GET` | `/api/v1/documents/jobs/{job_id}` | **Job Status Snapshot**: Poll current ingestion status, processed file count, and errors | Path: `job_id` | `JobStatusResponse` (200) |
| `POST` | `/api/v1/documents` | Create raw text/markdown document directly in knowledge base | JSON (`title`, `content`, `session_id`, `user_id`) | `DocumentResponse` (201) |
| `GET` | `/api/v1/documents` | List indexed documents for the active tenant session | Query: `limit`, `offset`, `session_id`, `user_id`, `as_list` | `DocumentListResponse` (200) |
| `GET` | `/api/v1/documents/{document_id}` | Inspect document details, total pages, and chunk breakdown | Path: `document_id` | `DocumentResponse` (200) |
| `DELETE` | `/api/v1/documents/{document_id}` | Cascade purge: deletes document record, chunk metadata, and ChromaDB vectors | Path: `document_id` | Status 204 No Content |
| `POST` | `/api/v1/ask` | 3-stage grounded question answering with reranking and citations | JSON (`question`, `session_id`, `user_id`) | `AskResponse` (200) |
| `GET` | `/api/v1/history` | Retrieve persistent Q&A conversation and citation history for tenant session/user | Query: `session_id`, `user_id`, `limit` | `QuestionHistoryResponse` (200) |
| `DELETE` | `/api/v1/history` | Clear Q&A conversation history for tenant session or user | Query: `session_id`, `user_id` | Status 204 No Content |

### Quick cURL Examples

```bash
# 1. Check Service Health
curl -s http://localhost:8000/api/v1/health | jq .

# 2. Upload Document with Asynchronous Decoupled Ingestion (HTTP 202 Accepted)
curl -X POST "http://localhost:8000/api/v1/documents/upload?background=true" \
  -H "X-Session-ID: sess_demo" \
  -F "files=@sample.pdf"

# 3. Stream Real-Time Ingestion Progress over Server-Sent Events (SSE)
curl -N "http://localhost:8000/api/v1/documents/jobs/{job_id}/events"

# 4. Ask Grounded Question with Citations
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: sess_demo" \
  -d '{"question": "Summarize the primary revenue trends."}'

# 5. Retrieve Q&A Conversation & Source Citation History
curl -s "http://localhost:8000/api/v1/history?session_id=sess_demo" | jq .

# 6. Clear Session Conversation History
curl -X DELETE "http://localhost:8000/api/v1/history?session_id=sess_demo"
```

---

## 🤖 AI Coding Tools Disclosure

In accordance with the candidate challenge guidelines, this repository includes a dedicated disclosure document:
👉 **[AI_USAGE.md](AI_USAGE.md)**

It documents:
1. Specific AI tools utilized (**Antigravity IDE**, Google Gemini 2.5 Flash, Gemini Embedding-2, Groq Llama 3 / GPT-OSS-120B).
2. Key architectural influences (3-stage hybrid RAG funnel, decoupled asynchronous ingestion with SSE telemetry, enterprise 9-column context-aware logging, and secret masking).
3. Concrete examples of AI suggestions reviewed, changed, or rejected (e.g., rejecting concurrent `asyncio.gather` file ingestion due to Gemini API rate limits and SQLite locking, redesigning blocking modal overlays into non-intrusive in-table status tracking, and normalizing RRF scores).
4. Engineering ownership and technical accountability affirmation.

---

| Requirement | Implementation Details |
|---|---|
| **Multi-File Ingestion** | Batch upload up to 5 files (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.csv`, max 5MB each) with asynchronous decoupled indexing (`HTTP 202 Accepted`), real-time Server-Sent Events (SSE) telemetry, polling fallback, and individual document failure isolation. |
| **Hybrid RAG Funnel** | Vector Store Dense Search (3072-d) + BM25L Lexical Search → RRF k=60 → Groq LLM Cross-Attention Reranking → Grounded Synthesis. |
| **Multi-Tenant Isolation** | Scoped database records and compound vector store filters on `{user_id, session_id}`. |
| **Grounded Citations** | Expandable source cards with page numbers, chunk snippets, retrieval channels, and relevance scores. |
| **Thinking UI** | Rotating thought phase indicator (`✦ Searching... Querying... Synthesizing...`). |
| **Mobile Responsiveness** | Slide-over drawer with close button and backdrop dismissal across iOS and Android mobile viewports without breaking desktop split view. |
| **Observability & Tracing** | Native LangSmith tracing with zero-secret log masking (`sk-***`, `gsk_***`, `AIzaSy***`). |
| **Scalability Note** | Concrete architecture strategies for 10K, 100K, and 1M concurrent requests documented in `ARCHITECTURE.md`. |
