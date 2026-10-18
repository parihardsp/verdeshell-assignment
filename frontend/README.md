# Enterprise Knowledge Assistant — React Frontend

A production-grade, modern React Single-Page Application (SPA) designed to interface with the Enterprise RAG backend service. Built with **React 18** and **Vite** with zero third-party component library bloat, featuring high-fidelity dark mode, glassmorphism, and complete mobile and desktop responsiveness.

---

## 🌟 Key Features

### 1. Grounded Question Answering (`AI Chatbot` Tab)
- **Dynamic Thought Phase Indicator**: Live animated rotating star (`✦ Searching... Querying... Synthesizing...`) displayed while the backend executes hybrid retrieval, neural reranking, and grounded synthesis.
- **Fixed Bottom Query Bar**: The query input box remains pinned at the bottom of the screen with automatic scroll-to-bottom as conversation flows.
- **Verified Source Citations**: Expandable source citation cards (collapsed by default, click to expand) showing document titles, page numbers, text snippets, retrieval channels (`Hybrid`, `Dense`, `BM25`), and Groq reranker relevance scores.
- **Confidence & Latency Badges**: Real-time round-trip latency (`⚡ xx ms`) and grounded confidence ratings (`🎯 xx%`).
- **HTTP 429 Rate Limit Visual Countdown**: Live countdown timer (`Retry in Xs`) with automatic re-activation when the rate limit window resets.
- **In-App Confirmation Dialog**: Beautiful modal dialog for clearing conversation threads, replacing native browser `window.confirm` popups.

### 2. Document Hub & Ingestion (`Document Hub` Tab)
- **Dual Ingestion Modes**:
  1. **Upload Document Files**: Drag-and-drop dropzone supporting batch upload of up to 5 files (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.csv`, max 5MB each) with asynchronous decoupled background indexing (`HTTP 202 Accepted`).
  2. **Manual Document Entry**: Form to add knowledge articles directly with custom title and content.
- **In-Table Real-Time Ingestion Status (Zero Disruptive Overlays)**:
  - **Optimistic Table Rows**: Dropping files immediately inserts pending rows at the top of the table in `○ Queued` state.
  - **Live Telemetry (SSE + Polling Fallback)**: The actively indexing file renders `● Processing {percent}%` with an animated pulsing amber beacon, pipeline stage badge (`Stage: EXTRACTION / CHUNKING / EMBEDDING / VECTOR_DB`), and smooth micro-progress bar.
  - **Non-Blocking Operation**: Eliminates all blocking modals or banner overlays, allowing users to freely browse, search, and inspect existing documents during ingestion.
  - **Granular Error Isolation**: Failed documents display `● Failed / Skipped` with specific error tooltips without failing or blocking other documents in the batch.
- **Document Inventory Table**:
  - Displays all indexed documents for the active session.
  - Format-specific icons: `📄` PDF, `📘` Word (`.docx`, `.doc`), `📝` Markdown, `📃` Plain Text, `📊` CSV.
  - Chunk counts, total pages, and indexing timestamps.
  - Horizontal scroll protection (`min-width: 620px`) ensuring table readability on mobile devices.
  - Search filter input with instant client-side filtering.
- **Document Inspector Modal**: View full extracted text, chunks, and metadata directly from SQLite.

### 3. User & Session Intelligence Hub Sidebar
- **User Profile**: Initials avatar and inline display name editing.
- **Active Session Management**: Copyable Session ID, session switcher dropdown, and "+ New Session" button for isolated testing.
- **Indexed Document List**: Displays all active documents in the session with quick click-to-preview links and "+ Add" shortcut.

### 4. Mobile Responsiveness & Viewport Adaptations (iOS & Android)
- **Mobile Drawer Architecture**:
  - On screens <= 820px, the sidebar transforms from a desktop 2-column pinned panel into a slide-over fixed drawer (`88vw`, max 370px).
  - Defaults to **closed** on mobile so it never covers the main chat or document content upon opening.
  - High-contrast top close button (`✕`) inside `.sidebar-mobile-header`.
  - Dark blur backdrop overlay (`.sidebar-backdrop`) closing the drawer on outside tap.
  - Auto-dismisses when switching tabs or selecting a session/document.
- **Responsive Header**:
  - Dynamically condenses brand title to `Knowledge Assistant` on mobile screens.
  - Abbreviates tabs to `💬 Chat` and `📁 Docs`.
  - Keeps the Session Hub toggle button (`◫`) permanently accessible in the top right.
  - Condenses health pill to a pulsing status dot.
- **Mobile Browser Auto-Zoom Prevention**:
  - Input fields and textareas are explicitly scaled to `16px` on mobile, preventing mobile web browsers (such as iOS Safari) from auto-zooming and distorting page layout.
  - Safe area handling via `viewport-fit=cover` in `index.html`.

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and `npm`.
- FastAPI backend running on `http://localhost:8000`.

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```

The app will start with Vite Hot Module Replacement (HMR) at:
👉 **`http://localhost:5173`**

The development server includes a built-in reverse proxy in `vite.config.js` forwarding `/api` and `/health` requests directly to `http://localhost:8000`.

### 3. Production Build
```bash
npm run build
npm run preview
```

---

## 📁 Directory Structure

```text
frontend/src/
├── api/
│   ├── client.js                # Base fetch wrapper with tenant headers & timeout handling
│   └── endpoints.js             # API calls (askQuestion, uploadFiles, getDocuments, deleteDocument)
├── components/
│   ├── chat/
│   │   ├── ChatView.jsx         # Chat layout (initial screen vs active thread)
│   │   ├── ChatMessage.jsx      # Message bubble with expandable source cards
│   │   ├── QueryInput.jsx       # Fixed bottom query textarea and send button
│   │   └── RateLimitCountdown.jsx # Rate limit cooldown countdown banner
│   ├── documents/
│   │   ├── DocumentsView.jsx    # Dual ingestion mode tab switcher
│   │   ├── FileDropzone.jsx     # Drag & drop upload zone with multi-format validation
│   │   ├── DocumentTable.jsx    # Document table with in-table real-time telemetry, stage tags, and filter
│   │   └── DocumentPreviewModal.jsx # Full-screen document inspector dialog
│   ├── Header.jsx               # Responsive brand header & mobile drawer toggle
│   ├── UserSessionSidebar.jsx   # Pinned desktop sidebar & mobile slide-out drawer
│   ├── ConfirmModal.jsx         # In-app confirmation dialog
│   └── Toast.jsx                # Non-intrusive alert notification banners
├── utils/
│   ├── formatters.js            # Date and byte formatters
│   └── session.js               # Tab-scoped sessionStorage persistence
├── App.jsx                      # Main application shell with drawer & tab routing
├── App.css                      # Design system, glassmorphism, and responsive media queries
└── main.jsx                     # React DOM root entry point
```
