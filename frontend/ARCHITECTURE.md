# Frontend Architecture Specification — Enterprise Knowledge Assistant Client

This document outlines the architectural blueprint, component design, responsive drawer layout, state management, and error-handling strategies for the **Enterprise Knowledge Assistant React Frontend**.

---

## 1. Architectural Philosophy & Technology Stack

The frontend is constructed as a modern, lightweight, production-grade Single-Page Application (SPA) designed to interface with the FastAPI RAG backend.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        React SPA (Vite + React 18)                     │
│                                                                        │
│  ┌────────────────────┐   ┌───────────────────┐   ┌─────────────────┐  │
│  │     App Shell      │   │   Chat & Q&A      │   │  Document Hub   │  │
│  │ (Session, Health)  │   │  (Grounded View)  │   │  (Upload & List)│  │
│  └─────────┬──────────┘   └─────────┬─────────┘   └────────┬────────┘  │
│            │                        │                      │           │
│  ┌─────────┴────────────────────────┴──────────────────────┴────────┐  │
│  │                   Centralized API Client Layer                    │  │
│  │        (Fetch Wrapper, Tenant Headers, RFC 7807 Parsing)         │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────────┼──────────────────────────────────┘
                                      │ HTTP / JSON (Port 8000)
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  FastAPI Backend Server (Port 8000)                    │
│      GET /health | POST /api/v1/ask | POST /api/v1/documents ...       │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Technologies
- **Runtime & Build Tool**: [Vite](https://vitejs.dev/) + React 18.
- **Styling Architecture**: Vanilla CSS using CSS Custom Properties (CSS variables) for modern typography (Inter), glassmorphism, responsive flex/grid layouts, and sleek dark mode integration with zero UI library lock-in.
- **Networking**: Native `fetch` with `AbortController` timeouts, automatic tenant headers (`X-User-ID`, `X-Session-ID`), and RFC 7807 sanitized error shielding.
- **State Management**: Reactive React state hooks (`useState`, `useCallback`, `useEffect`) with `sessionStorage` persistence for tab-scoped isolation.

---

## 2. Component Hierarchy & Layout Modes

```text
App.jsx (Root Shell)
├── Header.jsx (Responsive Brand, Health Pill, Tabs Nav, Sidebar Toggle)
├── workspace-layout (Container)
│   ├── main-content-zone (Active Tab Content)
│   │   ├── ChatView.jsx (when activeTab === "chat")
│   │   │   ├── chat-initial-screen (Empty state search console)
│   │   │   └── chat-active-screen (Conversation stream)
│   │   │       ├── chat-sub-header (Document count badge & Clear Chat)
│   │   │       ├── chat-history (Scrollable messages container)
│   │   │       │   ├── ChatMessage.jsx (User / Assistant bubble & Expandable Citations)
│   │   │       │   └── ThinkingIndicator (Rotating star ✦ with live phases)
│   │   │       ├── RateLimitCountdown.jsx (Visual 429 countdown banner)
│   │   │       └── QueryInput.jsx (Fixed bottom query textarea & submit button)
│   │   │
│   │   └── DocumentsView.jsx (when activeTab === "documents")
│   │       ├── doc-actions-bar (Section title & Refresh button)
│   │       ├── ingest-mode-tabs (Segmented switcher: Files vs Manual Text)
│   │       │   ├── FileDropzone.jsx (Drag & drop multi-format batch upload)
│   │       │   └── manual-doc-form (Title and content input form)
│   │       ├── DocumentTable.jsx (Session knowledge inventory with inline telemetry, stage tags & filter)
│   │       └── DocumentPreviewModal.jsx (Full-text inspector dialog)
│   │
│   ├── sidebar-backdrop (Mobile blur overlay, dismisses drawer on outside tap)
│   └── UserSessionSidebar.jsx (Pinned 2-column on desktop / Slide-over drawer on mobile)
│       ├── sidebar-mobile-header (Mobile-only ✕ close button)
│       ├── user-profile-card (Avatar & editable display name)
│       ├── session-intelligence-card (Session ID, copy button & session switcher)
│       └── session-documents-card (Active document index with + Add shortcut)
│
├── ConfirmModal.jsx (In-app confirmation dialog for Clear Chat / actions)
└── Toast.jsx (Floating non-intrusive notification banner)
```

---

## 3. Responsive Design Architecture: Desktop vs Mobile

### A. Desktop Workspace Layout (`> 820px`)
- **2-Column Split View**:
  - Center Workspace (`.main-content-zone`): Takes flexible remaining space (`flex: 1`).
  - Session Sidebar (`.user-session-sidebar`): Fixed width (`350px`), pinned to the right edge.
- **Header**: Displays full brand title (`ENTERPRISE INTELLIGENCE Enterprise Knowledge Assistant`), full tab labels (`💬 AI Chatbot`, `📁 Document Hub`), and `System Ready` status text.
- **Sidebar Elements**: `.sidebar-mobile-header` and `.sidebar-backdrop` are strictly hidden (`display: none !important`).

### B. Mobile Overlay Drawer Architecture (`<= 820px` & Mobile Viewports)
- **Default Closed**: Sidebar state initializes to `false` on mobile viewports so that the main screen is never obstructed when opening the app.
- **Slide-Over Drawer**:
  - Styled as an overlay drawer (`position: fixed; top: 0; right: 0; bottom: 0; width: 88vw; max-width: 370px; z-index: 1001; height: 100dvh;`).
  - Slides in smoothly with cubic-bezier animation (`animation: slideInDrawer 0.25s`).
- **Dismissal Controls**:
  - High-contrast `✕` close button in `.sidebar-mobile-header`.
  - Dark blur backdrop (`.sidebar-backdrop`, `background: rgba(10, 15, 29, 0.72); backdrop-filter: blur(5px); z-index: 1000;`) dismissing the drawer on outside tap.
  - Auto-dismisses whenever the user switches tabs or selects a session/document.
- **Condensed Header**:
  - Brand title condenses to `Knowledge Assistant`.
  - Tabs condense to `💬 Chat` and `📁 Docs`.
  - Status pill condenses to a pulsing emerald dot (`●`).
  - Toggle button (`◫`) remains accessible in the top right.
- **Mobile Browser Auto-Zoom Prevention**:
  - All input fields and textareas (`.query-textarea`, `.table-search-input`, `.form-input`) are set to `font-size: 16px !important` on mobile, preventing mobile browsers from auto-zooming into the viewport.
  - `viewport-fit=cover` handles mobile device notches, Dynamic Island, and bottom safe areas.
  - Document table includes `min-width: 620px` inside `.table-scroll-wrapper` to enable smooth horizontal scrolling without crushing column layout.

---

## 4. Multi-Tenant State & Storage Management

All tenant identifiers are generated once per browser tab and persisted in `sessionStorage`:

```javascript
// frontend/src/utils/session.js
export function getOrCreateSessionId() {
  let sId = sessionStorage.getItem("verdeshell_session_id");
  if (!sId) {
    sId = "sess_" + Math.random().toString(36).substring(2, 12);
    sessionStorage.setItem("verdeshell_session_id", sId);
  }
  return sId;
}

export function getOrCreateUserId() {
  let uId = sessionStorage.getItem("verdeshell_user_id");
  if (!uId) {
    uId = "user_" + Math.random().toString(36).substring(2, 12);
    sessionStorage.setItem("verdeshell_user_id", uId);
  }
  return uId;
}
```

- **Per-Session Chat History**: Chat messages are persisted to `sessionStorage` under `chat_user_{userId}_sess_{sessionId}`, allowing instant conversation restore when switching between sessions.
- **Automatic Request Headers**: `api/client.js` injects `X-User-ID` and `X-Session-ID` into every HTTP request.

---

## 5. In-App Confirmation Modal & Toast System

- **Zero Native Browser Popups**: Browser-native `window.confirm()` and `window.alert()` popups have been completely replaced with a custom in-app `ConfirmModal.jsx` featuring dangerous/safe action styling, modal subtitle, and keyboard escape dismissal.
- **Toast Notifications**: Non-intrusive alert toasts (`Toast.jsx`) anchored to the bottom-left of the viewport with a 3.5-second auto-diminish progress bar and manual close button.

---

## 6. Real-Time Asynchronous Ingestion & Telemetry Architecture

To handle multi-file uploads (up to 5 files, 5MB each) without blocking user interactions or triggering HTTP connection timeouts:

```text
[FileDropzone] ──► 1. Optimistic Rows Inserted (○ Queued)
               ──► 2. POST /documents/upload?background=true ──► HTTP 202 Accepted { job_id }
               ──► 3. Open SSE Stream: GET /documents/jobs/{job_id}/events
                      │
                      ├─► Event { stage: 'EXTRACTION', current_file: 'doc1.pdf', percent: 25 }
                      │   └─► Row updates to: ● Processing 25% + Stage: EXTRACTION + micro-bar
                      │
                      ├─► Event { stage: 'EMBEDDING', current_file: 'doc1.pdf', percent: 60 }
                      │   └─► Micro-bar slides smoothly to 60%
                      │
                      ├─► Event { results: [...], errors: [...] }
                      │   ├─► doc1.pdf transitions to ● Indexed (12 chunks)
                      │   └─► doc2.pdf (if corrupt) transitions to ● Failed / Skipped
                      │
                      └─► Fallback: If SSE fails, auto-switches to GET /jobs/{job_id} every 1.2s
```

---

## 7. Frontend Architectural Trade-Offs & Design Rationale

| Area | Decision Made | Rationale (Why this choice in V1) | Trade-Offs / Limitations | Production Evolution Path |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion Feedback** | **In-Table Real-Time Status** (eliminating overlays) | Unobtrusive UX. The user can browse existing documents, read previews, or search while files index in the background. | Table rows must manage optimistic temporary state before backend persistence is confirmed. | Maintain optimistic state manager with WebSocket reconnect and persistent job history across page reloads. |
| **State Management** | **React Hooks + `sessionStorage`** | Zero dependencies, deterministic tab isolation, instant boot time, zero Redux/Zustand boilerplate. | State does not persist across separate browser windows or devices unless synced to a remote database. | Migrate to Zustand with IndexedDB or centralized user profile synchronization via GraphQL / REST. |
| **Telemetry Transport** | **Server-Sent Events (SSE)** with polling fallback | Lightweight unidirectional text stream natively supported via standard HTTP/2 and simple `fetch` streaming; avoids WebSocket handshake overhead. | Unidirectional only (client cannot send control messages over the same stream without separate POST). | Full-duplex WebSockets for collaborative multi-user editing, shared annotations, and interactive indexing cancellation. |
| **Styling Architecture** | **Vanilla CSS Custom Properties (CSS Variables)** | Complete design control, instant compilation, zero bundle size overhead, full adherence to custom design tokens. | Requires maintaining global theme classes rather than rapid utility-class prototyping. | Maintain shared design tokens packaged as a private npm design system library (`@company/design-tokens`). |
| **File Persistence** | **Ephemeral Optimistic Prepend** | Instantaneous zero-latency feedback on file drop without waiting for backend network roundtrip. | Brief UI flicker if network fails before HTTP 202 is received (handled via error catch rollback). | Optimistic queue with retry actions and offline local buffer via IndexedDB. |

