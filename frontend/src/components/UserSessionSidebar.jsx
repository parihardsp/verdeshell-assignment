import React, { useState } from "react";

export default function UserSessionSidebar({
  userId,
  sessionId,
  userDisplayName,
  onUpdateDisplayName,
  documents,
  health,
  onRefreshSession,
  onNewSession,
  onSelectSession,
  availableSessions = [],
  onSelectDocument,
  onSwitchToDocsTab,
  isRefreshing,
  onClose,
}) {
  const [isEditingName, setIsEditingName] = useState(false);
  const [tempName, setTempName] = useState(userDisplayName);
  const [copiedKey, setCopiedKey] = useState(null);

  const handleCopy = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleSaveName = (e) => {
    e.preventDefault();
    if (tempName.trim()) {
      onUpdateDisplayName(tempName.trim());
      setIsEditingName(false);
    }
  };

  const getInitials = (name) => {
    if (!name) return "US";
    const parts = name.trim().split(" ");
    if (parts.length >= 2) {
      return (parts[0][0] + parts[1][0]).toUpperCase();
    }
    return name.slice(0, 2).toUpperCase();
  };

  const indexedDocs = documents.filter((d) => d.status === "indexed");
  const failedDocs = documents.filter((d) => d.status === "failed");
  const totalChunks = indexedDocs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  return (
    <aside className="user-session-sidebar" aria-label="User and Session Intelligence Hub">
      {/* Mobile-Only Drawer Header with prominent close button */}
      <div className="sidebar-mobile-header">
        <div className="sidebar-mobile-header-left">
          <span className="sidebar-mobile-icon">◫</span>
          <span className="sidebar-mobile-title">Session & Intelligence Hub</span>
        </div>
        <button
          type="button"
          className="sidebar-mobile-close-btn"
          onClick={onClose}
          aria-label="Close sidebar panel"
          title="Close session panel"
        >
          ✕
        </button>
      </div>

      {/* 1. User Profile Card */}
      <div className="sidebar-card user-profile-card">
        <div className="user-profile-header">
          <div className="user-avatar" title={`Active User: ${userId}`}>
            {getInitials(userDisplayName)}
          </div>
          <div className="user-meta">
            {isEditingName ? (
              <form onSubmit={handleSaveName} className="name-edit-form">
                <input
                  type="text"
                  value={tempName}
                  onChange={(e) => setTempName(e.target.value)}
                  className="name-edit-input"
                  autoFocus
                  maxLength={30}
                />
                <button type="submit" className="name-edit-save-btn" title="Save name">
                  ✓
                </button>
              </form>
            ) : (
              <div className="name-display-row">
                <h3 className="user-display-name" title={userDisplayName}>
                  {userDisplayName}
                </h3>
                <button
                  type="button"
                  className="icon-btn edit-name-btn"
                  onClick={() => {
                    setTempName(userDisplayName);
                    setIsEditingName(true);
                  }}
                  title="Edit display name"
                >
                  ✎
                </button>
              </div>
            )}
            <div className="tenant-id-row">
              <span className="tenant-tag" title="User Tenant Isolation ID">
                ID: {userId}
              </span>
              <button
                type="button"
                className="icon-copy-btn"
                onClick={() => handleCopy(userId, "user")}
                title="Copy User ID"
              >
                {copiedKey === "user" ? "✓" : "📋"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Active Session Card */}
      <div className="sidebar-card session-manager-card">
        <div className="sidebar-section-title-row">
          <span className="section-title">ACTIVE SESSION</span>
          <div className="session-action-group">
            <button
              type="button"
              className={`sidebar-action-btn refresh-btn ${isRefreshing ? "spin" : ""}`}
              onClick={onRefreshSession}
              title="Refresh conversation & documents for this session"
              disabled={isRefreshing}
            >
              ↻ Refresh
            </button>
            <button
              type="button"
              className="sidebar-action-btn new-session-btn"
              onClick={onNewSession}
              title="Start a fresh, isolated session"
            >
              + New
            </button>
          </div>
        </div>

        <div className="active-session-pill-box">
          <div className="session-pill">
            <span className="session-status-dot pulse-emerald"></span>
            <span className="session-id-text" title={sessionId}>
              {sessionId}
            </span>
          </div>
          <button
            type="button"
            className="icon-copy-btn"
            onClick={() => handleCopy(sessionId, "sess")}
            title="Copy Session ID"
          >
            {copiedKey === "sess" ? "✓" : "📋"}
          </button>
        </div>

        {/* Recent user sessions switcher */}
        {availableSessions.length > 1 && (
          <div className="session-switch-wrapper">
            <label htmlFor="session-switch-select" className="session-switch-label">
              Switch Session:
            </label>
            <select
              id="session-switch-select"
              value={sessionId}
              onChange={(e) => onSelectSession(e.target.value)}
              className="session-switch-select"
            >
              {availableSessions.map((sId) => (
                <option key={sId} value={sId}>
                  {sId === sessionId ? `● ${sId} (Current)` : sId}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* 3. Session Document Index */}
      <div className="sidebar-card session-documents-card">
        <div className="sidebar-section-title-row">
          <div className="sidebar-title-badge-group">
            <span className="section-title">INDEXED DOCUMENTS</span>
            <span className={`section-count-pill ${indexedDocs.length > 0 ? "has-docs" : ""}`}>
              {indexedDocs.length}
            </span>
          </div>
          <button
            type="button"
            className="sidebar-action-btn add-doc-btn"
            onClick={onSwitchToDocsTab}
            title="Upload or add documents to this session"
          >
            + Add
          </button>
        </div>

        <div className="session-doc-list">
          {documents.length === 0 ? (
            <div className="empty-docs-box">
              <div className="empty-docs-icon-badge">
                <span className="empty-docs-icon">📂</span>
              </div>
              <p className="empty-docs-headline">No documents in this session</p>
              <p className="empty-docs-subtext">Add PDFs or markdown notes to ground AI responses.</p>
              <button
                type="button"
                className="sidebar-upload-cta"
                onClick={onSwitchToDocsTab}
              >
                <span>Upload PDFs or notes</span>
                <span className="cta-arrow-icon">→</span>
              </button>
            </div>
          ) : (
            documents.map((doc) => {
              const lowerName = (doc.file_name || "").toLowerCase();
              const docIcon = lowerName.endsWith(".pdf")
                ? "📄"
                : lowerName.endsWith(".docx") || lowerName.endsWith(".doc")
                ? "📘"
                : lowerName.endsWith(".csv")
                ? "📊"
                : lowerName.endsWith(".md")
                ? "📝"
                : "📃";
              const isIndexed = doc.status === "indexed";
              return (
                <div
                  key={doc.document_id}
                  className={`session-doc-item ${isIndexed ? "item-indexed" : "item-failed"}`}
                  onClick={() => onSelectDocument && onSelectDocument(doc.document_id)}
                  title={`${doc.title || doc.file_name} (${doc.chunk_count || 0} chunks) - Click to preview`}
                >
                  <span className="doc-type-icon">{docIcon}</span>
                  <div className="doc-info-col">
                    <span className="doc-title-text">{doc.title || doc.file_name}</span>
                    <span className="doc-meta-subtext">
                      {isIndexed ? (
                        <>
                          <span className="dot-ok">●</span> {doc.chunk_count || 0} chunks
                          {doc.total_pages ? ` • ${doc.total_pages}p` : ""}
                        </>
                      ) : (
                        <span className="dot-err">● Failed / Image-only</span>
                      )}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {indexedDocs.length > 0 && (
          <div className="doc-stats-summary-footer">
            <div className="stat-pill">
              <span className="stat-num">{indexedDocs.length}</span>
              <span className="stat-lbl">Active Docs</span>
            </div>
            <div className="stat-pill">
              <span className="stat-num">{totalChunks}</span>
              <span className="stat-lbl">Vectors</span>
            </div>
            {failedDocs.length > 0 && (
              <div className="stat-pill warn">
                <span className="stat-num">{failedDocs.length}</span>
                <span className="stat-lbl">Skipped</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 4. Enterprise System & Model Telemetry */}
      <div className="sidebar-card telemetry-card">
        <span className="section-title">SYSTEM INFRASTRUCTURE</span>
        <div className="telemetry-rows">
          <div className="telemetry-row">
            <span className="telemetry-key">Embeddings:</span>
            <span className="telemetry-val cyan-glow">
              Gemini 3072d
            </span>
          </div>
          <div className="telemetry-row">
            <span className="telemetry-key">LLM Synthesis:</span>
            <span className="telemetry-val purple-glow">
              {health?.llm_provider || "Groq / Gemini"}
            </span>
          </div>
          <div className="telemetry-row">
            <span className="telemetry-key">Vector Store:</span>
            <span className="telemetry-val">ChromaDB (HNSW)</span>
          </div>
          <div className="telemetry-row">
            <span className="telemetry-key">Multi-Tenant:</span>
            <span className="telemetry-val ok">Isolated</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
