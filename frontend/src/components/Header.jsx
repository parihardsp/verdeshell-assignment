import React from "react";

export default function Header({
  health,
  docCount = 0,
  activeTab,
  setActiveTab,
  onToggleSidebar,
  sidebarOpen,
}) {
  const isOnline = health && (health.status === "healthy" || health.status === "ok");

  return (
    <header className="app-header">
      <div className="header-brand-block">
        <div className="brand-badge-icon">🌿</div>
        <div className="brand-titles">
          <div className="brand-supertitle">ENTERPRISE INTELLIGENCE</div>
          <h1 className="brand-main-title">
            <span className="desktop-title">Enterprise Knowledge Assistant</span>
            <span className="mobile-title">Knowledge Assistant</span>
          </h1>
        </div>
      </div>

      <nav className="header-tabs-nav" aria-label="Main Navigation">
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          <span className="tab-icon">💬</span>
          <span className="tab-label desktop-label">AI Chatbot</span>
          <span className="tab-label mobile-label">Chat</span>
        </button>
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "documents" ? "active" : ""}`}
          onClick={() => setActiveTab("documents")}
        >
          <span className="tab-icon">📁</span>
          <span className="tab-label desktop-label">Document Hub</span>
          <span className="tab-label mobile-label">Docs</span>
          {docCount > 0 && <span className="tab-count-badge">{docCount}</span>}
        </button>
      </nav>

      <div className="header-meta-controls">
        <div
          className={`health-pill ${isOnline ? "health-online" : "health-offline"}`}
          title={isOnline ? "Backend and Vector Store healthy" : "Backend unreachable"}
        >
          <span className="health-dot"></span>
          <span className="health-text">
            {isOnline ? "System Ready" : "System Offline"}
          </span>
        </div>

        <button
          type="button"
          className={`sidebar-toggle-btn ${sidebarOpen ? "active" : ""}`}
          onClick={onToggleSidebar}
          title={sidebarOpen ? "Hide session panel" : "Show session panel"}
        >
          <span className="sidebar-toggle-icon">◫</span>
          <span className="sidebar-toggle-label">Session Hub</span>
        </button>
      </div>
    </header>
  );
}
