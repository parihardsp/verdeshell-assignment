import React, { useState } from "react";

export default function TextIngestModal({ isOpen, onClose, onSave, isSaving }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim()) {
      setError("Document title is required.");
      return;
    }
    if (!content.trim() || content.trim().length < 10) {
      setError("Content must contain at least 10 characters.");
      return;
    }
    setError("");
    onSave(title.trim(), content.trim());
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">📝 Ingest Raw Text Document</div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#94a3b8",
              fontSize: "1.25rem",
              cursor: "pointer",
            }}
          >
            ×
          </button>
        </div>

        {error && (
          <div style={{ color: "#f87171", fontSize: "0.82rem", marginBottom: "0.75rem" }}>
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Article / Policy Title</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g., Global Remote Work Policy 2026"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={isSaving}
              autoFocus
            />
          </div>

          <div className="form-group">
            <label className="form-label">Document Text Content</label>
            <textarea
              className="form-textarea"
              rows={8}
              placeholder="Paste article text or policy clauses here..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              disabled={isSaving}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem", marginTop: "1rem" }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={isSaving}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSaving}
            >
              {isSaving ? "Chunking & Indexing..." : "Index Document"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
