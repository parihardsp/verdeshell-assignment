import React, { useEffect, useState } from "react";
import { getDocumentDetails } from "../../api/endpoints";

export default function DocumentPreviewModal({ documentId, onClose }) {
  const [doc, setDoc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!documentId) return;
    setLoading(true);
    setError(null);

    getDocumentDetails(documentId)
      .then((data) => {
        setDoc(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load document content.");
        setLoading(false);
      });
  }, [documentId]);

  if (!documentId) return null;

  const handleCopy = () => {
    if (doc?.content) {
      navigator.clipboard.writeText(doc.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-container doc-preview-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-header">
          <div className="doc-preview-title-col">
            <span className="modal-subtitle">DOCUMENT INSPECTOR</span>
            <h2 className="modal-title">{doc?.title || "Document Details"}</h2>
          </div>
          <button
            type="button"
            className="icon-btn close-modal-btn"
            onClick={onClose}
            title="Close Preview"
          >
            ✕
          </button>
        </div>

        {loading ? (
          <div className="modal-loading-box">
            <div className="simple-spinner"></div>
            <span>Fetching parsed document from SQLite...</span>
          </div>
        ) : error ? (
          <div className="modal-error-box">
            <span className="error-icon">⚠️</span>
            <p>{error}</p>
          </div>
        ) : doc ? (
          <>
            <div className="doc-meta-strip">
              <span className="meta-badge ok">Status: {doc.status}</span>
              <span className="meta-badge">Chunks: {doc.chunk_count || 0}</span>
              {doc.total_pages && (
                <span className="meta-badge">Pages: {doc.total_pages}</span>
              )}
              {doc.file_name && (
                <span className="meta-badge file-name" title={doc.file_name}>
                  File: {doc.file_name}
                </span>
              )}
              <span className="meta-badge id" title={`ID: ${doc.document_id}`}>
                ID: {doc.document_id}
              </span>
            </div>

            <div className="doc-content-viewer-box">
              <div className="viewer-toolbar">
                <span className="viewer-char-count">
                  {doc.content ? `${doc.content.length.toLocaleString()} characters` : "No text content"}
                </span>
                <button
                  type="button"
                  className="btn-secondary copy-content-btn"
                  onClick={handleCopy}
                  disabled={!doc.content}
                >
                  {copied ? "✓ Copied Content" : "📋 Copy Full Text"}
                </button>
              </div>

              <div className="doc-text-scroll">
                <pre className="doc-pre-text">
                  {doc.content || "No textual content available for this document."}
                </pre>
              </div>
            </div>
          </>
        ) : null}

        <div className="modal-actions-footer">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
