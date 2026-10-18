import React, { useState } from "react";
import { formatDate } from "../../utils/formatters";

export default function DocumentTable({
  documents = [],
  pendingUploads = [],
  onDelete,
  onPreview,
  isDeletingId,
}) {
  const [filterQuery, setFilterQuery] = useState("");

  // Merge pending uploads that are not yet persisted in documents
  const unpersistedPending = (pendingUploads || []).filter(
    (p) => !documents.some((d) => d.file_name === p.file_name && d.status === "indexed")
  );
  const combinedDocs = [...unpersistedPending, ...(documents || [])];

  if (combinedDocs.length === 0) {
    return (
      <div className="empty-table-state">
        <div className="empty-table-icon">📂</div>
        <div className="empty-table-title">No Documents in This Session</div>
        <div className="empty-table-subtitle">
          Upload documents above to populate the knowledge base.
        </div>
      </div>
    );
  }

  const filteredDocs = combinedDocs.filter((d) => {
    if (!filterQuery.trim()) return true;
    const q = filterQuery.toLowerCase();
    return (
      (d.title || "").toLowerCase().includes(q) ||
      (d.file_name || "").toLowerCase().includes(q) ||
      (d.document_id || d.temp_id || "").toLowerCase().includes(q)
    );
  });

  return (
    <div className="document-table-container">
      <div className="table-header-title-bar">
        <div className="table-header-left">
          <span className="table-title">
            📑 Session Knowledge Inventory ({combinedDocs.length})
          </span>
          <span className="table-subtitle">
            HNSW-indexed in ChromaDB with page boundary metadata
          </span>
        </div>

        <div className="table-filter-box">
          <span className="search-lens-icon">🔍</span>
          <input
            type="text"
            className="table-search-input"
            placeholder="Filter documents..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
          />
          {filterQuery && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => setFilterQuery("")}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="table-scroll-wrapper">
        <table className="doc-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Source File</th>
              <th>Status</th>
              <th>Chunks & Pages</th>
              <th>Indexed At</th>
              <th style={{ textAlign: "right" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredDocs.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", padding: "2rem", color: "#94a3b8" }}>
                  No documents match filter "{filterQuery}"
                </td>
              </tr>
            ) : (
              filteredDocs.map((doc) => {
                const isDeleting = isDeletingId === doc.document_id;
                const isPending = doc.status === "processing" || doc.status === "queued";
                const isIndexed = doc.status === "indexed";
                const isFailed = doc.status === "failed";
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

                const rowKey = doc.document_id || doc.temp_id || doc.file_name;

                return (
                  <tr key={rowKey} className={isPending ? "pending-row" : ""}>
                    <td>
                      <div className="doc-title-cell">
                        <span className="doc-icon">{docIcon}</span>
                        <div>
                          <strong className="doc-title-text">{doc.title}</strong>
                          <div className="doc-id-subtext">{doc.document_id || doc.temp_id || "staging"}</div>
                        </div>
                      </div>
                    </td>
                    <td className="doc-source-cell">
                      {doc.file_name || "Direct Text Article"}
                    </td>
                    <td>
                      {doc.status === "processing" ? (
                        <div className="status-processing-container">
                          <span className="status-pill status-processing">
                            <span className="pulsing-amber-dot">●</span> Processing {doc.percent || 0}%
                          </span>
                          {doc.stage && (
                            <span className="status-stage-label">
                              Stage: {doc.stage}
                            </span>
                          )}
                          <div className="micro-progress-track">
                            <div
                              className="micro-progress-fill"
                              style={{ width: `${Math.min(100, Math.max(5, doc.percent || 5))}%` }}
                            />
                          </div>
                          {doc.message && (
                            <span className="status-subtext-msg" title={doc.message}>
                              {doc.message}
                            </span>
                          )}
                        </div>
                      ) : doc.status === "queued" ? (
                        <div className="status-queued-container">
                          <span className="status-pill status-queued">
                            <span>○</span> Queued
                          </span>
                          <span className="status-subtext-msg">Waiting in queue...</span>
                        </div>
                      ) : isFailed ? (
                        <div className="status-failed-container">
                          <span className="status-pill status-failed">
                            ● Failed / Skipped
                          </span>
                          {doc.error_message && (
                            <span className="status-error-subtext" title={doc.error_message}>
                              {doc.error_message}
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="status-pill status-indexed">
                          ● Indexed
                        </span>
                      )}
                    </td>
                    <td>
                      {isPending ? (
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontStyle: "italic" }}>
                          Computing vectors...
                        </span>
                      ) : isFailed ? (
                        <span style={{ fontSize: "0.75rem", color: "var(--text-faint)" }}>
                          0 chunks
                        </span>
                      ) : (
                        <div className="chunks-badge-cell">
                          <span className="chunk-count-badge">
                            {doc.chunk_count || 0} chunks
                          </span>
                          {doc.total_pages ? (
                            <span className="page-count-badge">
                              {doc.total_pages} {doc.total_pages === 1 ? "page" : "pages"}
                            </span>
                          ) : null}
                        </div>
                      )}
                    </td>
                    <td className="doc-date-cell">
                      {isPending ? (
                        <span style={{ fontSize: "0.75rem", color: "#f59e0b", fontWeight: 600 }}>
                          Just now
                        </span>
                      ) : (
                        formatDate(doc.created_at)
                      )}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      {isPending ? (
                        <span className="indexing-actions-badge">
                          <span>⏳</span> {doc.status === "processing" ? "Indexing..." : "Queued"}
                        </span>
                      ) : (
                        <div className="row-action-btns">
                          <button
                            type="button"
                            className="btn-action preview-action-btn"
                            onClick={() => onPreview && onPreview(doc.document_id)}
                            disabled={isFailed}
                            title="Inspect parsed text content and chunks"
                          >
                            👁️ View
                          </button>
                          <button
                            type="button"
                            className="btn-action delete-action-btn"
                            onClick={() => onDelete(doc.document_id, doc.title)}
                            disabled={isDeleting || !doc.document_id || doc.document_id.startsWith("failed_")}
                            title="Purge document and delete its vector chunks from ChromaDB"
                          >
                            {isDeleting ? "Deleting..." : "🗑️ Delete"}
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {filteredDocs.length > 0 && (
        <div className="table-footer-status-bar">
          <span className="footer-status-text">
            Showing {filteredDocs.length} of {combinedDocs.length} document(s) in session
          </span>
          {combinedDocs.length > 2 && (
            <span className="footer-scroll-hint">
              ↕ Scroll to view all documents
            </span>
          )}
        </div>
      )}
    </div>
  );
}
