import React, { useState } from "react";
import FileDropzone from "./FileDropzone";
import DocumentTable from "./DocumentTable";
import DocumentPreviewModal from "./DocumentPreviewModal";

export default function DocumentsView({
  documents,
  pendingUploads = [],
  onUploadFiles,
  onCreateTextDoc,
  onDeleteDocument,
  isUploading,
  isSavingText,
  isDeletingId,
  onRefreshDocs,
  sessionId,
  selectedDocId,
  onSelectDocId,
}) {
  const [ingestMode, setIngestMode] = useState("files"); // "files" | "text"
  const [manualTitle, setManualTitle] = useState("");
  const [manualContent, setManualContent] = useState("");
  const [previewDocId, setPreviewDocId] = useState(selectedDocId || null);
  const [uploadFileCount, setUploadFileCount] = useState(1);

  const isIndexing = isUploading || isSavingText;
  const activeDocForPreview = selectedDocId || previewDocId;

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    if (!manualTitle.trim() || !manualContent.trim() || isIndexing) return;
    await onCreateTextDoc(manualTitle.trim(), manualContent.trim());
    setManualTitle("");
    setManualContent("");
  };

  const handleFilesUploadProxy = (files) => {
    setUploadFileCount(files.length);
    onUploadFiles(files);
  };

  const indexedDocs = documents.filter((d) => d.status === "indexed");
  const totalChunks = indexedDocs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  return (
    <div className="documents-view">
      {/* Document Inspector Modal */}
      {activeDocForPreview && (
        <DocumentPreviewModal
          documentId={activeDocForPreview}
          onClose={() => {
            setPreviewDocId(null);
            if (onSelectDocId) onSelectDocId(null);
          }}
        />
      )}

      {/* Header bar */}
      <div className="doc-actions-bar">
        <div className="doc-header-text">
          <h2 className="doc-section-title">Knowledge Base Ingestion Hub</h2>
          <div className="doc-section-subtitle">
            Upload files or manually enter content to index into session{" "}
            <code className="session-code-tag">{sessionId}</code>.
          </div>
        </div>

        <div className="doc-toolbar-buttons">
          <button
            type="button"
            className="btn btn-secondary toolbar-btn"
            onClick={onRefreshDocs}
            title="Refresh document inventory from database"
          >
            ↻ Refresh Docs
          </button>
        </div>
      </div>

      {/* Ingestion Mode Segmented Switcher */}
      <div className="ingest-mode-switcher-container">
        <div className="ingest-mode-tabs" role="tablist">
          <button
            type="button"
            className={`ingest-tab-btn ${ingestMode === "files" ? "active" : ""}`}
            onClick={() => setIngestMode("files")}
            role="tab"
            aria-selected={ingestMode === "files"}
          >
            <span className="tab-icon">📁</span>
            <span>Upload Document Files</span>
          </button>

          <button
            type="button"
            className={`ingest-tab-btn ${ingestMode === "text" ? "active" : ""}`}
            onClick={() => setIngestMode("text")}
            role="tab"
            aria-selected={ingestMode === "text"}
          >
            <span className="tab-icon">✍️</span>
            <span>Enter Document (Title & Content)</span>
          </button>
        </div>

        {/* Option 1: File Upload & Drag & Drop */}
        {ingestMode === "files" && (
          <div className="ingest-panel-box">
            <FileDropzone
              onUpload={handleFilesUploadProxy}
              isUploading={isIndexing}
            />
          </div>
        )}

        {/* Option 2: Manual Document Entry (Title & Content) */}
        {ingestMode === "text" && (
          <div className="ingest-panel-box manual-entry-box">
            <div className="manual-entry-header">
              <h3 className="manual-entry-title">Manual Document Entry</h3>
              <p className="manual-entry-desc">
                Provide a title and document content to segment, compute embeddings, and index into the vector store.
              </p>
            </div>

            <form onSubmit={handleManualSubmit} className="manual-doc-form">
              <div className="form-group">
                <label htmlFor="manual-doc-title" className="form-label">
                  Document Title <span className="req-star">*</span>
                </label>
                <input
                  id="manual-doc-title"
                  type="text"
                  className="form-input"
                  placeholder="e.g. SLA Terms and Maintenance Protocol"
                  value={manualTitle}
                  onChange={(e) => setManualTitle(e.target.value)}
                  disabled={isIndexing}
                  required
                />
              </div>

              <div className="form-group">
                <div className="label-with-counter">
                  <label htmlFor="manual-doc-content" className="form-label">
                    Document Content <span className="req-star">*</span>
                  </label>
                  <span className="char-counter">
                    {manualContent.length.toLocaleString()} characters
                  </span>
                </div>
                <textarea
                  id="manual-doc-content"
                  className="form-textarea"
                  rows={8}
                  placeholder="Paste or write document text here. Supports Markdown formatting, paragraphs, and bullet points..."
                  value={manualContent}
                  onChange={(e) => setManualContent(e.target.value)}
                  disabled={isIndexing}
                  required
                />
              </div>

              <div className="form-actions-row">
                <button
                  type="submit"
                  className="btn btn-primary submit-doc-btn"
                  disabled={!manualTitle.trim() || !manualContent.trim() || isIndexing}
                >
                  {isIndexing ? (
                    <>
                      <span className="spinner-dots">●</span>
                      <span>Indexing & Computing Embeddings...</span>
                    </>
                  ) : (
                    <span>🚀 Ingest & Index Document</span>
                  )}
                </button>
                {manualContent && (
                  <button
                    type="button"
                    className="btn btn-secondary clear-form-btn"
                    onClick={() => {
                      setManualTitle("");
                      setManualContent("");
                    }}
                    disabled={isIndexing}
                  >
                    Clear Fields
                  </button>
                )}
              </div>
            </form>
          </div>
        )}
      </div>

      {/* Session Knowledge Inventory Table */}
      <DocumentTable
        documents={documents}
        pendingUploads={pendingUploads}
        onDelete={onDeleteDocument}
        onPreview={(docId) => setPreviewDocId(docId)}
        isDeletingId={isDeletingId}
      />
    </div>
  );
}
