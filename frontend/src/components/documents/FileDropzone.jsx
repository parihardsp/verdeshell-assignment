import React, { useState, useRef } from "react";
import { formatBytes } from "../../utils/formatters";

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md"];
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const MAX_FILES = 5;

export default function FileDropzone({ onUpload, isUploading }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [validationError, setValidationError] = useState("");
  const inputRef = useRef(null);

  const validateFiles = (fileList) => {
    setValidationError("");
    const valid = [];

    if (fileList.length > MAX_FILES) {
      setValidationError(`Maximum ${MAX_FILES} files allowed per batch upload.`);
      return [];
    }

    for (const f of fileList) {
      const ext = "." + f.name.split(".").pop().toLowerCase();
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        setValidationError(
          `"${f.name}" has an unsupported format. Allowed formats: ${ALLOWED_EXTENSIONS.join(", ")}`
        );
        return [];
      }
      if (f.size > MAX_FILE_SIZE) {
        setValidationError(
          `"${f.name}" exceeds the 5MB size limit (${formatBytes(f.size)}).`
        );
        return [];
      }
      valid.push(f);
    }
    return valid;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const valid = validateFiles(Array.from(e.dataTransfer.files));
      if (valid.length > 0) setSelectedFiles(valid);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files.length > 0) {
      const valid = validateFiles(Array.from(e.target.files));
      if (valid.length > 0) setSelectedFiles(valid);
    }
  };

  const removeFile = (idx) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUploadClick = () => {
    if (selectedFiles.length === 0 || isUploading) return;
    onUpload(selectedFiles);
    setSelectedFiles([]);
  };

  return (
    <div className="file-dropzone-wrapper">
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.txt,.md"
        style={{ display: "none" }}
        onChange={handleChange}
      />

      {/* Main Drag & Drop Zone */}
      <div
        className={`file-dropzone-container ${dragActive ? "is-drag-over" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <div className="dropzone-icon">📤</div>
        <h3 className="dropzone-title">Drag & Drop Document Files Here</h3>
        <p className="dropzone-instruction">
          Supports <strong>.pdf</strong>, <strong>.docx</strong>, <strong>.doc</strong>, <strong>.txt</strong>, <strong>.md</strong> (up to 5 files, 5MB each)
        </p>

        {/* Clear Browse & Upload Button */}
        <div className="dropzone-btn-row">
          <button
            type="button"
            className="btn btn-primary dropzone-browse-btn"
            onClick={(e) => {
              e.stopPropagation();
              inputRef.current?.click();
            }}
            disabled={isUploading}
          >
            <span>📁 Browse & Select Files</span>
          </button>
        </div>

        <span className="dropzone-constraints-tag">
          Page-preserving chunker • Gemini 3072d Dense Vector Index
        </span>
      </div>

      {validationError && (
        <div className="validation-error-banner">
          <span>⚠️</span>
          <span>{validationError}</span>
        </div>
      )}

      {/* Selected Files Queue */}
      {selectedFiles.length > 0 && (
        <div className="selected-files-queue-box">
          <div className="queue-header-row">
            <span className="queue-title">
              Ready for Ingestion ({selectedFiles.length} file{selectedFiles.length > 1 ? "s" : ""}):
            </span>
            <button
              type="button"
              className="btn-link clear-all-files-btn"
              onClick={() => setSelectedFiles([])}
              disabled={isUploading}
            >
              Clear All
            </button>
          </div>

          <div className="selected-files-grid">
            {selectedFiles.map((file, idx) => (
              <div key={idx} className="selected-file-chip">
                <span className="chip-file-icon">📄</span>
                <div className="chip-file-info">
                  <span className="chip-file-name" title={file.name}>
                    {file.name}
                  </span>
                  <span className="chip-file-size">{formatBytes(file.size)}</span>
                </div>
                <button
                  type="button"
                  className="chip-remove-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(idx);
                  }}
                  disabled={isUploading}
                  title="Remove this file"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          <div className="queue-actions-bar">
            <button
              type="button"
              className="btn btn-primary start-ingest-btn"
              onClick={handleUploadClick}
              disabled={isUploading}
            >
              {isUploading ? (
                <>
                  <span className="spinner-dots">●</span>
                  <span>Ingesting & Indexing...</span>
                </>
              ) : (
                <span>🚀 Ingest & Index {selectedFiles.length} File{selectedFiles.length > 1 ? "s" : ""}</span>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
