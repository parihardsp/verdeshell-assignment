import React, { useEffect, useState } from "react";

const ANTHROPIC_STAGES = [
  "Reading document format and verifying constraints...",
  "Extracting text & page structures via PyMuPDF...",
  "Recursive chunking into semantic text passages...",
  "Generating 3072-dimensional vector embeddings with Gemini...",
  "Persisting vectors & metadata into ChromaDB collection...",
  "Finalizing session-scoped knowledge index...",
];

export default function IndexingProgress({
  isIndexing = false,
  fileCount = 1,
  mode = "files", // "files" | "text"
  progressData = null, // { percent, stage, message, current_file, processed_files, total_files }
}) {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    if (!isIndexing || progressData) {
      if (!isIndexing) setStageIndex(0);
      return;
    }

    const interval = setInterval(() => {
      setStageIndex((prev) => (prev < ANTHROPIC_STAGES.length - 1 ? prev + 1 : prev));
    }, 1200);

    return () => clearInterval(interval);
  }, [isIndexing, progressData]);

  if (!isIndexing) return null;

  const activePercent =
    progressData && typeof progressData.percent === "number"
      ? progressData.percent
      : Math.min(100, Math.round(((stageIndex + 1) / ANTHROPIC_STAGES.length) * 100));

  const activeMessage =
    progressData?.message ||
    ANTHROPIC_STAGES[stageIndex];

  const totalFiles = progressData?.total_files || fileCount;
  const currentFileName = progressData?.current_file;

  return (
    <div className="anthropic-spinner-overlay" role="status" aria-live="polite">
      <div className="anthropic-spinner-card">
        {/* Breathing Anthropic-style starlight / orb */}
        <div className="anthropic-halo-container">
          <div className="anthropic-pulsing-halo"></div>
          <div className="anthropic-inner-halo"></div>
          <div className="anthropic-star-glyph">✦</div>
        </div>

        {/* Dynamic status text */}
        <div className="anthropic-status-block">
          <h3 className="anthropic-indexing-heading">
            {mode === "files"
              ? `Indexing & Ingesting ${totalFiles} File Document${totalFiles > 1 ? "s" : ""}...`
              : "Indexing & Ingesting Document Article..."}
          </h3>

          {currentFileName && (
            <div
              style={{
                fontSize: "0.82rem",
                color: "var(--accent, #d97706)",
                fontWeight: 600,
                marginTop: "0.2rem",
                letterSpacing: "0.01em",
              }}
            >
              Processing: {currentFileName}
            </div>
          )}

          <div className="anthropic-phase-indicator">
            <span className="anthropic-phase-dot">●</span>
            <span className="anthropic-phase-text keyframe-fade">
              {activeMessage}
            </span>
          </div>
        </div>

        {/* Animated Progress Shimmer Line & Percentage */}
        <div style={{ width: "100%", marginTop: "0.25rem" }}>
          <div className="anthropic-shimmer-track">
            <div
              className="anthropic-shimmer-fill"
              style={{
                width: `${activePercent}%`,
                transition: "width 0.4s ease-out",
              }}
            ></div>
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "0.75rem",
              color: "var(--text-muted, #71717a)",
              marginTop: "0.35rem",
              padding: "0 0.2rem",
            }}
          >
            <span>{progressData?.stage ? `Stage: ${progressData.stage}` : "Semantic Pipeline"}</span>
            <span style={{ fontWeight: 600, color: "var(--text-primary, #e4e4e7)" }}>
              {activePercent}%
            </span>
          </div>
        </div>

        <p className="anthropic-wait-footnote">
          Asynchronous background pipeline with real-time SSE telemetry & vector persistence.
        </p>
      </div>
    </div>
  );
}
