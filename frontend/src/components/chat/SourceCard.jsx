import React from "react";

export default function SourceCard({ source, index }) {
  const {
    title,
    file_name,
    page_number,
    retrieval_channel = "dense",
    similarity_score,
    reranker_score,
    snippet,
  } = source;

  const channelClass =
    retrieval_channel === "hybrid"
      ? "hybrid"
      : retrieval_channel === "bm25"
      ? "bm25"
      : "dense";

  const channelLabel =
    retrieval_channel === "hybrid"
      ? "🟣 Hybrid"
      : retrieval_channel === "bm25"
      ? "🟢 BM25"
      : "🔵 Dense";

  const scoreText =
    reranker_score !== null && reranker_score !== undefined && reranker_score > 0
      ? `Groq Score: ${reranker_score.toFixed(1)}/10`
      : `Score: ${(similarity_score || 0).toFixed(4)}`;

  return (
    <div className="source-card">
      <div className="source-header">
        <div className="source-title">
          [{index}] {title}
          {page_number && <span style={{ color: "#94a3b8" }}> • Page {page_number}</span>}
        </div>
        <div className="source-badges">
          <span className={`channel-badge ${channelClass}`}>{channelLabel}</span>
        </div>
      </div>

      <div style={{ fontSize: "0.74rem", color: "#64748b" }}>
        {file_name && <span>{file_name} • </span>}
        <span style={{ fontWeight: 600, color: "#cbd5e1" }}>{scoreText}</span>
      </div>

      <div className="source-snippet">
        "{snippet}"
      </div>
    </div>
  );
}
