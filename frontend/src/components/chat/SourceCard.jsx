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
      : `Score: ${(similarity_score || 0).toFixed(3)}`;

  // Sanitize raw HTML tags (e.g. <br>), unicode replacement chars (), and excess whitespace
  const cleanSnippet = (snippet || "")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();

  return (
    <div className="source-card">
      <div className="source-header">
        <div className="source-title">
          [{index}] {title}
          {page_number && <span className="source-page-text"> • Page {page_number}</span>}
        </div>
        <div className="source-badges">
          <span className={`channel-badge ${channelClass}`}>{channelLabel}</span>
        </div>
      </div>

      <div className="source-meta-row">
        {file_name && <span className="source-filename">{file_name} • </span>}
        <span className="source-score">{scoreText}</span>
      </div>

      <div className="source-snippet">
        "{cleanSnippet}"
      </div>
    </div>
  );
}
