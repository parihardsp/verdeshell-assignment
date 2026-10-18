import React, { useState } from "react";
import { parseBotAnswer, formatLatency } from "../../utils/formatters";
import SourceCard from "./SourceCard";

export default function BotAnswerCard({
  answer,
  sources = [],
  latencyMs,
  confidenceScore,
  defaultExpandedSources = false,
}) {
  const [showSources, setShowSources] = useState(defaultExpandedSources);
  const { summary, breakdown, isRefusal } = parseBotAnswer(answer);

  if (isRefusal) {
    return (
      <div className="bot-answer-card">
        <div
          style={{
            background: "rgba(245, 158, 11, 0.1)",
            border: "1px solid rgba(245, 158, 11, 0.3)",
            borderRadius: "10px",
            padding: "1rem 1.25rem",
            color: "#fde68a",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 700 }}>
            <span>ℹ️</span>
            <span>Information Not Found in Knowledge Base</span>
          </div>
          <p style={{ marginTop: "0.4rem", fontSize: "0.92rem", color: "#fef3c7" }}>
            {summary}
          </p>
          <div style={{ marginTop: "0.5rem", fontSize: "0.78rem", color: "#fef08a" }}>
            💡 <em>Tip: Upload the relevant PDF, Markdown, or text documents in the "Documents Hub" tab.</em>
          </div>
        </div>

        {latencyMs && (
          <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
            ⚡ Latency: {formatLatency(latencyMs)}
          </div>
        )}
      </div>
    );
  }

  // Format bullet points and numbered steps in breakdown
  const renderBreakdownContent = (text) => {
    if (!text) return null;
    const lines = text.split("\n").filter((l) => l.trim().length > 0);

    return (
      <div className="bot-breakdown-flow">
        <ul className="bot-breakdown-list">
          {lines.map((line, idx) => {
            const trimmed = line.trim();
            // Check if numbered step: 1. or 1)
            const numberMatch = trimmed.match(/^(\d+)[.)]\s*(.*)$/);
            // Check if bullet: - or * or •
            const bulletMatch = trimmed.match(/^[-*•]\s*(.*)$/);

            const contentText = numberMatch
              ? numberMatch[2]
              : bulletMatch
              ? bulletMatch[1]
              : trimmed;

            // Split bold title if present: **Title:** Rest
            const boldMatch = contentText.match(/^\*\*(.*?)\*\*:?\s*(.*)$/);

            if (numberMatch) {
              return (
                <li key={idx} className="breakdown-item is-step">
                  <span className="step-badge">{numberMatch[1]}</span>
                  <div className="breakdown-text">
                    {boldMatch ? (
                      <>
                        <strong>{boldMatch[1]}:</strong> {boldMatch[2]}
                      </>
                    ) : (
                      contentText
                    )}
                  </div>
                </li>
              );
            }

            return (
              <li key={idx} className="breakdown-item is-bullet">
                <span className="bullet-marker">•</span>
                <div className="breakdown-text">
                  {boldMatch ? (
                    <>
                      <strong>{boldMatch[1]}:</strong> {boldMatch[2]}
                    </>
                  ) : (
                    contentText
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    );
  };

  const confidencePct =
    confidenceScore !== null && confidenceScore !== undefined && confidenceScore > 0
      ? Math.round(confidenceScore * 100)
      : null;

  return (
    <div className="bot-answer-card">
      {/* 1. Natural Summary Paragraph */}
      {summary && (
        <div className="bot-summary-paragraph">
          <p>{summary}</p>
        </div>
      )}

      {/* 2. Structured Breakdown Points & Steps */}
      {breakdown && renderBreakdownContent(breakdown)}

      {/* 3. Metadata Bar */}
      <div className="bot-metadata-bar">
        <span className="meta-pill grounded">🟢 Grounded in Session Docs</span>
        {latencyMs && <span className="meta-pill">⚡ {formatLatency(latencyMs)}</span>}
        {confidencePct && <span className="meta-pill">🎯 {confidencePct}% Confidence</span>}
        {sources && sources.length > 0 && (
          <span className="meta-pill">📑 {sources.length} Sources Cited</span>
        )}
      </div>

      {/* 4. Verified Sources Accordion (Closed by default, click to extend) */}
      {sources && sources.length > 0 && (
        <div className="sources-section">
          <button
            type="button"
            className="sources-toggle-btn"
            onClick={() => setShowSources((prev) => !prev)}
            aria-expanded={showSources}
          >
            <span className="sources-chevron">{showSources ? "▼" : "▶"}</span>
            <span>📑 Verified Grounding Sources ({sources.length})</span>
          </button>

          {showSources && (
            <div className="sources-grid">
              {sources.map((source, idx) => (
                <SourceCard key={idx} source={source} index={idx + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
