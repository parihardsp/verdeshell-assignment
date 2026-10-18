import React, { useState } from "react";
import { parseBotAnswer, formatLatency } from "../../utils/formatters";
import SourceCard from "./SourceCard";

export default function BotAnswerCard({
  answer,
  sources = [],
  latencyMs,
  confidenceScore,
  responseType,
  defaultExpandedSources = false,
  onSwitchToDocsTab,
}) {
  const [showSources, setShowSources] = useState(defaultExpandedSources);
  const { summary, breakdown, isRefusal, refusalType } = parseBotAnswer(answer);

  const effectiveType = responseType || refusalType;
  const isActuallyRefusal = isRefusal || effectiveType === "empty_session" || effectiveType === "no_relevant_info";

  // Format confidence text concisely for the single-line footer
  const confidenceText = (() => {
    if (confidenceScore === null || confidenceScore === undefined || confidenceScore <= 0) {
      return null;
    }
    const pct = Math.round(confidenceScore * 100);
    if (pct >= 80) return "High Confidence";
    if (pct >= 50) return "Moderate Confidence";
    return `${pct}% Confidence`;
  })();

  if (isActuallyRefusal) {
    if (effectiveType === "empty_session") {
      return (
        <div className="bot-answer-card empty-session-card">
          <div className="card-header-row">
            <div className="card-title-group">
              <span className="card-icon">📁</span>
              <span className="card-title">Session Knowledge Base is Empty</span>
            </div>
          </div>

          <p className="card-body-text">
            {summary || "No documents, notes, or files have been added to this session yet. Please upload content first so answers can be grounded in your data."}
          </p>

          {onSwitchToDocsTab && (
            <button
              type="button"
              className="card-action-btn"
              onClick={onSwitchToDocsTab}
              title="Go to Document Hub"
            >
              <span>+ Upload Documents in Document Hub</span>
              <span className="btn-arrow">→</span>
            </button>
          )}

          <div className="card-divider" />

          <div className="bot-footer-line">
            {latencyMs !== null && latencyMs !== undefined && (
              <>
                <span className="footer-meta-item">⚡ {formatLatency(latencyMs)}</span>
                <span className="footer-dot">•</span>
              </>
            )}
            <span className="footer-meta-item">🛡️ Zero API Quota Used</span>
            <span className="footer-dot">•</span>
            <span className="footer-meta-item">⚡ Instant Guard</span>
          </div>
        </div>
      );
    }

    // Case 2: Documents exist in session, but no matching context found for this question
    return (
      <div className="bot-answer-card no-match-card">
        <div className="card-header-row">
          <div className="card-title-group">
            <span className="card-icon">🔍</span>
            <span className="card-title">Information Not Found in Knowledge Base</span>
          </div>
        </div>

        <p className="card-body-text">
          {summary || "I searched through your uploaded session documents, but could not find relevant information to answer this question."}
        </p>

        <div className="card-tip-box">
          <span className="tip-icon">💡</span>
          <span className="tip-text">
            <strong>Tip:</strong> Try rephrasing your search keywords or upload additional documents covering this topic in the Document Hub.
          </span>
        </div>

        {onSwitchToDocsTab && (
          <button
            type="button"
            className="card-action-btn"
            onClick={onSwitchToDocsTab}
            title="Add more documents"
          >
            <span>📁 Add Supporting Documents in Document Hub</span>
            <span className="btn-arrow">→</span>
          </button>
        )}

        <div className="card-divider" />

        <div className="bot-footer-line">
          {latencyMs !== null && latencyMs !== undefined && (
            <>
              <span className="footer-meta-item">⚡ {formatLatency(latencyMs)}</span>
              <span className="footer-dot">•</span>
            </>
          )}
          <span className="footer-meta-item">🎯 0% Match Confidence</span>
          <span className="footer-dot">•</span>
          <span className="footer-meta-item">📑 Scanned Session Docs</span>
        </div>
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

  return (
    <div className="bot-answer-card grounded-card">
      {/* 1. Answer Content (Dominant Focus: 70–80% emphasis) */}
      <div className="bot-answer-primary">
        {summary && (
          <div className="bot-summary-paragraph">
            <p>{summary}</p>
          </div>
        )}

        {breakdown && renderBreakdownContent(breakdown)}
      </div>

      {/* 2. Verified Sources (Single Expandable Section immediately below answer) */}
      {sources && sources.length > 0 && (
        <div className="sources-expandable-section">
          <div className="card-divider" />
          <button
            type="button"
            className="sources-toggle-btn"
            onClick={() => setShowSources((prev) => !prev)}
            aria-expanded={showSources}
          >
            <span className="sources-label">📚 Verified Sources ({sources.length})</span>
            <span className="sources-chevron">{showSources ? "▲" : "▼"}</span>
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

      {/* 3. Metadata (Single-line subtle footer) */}
      <div className="card-divider" />
      <div className="bot-footer-line">
        {sources && sources.length > 0 && (
          <span className="footer-meta-item is-grounded">🟢 Grounded</span>
        )}
        {sources && sources.length > 0 && latencyMs !== null && latencyMs !== undefined && (
          <span className="footer-dot">•</span>
        )}
        {latencyMs !== null && latencyMs !== undefined && (
          <span className="footer-meta-item">⚡ {formatLatency(latencyMs)}</span>
        )}
        {confidenceText && (
          <>
            <span className="footer-dot">•</span>
            <span className="footer-meta-item">🎯 {confidenceText}</span>
          </>
        )}
        {sources && sources.length > 0 && (
          <>
            <span className="footer-dot">•</span>
            <span className="footer-meta-item">
              📑 {sources.length} {sources.length === 1 ? "Source" : "Sources"}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
