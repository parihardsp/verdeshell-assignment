import React, { useState, useRef, useEffect } from "react";

export default function QueryInput({ onSend, isLoading, isRateLimited, docCount = 0 }) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef(null);

  // Auto-resize textarea to expand flexibly up to 4-5 lines (~130px), then scroll
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;

    el.style.height = "auto";
    const maxHeight = 130; // ~4-5 lines at 1.5 line-height
    const scrollHeight = el.scrollHeight;

    if (scrollHeight > maxHeight) {
      el.style.height = `${maxHeight}px`;
      el.style.overflowY = "auto";
    } else {
      el.style.height = `${Math.max(scrollHeight, 24)}px`;
      el.style.overflowY = "hidden";
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!query.trim() || isLoading || isRateLimited) return;
    onSend(query.trim());
    setQuery("");
  };

  return (
    <div className="query-input-container">
      <div className="query-input-wrapper">
        <textarea
          ref={textareaRef}
          className="query-textarea"
          rows={1}
          placeholder={
            isRateLimited
              ? "Rate limit cooldown active. Please wait..."
              : docCount === 0
              ? "Session is empty. Upload documents or ask a question..."
              : "Ask any question from your knowledge base..."
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading || isRateLimited}
        />

        <button
          type="button"
          className="query-send-btn"
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading || isRateLimited}
          title="Send question (Enter)"
        >
          {isLoading ? (
            <span className="spinner-dots">●</span>
          ) : (
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          )}
        </button>
      </div>

      <div className="input-keyboard-hint">
        <span>Enter ↵ to send • Shift + Enter for new line</span>
      </div>
    </div>
  );
}
