import React, { useRef, useEffect, useState } from "react";
import ChatMessage from "./ChatMessage";
import QueryInput from "./QueryInput";
import RateLimitCountdown from "./RateLimitCountdown";

const THINKING_STEPS = [
  "Searching...",
  "Querying...",
  "Pondering...",
  "Synthesizing...",
];

function ThinkingIndicator() {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStepIdx((prev) => (prev + 1) % THINKING_STEPS.length);
    }, 1200);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="chat-thinking-row" role="status" aria-live="polite">
      <div className="thinking-indicator-container">
        <span className="thinking-indicator-star" aria-hidden="true">✦</span>
        <span className="thinking-indicator-text">
          {THINKING_STEPS[stepIdx]}
        </span>
      </div>
    </div>
  );
}

export default function ChatView({
  messages,
  onSendMessage,
  onClearChat,
  isLoading,
  rateLimitSeconds,
  onRateLimitReset,
  sessionId,
  docCount,
  onSwitchToDocsTab,
}) {
  const historyEndRef = useRef(null);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-view">
      {/* 1. Initial Empty State: Clean search console view with query box */}
      {messages.length === 0 ? (
        <div className="chat-initial-screen">
          <div className="chat-initial-center-box">
            <div className="initial-logo-wrapper">
              <span className="initial-logo-icon">🌿</span>
            </div>
            <h1 className="initial-title">Enterprise Knowledge Assistant</h1>
            <p className="initial-subtitle">
              Upload documents in the Document Hub or query the active knowledge base below.
            </p>

            {/* Central Search Box */}
            <div className="initial-query-box">
              <QueryInput
                onSend={onSendMessage}
                isLoading={isLoading}
                isRateLimited={rateLimitSeconds > 0}
              />
            </div>

            {docCount > 0 ? (
              <div className="initial-doc-status-pill">
                <span className="status-dot-emerald">●</span>
                <span>{docCount} document(s) indexed & ready for grounded search</span>
              </div>
            ) : (
              <button
                type="button"
                className="initial-upload-link-btn"
                onClick={onSwitchToDocsTab}
              >
                📁 Go to Document Hub to upload documents or notes →
              </button>
            )}
          </div>
        </div>
      ) : (
        /* 2. Active Conversation Stream */
        <div className="chat-active-screen">
          {/* Sub-header context banner */}
          <div className="chat-sub-header">
            <div className="chat-context-badge">
              <span className="dot-pulse">●</span>
              <span>
                {docCount > 0
                  ? `${docCount} document(s) active in session knowledge base`
                  : "Searching enterprise knowledge base"}
              </span>
            </div>

            <button
              type="button"
              className="clear-chat-btn"
              onClick={onClearChat}
              title="Clear current conversation"
            >
              🗑️ Clear Chat
            </button>
          </div>

          {/* Messages Thread (Scrollable Area) */}
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <ChatMessage
                key={idx}
                message={msg}
                isLatest={idx === messages.length - 1 && msg.role === "assistant"}
              />
            ))}

            {/* Dynamic synthesis / retrieval indicator */}
            {isLoading && <ThinkingIndicator />}

            <div ref={historyEndRef} />
          </div>

          {/* Rate Limit Active Banner */}
          {rateLimitSeconds > 0 && (
            <RateLimitCountdown
              initialSeconds={rateLimitSeconds}
              onReset={onRateLimitReset}
            />
          )}

          {/* Fixed Bottom Input Bar */}
          <QueryInput
            onSend={onSendMessage}
            isLoading={isLoading}
            isRateLimited={rateLimitSeconds > 0}
          />
        </div>
      )}
    </div>
  );
}
