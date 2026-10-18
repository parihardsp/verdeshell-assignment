import React from "react";
import BotAnswerCard from "./BotAnswerCard";

export default function ChatMessage({ message, isLatest, onSwitchToDocsTab }) {
  const isUser = message.role === "user";

  return (
    <div className={`chat-message ${isUser ? "user-message" : "assistant-message"}`}>
      <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "👤" : "🌿"}
      </div>

      <div className="message-content-wrapper">
        {isUser ? (
          <div className="message-bubble user-bubble">{message.content}</div>
        ) : (
          <BotAnswerCard
            answer={message.content}
            sources={message.sources || []}
            latencyMs={message.latency_ms}
            confidenceScore={message.confidence_score}
            responseType={message.response_type}
            defaultExpandedSources={false}
            onSwitchToDocsTab={onSwitchToDocsTab}
          />
        )}
      </div>
    </div>
  );
}
