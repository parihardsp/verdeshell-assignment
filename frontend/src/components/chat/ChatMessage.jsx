import React from "react";
import BotAnswerCard from "./BotAnswerCard";

export default function ChatMessage({ message, isLatest }) {
  const isUser = message.role === "user";

  return (
    <div className="chat-message">
      <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "👤" : "🌿"}
      </div>

      <div className="message-content-wrapper">
        {isUser ? (
          <div className="message-bubble-user">{message.content}</div>
        ) : (
          <BotAnswerCard
            answer={message.content}
            sources={message.sources || []}
            latencyMs={message.latency_ms}
            confidenceScore={message.confidence_score}
            defaultExpandedSources={false}
          />
        )}
      </div>
    </div>
  );
}
