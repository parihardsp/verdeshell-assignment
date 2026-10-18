import React, { useEffect, useState } from "react";

export default function Toast({ message, type = "info", onClose, duration = 3500 }) {
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    // 1. After 3.5 seconds, start the smooth diminishing fade-out
    const exitTimer = setTimeout(() => {
      setIsExiting(true);
    }, duration);

    // 2. After diminishing transition completes (1.2s), unmount from DOM
    const removeTimer = setTimeout(() => {
      onClose();
    }, duration + 1200);

    return () => {
      clearTimeout(exitTimer);
      clearTimeout(removeTimer);
    };
  }, [onClose, duration]);

  const handleManualClose = () => {
    setIsExiting(true);
    setTimeout(onClose, 250);
  };

  const icons = {
    success: "✓",
    error: "✕",
    warning: "!",
    info: "ℹ",
  };

  return (
    <div
      className={`toast-container ${isExiting ? "toast-exiting" : "toast-entering"}`}
      role="status"
      aria-live="polite"
    >
      <div className={`toast-card toast-${type}`}>
        <div className={`toast-icon-badge ${type}`}>
          <span className="toast-icon-symbol">{icons[type] || "ℹ"}</span>
        </div>

        <div className="toast-message-content">
          <p className="toast-message-text">{message}</p>
        </div>

        <button
          type="button"
          className="toast-close-btn"
          onClick={handleManualClose}
          aria-label="Close notification"
          title="Dismiss notification"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>

        {/* Subtle timer progress bar */}
        <div
          className={`toast-progress-bar ${type}`}
          style={{ animationDuration: `${duration}ms` }}
        />
      </div>
    </div>
  );
}
