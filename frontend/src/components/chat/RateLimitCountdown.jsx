import React, { useState, useEffect } from "react";

export default function RateLimitCountdown({ initialSeconds = 60, onReset }) {
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds);

  useEffect(() => {
    if (secondsLeft <= 0) {
      if (onReset) onReset();
      return;
    }

    const timer = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          if (onReset) onReset();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [secondsLeft, onReset]);

  return (
    <div className="rate-limit-banner">
      <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
        <span>⚠️</span>
        <div>
          <strong>Rate Limit Exceeded (20 req / 60s per client).</strong>
          <div style={{ fontSize: "0.8rem", color: "#fef9c3", marginTop: "2px" }}>
            The API enforces fair-usage sliding-window rate limiting.
          </div>
        </div>
      </div>
      <div className="rate-limit-countdown">
        Retry in {secondsLeft}s
      </div>
    </div>
  );
}
