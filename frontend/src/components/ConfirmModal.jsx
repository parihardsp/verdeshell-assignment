import React, { useEffect } from "react";

export default function ConfirmModal({
  isOpen,
  title = "Confirm Action",
  subtitle = "CONFIRMATION",
  message = "Are you sure you want to proceed?",
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  confirmVariant = "danger", // "danger" | "primary"
  icon = "🗑️",
  onConfirm,
  onCancel,
}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onCancel();
      }
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal-container confirm-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-modal-title"
      >
        <div className="confirm-modal-body">
          <div className={`confirm-icon-badge ${confirmVariant}`}>
            <span className="confirm-icon">{icon}</span>
          </div>

          <div className="confirm-content">
            <span className="modal-subtitle">{subtitle}</span>
            <h3 id="confirm-modal-title" className="confirm-modal-title">
              {title}
            </h3>
            <p className="confirm-modal-message">{message}</p>
          </div>
        </div>

        <div className="confirm-modal-actions">
          <button
            type="button"
            className="confirm-btn-cancel"
            onClick={onCancel}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            className={`confirm-btn-action ${confirmVariant}`}
            onClick={onConfirm}
            autoFocus
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
