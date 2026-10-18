/**
 * Strongly-typed API endpoints consuming the FastAPI backend services.
 */

import { request } from "./client";

/**
 * Health check endpoint: verifies backend status, LLM provider, and embedding provider.
 */
export async function getHealth() {
  return request("/health", { method: "GET", timeoutMs: 5000 });
}

/**
 * Grounded Q&A endpoint: answers user questions strictly using session documents.
 * Protected by 20 req/min sliding-window rate limiter.
 */
export async function askQuestion(question, sessionId, userId) {
  return request("/api/v1/ask", {
    method: "POST",
    body: {
      question,
      session_id: sessionId,
      user_id: userId,
    },
    timeoutMs: 45000,
  });
}

/**
 * Upload up to 5 documents (.pdf, .txt, .md, .docx) for PyMuPDF parsing and ChromaDB indexing.
 * Supports asynchronous background indexing (HTTP 202) with real-time SSE progress streaming.
 */
export async function uploadFiles(files, sessionId, userId, { background = true } = {}) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  if (sessionId) formData.append("session_id", sessionId);
  if (userId) formData.append("user_id", userId);

  const url = background
    ? "/api/v1/documents/upload?background=true"
    : "/api/v1/documents/upload";

  return request(url, {
    method: "POST",
    body: formData,
    timeoutMs: background ? 20000 : 90000,
  });
}

/**
 * Fetch snapshot status for an asynchronous ingestion job.
 */
export async function getJobStatus(jobId) {
  return request(`/api/v1/documents/jobs/${jobId}`, {
    method: "GET",
    timeoutMs: 10000,
  });
}

/**
 * Stream real-time ingestion progress with SSE and graceful polling fallback.
 *
 * @param {string} jobId - Unique job ID returned from POST /documents/upload
 * @param {Object} handlers - Event callbacks: { onProgress(data), onComplete(data), onError(err) }
 * @returns {Function} cleanup function to close SSE connection or stop polling
 */
export function subscribeJobProgress(jobId, { onProgress, onComplete, onError }) {
  let isClosed = false;
  let eventSource = null;
  let pollingInterval = null;

  const cleanup = () => {
    isClosed = true;
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  };

  const startPollingFallback = () => {
    if (isClosed || pollingInterval) return;
    pollingInterval = setInterval(async () => {
      try {
        const snap = await getJobStatus(jobId);
        if (isClosed) return;
        if (onProgress) onProgress(snap);
        if (["completed", "failed", "completed_with_errors"].includes(snap.status)) {
          cleanup();
          if (onComplete) onComplete(snap);
        }
      } catch (err) {
        if (isClosed) return;
        cleanup();
        if (onError) onError(err);
      }
    }, 1200);
  };

  if (typeof window !== "undefined" && window.EventSource) {
    try {
      eventSource = new EventSource(`/api/v1/documents/jobs/${jobId}/events`);

      eventSource.onmessage = (event) => {
        if (isClosed) return;
        try {
          const data = JSON.parse(event.data);
          if (onProgress) onProgress(data);
          if (["completed", "failed", "completed_with_errors"].includes(data.status)) {
            cleanup();
            if (onComplete) onComplete(data);
          }
        } catch (parseErr) {
          console.warn("Failed to parse SSE event data:", parseErr);
        }
      };

      eventSource.onerror = () => {
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        if (!isClosed) {
          startPollingFallback();
        }
      };
    } catch {
      startPollingFallback();
    }
  } else {
    startPollingFallback();
  }

  return cleanup;
}

/**
 * Ingest a raw text article directly into ChromaDB vector store.
 */
export async function createDocument(title, content, sessionId, userId) {
  return request("/api/v1/documents", {
    method: "POST",
    body: {
      title,
      content,
      session_id: sessionId,
      user_id: userId,
    },
    timeoutMs: 30000,
  });
}

/**
 * List indexed documents scoped to the active user and session.
 */
export async function getDocuments(sessionId, userId, skip = 0, limit = 50) {
  const params = new URLSearchParams();
  if (sessionId) params.append("session_id", sessionId);
  if (userId) params.append("user_id", userId);
  params.append("skip", String(skip));
  params.append("limit", String(limit));

  return request(`/api/v1/documents?${params.toString()}`, {
    method: "GET",
    timeoutMs: 10000,
  });
}

/**
 * Fetch full document details including parsed text content.
 */
export async function getDocumentDetails(documentId) {
  return request(`/api/v1/documents/${documentId}`, {
    method: "GET",
    timeoutMs: 10000,
  });
}

/**
 * Delete a document and purge all its chunk vectors from ChromaDB.
 */
export async function deleteDocument(documentId) {
  return request(`/api/v1/documents/${documentId}`, {
    method: "DELETE",
    timeoutMs: 10000,
  });
}
