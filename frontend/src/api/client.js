/**
 * Centralized HTTP Client with tenant scoping, timeouts, and error shielding.
 */

import { getOrCreateSessionId, getOrCreateUserId } from "../utils/session";

const API_TIMEOUT_MS = 45000;

export class ApiError extends Error {
  constructor(message, status, data = null, retryAfter = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
    this.retryAfter = retryAfter;
  }
}

export async function request(path, options = {}) {
  const {
    method = "GET",
    body = null,
    headers = {},
    timeoutMs = API_TIMEOUT_MS,
    ...rest
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const sessionId = getOrCreateSessionId();
  const userId = getOrCreateUserId();

  const reqHeaders = {
    "X-Session-ID": sessionId,
    "X-User-ID": userId,
    ...headers,
  };

  // Only set Content-Type to application/json if body is not FormData
  if (body && !(body instanceof FormData)) {
    reqHeaders["Content-Type"] = "application/json";
  }

  try {
    const response = await fetch(path, {
      method,
      headers: reqHeaders,
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      ...rest,
    });

    clearTimeout(timeoutId);

    // Extract Retry-After header if present on 429
    const retryAfterHeader = response.headers.get("Retry-After");
    const retryAfterSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : null;

    if (!response.ok) {
      let errorDetail = `Request failed with status ${response.status}`;
      let errorData = null;

      try {
        errorData = await response.json();
        if (typeof errorData.detail === "string") {
          errorDetail = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorDetail = errorData.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
        } else if (errorData.message) {
          errorDetail = errorData.message;
        }
      } catch {
        errorDetail = await response.text().catch(() => errorDetail);
      }

      throw new ApiError(errorDetail, response.status, errorData, retryAfterSeconds);
    }

    // 204 No Content
    if (response.status === 204) {
      return null;
    }

    return await response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.", 408);
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err.message || "Failed to communicate with backend server.",
      0
    );
  }
}
