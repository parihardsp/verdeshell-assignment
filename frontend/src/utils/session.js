/**
 * Session and User tenant identity utilities.
 * Persists tab-scoped session IDs, user IDs, custom display names, and chat histories.
 */

const SESSION_KEY = "verdeshell_session_id";
const USER_KEY = "verdeshell_user_id";
const DISPLAY_NAME_KEY = "verdeshell_display_name";
const CHAT_HISTORY_PREFIX = "verdeshell_chat_history_";
const SESSIONS_INDEX_KEY = "verdeshell_user_sessions_";

function generateShortId(prefix = "id") {
  // Industry-standard 6-character hex suffix (16.7M combinations)
  const hex = Math.floor(Math.random() * 0xffffff)
    .toString(16)
    .padStart(6, "0");
  return `${prefix}_${hex}`;
}

export function getOrCreateSessionId() {
  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = generateShortId("sess");
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

export function getOrCreateUserId() {
  let userId = localStorage.getItem(USER_KEY);
  // Auto-migrate legacy verbose IDs (e.g. "user_5liuvhkruc") to clean "usr_xxxxxx"
  if (!userId || userId.startsWith("user_")) {
    userId = generateShortId("usr");
    localStorage.setItem(USER_KEY, userId);
  }
  return userId;
}

export function getUserDisplayName() {
  return localStorage.getItem(DISPLAY_NAME_KEY) || "Knowledge Explorer";
}

export function setUserDisplayName(name) {
  if (name && name.trim()) {
    localStorage.setItem(DISPLAY_NAME_KEY, name.trim());
  }
}

export function setSessionId(newSessionId) {
  if (newSessionId && newSessionId.trim()) {
    sessionStorage.setItem(SESSION_KEY, newSessionId.trim());
  }
}

export function setUserId(newUserId) {
  if (newUserId && newUserId.trim()) {
    localStorage.setItem(USER_KEY, newUserId.trim());
  }
}

export function createNewSession() {
  const newId = generateShortId("sess");
  sessionStorage.setItem(SESSION_KEY, newId);
  return newId;
}

export function recordSessionForUser(userId, sessionId) {
  try {
    const key = `${SESSIONS_INDEX_KEY}${userId}`;
    const raw = localStorage.getItem(key);
    const list = raw ? JSON.parse(raw) : [];
    if (!list.includes(sessionId)) {
      list.unshift(sessionId);
      localStorage.setItem(key, JSON.stringify(list.slice(0, 20)));
    }
  } catch {
    // Ignore storage quota errors
  }
}

export function listUserSessions(userId) {
  try {
    const key = `${SESSIONS_INDEX_KEY}${userId}`;
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function loadSessionChat(userId, sessionId) {
  try {
    const key = `${CHAT_HISTORY_PREFIX}${userId}_${sessionId}`;
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveSessionChat(userId, sessionId, messages) {
  try {
    const key = `${CHAT_HISTORY_PREFIX}${userId}_${sessionId}`;
    localStorage.setItem(key, JSON.stringify(messages));
  } catch {
    // Ignore storage quota errors
  }
}

export function clearSessionChat(userId, sessionId) {
  try {
    const key = `${CHAT_HISTORY_PREFIX}${userId}_${sessionId}`;
    localStorage.removeItem(key);
  } catch {
    // Ignore
  }
}
