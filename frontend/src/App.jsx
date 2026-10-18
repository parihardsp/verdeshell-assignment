import React, { useState, useEffect, useCallback } from "react";
import Header from "./components/Header";
import Toast from "./components/Toast";
import ConfirmModal from "./components/ConfirmModal";
import UserSessionSidebar from "./components/UserSessionSidebar";
import ChatView from "./components/chat/ChatView";
import DocumentsView from "./components/documents/DocumentsView";
import {
  getHealth,
  askQuestion,
  uploadFiles,
  subscribeJobProgress,
  createDocument,
  getDocuments,
  deleteDocument,
} from "./api/endpoints";
import {
  getOrCreateSessionId,
  getOrCreateUserId,
  getUserDisplayName,
  setUserDisplayName,
  createNewSession,
  setSessionId,
  recordSessionForUser,
  listUserSessions,
  loadSessionChat,
  saveSessionChat,
  clearSessionChat,
} from "./utils/session";
import "./App.css";

export default function App() {
  const [userId] = useState(getOrCreateUserId);
  const [sessionId, setSessionIdState] = useState(getOrCreateSessionId);
  const [userDisplayName, setDisplayNameState] = useState(getUserDisplayName);
  const [availableSessions, setAvailableSessions] = useState(() => listUserSessions(userId));

  const [activeTab, setActiveTab] = useState("chat");
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window !== "undefined") {
      return window.innerWidth > 820;
    }
    return true;
  });
  const [selectedDocId, setSelectedDocId] = useState(null);

  const closeSidebarOnMobile = useCallback(() => {
    if (typeof window !== "undefined" && window.innerWidth <= 820) {
      setSidebarOpen(false);
    }
  }, []);

  const handleTabSwitch = useCallback(
    (tab) => {
      setActiveTab(tab);
      closeSidebarOnMobile();
    },
    [closeSidebarOnMobile]
  );

  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState(() => loadSessionChat(userId, sessionId));
  const [documents, setDocuments] = useState([]);
  const [toast, setToast] = useState(null);
  const [confirmModalConfig, setConfirmModalConfig] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [pendingUploads, setPendingUploads] = useState([]);
  const [isSavingText, setIsSavingText] = useState(false);
  const [isDeletingId, setIsDeletingId] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [rateLimitSeconds, setRateLimitSeconds] = useState(0);

  const showToast = useCallback((message, type = "info") => {
    setToast({ message, type });
  }, []);

  // 1. Check health on mount
  useEffect(() => {
    getHealth()
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  // 2. Persist chat messages whenever they change
  useEffect(() => {
    saveSessionChat(userId, sessionId, messages);
    if (messages.length > 0) {
      recordSessionForUser(userId, sessionId);
      setAvailableSessions(listUserSessions(userId));
    }
  }, [userId, sessionId, messages]);

  // 3. Fetch documents for active session
  const fetchSessionDocuments = useCallback(async () => {
    try {
      const res = await getDocuments(sessionId, userId);
      const items = res.items || [];
      setDocuments(items);
      if (items.length > 0) {
        recordSessionForUser(userId, sessionId);
        setAvailableSessions(listUserSessions(userId));
      }
    } catch {
      setDocuments([]);
    }
  }, [sessionId, userId]);

  useEffect(() => {
    fetchSessionDocuments();
  }, [fetchSessionDocuments]);

  // 4. Update display name handler
  const handleUpdateDisplayName = (newName) => {
    setUserDisplayName(newName);
    setDisplayNameState(newName);
    showToast(`Display name updated to "${newName}".`, "success");
  };

  // 5. Handle sending a chat question
  const handleSendMessage = async (question) => {
    if (!question || !question.trim()) return;
    const userMsg = { role: "user", content: question.trim() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setIsLoading(true);

    try {
      const data = await askQuestion(question.trim(), sessionId, userId);
      const botMsg = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        latency_ms: data.latency_ms,
        confidence_score: data.confidence_score,
        response_type: data.response_type,
      };
      setMessages([...updatedMessages, botMsg]);
    } catch (err) {
      if (err.status === 429) {
        const cooldown = err.retryAfter || 60;
        setRateLimitSeconds(cooldown);
        showToast(`Rate limit reached (20 req/min). Cooldown for ${cooldown}s.`, "warning");
      } else {
        showToast(err.message || "Failed to formulate an answer.", "error");
        setMessages([
          ...updatedMessages,
          {
            role: "assistant",
            content: `The assistant encountered an error: ${err.message}`,
            sources: [],
            latency_ms: 0,
            confidence_score: 0.0,
          },
        ]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // 6. Handle batch file upload with live in-table progress rows
  const handleUploadFiles = async (files) => {
    setIsUploading(true);

    // Optimistically insert pending rows directly into the table
    const initialPending = files.map((file, idx) => ({
      temp_id: `pending_${Date.now()}_${idx}`,
      file_name: file.name,
      title: file.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, " "),
      status: "queued",
      stage: "QUEUED",
      percent: 0,
      message: "Queued for indexing...",
      chunk_count: null,
      total_pages: null,
      created_at: new Date().toISOString(),
    }));
    setPendingUploads(initialPending);

    try {
      const res = await uploadFiles(files, sessionId, userId, { background: true });

      // If async background job was registered (HTTP 202)
      if (res && res.job_id) {
        // Subscribe to real-time SSE progress stream with polling fallback
        await new Promise((resolve) => {
          const unsubscribe = subscribeJobProgress(res.job_id, {
            onProgress: (snap) => {
              setPendingUploads((prev) =>
                prev.map((item) => {
                  // Check if this file has completed
                  const matchingResult = snap.results?.find((r) => r.file_name === item.file_name);
                  if (matchingResult) {
                    return {
                      ...item,
                      status: "indexed",
                      percent: 100,
                      chunk_count: matchingResult.chunk_count,
                      total_pages: matchingResult.total_pages,
                      document_id: matchingResult.document_id,
                      message: "Indexed successfully",
                    };
                  }
                  // Check if this file failed
                  const matchingError = snap.errors?.find((e) => e.file_name === item.file_name);
                  if (matchingError) {
                    return {
                      ...item,
                      status: "failed",
                      error_message: matchingError.error_message || "Indexing failed",
                      message: matchingError.error_message || "Failed",
                    };
                  }
                  // Check if this file is actively being processed
                  if (snap.current_file === item.file_name) {
                    return {
                      ...item,
                      status: "processing",
                      stage: snap.stage,
                      percent: snap.percent,
                      message: snap.message,
                    };
                  }
                  // Still queued waiting for sequential turn
                  return item;
                })
              );
            },
            onComplete: (snap) => {
              unsubscribe();
              const results = snap.results || [];
              const errors = snap.errors || [];
              const indexedCount = results.filter((d) => d.status === "indexed").length;
              const failedDocs = errors.length > 0 ? errors : results.filter((d) => d.status === "failed");

              // Mark all items according to final snapshot
              setPendingUploads((prev) =>
                prev.map((item) => {
                  const matchingResult = results.find((r) => r.file_name === item.file_name);
                  if (matchingResult) {
                    return {
                      ...item,
                      status: "indexed",
                      percent: 100,
                      chunk_count: matchingResult.chunk_count,
                      total_pages: matchingResult.total_pages,
                      document_id: matchingResult.document_id,
                    };
                  }
                  const matchingError = errors.find((e) => e.file_name === item.file_name);
                  if (matchingError) {
                    return {
                      ...item,
                      status: "failed",
                      error_message: matchingError.error_message || "Indexing failed",
                    };
                  }
                  return item;
                })
              );

              if (failedDocs.length > 0 && indexedCount > 0) {
                const failedDetails = failedDocs
                  .map((d) => `${d.file_name} (${d.error_message || "no extractable text"})`)
                  .join(", ");
                showToast(
                  `Indexed ${indexedCount} file(s). Failed/Skipped ${failedDocs.length}: ${failedDetails}`,
                  "warning"
                );
              } else if (failedDocs.length > 0 && indexedCount === 0) {
                const failedDetails = failedDocs
                  .map((d) => `${d.file_name} (${d.error_message || "unreadable format"})`)
                  .join(", ");
                showToast(
                  `Indexing failed for all file(s): ${failedDetails}`,
                  "error"
                );
              } else {
                showToast(`Successfully indexed ${indexedCount || files.length} document(s).`, "success");
              }

              // Fetch persisted records from backend, then clear optimistic pending items
              fetchSessionDocuments().then(() => {
                setTimeout(() => {
                  setPendingUploads([]);
                }, 1500);
              });
              resolve();
            },
            onError: (err) => {
              unsubscribe();
              showToast(err.message || "Background indexing tracking encountered an error.", "error");
              setPendingUploads([]);
              resolve();
            },
          });
        });

        return;
      }

      // Synchronous fallback (if backend returned direct 201)
      const indexedCount =
        typeof res.indexed_count === "number"
          ? res.indexed_count
          : res.documents?.filter((d) => d.status === "indexed").length || 0;
      const failedDocs = res.documents?.filter((d) => d.status === "failed") || [];

      if (failedDocs.length > 0 && indexedCount > 0) {
        const skippedNames = failedDocs.map((d) => d.file_name).join(", ");
        showToast(
          `Indexed ${indexedCount} file(s). Skipped ${failedDocs.length} file(s): ${skippedNames}`,
          "warning"
        );
      } else if (failedDocs.length > 0 && indexedCount === 0) {
        showToast(`No documents indexed. File(s) contain no readable text.`, "error");
      } else {
        showToast(`Successfully indexed ${indexedCount || files.length} document(s).`, "success");
      }
      await fetchSessionDocuments();
    } catch (err) {
      showToast(err.message || "File upload failed.", "error");
      setPendingUploads([]);
    } finally {
      setIsUploading(false);
    }
  };

  // 7. Handle direct raw text ingestion
  const handleCreateTextDoc = async (title, content) => {
    setIsSavingText(true);
    try {
      await createDocument(title, content, sessionId, userId);
      recordSessionForUser(userId, sessionId);
      setAvailableSessions(listUserSessions(userId));
      showToast(`Indexed article "${title}" successfully.`, "success");
      await fetchSessionDocuments();
    } catch (err) {
      showToast(err.message || "Failed to index text document.", "error");
    } finally {
      setIsSavingText(false);
    }
  };

  // 8. Handle document deletion via In-App Confirm Dialog
  const handleDeleteDoc = (docId, title) => {
    setConfirmModalConfig({
      title: `Delete "${title}"?`,
      subtitle: "PURGE DOCUMENT & VECTORS",
      message: `Are you sure you want to delete "${title}"? Its text chunks and dense vector embeddings will be permanently purged from ChromaDB.`,
      confirmLabel: "Delete Document",
      confirmVariant: "danger",
      icon: "🗑️",
      onConfirm: async () => {
        setConfirmModalConfig(null);
        setIsDeletingId(docId);
        try {
          await deleteDocument(docId);
          showToast(`Deleted "${title}" successfully.`, "info");
          await fetchSessionDocuments();
        } catch (err) {
          showToast(err.message || "Failed to delete document.", "error");
        } finally {
          setIsDeletingId(null);
        }
      },
    });
  };

  // 9. Handle new session generation
  const handleNewSession = () => {
    // Avoid ghost sessions: if current session is already fresh and empty, reuse it
    const hasMessages = messages.length > 0;
    const hasDocuments = documents.length > 0;
    if (!hasMessages && !hasDocuments) {
      showToast(`Current session (${sessionId}) is already fresh and empty.`, "info");
      return;
    }

    const newSessId = createNewSession();
    setSessionIdState(newSessId);
    setMessages([]);
    setDocuments([]);
    setAvailableSessions(listUserSessions(userId));
    showToast(`Switched to new isolated session: ${newSessId}`, "info");
  };

  // 10. Handle switching to a past session
  const handleSelectSession = (targetSessionId) => {
    if (targetSessionId === sessionId) return;
    setSessionId(targetSessionId);
    setSessionIdState(targetSessionId);
    const restoredChat = loadSessionChat(userId, targetSessionId);
    setMessages(restoredChat);
    showToast(`Loaded session ${targetSessionId} with ${restoredChat.length} past message(s).`, "info");
  };

  // 11. Handle refreshing session & conversation
  const handleRefreshSession = async () => {
    setIsRefreshing(true);
    try {
      const restoredChat = loadSessionChat(userId, sessionId);
      setMessages(restoredChat);
      await fetchSessionDocuments();
      const healthData = await getHealth().catch(() => ({ status: "offline" }));
      setHealth(healthData);
      setAvailableSessions(listUserSessions(userId));
      showToast(`Session ${sessionId} synced: documents and past conversation updated.`, "success");
    } finally {
      setIsRefreshing(false);
    }
  };

  // 12. Handle clearing chat via In-App Confirm Dialog
  const handleClearChat = () => {
    setConfirmModalConfig({
      title: "Clear Current Conversation?",
      subtitle: "CLEAR CHAT THREAD",
      message: "Are you sure you want to clear all messages in this session? This will remove your active conversation thread from view. Your session ID and indexed documents remain intact.",
      confirmLabel: "Clear Messages",
      confirmVariant: "danger",
      icon: "🗑️",
      onConfirm: () => {
        clearSessionChat(userId, sessionId);
        setMessages([]);
        showToast("Conversation cleared for this session.", "info");
        setConfirmModalConfig(null);
      },
    });
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        health={health}
        docCount={documents.filter((d) => d.status === "indexed").length}
        activeTab={activeTab}
        setActiveTab={handleTabSwitch}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
        sidebarOpen={sidebarOpen}
      />

      {/* Main Workspace Layout */}
      <div className={`workspace-layout ${sidebarOpen ? "with-sidebar" : "full-width"}`}>
        {/* Center Workspace (Tabs) */}
        <main
          className={`main-content-zone ${
            activeTab === "documents" ? "documents-mode" : "chat-mode"
          }`}
        >
          {activeTab === "chat" ? (
            <ChatView
              messages={messages}
              onSendMessage={handleSendMessage}
              onClearChat={handleClearChat}
              isLoading={isLoading}
              rateLimitSeconds={rateLimitSeconds}
              onRateLimitReset={() => setRateLimitSeconds(0)}
              sessionId={sessionId}
              docCount={documents.filter((d) => d.status === "indexed").length}
              onSwitchToDocsTab={() => handleTabSwitch("documents")}
            />
          ) : (
            <DocumentsView
              documents={documents}
              onUploadFiles={handleUploadFiles}
              onCreateTextDoc={handleCreateTextDoc}
              onDeleteDocument={handleDeleteDoc}
              isUploading={isUploading}
              isSavingText={isSavingText}
              isDeletingId={isDeletingId}
              onRefreshDocs={fetchSessionDocuments}
              sessionId={sessionId}
              selectedDocId={selectedDocId}
              onSelectDocId={setSelectedDocId}
              pendingUploads={pendingUploads}
            />
          )}
        </main>

        {/* Mobile Backdrop Overlay (Closes drawer on outside tap) */}
        {sidebarOpen && (
          <div
            className="sidebar-backdrop"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close session panel"
          />
        )}

        {/* Right User & Session Intelligence Sidebar (Pinned desktop 2nd column / Mobile slide-over drawer) */}
        {sidebarOpen && (
          <UserSessionSidebar
            userId={userId}
            sessionId={sessionId}
            userDisplayName={userDisplayName}
            onUpdateDisplayName={handleUpdateDisplayName}
            documents={documents}
            health={health}
            onRefreshSession={handleRefreshSession}
            onNewSession={() => {
              handleNewSession();
              closeSidebarOnMobile();
            }}
            onSelectSession={(targetId) => {
              handleSelectSession(targetId);
              closeSidebarOnMobile();
            }}
            availableSessions={availableSessions}
            onSelectDocument={(docId) => {
              setSelectedDocId(docId);
              setActiveTab("documents");
              closeSidebarOnMobile();
            }}
            onSwitchToDocsTab={() => {
              setActiveTab("documents");
              closeSidebarOnMobile();
            }}
            isRefreshing={isRefreshing}
            onClose={() => setSidebarOpen(false)}
          />
        )}
      </div>

      {/* In-App Confirmation Dialog (Replaces native browser window.confirm popups) */}
      {confirmModalConfig && (
        <ConfirmModal
          isOpen={true}
          title={confirmModalConfig.title}
          subtitle={confirmModalConfig.subtitle}
          message={confirmModalConfig.message}
          confirmLabel={confirmModalConfig.confirmLabel}
          confirmVariant={confirmModalConfig.confirmVariant}
          icon={confirmModalConfig.icon}
          onConfirm={confirmModalConfig.onConfirm}
          onCancel={() => setConfirmModalConfig(null)}
        />
      )}

      {/* Global Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}
