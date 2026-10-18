/**
 * Text and metric formatters for RAG responses and document management.
 */

export function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function formatDate(dateString) {
  if (!dateString) return "Just now";
  try {
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return dateString;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) +
      ", " +
      d.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return dateString;
  }
}

export function formatLatency(ms) {
  if (ms === null || ms === undefined) return "";
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Parses grounded LLM output into a natural summary paragraph and bulleted/stepwise points.
 * Ensures that headers like "📌 Quick Summary" or "📋 Key Details & Steps" are cleanly stripped.
 */
export function parseBotAnswer(rawText) {
  if (!rawText) return { summary: "", breakdown: null, isRefusal: false };

  // Strip explicit section headers / emoji titles that might be generated
  let cleaned = rawText
    .replace(/(?:^|\n)\s*(?:[#*]{1,4}\s*)?(?:[📌📋📝🔹•-]\s*)?(?:quick\s*summary|key\s*(?:details|points|steps|takeaways|breakdown))(?:\s*&?\s*(?:steps|details|breakdown))?:?\s*(?:[*#]{1,4})?\s*(?:\n|$)/gi, "\n")
    .trim();

  const lower = cleaned.toLowerCase();

  const isEmptySession =
    lower.includes("does not contain any uploaded documents") ||
    lower.includes("no documents, pdfs, or notes have been uploaded") ||
    lower.includes("no documents or notes have been uploaded") ||
    lower.includes("session is empty") ||
    lower.includes("session knowledge base is empty") ||
    lower.includes("no documents or notes have been added");

  const isNoMatch =
    lower.includes("does not contain sufficient information") ||
    lower.includes("does not contain information") ||
    lower.includes("insufficient information");

  const isRefusal = isEmptySession || isNoMatch;

  if (isRefusal) {
    return {
      summary: cleaned,
      breakdown: null,
      isRefusal: true,
      refusalType: isEmptySession ? "empty_session" : "no_relevant_info",
    };
  }

  // Check for double newline paragraphs
  const paragraphs = cleaned.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);

  if (paragraphs.length > 1) {
    let summary = paragraphs[0];
    summary = summary.replace(/^\*\*(?:quick\s*)?summary:?\*\*\s*/i, "");
    const breakdown = paragraphs.slice(1).join("\n\n");
    return { summary, breakdown, isRefusal: false };
  }

  // Check single block with embedded bullets
  const lines = cleaned.split("\n");
  const firstBulletIdx = lines.findIndex((l) =>
    /^[-*•]|\d+\./.test(l.trim())
  );

  if (firstBulletIdx > 0) {
    let summary = lines.slice(0, firstBulletIdx).join(" ").trim();
    summary = summary.replace(/^\*\*(?:quick\s*)?summary:?\*\*\s*/i, "");
    const breakdown = lines.slice(firstBulletIdx).join("\n").trim();
    return { summary, breakdown, isRefusal: false };
  }

  return { summary: cleaned, breakdown: null, isRefusal: false };
}
