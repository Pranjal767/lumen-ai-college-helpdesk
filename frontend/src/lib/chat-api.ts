export type ChatRole = "user" | "assistant";
export type ChatSource = {
  document_id: number;
  title: string;
  url: string | null;
  page: number | null;
  snippet: string;
  source_type: string;
  record_id: string | null;
  category: string | null;
};
export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: ChatSource[];
};
export type ConversationSummary = { id: string; title: string; updatedAt: number };

const API_ROOT = "/api/v1";
const SESSION_KEY = "lumen.ai.session";
const LEGACY_SESSION_KEY = "lumenen.ai.session";

function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY) || localStorage.getItem(LEGACY_SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
  }
  localStorage.setItem(SESSION_KEY, id);
  localStorage.removeItem(LEGACY_SESSION_KEY);
  return id;
}

const sessionId = getSessionId();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const data: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const payload = data as { error?: { message?: string } };
    throw new Error(payload.error?.message || `Chat request failed (${response.status}).`);
  }
  return data as T;
}

export async function checkBackend(): Promise<boolean> {
  try {
    const data = await request<{ status?: string }>("/health", { signal: AbortSignal.timeout(5000) });
    return data.status === "healthy";
  } catch {
    return false;
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const data = await request<{ conversations: Array<{ id: string; title: string; updated_at: string }> }>(
    `/conversations?session_id=${encodeURIComponent(sessionId)}`,
  );
  return (data.conversations || []).map((item) => ({
    id: item.id,
    title: item.title || "College question",
    updatedAt: new Date(item.updated_at).getTime(),
  }));
}

export async function loadConversation(id: string): Promise<ChatMessage[]> {
  const data = await request<{ conversation: { messages: Array<{ id: number; role: ChatRole; content: string; created_at: string }> } }>(
    `/conversations/${encodeURIComponent(id)}?session_id=${encodeURIComponent(sessionId)}`,
  );
  return (data.conversation.messages || []).map((message) => ({
    id: String(message.id),
    role: message.role,
    content: message.content,
    createdAt: new Date(message.created_at).getTime(),
    sources: [],
  }));
}

export async function sendChatMessage(
  question: string,
  conversationId: string | null,
): Promise<{ answer: string; conversationId: string; sources: ChatSource[] }> {
  const data = await request<{
    answer: string;
    conversation_id: string;
    sources: unknown;
  }>("/chat", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: sessionId,
      conversation_id: conversationId,
    }),
  });
  if (typeof data.answer !== "string" || typeof data.conversation_id !== "string") {
    throw new Error("Lumen AI returned an incomplete response. Please try again.");
  }
  const sources = Array.isArray(data.sources) ? data.sources : [];
  return {
    answer: data.answer,
    conversationId: data.conversation_id,
    sources: sources.map((value) => {
      const source = (value && typeof value === "object" ? value : {}) as Partial<ChatSource>;
      return {
        document_id: Number(source.document_id) || 0,
        title: typeof source.title === "string" ? source.title : "Untitled source",
        url: typeof source.url === "string" ? source.url : null,
        page: typeof source.page === "number" ? source.page : null,
        snippet: typeof source.snippet === "string" ? source.snippet : "",
        source_type: typeof source.source_type === "string" ? source.source_type : "unknown",
        record_id: typeof source.record_id === "string" ? source.record_id : null,
        category: typeof source.category === "string" ? source.category : null,
      };
    }),
  };
}
