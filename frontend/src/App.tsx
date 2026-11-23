import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  ArrowUp, BookOpen, CalendarDays, Check, ChevronDown, Clock3, GraduationCap,
  History, Menu, MessageCircle, NotebookText, Search, Send, Sparkles,
  UsersRound, X,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { Textarea } from "./components/ui/textarea";
import { checkBackend, listConversations, loadConversation, sendChatMessage, type ChatMessage, type ConversationSummary } from "./lib/chat-api";

const shortcuts = [
  { label: "Upcoming Deadlines?", prompt: "What are my upcoming college deadlines?", icon: CalendarDays, tone: "cyan" },
  { label: "Events & Notices?", prompt: "Summarize the latest college notices and upcoming campus events from the annual calendar, highlighting dates and details.", icon: NotebookText, tone: "violet" },
  { label: "Exam Schedule?", prompt: "Help me find my exam schedule.", icon: GraduationCap, tone: "blue" },
  { label: "Course Information?", prompt: "List the available programs, semester course examples, and any documented elective options.", icon: BookOpen, tone: "teal" },
  { label: "Internship / Placement Notices?", prompt: "Are there new internship or placement opportunities?", icon: UsersRound, tone: "pink" },
];

const starters = [
  { title: "Upcoming deadlines", date: "Today", icon: CalendarDays, prompt: "What are my upcoming college deadlines?" },
  { title: "Exam schedule", date: "Today", icon: GraduationCap, prompt: "Help me find my exam schedule." },
  { title: "Elective options", date: "Yesterday", icon: BookOpen, prompt: "I need information about my courses and electives." },
  { title: "Internship notices", date: "Yesterday", icon: UsersRound, prompt: "Are there new internship or placement opportunities?" },
  { title: "Bonafide application", date: "Nov 14", icon: NotebookText, prompt: "How do I apply for a bonafide certificate?" },
  { title: "Semester dates", date: "Nov 13", icon: Clock3, prompt: "When will the next semester begin?" },
];

function makeMessage(role: ChatMessage["role"], content: string, sources: ChatMessage["sources"] = []): ChatMessage {
  return { id: crypto.randomUUID(), role, content, createdAt: Date.now(), sources };
}

function getIndiaGreeting(): string {
  const hour = Number(new Intl.DateTimeFormat("en-IN", {
    hour: "numeric",
    hourCycle: "h23",
    timeZone: "Asia/Kolkata",
  }).format(new Date()));
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const [activePrompt, setActivePrompt] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const conversationListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const list = conversationListRef.current;
    if (list) list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function refreshHistory() {
    try { setConversations(await listConversations()); }
    catch { setNotice("Conversation history is temporarily unavailable."); }
    finally { setHistoryLoading(false); }
  }

  useEffect(() => { void refreshHistory(); }, []);
  useEffect(() => {
    let active = true;
    const ping = async () => { const online = await checkBackend(); if (active) setBackendOnline(online); };
    void ping();
    const timer = window.setInterval(() => void ping(), 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 1800);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function openConversation(id: string) {
    setMobileSidebar(false);
    setNotice("");
    setLoading(true);
    try {
      setMessages(await loadConversation(id));
      setConversationId(id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Unable to open that conversation.");
    } finally { setLoading(false); }
  }

  async function submit(content = draft) {
    const text = content.trim();
    if (!text || loading) return;
    const userMessage = makeMessage("user", text);
    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setActivePrompt(null);
    setLoading(true);
    try {
      const result = await sendChatMessage(text, conversationId);
      setConversationId(result.conversationId);
      setMessages((current) => [...current, makeMessage("assistant", result.answer, result.sources)]);
      await refreshHistory();
    } catch (error) {
      setMessages((current) => [...current, makeMessage("assistant", `I couldn't complete that request. ${error instanceof Error ? error.message : "Please try again."}`)]);
      setNotice(error instanceof Error ? error.message : "I couldn't reach Lumen AI. Please try again.");
    } finally { setLoading(false); }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void submit(); }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); }
  }
  function startNewChat() {
    setMessages([]);
    setConversationId(null);
    setDraft("");
    setActivePrompt(null);
    setMobileSidebar(false);
    textareaRef.current?.focus();
  }
  async function copyAnswer(message: ChatMessage) {
    try { await navigator.clipboard.writeText(message.content); setCopied(message.id); window.setTimeout(() => setCopied(null), 1600); }
    catch { setNotice("Clipboard access isn’t available here."); window.setTimeout(() => setNotice(""), 2200); }
  }

  return (
    <div className="lumen-app">
      <header className="topbar">
        <div className="brand-lockup">
          <button className="mobile-toggle" onClick={() => setMobileSidebar((open) => !open)} aria-label="Toggle history"><Menu size={20} /></button>
          <div className="brand-emblem"><BookOpen size={27} strokeWidth={1.7} /></div>
          <span className="brand-name">Lumen AI</span>
          <span className="brand-divider" />
          <span className="brand-tagline">Your Intelligent College Companion</span>
        </div>
        <div className="topbar-tools"><span className={`preview-badge backend-status ${backendOnline === false ? "offline" : backendOnline ? "online" : "checking"}`} title={backendOnline ? "Connected to Lumen AI backend" : backendOnline === false ? "Lumen AI backend is unavailable" : "Checking backend connection"}><span />{backendOnline ? "Connected" : backendOnline === false ? "Reconnecting" : "Connecting"}</span><button className="search-button" onClick={() => { textareaRef.current?.focus(); }} aria-label="Focus chat"><Search size={17} /></button></div>
      </header>

      <div className="app-body">
        <AnimatePresence>
          {(mobileSidebar || window.innerWidth > 760) && <motion.aside className={`history-sidebar ${mobileSidebar ? "mobile-visible" : ""}`} initial={{ x: -16, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: -16, opacity: 0 }}>
            <div className="history-heading"><div><History size={21} /><h2>History</h2></div><button aria-label="Close history" className="close-mobile" onClick={() => setMobileSidebar(false)}><X size={18} /></button></div>
            <Button variant="outline" className="new-chat-button" onClick={startNewChat}><span>＋</span> New conversation</Button>
            <div className="history-label">{conversations.length ? "RECENT CONVERSATIONS" : "SUGGESTED QUESTIONS"}</div>
            <div className="history-list">
              {historyLoading ? <div className="history-empty">Loading your conversations...</div> : conversations.length ? conversations.map((item) => <button className={`history-item ${conversationId === item.id ? "current" : ""}`} key={item.id} onClick={() => void openConversation(item.id)}><MessageCircle size={17} /><span><strong>{item.title}</strong><small>{Number.isFinite(item.updatedAt) ? new Date(item.updatedAt).toLocaleDateString() : "Recent"}</small></span></button>) : starters.map((item) => <button className="history-item" key={item.title} onClick={() => { setDraft(item.prompt); setMobileSidebar(false); textareaRef.current?.focus(); }}><MessageCircle size={18} /><span><strong>{item.title}</strong><small>Ask Lumen AI</small></span></button>)}
            </div>
            <div className="sidebar-bottom"><div className="campus-card"><div className="campus-card-icon"><Sparkles size={18} /></div><div><strong>A little more clarity.</strong><small>Your campus, in one place.</small></div><span className="campus-card-glow" /></div><button className="student-profile"><span className="profile-avatar">S</span><span><strong>Student</strong><small>College information</small></span><ChevronDown size={16} /></button></div>
          </motion.aside>}
        </AnimatePresence>

        <main className="main-stage" onMouseLeave={() => setNotice("")}>
          <div className="ambient ambient-a" /><div className="ambient ambient-b" />
          <svg className="desk-prop desk-prop-left" viewBox="0 0 220 190" fill="none" aria-hidden="true"><path d="M0 165h220M24 165l19-18h104l19 18M34 147h110v8H34zM56 144l11-9h89l-9 9M54 165v16m120-16v16M72 82v63m0-43C42 89 41 62 44 48c22 6 35 22 31 43m-3 13c-3-28 10-47 27-55 8 24-1 44-24 58m25 12c0-24 13-41 30-45 3 22-8 38-30 50m-29-5c-21-5-31-21-29-39 19 2 31 14 34 35" stroke="currentColor" strokeWidth="2"/><path d="M43 48c19 8 29 21 32 43m21-45c-11 20-18 39-21 58m52 1c-13 10-22 19-29 32" stroke="currentColor" strokeWidth="1"/><path d="M37 165h142l-10 12H47z" fill="#101d36" stroke="currentColor" strokeWidth="1"/><path d="M54 178h108v5H54z" fill="#172845" stroke="currentColor" strokeWidth="1"/></svg>
          <svg className="desk-prop desk-prop-right" viewBox="0 0 280 145" fill="none" aria-hidden="true"><path d="M11 117h246l-16 15H28z" fill="#111d36" stroke="currentColor" strokeWidth="1.4"/><path d="M25 99h224l-8 17H32z" fill="#172744" stroke="currentColor" strokeWidth="1.4"/><path d="M33 82h208l8 16H26z" fill="#101d36" stroke="currentColor" strokeWidth="1.4"/><path d="M45 66h191v14H40z" fill="#1a2947" stroke="currentColor" strokeWidth="1.4"/><path d="M193 132v7m39-7v7m-40 0h41" stroke="currentColor" strokeWidth="1.4"/><path d="M47 70h170m-177 15h189m-192 16h192m-196 18h213" stroke="currentColor" strokeWidth=".7" opacity=".7"/></svg>
          <svg className="campus-sketch" viewBox="0 0 440 230" fill="none" aria-hidden="true"><path d="M23 177h388M53 177v-70l58-37 61 37v70M74 177v-48h76v48M97 129v48M53 107h119M111 70V45h21v25M100 45h43M278 177v-54h99v54M292 123V90l35-25 36 25v33M311 91v32m33-32v32M327 65V37m-10 15 10-16 10 16M263 177h133M38 190h371" stroke="currentColor" strokeWidth="1.2"/><circle cx="111" cy="102" r="9" stroke="currentColor"/><path d="M107 102h8m-4-4v8" stroke="currentColor" strokeWidth="1.1" /></svg>
          <div className="ambient-lamp" aria-hidden="true"><span /></div>

          <AnimatePresence initial={false}>
            {messages.length === 0 ? <motion.section key="welcome" className="welcome-view" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: .42 }}>
              <div className="welcome-orbit"><div className="orbit-ring ring-outer" /><div className="orbit-ring ring-inner" /><div className="welcome-spark spark-top">✦</div><div className="welcome-spark spark-side">✧</div><div className="welcome-book"><BookOpen size={70} strokeWidth={1.25} /></div></div>
              <div className="welcome-copy"><h1>{getIndiaGreeting()}</h1><p className="welcome-subtitle">How can I assist you today?</p><div className="welcome-rule"><span /><i>✦</i><span /></div></div>
              <div className="shortcut-grid">{shortcuts.map(({ label, prompt, icon: Icon, tone }, index) => <motion.button key={label} className={`shortcut shortcut-${tone} shortcut-${index + 1}`} whileHover={{ y: -3, scale: 1.015 }} whileTap={{ scale: .985 }} onClick={() => { setDraft(prompt); setActivePrompt(label); textareaRef.current?.focus(); }}><span className="shortcut-icon"><Icon size={24} strokeWidth={1.8} /></span><span>{label}</span><ArrowUp className="shortcut-arrow" size={13} /></motion.button>)}</div>
            </motion.section> : <motion.section key="conversation" className="conversation-view" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="conversation-heading"><span className="conversation-sigil"><Sparkles size={17} /></span><div><small>YOUR CAMPUS COMPANION</small><h1>Let’s figure it out.</h1></div></div>
              <div className="conversation-list" ref={conversationListRef}>{messages.map((message, index) => <motion.article key={message.id} className={`chat-message ${message.role}`} initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index === messages.length - 1 ? .04 : 0 }}><div className="message-avatar">{message.role === "assistant" ? <Sparkles size={16} /> : "S"}</div><div className="message-content"><div className="message-byline">{message.role === "assistant" ? "Lumen AI" : "You"} <span>{"\u00b7"}</span> {new Date(message.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</div><p>{message.content}</p>{message.role === "assistant" && message.sources !== undefined && <details className="answer-sources"><summary>Sources <span>{message.sources.length}</span></summary><div className="answer-source-list">{message.sources.map((source, sourceIndex) => <div className="answer-source" key={`${source.document_id}-${sourceIndex}`}><span className={`source-badge ${source.source_type === "simulated_demo" ? "demo" : "official"}`}>{source.source_type === "simulated_demo" ? "DEMO DATA \u00b7 FICTIONAL \u00b7 NOT OFFICIAL" : "OFFICIAL COLLEGE SOURCE"}</span><strong>{source.title.replace(/^\[DEMO DATA\]\s*/i, "")}</strong>{source.category && <small>{source.category}</small>}<details className="source-excerpt"><summary>View supporting excerpt</summary><p>{source.snippet.replace(/^DEMO DATA\s*[-:]\s*FICTIONAL AIT COLLEGE SCENARIO\.[^\n]*\n*/i, "")}</p></details></div>)}</div></details>}{message.role === "assistant" && <button className="copy-answer" onClick={() => void copyAnswer(message)}>{copied === message.id ? <Check size={13} /> : <NotebookText size={13} />}{copied === message.id ? "Copied" : "Copy answer"}</button>}</div></motion.article>)}{loading && <div className="chat-message assistant"><div className="message-avatar"><Sparkles size={16} /></div><div className="typing-indicator"><i /><i /><i /></div></div>}</div>
            </motion.section>}
          </AnimatePresence>

          <div className={`composer-wrap ${messages.length ? "composer-in-conversation" : ""}`}>
            <form className="chat-composer" onSubmit={onSubmit}>
              {activePrompt && <div className="active-prompt"><span>{activePrompt}</span><button type="button" onClick={() => setActivePrompt(null)} aria-label="Clear selected prompt"><X size={14} /></button></div>}
              <Textarea ref={textareaRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask anything about your college…" aria-label="Ask Lumen AI a question" rows={1} />
              <div className="composer-controls"><div className="composer-left"><span className="composer-hint">Your campus companion, whenever you need it.</span></div><div className="composer-right"><button className="mode-button" type="button" disabled aria-disabled="true"><Sparkles size={13} /> Campus mode <ChevronDown size={13} /></button><Button className="send-button" type="submit" aria-label="Send message" disabled={!draft.trim() || loading}><Send size={17} /></Button></div></div>
            </form>
            <div className="composer-disclaimer">Lumen AI can make mistakes. Confirm important dates and policies with your college.</div>
          </div>
          <div className="desk-glow" aria-hidden="true" /><div className="desk-line" aria-hidden="true" />
          <div className="stage-footer"><span><span className="footer-star">✦</span> Here for the big questions &amp; the small ones.</span><span>STAY CURIOUS <i>·</i> YOU’VE GOT THIS</span></div>
          <AnimatePresence>{notice && <motion.div className="toast-notice" onMouseLeave={() => setNotice("")} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>{notice}</motion.div>}</AnimatePresence>
        </main>
      </div>
    </div>
  );
}
