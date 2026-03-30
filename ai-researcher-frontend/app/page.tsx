"use client";

import { useState, useRef, useEffect, type ComponentProps, type DragEvent, type KeyboardEvent } from "react";
import {
  Send, Paperclip, Plus, Search, Bot, User, Sparkles,
  UploadCloud, Cpu, ChevronDown, ChevronUp, ExternalLink,
  BookOpen, FileText, Loader2, Library, Trash2, AlertCircle, X, Download
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const ENV_DEFAULT_PROJECT_ID = process.env.NEXT_PUBLIC_DEFAULT_PROJECT_ID || "";

// ── Types ──────────────────────────────────────────────────────────────────────

type Source = {
  id: string;
  type: string;
  title?: string;
  source_id?: string;
  arxiv_id?: string;
  pdf_url?: string;
  abs_url?: string;
  link?: string;
  source?: string;
  page?: number;
  citation_count?: number;
  journal?: string;
  metadata?: { page?: number; page_number?: number; [key: string]: unknown };
  [key: string]: unknown;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: string;
  sources?: Source[];
  suggestions?: string[];
};

type LibraryKind = "pdf_upload" | "arxiv" | "semantic_scholar" | "pubmed" | "openalex" | "web";

type LibraryItem = {
  id: string;
  title: string;
  type: LibraryKind;
  date: string;
  abs_url?: string;
  pdf_url?: string;
  is_metadata_only?: boolean;
};

type ProjectStats = {
  project_id: string;
  paper_count: number;
  chunk_count: number;
  papers_with_pdf: number;
  papers_with_links: number;
  source_counts: Record<string, number>;
};

type PaperResult = {
  title?: string;
  authors?: string[];
  summary?: string;
  published?: string;
  source?: string;
  source_id?: string;
  arxiv_id?: string;
  abs_url?: string;
  pdf_url?: string;
  citation_count?: number;
  journal?: string;
};

type LibraryResponse = {
  papers?: Array<{
    paper_id?: string;
    id?: string;
    title?: string;
    source?: string;
    arxiv_id?: string;
    abs_url?: string;
    pdf_url?: string;
    published?: string;
    is_metadata_only?: boolean;
  }>;
};

type PaperSearchResponse = {
  papers?: PaperResult[];
};

type NotebookResponse = {
  status: string;
  paper_id: string;
  title: string;
  file_name: string;
  notebook_json: string;
  preview_markdown: string;
  dependencies: string[];
  source_url?: string;
  generated_with_model: string;
  colab_ready: boolean;
  artifact_summary: string[];
  study_questions: string[];
  reproducibility_checklist: string[];
  risk_notes: string[];
  generation_goal: string;
  compute_profile: string;
};

type GenerateNotebookRequest = {
  api_key: string;
  model: string;
  generation_goal: string;
  compute_profile: string;
  include_study_questions: boolean;
  include_reproducibility_checklist: boolean;
  include_risk_notes: boolean;
};

type ApiError = {
  detail?: string;
};

type MarkdownComponents = ComponentProps<typeof ReactMarkdown>["components"];

// ── Helpers ────────────────────────────────────────────────────────────────────

function createNewProjectId() {
  return "proj-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

function cleanAnswer(text: string): string {
  return text
    .replace(/\[DONE\]/gi, "")
    .replace(/\[LOCAL\s+\d+(?:,\s*LOCAL\s+\d+)*\]/g, "")
    .trim();
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unknown error";
}

function normalizeLibraryType(source?: string, hasAbstractLink = false): LibraryKind {
  switch (source) {
    case "arxiv":
    case "semantic_scholar":
    case "pubmed":
    case "openalex":
    case "pdf_upload":
      return source;
    default:
      return hasAbstractLink ? "web" : "pdf_upload";
  }
}

function looksLikeArxivId(value?: string): boolean {
  if (!value) return false;
  return /^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+\/\d+(?:v\d+)?)$/i.test(value.trim());
}

/** Extract the best identifier from a paper result (works with old + new research_tool) */
function getPaperId(paper: PaperResult): string {
  if (looksLikeArxivId(paper.arxiv_id)) {
    return paper.arxiv_id || "";
  }
  return paper.source_id || paper.arxiv_id || "";
}

/** Returns the best URL to open for a paper */
function getPaperAbsUrl(paper: Pick<PaperResult, "abs_url" | "arxiv_id">): string {
  return paper.abs_url || (paper.arxiv_id ? `https://arxiv.org/abs/${paper.arxiv_id}` : "");
}

function formatLibraryType(type: LibraryKind): string {
  switch (type) {
    case "pdf_upload":
      return "uploaded pdf";
    case "semantic_scholar":
      return "semantic scholar";
    default:
      return type;
  }
}

// ── Main Page ──────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [mounted, setMounted] = useState(false);

  const [projectId, setProjectId] = useState<string>(() => {
    if (ENV_DEFAULT_PROJECT_ID) return ENV_DEFAULT_PROJECT_ID;
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ai_researcher_project_id");
      if (saved) return saved;
    }
    return createNewProjectId();
  });

  // session_id = stable per browser tab, used to scope chat history on backend
  const [sessionId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      let sid = sessionStorage.getItem("ai_researcher_session_id");
      if (!sid) {
        sid = "sess-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
        sessionStorage.setItem("ai_researcher_session_id", sid);
      }
      return sid;
    }
    return "sess-default";
  });

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isChatSending, setIsChatSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<"add" | "library">("add");
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>([]);
  const [projectStats, setProjectStats] = useState<ProjectStats | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [importingId, setImportingId] = useState<string | null>(null);
  const [failedId, setFailedId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<PaperResult[]>([]);
  const [removingId, setRemovingId] = useState<string | null>(null);

  const [generatedNotebook, setGeneratedNotebook] = useState<NotebookResponse | null>(null);
  const [isGeneratingNotebook, setIsGeneratingNotebook] = useState(false);
  const [showNotebookModal, setShowNotebookModal] = useState(false);
  const [notebookError, setNotebookError] = useState<string | null>(null);
  const [activeNotebookPaperId, setActiveNotebookPaperId] = useState<string | null>(null);
  const [selectedNotebookPaper, setSelectedNotebookPaper] = useState<LibraryItem | null>(null);
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [geminiModel, setGeminiModel] = useState("gemini-2.5-pro");
  const [notebookGoal, setNotebookGoal] = useState("teaching");
  const [computeProfile, setComputeProfile] = useState("balanced");
  const [includeStudyQuestions, setIncludeStudyQuestions] = useState(true);
  const [includeReproChecklist, setIncludeReproChecklist] = useState(true);
  const [includeRiskNotes, setIncludeRiskNotes] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const dragCounter = useRef(0);

  useEffect(() => {
    setMounted(true);
    document.documentElement.classList.add("dark");
    if (typeof window !== "undefined") {
      setGeminiApiKey(localStorage.getItem("ai_researcher_gemini_api_key") || "");
      setGeminiModel(localStorage.getItem("ai_researcher_gemini_model") || "gemini-2.5-pro");
      setNotebookGoal(localStorage.getItem("ai_researcher_notebook_goal") || "teaching");
      setComputeProfile(localStorage.getItem("ai_researcher_compute_profile") || "balanced");
      setIncludeStudyQuestions(localStorage.getItem("ai_researcher_study_questions") !== "false");
      setIncludeReproChecklist(localStorage.getItem("ai_researcher_repro_checklist") !== "false");
      setIncludeRiskNotes(localStorage.getItem("ai_researcher_risk_notes") !== "false");
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined" && !ENV_DEFAULT_PROJECT_ID) {
      localStorage.setItem("ai_researcher_project_id", projectId);
    }
  }, [projectId]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("ai_researcher_gemini_api_key", geminiApiKey);
      localStorage.setItem("ai_researcher_gemini_model", geminiModel);
      localStorage.setItem("ai_researcher_notebook_goal", notebookGoal);
      localStorage.setItem("ai_researcher_compute_profile", computeProfile);
      localStorage.setItem("ai_researcher_study_questions", String(includeStudyQuestions));
      localStorage.setItem("ai_researcher_repro_checklist", String(includeReproChecklist));
      localStorage.setItem("ai_researcher_risk_notes", String(includeRiskNotes));
    }
  }, [geminiApiKey, geminiModel, notebookGoal, computeProfile, includeStudyQuestions, includeReproChecklist, includeRiskNotes]);

  async function refreshProjectData(activeProjectId: string) {
    const [papersResponse, statsResponse] = await Promise.all([
      fetch(`${API_BASE}/projects/${encodeURIComponent(activeProjectId)}/papers`),
      fetch(`${API_BASE}/projects/${encodeURIComponent(activeProjectId)}/stats`),
    ]);

    if (!papersResponse.ok) {
      throw new Error(`Failed to load library: ${papersResponse.statusText}`);
    }

    const papersData = await papersResponse.json() as LibraryResponse;
    const items: LibraryItem[] = (papersData.papers || []).map((paper) => ({
      id: paper.paper_id || paper.id || "",
      title: paper.title || paper.paper_id || "Untitled",
      type: normalizeLibraryType(paper.source, Boolean(paper.abs_url)),
      date: paper.published || "",
      abs_url: paper.abs_url,
      pdf_url: paper.pdf_url,
      is_metadata_only: Boolean(paper.is_metadata_only),
    }));
    setLibraryItems(items);

    if (statsResponse.ok) {
      const statsData = await statsResponse.json() as ProjectStats;
      setProjectStats(statsData);
    } else {
      setProjectStats(null);
    }
  }

  // Load library from backend on mount / project change
  useEffect(() => {
    if (!projectId) return;

    void refreshProjectData(projectId).catch(() => {
      setLibraryItems([]);
      setProjectStats(null);
    });
  }, [projectId]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages, isChatSending]);

  function nowLabel() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  function generateMsgId() {
    return Math.random().toString(36).substring(7);
  }

  // ── Streaming ──────────────────────────────────────────────────────────────

  async function readStream(
    response: Response,
    onChunk: (text: string) => void,
    onEvent?: (type: "SOURCES" | "SUGGESTIONS", data: Source[] | string[]) => void,
    onError?: (message: string) => void
  ) {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;

    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data.startsWith("[SOURCES]")) {
          if (onEvent) onEvent("SOURCES", JSON.parse(data.slice(9)) as Source[]);
        } else if (data.startsWith("[SUGGESTIONS]")) {
          if (onEvent) onEvent("SUGGESTIONS", JSON.parse(data.slice(13)) as string[]);
        } else if (data.startsWith("[ERROR]")) {
          const message = data.slice(7);
          console.error("Stream Error:", message);
          onError?.(message);
        } else if (data.trim() !== "[DONE]") {
          let parsedText = data;
          try {
            if (data.startsWith('"')) parsedText = JSON.parse(data);
          } catch {}
          if (parsedText) onChunk(parsedText);
        }
      }
    }
  }

  // ── Upload PDF ─────────────────────────────────────────────────────────────

  async function handleUploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      console.error("Only PDF files are allowed");
      return;
    }
    const formData = new FormData();
    formData.append("project_id", projectId);
    formData.append("file", file);

    setIsUploading(true);
    try {
      const res = await fetch(
        `${API_BASE}/ingest-pdf?project_id=${encodeURIComponent(projectId)}`,
        { method: "POST", body: formData }
      );
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
      const data = await res.json() as { paper_id?: string };

      const newItem: LibraryItem = {
        id:    data.paper_id || file.name,
        title: file.name,
        type:  "pdf_upload",
        date:  nowLabel(),
      };
      setLibraryItems((prev) => [newItem, ...prev]);
      await refreshProjectData(projectId);
      setMessages((prev) => [
        ...prev,
        { id: generateMsgId(), role: "assistant", content: `✅ PDF **"${file.name}"** ingested successfully. You can now ask questions about it!`, ts: nowLabel() },
      ]);
      setSidebarTab("library");
    } catch (error: unknown) {
      setMessages((prev) => [
        ...prev,
        { id: generateMsgId(), role: "assistant", content: `❌ Upload error: ${getErrorMessage(error)}`, ts: nowLabel() },
      ]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) await handleUploadFile(files[0]);
  }

  // ── Chat ───────────────────────────────────────────────────────────────────

  async function callChatAPI(question: string) {
    setIsChatSending(true);
    const userMsgId = generateMsgId();
    const botMsgId  = generateMsgId();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user",      content: question, ts: nowLabel() },
      { id: botMsgId,  role: "assistant", content: "",        ts: nowLabel() },
    ]);

    try {
      const res = await fetch(`${API_BASE}/chat-stream`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          project_id: projectId,
          question,
          session_id: sessionId,   // ← MEMORY: pass session_id so backend maintains history
        }),
      });
      if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);

      await readStream(
        res,
        (text) => {
          setMessages((prev) =>
            prev.map((m) => m.id === botMsgId ? { ...m, content: m.content + text } : m)
          );
        },
        (type, data) => {
          if (type === "SOURCES") {
            setMessages((prev) =>
              prev.map((m) => m.id === botMsgId ? { ...m, sources: data as Source[] } : m)
            );
          } else if (type === "SUGGESTIONS") {
            setMessages((prev) =>
              prev.map((m) => m.id === botMsgId ? { ...m, suggestions: data as string[] } : m)
            );
          }
        },
        (message: string) => {
          setMessages((prev) =>
            prev.map((m) => m.id === botMsgId ? { ...m, content: `❌ Error: ${message}` } : m)
          );
        }
      );

      // Final cleanup after stream ends
      setMessages((prev) =>
        prev.map((m) => m.id === botMsgId ? { ...m, content: cleanAnswer(m.content) } : m)
      );
    } catch (error: unknown) {
      setMessages((prev) =>
        prev.map((m) => m.id === botMsgId ? { ...m, content: `❌ Error: ${getErrorMessage(error)}` } : m)
      );
    } finally {
      setIsChatSending(false);
    }
  }

  function downloadNotebook(notebook: NotebookResponse) {
    const blob = new Blob([notebook.notebook_json], { type: "application/x-ipynb+json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = notebook.file_name || `${notebook.paper_id}.ipynb`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function downloadAndOpenColab(notebook: NotebookResponse) {
    downloadNotebook(notebook);
    window.open("https://colab.research.google.com/", "_blank", "noopener,noreferrer");
  }

  function openNotebookModal(item: LibraryItem) {
    if (item.is_metadata_only) return;
    setSelectedNotebookPaper(item);
    setActiveNotebookPaperId(item.id);
    setNotebookError(null);
    setGeneratedNotebook(null);
    setShowNotebookModal(true);
  }

  async function handleGenerateNotebook() {
    if (!selectedNotebookPaper || isGeneratingNotebook) return;
    if (!geminiApiKey.trim()) {
      setNotebookError("Gemini API key is required to generate a notebook.");
      return;
    }

    setNotebookError(null);
    setGeneratedNotebook(null);
    setIsGeneratingNotebook(true);

    try {
      const payload: GenerateNotebookRequest = {
        api_key: geminiApiKey.trim(),
        model: geminiModel.trim() || "gemini-2.5-pro",
        generation_goal: notebookGoal,
        compute_profile: computeProfile,
        include_study_questions: includeStudyQuestions,
        include_reproducibility_checklist: includeReproChecklist,
        include_risk_notes: includeRiskNotes,
      };
      const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/papers/${encodeURIComponent(selectedNotebookPaper.id)}/generate-notebook`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({} as ApiError)) as ApiError;
        throw new Error(errData.detail || `Notebook generation failed (${res.status})`);
      }

      const data = await res.json() as NotebookResponse;
      setGeneratedNotebook(data);
      setMessages((prev) => [
        ...prev,
        {
          id: generateMsgId(),
          role: "assistant",
          content: `✅ Built a Research Lab Pack for **"${selectedNotebookPaper.title}"** with **${data.generated_with_model}**. Goal: **${data.generation_goal}**. You can preview the notebook, study questions, and reproducibility notes now.`,
          ts: nowLabel(),
        },
      ]);
    } catch (error: unknown) {
      const message = getErrorMessage(error);
      setNotebookError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: generateMsgId(),
          role: "assistant",
          content: `❌ Notebook generation failed for **"${selectedNotebookPaper.title}"**: ${message}`,
          ts: nowLabel(),
        },
      ]);
    } finally {
      setIsGeneratingNotebook(false);
    }
  }

  async function sendCurrentInput() {
    if (!input.trim() || isChatSending) return;
    const q = input.trim();
    setInput("");
    await callChatAPI(q);
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    await sendCurrentInput();
  }

  // ── Drag & Drop ────────────────────────────────────────────────────────────

  function handleDragOver(e: DragEvent)  { e.preventDefault(); e.stopPropagation(); }
  function handleDragEnter(e: DragEvent) {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) setIsDragging(true);
  }
  function handleDragLeave(e: DragEvent) {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  }
  async function handleDrop(e: DragEvent) {
    e.preventDefault(); e.stopPropagation();
    setIsDragging(false); dragCounter.current = 0;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleUploadFile(e.dataTransfer.files[0]);
    }
  }

  // ── Paper Search (multi-source) ────────────────────────────────────────────

  async function handlePaperSearch(topic: string) {
    if (!topic.trim()) return;
    setIsSearching(true);
    setSearchResults([]);
    try {
      // Use new /paper-search endpoint (multi-source: arXiv + Semantic Scholar + PubMed)
      const res = await fetch(`${API_BASE}/paper-search`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query: topic.trim(), max_results: 8 }),
      });
      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data = await res.json() as PaperSearchResponse;
      setSearchResults(data.papers || []);
    } catch (error: unknown) {
      console.error("Paper search error:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }

  // ── Import Paper ───────────────────────────────────────────────────────────

  async function handleImportPaper(paper: PaperResult) {
    const id    = getPaperId(paper);
    const title = paper.title || "Untitled";

    if (!id) {
      console.error("Cannot import paper: no source_id or arxiv_id found", paper);
      return;
    }

    setImportingId(id);
    setFailedId(null);

    try {
      const res = await fetch(`${API_BASE}/arxiv-import`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          project_id: projectId,
          arxiv_id: id,
          title: paper.title,
          authors: paper.authors,
          summary: paper.summary,
          published: paper.published,
          source: paper.source,
          source_id: paper.source_id,
          abs_url: paper.abs_url,
          pdf_url: paper.pdf_url,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({} as ApiError)) as ApiError;
        throw new Error(errData.detail || `Import failed (${res.status})`);
      }

      const absUrl = getPaperAbsUrl(paper);
      setLibraryItems((prev) => [
        {
          id,
          title,
          type:    normalizeLibraryType(paper.source, Boolean(absUrl)),
          date:    paper.published?.slice(0, 4) || nowLabel(),
          abs_url: absUrl,
          pdf_url: paper.pdf_url,
          is_metadata_only: false,
        },
        ...prev.filter((x) => x.id !== id),   // prevent duplicates
      ]);
      await refreshProjectData(projectId);

      setMessages((prev) => [
        ...prev,
        {
          id:      generateMsgId(),
          role:    "assistant",
          content: `✅ Added **"${title}"** to your library. You can now ask questions about it!`,
          ts:      nowLabel(),
        },
      ]);
      setSidebarTab("library");
    } catch (error: unknown) {
      console.error("Import failed:", error);
      setFailedId(id);
      setTimeout(() => setFailedId(null), 4000);
    } finally {
      setImportingId(null);
    }
  }

  // ── Delete Paper ───────────────────────────────────────────────────────────

  async function handleDeleteLibraryItem(id: string) {
    setRemovingId(id);
    try {
      const res = await fetch(`${API_BASE}/remove-paper`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ project_id: projectId, paper_id: id }),
      });
      if (!res.ok) throw new Error("Remove failed");
      setLibraryItems((prev) => prev.filter((item) => item.id !== id));
      await refreshProjectData(projectId);
      setMessages((prev) => [
        ...prev,
        { id: generateMsgId(), role: "assistant", content: "🗑️ Paper removed from library.", ts: nowLabel() },
      ]);
    } catch (error: unknown) {
      console.error("Remove paper failed:", error);
      setMessages((prev) => [
        ...prev,
        { id: generateMsgId(), role: "assistant", content: `❌ Failed to remove paper: ${getErrorMessage(error)}`, ts: nowLabel() },
      ]);
    } finally {
      setRemovingId(null);
    }
  }

  // ── New Workspace ──────────────────────────────────────────────────────────

  async function handleNewWorkspace() {
    // Clear backend history for current session
    await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectId)}/clear-history?session_id=${encodeURIComponent(sessionId)}`,
      { method: "POST" }
    ).catch(() => {});

    const newId = createNewProjectId();
    setProjectId(newId);
    if (typeof window !== "undefined") {
      localStorage.setItem("ai_researcher_project_id", newId);
    }
    setMessages([]);
    setLibraryItems([]);
    setProjectStats(null);
    setSearchResults([]);
  }

  if (!mounted) return null;

  const noScroll = "[&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]";

  return (
    <div className="dark">
      <main className="flex h-screen flex-col bg-zinc-950 text-zinc-100 font-sans overflow-hidden selection:bg-indigo-500/30">

        {/* Drop overlay */}
        {isDragging && (
          <div
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-indigo-900/50 backdrop-blur-sm border-4 border-indigo-500 border-dashed m-6 rounded-3xl animate-in fade-in duration-200"
            onDragLeave={handleDragLeave} onDragOver={handleDragOver} onDrop={handleDrop}
          >
            <div className="animate-bounce bg-indigo-600 p-6 rounded-full text-white shadow-2xl">
              <UploadCloud className="h-16 w-16" />
            </div>
            <h3 className="mt-6 text-3xl font-bold text-white tracking-tight">Drop PDF to Ingest</h3>
            <p className="text-indigo-200 mt-2">Add this paper to your research context</p>
          </div>
        )}

        {/* Header */}
        <header className="z-10 flex items-center justify-between border-b border-white/10 bg-zinc-900/50 px-6 py-3 backdrop-blur-md">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 shadow-lg shadow-indigo-500/20">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-zinc-100">
                Research<span className="text-indigo-400">AI</span>
              </h1>
              <div className="flex items-center gap-2 mt-0.5">
                <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">System Online</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 transition-colors hover:border-white/20">
              <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Project</span>
              <input
                className="w-36 bg-transparent text-xs font-mono text-zinc-300 focus:outline-none"
                value={projectId}
                onChange={(e) => setProjectId(e.target.value)}
              />
            </div>
            <button
              onClick={handleNewWorkspace}
              className="flex items-center gap-2 rounded-lg bg-white text-zinc-900 px-3 py-1.5 text-xs font-bold hover:bg-zinc-200 transition shadow-lg shadow-white/5"
            >
              <Plus className="h-3.5 w-3.5" /><span>New Workspace</span>
            </button>
          </div>
        </header>

        <div className="z-10 flex flex-1 overflow-hidden">

          {/* Chat area */}
          <div className="flex flex-1 flex-col relative" onDragEnter={handleDragEnter} onDragOver={handleDragOver}>
            <div ref={chatRef} className={`flex-1 overflow-y-auto px-4 py-6 scroll-smooth ${noScroll}`}>
              <div className="mx-auto max-w-3xl space-y-8 pb-32">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-32 text-center animate-in fade-in zoom-in duration-500">
                    <div className="mb-6 bg-white/5 border border-white/10 p-5 rounded-3xl shadow-2xl">
                      <Cpu className="h-10 w-10 text-indigo-400" />
                    </div>
                    <h2 className="text-2xl font-bold text-zinc-100">Research Assistant Ready</h2>
                    <p className="text-sm text-zinc-500 max-w-sm mt-3 leading-relaxed">
                      Upload a PDF or search for papers in the sidebar to build your knowledge base, then ask questions here.
                    </p>
                    <div className="mt-6 grid grid-cols-2 gap-2 text-left w-full max-w-md">
                      {[
                        "Summarize the paper I just uploaded",
                        "Find 5 papers about transformers in NLP",
                        "Compare the two papers in my library",
                        "Find papers written by Yann LeCun",
                      ].map((suggestion) => (
                        <button
                          key={suggestion}
                          onClick={() => { setInput(suggestion); }}
                          className="text-left px-3 py-2.5 rounded-xl bg-white/5 border border-white/5 hover:border-indigo-500/40 hover:bg-indigo-500/10 text-[11px] text-zinc-400 hover:text-zinc-200 transition-all"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  messages.map((m) => {
                    const isLatest = messages[messages.length - 1].id === m.id;
                    return (
                      <div key={m.id} className="flex flex-col gap-2">
                        <ChatBubble message={m} isLatest={isLatest} isStreaming={isChatSending} />
                        {!isChatSending && isLatest && m.suggestions && m.suggestions.length > 0 && (
                          <div className="flex flex-wrap gap-2 ml-12 mt-1">
                            {m.suggestions.map((suggestion, idx) => (
                              <button
                                key={idx}
                                onClick={() => callChatAPI(suggestion)}
                                className="text-left px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/30 hover:border-indigo-400 hover:bg-indigo-500/20 text-[11px] text-indigo-300 hover:text-indigo-100 transition-all shadow-sm"
                              >
                                {suggestion}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
                {isChatSending && messages.length > 0 && messages[messages.length - 1].role === "user" && (
                  <div className="flex items-center gap-3 pl-4 animate-pulse opacity-50">
                    <div className="h-8 w-8 rounded-full bg-white/10 flex items-center justify-center">
                      <Bot className="h-4 w-4" />
                    </div>
                    <span className="text-xs text-zinc-400">Analyzing documents...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Input bar */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-zinc-950 via-zinc-950 to-transparent pt-10 pb-6 px-4">
              <div className="mx-auto max-w-3xl">
                <form
                  onSubmit={handleSend}
                  className="relative flex items-end gap-2 rounded-3xl border border-white/10 bg-zinc-900/80 p-2 shadow-2xl backdrop-blur-xl focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/50 transition-all"
                >
                  <input ref={fileInputRef} type="file" accept="application/pdf" className="hidden" onChange={handleFileChange} disabled={isUploading} />
                  <button
                    type="button" disabled={isUploading}
                    onClick={() => fileInputRef.current?.click()}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-zinc-400 hover:bg-white/10 hover:text-white transition"
                    title="Upload PDF"
                  >
                    {isUploading ? <Loader2 className="h-5 w-5 animate-spin text-indigo-400" /> : <Paperclip className="h-5 w-5" />}
                  </button>
                    <textarea
                      className={`max-h-40 min-h-[2.5rem] flex-1 bg-transparent py-3 text-sm text-zinc-200 placeholder:text-zinc-500 outline-none resize-none ${noScroll}`}
                      placeholder="Ask a question about your papers..."
                      value={input} onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          void sendCurrentInput();
                        }
                      }}
                      rows={1}
                    />
                  <button
                    type="submit" disabled={isChatSending || !input.trim()}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-lg hover:bg-indigo-500 hover:scale-105 transition disabled:opacity-50 disabled:scale-100"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </form>
                <p className="text-center text-[10px] text-zinc-600 mt-3">AI can make mistakes. Always verify citations.</p>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <aside className="hidden w-96 flex-col border-l border-white/5 bg-zinc-900/40 backdrop-blur-xl md:flex">
            <div className="px-4 py-4 border-b border-white/5">
              <div className="grid grid-cols-2 gap-1 bg-black/20 p-1 rounded-xl">
                <button
                  onClick={() => setSidebarTab("add")}
                  className={`flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg transition-all ${sidebarTab === "add" ? "bg-zinc-800 text-white shadow-md" : "text-zinc-500 hover:text-zinc-300"}`}
                >
                  <Search className="h-3.5 w-3.5" /> Add Knowledge
                </button>
                <button
                  onClick={() => setSidebarTab("library")}
                  className={`flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg transition-all ${sidebarTab === "library" ? "bg-zinc-800 text-white shadow-md" : "text-zinc-500 hover:text-zinc-300"}`}
                >
                  <Library className="h-3.5 w-3.5" /> My Library ({libraryItems.length})
                </button>
              </div>
            </div>

            {sidebarTab === "add" ? (
              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="p-4 space-y-2">
                  <PaperSearchBox isSearching={isSearching} onSearch={handlePaperSearch} />
                  {searchResults.length > 0 && (
                    <p className="text-[10px] text-zinc-500 text-center">
                      {searchResults.length} results from arXiv, Semantic Scholar & PubMed
                    </p>
                  )}
                </div>
                <div className={`flex-1 overflow-y-auto px-4 pb-4 space-y-3 ${noScroll}`}>
                  {searchResults.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-40 text-zinc-600">
                      <BookOpen className="h-8 w-8 mb-2 opacity-20" />
                      <p className="text-xs">Search papers by topic, author, or title</p>
                      <p className="text-[10px] opacity-50 mt-1">Searches arXiv, Semantic Scholar & PubMed</p>
                    </div>
                  )}
                  {searchResults.map((p, idx) => {
                    const pid = getPaperId(p);
                    return (
                      <PaperCard
                        key={pid || idx}
                        paper={p}
                        isLoading={importingId === pid}
                        isFailed={failedId === pid}
                        onImport={() => handleImportPaper(p)}
                      />
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className={`flex-1 overflow-y-auto p-4 space-y-2 ${noScroll}`}>
                {projectStats && (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-lg shadow-black/10">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Library Insights</p>
                        <h3 className="mt-1 text-sm font-semibold text-zinc-100">Evidence coverage for this workspace</h3>
                      </div>
                      <div className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-[10px] font-semibold text-emerald-300">
                        {projectStats.papers_with_links}/{projectStats.paper_count || 0} linked
                      </div>
                    </div>
                    <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                      <div className="rounded-xl bg-black/25 px-2 py-3">
                        <div className="text-lg font-semibold text-white">{projectStats.paper_count}</div>
                        <div className="text-[10px] uppercase tracking-wide text-zinc-500">papers</div>
                      </div>
                      <div className="rounded-xl bg-black/25 px-2 py-3">
                        <div className="text-lg font-semibold text-white">{projectStats.chunk_count}</div>
                        <div className="text-[10px] uppercase tracking-wide text-zinc-500">chunks</div>
                      </div>
                      <div className="rounded-xl bg-black/25 px-2 py-3">
                        <div className="text-lg font-semibold text-white">{projectStats.papers_with_pdf}</div>
                        <div className="text-[10px] uppercase tracking-wide text-zinc-500">pdfs</div>
                      </div>
                    </div>
                    {Object.keys(projectStats.source_counts).length > 0 && (
                      <div className="mt-4 flex flex-wrap gap-2">
                        {Object.entries(projectStats.source_counts).map(([source, count]) => (
                          <span key={source} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] font-medium text-zinc-300">
                            {formatLibraryType(normalizeLibraryType(source, source === "web"))}: {count}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {libraryItems.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-64 text-zinc-600">
                    <Library className="h-10 w-10 mb-3 opacity-20" />
                    <p className="text-xs">Library is empty</p>
                    <p className="text-[10px] opacity-50 mt-1">Upload PDFs or import from the search tab</p>
                  </div>
                )}
                {libraryItems.map((item) => (
                  <div key={item.id} className="group flex items-start justify-between gap-3 p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition">
                    <div className="flex items-start gap-3 overflow-hidden">
                      <div className="mt-1 p-1.5 rounded bg-indigo-500/20 text-indigo-400">
                        <FileText className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <div className="text-xs font-medium truncate text-zinc-200 leading-tight">{item.title}</div>
                        <div className="text-[10px] text-zinc-500 mt-1 uppercase font-bold tracking-wide">
                          {formatLibraryType(item.type)} {item.date ? `• ${item.date}` : ""}
                        </div>
                        {item.is_metadata_only && (
                          <div className="mt-1 inline-flex rounded-full border border-amber-500/25 bg-amber-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-300">
                            Metadata only
                          </div>
                        )}
                        {item.abs_url && (
                          <a
                            href={item.abs_url} target="_blank" rel="noreferrer"
                            className="text-[10px] text-indigo-400 hover:underline mt-0.5 flex items-center gap-1"
                          >
                            <ExternalLink className="h-2.5 w-2.5" /> View Paper
                          </a>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        onClick={() => openNotebookModal(item)}
                        disabled={item.is_metadata_only || isGeneratingNotebook}
                        className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2 text-indigo-300 transition hover:bg-indigo-500/20 hover:text-indigo-100 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-zinc-500"
                        title={item.is_metadata_only ? "Notebook generation needs full text" : "Generate notebook"}
                      >
                        {isGeneratingNotebook && activeNotebookPaperId === item.id
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <Cpu className="h-3.5 w-3.5" />
                        }
                      </button>
                      <button
                        onClick={() => handleDeleteLibraryItem(item.id)}
                        disabled={removingId === item.id}
                        className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 transition p-1 disabled:opacity-50"
                        title="Remove paper"
                      >
                        {removingId === item.id
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <Trash2 className="h-3.5 w-3.5" />
                        }
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>
        </div>
      </main>

      {/* Notebook Modal */}
      {showNotebookModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-zinc-900 border border-white/10 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-zinc-950/50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-500/20 text-indigo-400 rounded-lg">
                  {isGeneratingNotebook ? <Loader2 className="h-5 w-5 animate-spin" /> : <Cpu className="h-5 w-5" />}
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Research Lab Pack</h2>
                  <p className="text-xs text-zinc-400">
                    {generatedNotebook
                      ? `Generated with ${generatedNotebook.generated_with_model}`
                      : selectedNotebookPaper
                        ? `Selected paper: ${selectedNotebookPaper.title}`
                        : "Transforming your selected paper into a notebook, study guide, and reproducibility pack"}
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setShowNotebookModal(false);
                  setSelectedNotebookPaper(null);
                  setNotebookError(null);
                }}
                className="p-2 text-zinc-500 hover:text-white transition-colors rounded-lg hover:bg-white/5"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {generatedNotebook ? (
                <div className="prose prose-invert prose-indigo max-w-none text-sm leading-relaxed prose-headings:text-indigo-300 prose-headings:font-bold prose-a:text-indigo-400 hover:prose-a:text-indigo-300 prose-p:text-white/80 prose-li:text-white/80 prose-table:border-collapse prose-th:border prose-th:border-zinc-700 prose-th:bg-zinc-800 prose-th:px-3 prose-th:py-2 prose-td:border prose-td:border-zinc-800 prose-td:px-3 prose-td:py-2">
                  <div className="mb-5 grid gap-3 md:grid-cols-2 not-prose">
                    <div className="rounded-2xl border border-indigo-500/20 bg-indigo-500/10 p-4">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-200">Generation Profile</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-100">
                          Goal: {generatedNotebook.generation_goal}
                        </span>
                        <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-100">
                          Compute: {generatedNotebook.compute_profile}
                        </span>
                      </div>
                      {generatedNotebook.artifact_summary.length > 0 && (
                        <ul className="mt-3 space-y-2 text-sm text-zinc-200">
                          {generatedNotebook.artifact_summary.map((item) => (
                            <li key={item} className="flex gap-2">
                              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-indigo-300" />
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Companion Outputs</p>
                      <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                        <div className="rounded-xl bg-black/25 px-2 py-3">
                          <div className="text-lg font-semibold text-white">{generatedNotebook.study_questions.length}</div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500">questions</div>
                        </div>
                        <div className="rounded-xl bg-black/25 px-2 py-3">
                          <div className="text-lg font-semibold text-white">{generatedNotebook.reproducibility_checklist.length}</div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500">checks</div>
                        </div>
                        <div className="rounded-xl bg-black/25 px-2 py-3">
                          <div className="text-lg font-semibold text-white">{generatedNotebook.risk_notes.length}</div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500">notes</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mb-5 flex flex-wrap gap-2">
                    {generatedNotebook.dependencies.map((dependency) => (
                      <span key={dependency} className="rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-200">
                        {dependency}
                      </span>
                    ))}
                  </div>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{generatedNotebook.preview_markdown}</ReactMarkdown>
                </div>
              ) : isGeneratingNotebook ? (
                <div className="flex flex-col items-center justify-center h-full text-zinc-500 space-y-4">
                  <Loader2 className="h-8 w-8 animate-spin text-indigo-500/50" />
                  <p className="animate-pulse">Gemini 2.5 Pro is analyzing the PDF and building your Research Lab Pack...</p>
                </div>
              ) : notebookError ? (
                <div className="space-y-4">
                  <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-200">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                      <div>
                        <p className="font-semibold">Notebook generation failed</p>
                        <p className="mt-1 text-red-100/80">{notebookError}</p>
                      </div>
                    </div>
                  </div>
                  <NotebookSetupCard
                    selectedPaper={selectedNotebookPaper}
                    geminiApiKey={geminiApiKey}
                    geminiModel={geminiModel}
                    notebookGoal={notebookGoal}
                    computeProfile={computeProfile}
                    includeStudyQuestions={includeStudyQuestions}
                    includeReproChecklist={includeReproChecklist}
                    includeRiskNotes={includeRiskNotes}
                    onApiKeyChange={setGeminiApiKey}
                    onModelChange={setGeminiModel}
                    onGoalChange={setNotebookGoal}
                    onComputeProfileChange={setComputeProfile}
                    onIncludeStudyQuestionsChange={setIncludeStudyQuestions}
                    onIncludeReproChecklistChange={setIncludeReproChecklist}
                    onIncludeRiskNotesChange={setIncludeRiskNotes}
                  />
                </div>
              ) : (
                <NotebookSetupCard
                  selectedPaper={selectedNotebookPaper}
                  geminiApiKey={geminiApiKey}
                  geminiModel={geminiModel}
                  notebookGoal={notebookGoal}
                  computeProfile={computeProfile}
                  includeStudyQuestions={includeStudyQuestions}
                  includeReproChecklist={includeReproChecklist}
                  includeRiskNotes={includeRiskNotes}
                  onApiKeyChange={setGeminiApiKey}
                  onModelChange={setGeminiModel}
                  onGoalChange={setNotebookGoal}
                  onComputeProfileChange={setComputeProfile}
                  onIncludeStudyQuestionsChange={setIncludeStudyQuestions}
                  onIncludeReproChecklistChange={setIncludeReproChecklist}
                  onIncludeRiskNotesChange={setIncludeRiskNotes}
                />
              )}
            </div>
            <div className="p-4 border-t border-white/10 bg-zinc-950/50 flex justify-end gap-3">
              {!generatedNotebook && (
                <button
                  onClick={() => void handleGenerateNotebook()}
                  disabled={!selectedNotebookPaper || !geminiApiKey.trim() || isGeneratingNotebook}
                  className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-50 transition"
                >
                  {isGeneratingNotebook ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cpu className="h-4 w-4" />}
                  Generate Notebook
                </button>
              )}
              <button
                onClick={() => generatedNotebook && downloadNotebook(generatedNotebook)}
                disabled={!generatedNotebook || isGeneratingNotebook}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-zinc-800 text-white hover:bg-zinc-700 disabled:opacity-50 transition"
              >
                <Download className="h-4 w-4" /> Download `.ipynb`
              </button>
              <button
                onClick={() => generatedNotebook && downloadAndOpenColab(generatedNotebook)}
                disabled={!generatedNotebook || isGeneratingNotebook}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-white text-zinc-900 hover:bg-zinc-200 disabled:opacity-50 transition"
              >
                <ExternalLink className="h-4 w-4" /> Download + Open Colab
              </button>
              <button
                onClick={() => {
                  setShowNotebookModal(false);
                  setSelectedNotebookPaper(null);
                  setNotebookError(null);
                }}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function NotebookSetupCard({
  selectedPaper,
  geminiApiKey,
  geminiModel,
  notebookGoal,
  computeProfile,
  includeStudyQuestions,
  includeReproChecklist,
  includeRiskNotes,
  onApiKeyChange,
  onModelChange,
  onGoalChange,
  onComputeProfileChange,
  onIncludeStudyQuestionsChange,
  onIncludeReproChecklistChange,
  onIncludeRiskNotesChange,
}: {
  selectedPaper: LibraryItem | null;
  geminiApiKey: string;
  geminiModel: string;
  notebookGoal: string;
  computeProfile: string;
  includeStudyQuestions: boolean;
  includeReproChecklist: boolean;
  includeRiskNotes: boolean;
  onApiKeyChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onGoalChange: (value: string) => void;
  onComputeProfileChange: (value: string) => void;
  onIncludeStudyQuestionsChange: (value: boolean) => void;
  onIncludeReproChecklistChange: (value: boolean) => void;
  onIncludeRiskNotesChange: (value: boolean) => void;
}) {
  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-500">Research Lab Pack</p>
        <h3 className="mt-2 text-lg font-bold text-white">{selectedPaper?.title || "Select a paper"}</h3>
        <p className="mt-2 text-sm text-zinc-300 leading-relaxed">
          This feature uses <span className="font-semibold text-white">Gemini 2.5 Pro</span> on the real paper PDF so it can read equations,
          structure, and figures more faithfully. Instead of only generating code, it creates a full research lab pack with a runnable notebook,
          study prompts, and reproducibility guidance. Your Gemini key is used only here. Chat, search, import, and the rest of the
          app stay on Groq exactly as before.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl border border-white/10 bg-zinc-950/60 p-4">
          <label className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Gemini API Key</label>
          <input
            type="password"
            value={geminiApiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            placeholder="Paste your Gemini API key here"
            className="mt-2 w-full rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white outline-none transition focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
          />
          <p className="mt-2 text-xs text-zinc-400">
            Stored locally in your browser for convenience. It is only sent when you click Generate Notebook.
          </p>

          <label className="mt-4 block text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Model</label>
          <input
            value={geminiModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="mt-2 w-full rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white outline-none transition focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
          />
          <p className="mt-2 text-xs text-zinc-400">Recommended: <span className="font-mono text-zinc-200">gemini-2.5-pro</span></p>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <label className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Generation Goal</label>
              <select
                value={notebookGoal}
                onChange={(e) => onGoalChange(e.target.value)}
                className="mt-2 w-full rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white outline-none transition focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
              >
                <option value="teaching">Teaching first</option>
                <option value="replication">Replication focused</option>
                <option value="rapid_review">Rapid review</option>
              </select>
            </div>
            <div>
              <label className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Compute Profile</label>
              <select
                value={computeProfile}
                onChange={(e) => onComputeProfileChange(e.target.value)}
                className="mt-2 w-full rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white outline-none transition focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
              >
                <option value="balanced">Balanced</option>
                <option value="cpu_light">CPU light</option>
              </select>
            </div>
          </div>

          <div className="mt-4 grid gap-2">
            <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-200">
              <input type="checkbox" checked={includeStudyQuestions} onChange={(e) => onIncludeStudyQuestionsChange(e.target.checked)} />
              Include study questions
            </label>
            <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-200">
              <input type="checkbox" checked={includeReproChecklist} onChange={(e) => onIncludeReproChecklistChange(e.target.checked)} />
              Include reproducibility checklist
            </label>
            <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-200">
              <input type="checkbox" checked={includeRiskNotes} onChange={(e) => onIncludeRiskNotesChange(e.target.checked)} />
              Include risk and assumption notes
            </label>
          </div>
        </div>

        <div className="rounded-2xl border border-indigo-500/20 bg-indigo-500/10 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-200">How To Get A Key</p>
          <ol className="mt-3 space-y-3 text-sm text-indigo-50/90">
            <li>1. Open <a className="text-white underline hover:text-indigo-100" href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer">Google AI Studio API Keys</a>.</li>
            <li>2. Sign in, create an API key, and copy it.</li>
            <li>3. Paste it here, choose your goal, and click <span className="font-semibold">Generate Notebook</span>.</li>
          </ol>
          <p className="mt-4 text-xs leading-relaxed text-indigo-100/80">
            If a paper was added as metadata-only, re-upload or re-import the full PDF first so Gemini can analyze the actual document.
          </p>
        </div>
      </div>
    </div>
  );
}

function ChatBubble({ message, isLatest, isStreaming }: { message: Message; isLatest: boolean; isStreaming?: boolean }) {
  const isUser    = message.role === "user";
  const hasSources = message.sources && message.sources.length > 0;
  const showCursor = !isUser && isLatest && isStreaming && message.content.length > 0;
  const isEmpty   = !isUser && isLatest && isStreaming && message.content.length === 0;

  return (
    <div className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full shadow-lg ${isUser ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 border border-white/10"}`}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="max-w-[85%] min-w-0 flex flex-col">
        <div className={`relative rounded-2xl px-6 py-4 text-sm shadow-sm leading-relaxed ${isUser ? "bg-indigo-600 text-white rounded-tr-sm" : "bg-white/5 border border-white/5 text-zinc-100 rounded-tl-sm"}`}>
          {isEmpty ? (
            <div className="flex items-center gap-2 text-zinc-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span className="text-xs animate-pulse">Thinking...</span>
            </div>
          ) : (
            <>
              <FormattedMessage content={message.content} isUser={isUser} />
              {showCursor && (
                <span className="inline-block w-0.5 h-4 bg-indigo-400 ml-0.5 animate-pulse align-middle" />
              )}
            </>
          )}
        </div>
        {!isUser && hasSources && <SourcesDropdown sources={message.sources!} />}
      </div>
    </div>
  );
}

function SourcesDropdown({ sources }: { sources: Source[] }) {
  const [isOpen, setIsOpen] = useState(false);
  const uniqueSources = sources.filter((s, i, arr) => arr.findIndex((x) => x.id === s.id) === i);

  return (
    <div className="mt-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-500 hover:text-indigo-400 transition-colors"
      >
        {isOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        {isOpen ? "Hide Sources" : `View ${uniqueSources.length} Sources`}
      </button>
      {isOpen && (
        <div className="mt-2 grid gap-2 animate-in slide-in-from-top-2 duration-200">
          {uniqueSources.map((s, i) => {
            const pageNum   = s.page || s.metadata?.page || s.metadata?.page_number;
            const openUrl   = s.abs_url || s.pdf_url;
            const citations = s.citation_count;
            return (
              <div key={i} className="flex flex-col rounded-lg border border-white/10 bg-black/20 p-3 text-xs">
                <div className="flex justify-between items-start gap-2">
                  <div className="font-medium text-zinc-200 line-clamp-2">{s.title || "Untitled Document"}</div>
                  <div className="flex items-center gap-1.5 shrink-0 flex-wrap justify-end">
                    {pageNum   && <span className="px-1.5 py-0.5 rounded bg-white/10 text-[9px] font-mono text-zinc-300">Pg. {pageNum}</span>}
                    {citations && <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-[9px] text-amber-300">{citations} cites</span>}
                    <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-[9px] font-bold text-zinc-400 uppercase">{s.type}</span>
                  </div>
                </div>
                {s.journal && <p className="text-[10px] text-zinc-500 mt-1 italic">{s.journal}</p>}
                <div className="flex items-center gap-3 mt-2">
                  {openUrl && (
                    <a href={openUrl} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-[10px] font-medium text-indigo-400 hover:underline">
                      <ExternalLink className="h-3 w-3" /> View Paper
                    </a>
                  )}
                  {s.pdf_url && s.abs_url && (
                    <a href={s.pdf_url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-[10px] font-medium text-zinc-400 hover:underline">
                      <FileText className="h-3 w-3" /> PDF
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function FormattedMessage({ content, isUser }: { content: string; isUser: boolean }) {
  const clean = content
    .replace(/\[DONE\]/gi, "")
    .replace(/\[LOCAL\s+\d+(?:,\s*LOCAL\s+\d+)*\]/g, "")
    .trim();

  if (isUser) {
    return <p className="text-white/95 leading-relaxed break-words">{clean}</p>;
  }

  const markdownComponents: MarkdownComponents = {
    h1: ({ children }) => <h1 className="text-xl font-bold text-white mt-5 mb-2 border-b border-white/10 pb-1">{children}</h1>,
    h2: ({ children }) => <h2 className="text-lg font-bold text-white mt-5 mb-2 border-b border-white/10 pb-1">{children}</h2>,
    h3: ({ children }) => <h3 className="text-base font-semibold text-indigo-300 mt-4 mb-1">{children}</h3>,
    h4: ({ children }) => <h4 className="text-sm font-semibold text-indigo-200 mt-3 mb-1">{children}</h4>,
    p: ({ children }) => <p className="text-[15px] text-zinc-200 leading-relaxed mb-3 last:mb-0">{children}</p>,
    ul: ({ children }) => <ul className="my-3 space-y-2 pl-6 list-disc marker:text-indigo-400">{children}</ul>,
    ol: ({ children }) => <ol className="my-3 space-y-2 pl-6 list-decimal marker:text-indigo-400">{children}</ol>,
    li: ({ children }) => <li className="text-[15px] text-zinc-200 leading-relaxed pl-1">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
    em: ({ children }) => <em className="italic text-zinc-300">{children}</em>,
    a: ({ href, children }) => (
      <a href={href} target="_blank" rel="noopener noreferrer"
        className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2 transition-colors">
        {children}
      </a>
    ),
    code: ({ children, className, ...props }) =>
      className ? (
        <pre className="bg-black/40 border border-white/10 rounded-xl p-4 overflow-x-auto my-3 text-[13px]">
          <code className={`font-mono text-green-300 ${className}`.trim()} {...props}>{children}</code>
        </pre>
      ) : (
        <code className="bg-white/10 text-pink-300 px-1.5 py-0.5 rounded text-[13px] font-mono" {...props}>{children}</code>
      ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-indigo-500/60 pl-4 my-3 italic text-zinc-400">{children}</blockquote>
    ),
    hr: () => <hr className="border-white/10 my-4" />,
    table: ({ children }) => (
      <div className="overflow-x-auto my-4 rounded-xl border border-white/10">
        <table className="w-full text-sm border-collapse">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-indigo-900/40 border-b border-white/10">{children}</thead>,
    tbody: ({ children }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
    tr: ({ children }) => <tr className="hover:bg-white/5 transition-colors">{children}</tr>,
    th: ({ children }) => <th className="px-4 py-2.5 text-left text-xs font-semibold text-indigo-200 uppercase tracking-wide whitespace-nowrap">{children}</th>,
    td: ({ children }) => <td className="px-4 py-2.5 text-zinc-300 align-top text-[13px]">{children}</td>,
  };

  return (
    <div className="w-full overflow-hidden break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={markdownComponents}
      >
        {clean}
      </ReactMarkdown>
    </div>
  );
}

function PaperSearchBox({ isSearching, onSearch }: { isSearching: boolean; onSearch: (q: string) => void }) {
  const [val, setVal] = useState("");
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSearch(val); }} className="relative">
      <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
      <input
        className="w-full bg-black/20 border border-white/5 rounded-xl py-2 pl-9 pr-4 text-xs text-zinc-200 outline-none focus:ring-1 ring-indigo-500/50 transition placeholder:text-zinc-600"
        placeholder="Topic, author name, or paper title..."
        value={val}
        onChange={(e) => setVal(e.target.value)}
      />
      {isSearching && (
        <div className="absolute right-3 top-2.5">
          <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
        </div>
      )}
    </form>
  );
}

function PaperCard({
  paper, isLoading, isFailed, onImport,
}: {
  paper: PaperResult;
  isLoading: boolean;
  isFailed:  boolean;
  onImport:  () => void;
}) {
  const absUrl     = getPaperAbsUrl(paper);
  const sourceLabel = (paper.source || "arxiv").toUpperCase();
  const citations   = paper.citation_count;

  return (
    <div className="bg-white/5 border border-white/5 rounded-xl p-3 shadow-sm hover:border-indigo-500/30 transition group">
      <div className="flex justify-between items-start gap-2">
        <h4 className="text-xs font-bold text-zinc-200 leading-tight line-clamp-2">{paper.title}</h4>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-[9px] font-mono text-zinc-500">{paper.published?.slice(0, 4)}</span>
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 uppercase">{sourceLabel}</span>
        </div>
      </div>

      {paper.authors && paper.authors.length > 0 && (
        <p className="text-[10px] text-zinc-500 mt-1 truncate">
          {paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""}
        </p>
      )}

      <p className="text-[10px] text-zinc-500 mt-1.5 line-clamp-2 leading-relaxed">
        {paper.summary || "No abstract available."}
      </p>

      {citations != null && (
        <p className="text-[10px] text-amber-400/70 mt-1">{citations.toLocaleString()} citations</p>
      )}

      <div className="flex items-center gap-2 mt-3">
        <button
          onClick={onImport}
          disabled={isLoading || isFailed}
          className={`flex-1 flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-[10px] font-bold transition disabled:opacity-50 disabled:cursor-not-allowed
            ${isFailed
              ? "bg-red-500/10 text-red-400 border border-red-500/20"
              : "bg-indigo-600 hover:bg-indigo-500 text-white"}`}
        >
          {isLoading
            ? <><Loader2 className="h-3 w-3 animate-spin" /><span>Importing...</span></>
            : isFailed
            ? <><AlertCircle className="h-3 w-3" /><span>Failed – Retry?</span></>
            : <><Plus className="h-3 w-3" /><span>Add to Project</span></>}
        </button>
        {absUrl && (
          <a
            href={absUrl} target="_blank" rel="noreferrer"
            className="p-1.5 text-zinc-400 hover:text-indigo-400 hover:bg-white/5 border border-white/5 rounded-lg transition"
            title="Open abstract page"
          >
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </div>
  );
}
