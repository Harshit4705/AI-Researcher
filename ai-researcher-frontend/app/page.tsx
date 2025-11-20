"use client";

import { useState, useRef, useEffect, DragEvent } from "react";
import {
  Send,
  Paperclip,
  Plus,
  Search,
  Bot,
  User,
  Sparkles,
  UploadCloud,
  Cpu,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  BookOpen,
  FileText,
  Loader2,
  Library,
  Trash2,
  Link as LinkIcon,
  AlertCircle
} from "lucide-react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const ENV_DEFAULT_PROJECT_ID =
  process.env.NEXT_PUBLIC_DEFAULT_PROJECT_ID || "";

// --- Types ---
type AppMode = "chat" | "writer";

// UPDATED: More flexible Source type to catch nested metadata
type Source = {
  id: string;
  type: string;
  title?: string;
  arxiv_id?: string;
  pdf_url?: string;
  page?: number;
  metadata?: {
    page?: number;
    page_number?: number;
    [key: string]: any;
  };
  [key: string]: any;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: string;
  sources?: Source[];
};

type LibraryItem = {
  id: string;
  title: string;
  type: "pdf" | "arxiv";
  date: string;
  url?: string;
};

function createNewProjectId() {
  return (
    "proj-" +
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2, 8)
  );
}

export default function HomePage() {
  // --- State ---
  const [mounted, setMounted] = useState(false);
  const [projectId, setProjectId] = useState<string>(
    () => ENV_DEFAULT_PROJECT_ID || createNewProjectId()
  );

  // Chat State
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isChatSending, setIsChatSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Sidebar State
  const [sidebarTab, setSidebarTab] = useState<"add" | "library">("add");
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>([]);
  
  // ArXiv State
  const [isSearching, setIsSearching] = useState(false);
  const [importingArxivId, setImportingArxivId] = useState<string | null>(null);
  const [failedArxivId, setFailedArxivId] = useState<string | null>(null);
  const [arxivResults, setArxivResults] = useState<any[]>([]);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const dragCounter = useRef(0); 

  // --- Effects ---
  useEffect(() => {
    setMounted(true);
    document.documentElement.classList.add('dark');
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages, isChatSending]);

  // --- Helpers ---
  function nowLabel() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  function generateMsgId() {
    return Math.random().toString(36).substring(7);
  }

  // --- STREAMING LOGIC ---
  async function readStream(
    response: Response, 
    onChunk: (text: string) => void,
    onEvent?: (type: string, data: any) => void
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
        if (line.startsWith("data: ")) {
          const data = line.slice(6); 
          
          if (data.startsWith("[SOURCES]")) {
             if(onEvent) onEvent("SOURCES", JSON.parse(data.slice(9)));
          } 
          else if (data.startsWith("[ERROR]")) {
             console.error("Stream Error:", data.slice(7));
          }
          else {
            onChunk(data);
          }
        }
      }
    }
  }

  // --- HANDLERS ---

  async function handleUploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      // Console error instead of alert
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
      
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      
      const newItem: LibraryItem = {
        id: data.paper_id || file.name,
        title: file.name,
        type: "pdf",
        date: nowLabel()
      };
      setLibraryItems(prev => [newItem, ...prev]);

      setMessages((prev) => [
        ...prev,
        {
          id: generateMsgId(),
          role: "assistant",
          content: `PDF "${file.name}" ingested successfully.`,
          ts: nowLabel(),
        },
      ]);
      
      setSidebarTab("library");

    } catch (e: any) {
      setMessages((prev) => [
         ...prev,
         { id: generateMsgId(), role: "assistant", content: `Upload error: ${e.message}`, ts: nowLabel() },
      ]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      await handleUploadFile(files[0]);
    }
  }

  async function callChatAPI(question: string) {
    setIsChatSending(true);
    const userMsgId = generateMsgId();
    const botMsgId = generateMsgId();

    setMessages(prev => [
      ...prev, 
      { id: userMsgId, role: "user", content: question, ts: nowLabel() },
      { id: botMsgId, role: "assistant", content: "", ts: nowLabel() } 
    ]);

    try {
      const res = await fetch(`${API_BASE}/chat-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, question }),
      });

      if (!res.ok) throw new Error("Chat failed");

      await readStream(
        res,
        (text) => {
          const chunk = text + " "; 
          setMessages(prev => prev.map(m => 
            m.id === botMsgId ? { ...m, content: m.content + chunk } : m
          ));
        },
        (type, data) => {
          if (type === "SOURCES") {
            setMessages(prev => prev.map(m => 
              m.id === botMsgId ? { ...m, sources: data } : m
            ));
          }
        }
      );

    } catch (e: any) {
      setMessages(prev => prev.map(m => 
        m.id === botMsgId ? { ...m, content: `Error: ${e.message}` } : m
      ));
    } finally {
      setIsChatSending(false);
    }
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isChatSending) return;
    const q = input.trim();
    setInput("");
    await callChatAPI(q);
  }

  // --- DRAG & DROP ---
  function handleDragOver(e: DragEvent) { e.preventDefault(); e.stopPropagation(); }
  function handleDragEnter(e: DragEvent) { 
    e.preventDefault(); e.stopPropagation(); 
    dragCounter.current++; 
    if(e.dataTransfer.items && e.dataTransfer.items.length > 0) setIsDragging(true);
  }
  function handleDragLeave(e: DragEvent) { 
    e.preventDefault(); e.stopPropagation(); 
    dragCounter.current--;
    if(dragCounter.current === 0) setIsDragging(false);
  }
  async function handleDrop(e: DragEvent) { 
    e.preventDefault(); e.stopPropagation(); 
    setIsDragging(false); dragCounter.current = 0;
    if(e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleUploadFile(e.dataTransfer.files[0]);
    }
  }

  // --- ARXIV ---
  async function handleArxivSearch(topic: string) {
    if (!topic.trim()) return;
    setIsSearching(true);
    setArxivResults([]);
    try {
      const res = await fetch(`${API_BASE}/arxiv-search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: topic, max_results: 5 }),
      });
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      setArxivResults(data.papers || []);
    } catch (e) { console.error(e); } 
    finally { setIsSearching(false); }
  }

  async function handleImportArxiv(arxivId: string, title: string, pdfUrl?: string) {
    setImportingArxivId(arxivId);
    setFailedArxivId(null);
    try {
      const res = await fetch(`${API_BASE}/arxiv-import`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, arxiv_id: arxivId }),
      });
      
      if (!res.ok) throw new Error("Import failed");
      
      setLibraryItems(prev => [{ id: arxivId, title: title, type: "arxiv", date: nowLabel(), url: pdfUrl }, ...prev]);
      
      setMessages(prev => [...prev, { id: generateMsgId(), role: "assistant", content: `Added "${title}" to library.`, ts: nowLabel() }]);
      
    } catch (e: any) { 
      console.error("Import failed:", e);
      setFailedArxivId(arxivId); 
      setTimeout(() => setFailedArxivId(null), 4000);
    } finally { 
      setImportingArxivId(null); 
    }
  }

  function handleNewWorkspace() {
    setProjectId(createNewProjectId());
    setMessages([]);
    setLibraryItems([]);
    setArxivResults([]);
  }

  function handleDeleteLibraryItem(id: string) {
    setLibraryItems(prev => prev.filter(item => item.id !== id));
  }

  if (!mounted) return null;

  const noScroll = "[&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]";

  return (
    <div className="dark">
      <main className="flex h-screen flex-col bg-zinc-950 text-zinc-100 font-sans overflow-hidden selection:bg-indigo-500/30">
        
        {/* --- DRAG OVERLAY --- */}
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

        {/* --- HEADER --- */}
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
                 <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                   System Online
                 </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 transition-colors hover:border-white/20">
              <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Project ID</span>
              <input
                className="w-24 bg-transparent text-xs font-mono text-zinc-300 focus:outline-none"
                value={projectId} onChange={(e) => setProjectId(e.target.value)}
              />
            </div>
            
            <button
              onClick={handleNewWorkspace}
              className="flex items-center gap-2 rounded-lg bg-white text-zinc-900 px-3 py-1.5 text-xs font-bold hover:bg-zinc-200 transition shadow-lg shadow-white/5"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>New Workspace</span>
            </button>
          </div>
        </header>

        {/* --- CONTENT LAYOUT --- */}
        <div className="z-10 flex flex-1 overflow-hidden">
          
          {/* LEFT: CHAT AREA */}
          <div 
            className="flex flex-1 flex-col relative"
            onDragEnter={handleDragEnter} onDragOver={handleDragOver}
          >
            {/* Chat List */}
            <div ref={chatRef} className={`flex-1 overflow-y-auto px-4 py-6 scroll-smooth ${noScroll}`}>
              <div className="mx-auto max-w-3xl space-y-8 pb-32">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-32 text-center animate-in fade-in zoom-in duration-500">
                    <div className="mb-6 bg-white/5 border border-white/10 p-5 rounded-3xl shadow-2xl">
                       <Cpu className="h-10 w-10 text-indigo-400" />
                    </div>
                    <h2 className="text-2xl font-bold text-zinc-100">Research Assistant Ready</h2>
                    <p className="text-sm text-zinc-500 max-w-sm mt-3 leading-relaxed">
                      Upload a PDF or search arXiv in the sidebar to build your knowledge base, then ask questions here.
                    </p>
                  </div>
                ) : (
                  messages.map((m) => {
                     const isLatest = messages[messages.length - 1].id === m.id;
                     return <ChatBubble key={m.id} message={m} isLatest={isLatest} />;
                  })
                )}
                
                {isChatSending && messages.length > 0 && messages[messages.length-1].role === 'user' && (
                  <div className="flex items-center gap-3 pl-4 animate-pulse opacity-50">
                    <div className="h-8 w-8 rounded-full bg-white/10 flex items-center justify-center">
                       <Bot className="h-4 w-4" />
                    </div>
                    <span className="text-xs text-zinc-400">Analyzing documents...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Input Area */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-zinc-950 via-zinc-950 to-transparent pt-10 pb-6 px-4">
              <div className="mx-auto max-w-3xl">
                <form
                  onSubmit={handleSend}
                  className="relative flex items-end gap-2 rounded-3xl border border-white/10 bg-zinc-900/80 p-2 shadow-2xl backdrop-blur-xl focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/50 transition-all"
                >
                  <input
                    ref={fileInputRef} type="file" accept="application/pdf" className="hidden"
                    onChange={handleFileChange} disabled={isUploading}
                  />
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
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(e); } }}
                    rows={1}
                  />
                  
                  <button
                    type="submit" disabled={isChatSending || !input.trim()}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-lg hover:bg-indigo-500 hover:scale-105 transition disabled:opacity-50 disabled:scale-100"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </form>
                <p className="text-center text-[10px] text-zinc-600 mt-3">
                  AI can make mistakes. Always verify citations.
                </p>
              </div>
            </div>
          </div>

          {/* RIGHT: SIDEBAR */}
          <aside className="hidden w-96 flex-col border-l border-white/5 bg-zinc-900/40 backdrop-blur-xl md:flex">
            
            {/* Sidebar Tabs */}
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
                    <Library className="h-3.5 w-3.5" /> My Library
                  </button>
               </div>
            </div>

            {/* Content: Search */}
            {sidebarTab === "add" ? (
               <div className="flex-1 flex flex-col overflow-hidden">
                  <div className="p-4">
                    <ArxivSearchBox isSearching={isSearching} onSearch={handleArxivSearch} />
                  </div>
                  <div className={`flex-1 overflow-y-auto px-4 pb-4 space-y-3 ${noScroll}`}>
                     {arxivResults.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-40 text-zinc-600">
                           <BookOpen className="h-8 w-8 mb-2 opacity-20" />
                           <p className="text-xs">Search ArXiv for papers</p>
                        </div>
                     )}
                     {arxivResults.map((p, idx) => (
                        <ArxivCard 
                           key={idx} 
                           paper={p} 
                           isLoading={importingArxivId === p.arxiv_id} 
                           isFailed={failedArxivId === p.arxiv_id}
                           onImport={handleImportArxiv} 
                        />
                     ))}
                  </div>
               </div>
            ) : (
            /* Content: Library */
               <div className={`flex-1 overflow-y-auto p-4 space-y-2 ${noScroll}`}>
                  {libraryItems.length === 0 && (
                     <div className="flex flex-col items-center justify-center h-64 text-zinc-600">
                        <Library className="h-10 w-10 mb-3 opacity-20" />
                        <p className="text-xs">Library is empty</p>
                        <p className="text-[10px] opacity-50 mt-1">Upload PDFs or Import from ArXiv</p>
                     </div>
                  )}
                  {libraryItems.map(item => (
                     <div key={item.id} className="group flex items-start justify-between p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition">
                        <div className="flex items-start gap-3 overflow-hidden">
                           <div className="mt-1 p-1.5 rounded bg-indigo-500/20 text-indigo-400">
                              <FileText className="h-4 w-4" />
                           </div>
                           <div className="min-w-0">
                              <div className="text-xs font-medium truncate text-zinc-200 leading-tight">{item.title}</div>
                              <div className="text-[10px] text-zinc-500 mt-1 uppercase font-bold tracking-wide">{item.type} • {item.date}</div>
                           </div>
                        </div>
                        <button 
                           onClick={() => handleDeleteLibraryItem(item.id)} 
                           className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 transition p-1"
                        >
                           <Trash2 className="h-3.5 w-3.5" />
                        </button>
                     </div>
                  ))}
               </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}

/* --- SUB-COMPONENTS --- */

function ChatBubble({ message, isLatest }: { message: Message, isLatest: boolean }) {
  const isUser = message.role === "user";
  const hasSources = message.sources && message.sources.length > 0;
  const shouldAnimate = !isUser && isLatest;

  return (
    <div className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full shadow-lg ${isUser ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 border border-white/10"}`}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="max-w-[85%] flex flex-col">
        <div className={`relative rounded-2xl px-6 py-4 text-sm shadow-sm leading-relaxed ${
             isUser ? "bg-indigo-600 text-white rounded-tr-sm" : "bg-white/5 border border-white/5 text-zinc-100 rounded-tl-sm"
        }`}>
          {shouldAnimate ? (
             <TypewriterMessage content={message.content} />
          ) : (
             <FormattedMessage content={message.content} isUser={isUser} />
          )}
        </div>
        
        {!isUser && hasSources && (
           <SourcesDropdown sources={message.sources!} />
        )}
      </div>
    </div>
  );
}

function SourcesDropdown({ sources }: { sources: Source[] }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-500 hover:text-indigo-400 transition-colors"
      >
        {isOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        {isOpen ? "Hide Sources" : `View ${sources.length} Sources`}
      </button>

      {isOpen && (
        <div className="mt-2 grid gap-2 animate-in slide-in-from-top-2 duration-200">
          {sources.map((s, i) => {
            // SMART PAGE NUMBER FINDER
            const pageNum = s.page || s.metadata?.page || s.metadata?.page_number;
            
            return (
              <div key={i} className="flex flex-col rounded-lg border border-white/10 bg-black/20 p-3 text-xs">
                <div className="flex justify-between items-start gap-2">
                  <div className="font-medium text-zinc-200 line-clamp-1">
                    {s.title || "Untitled Document"}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                      {pageNum && (
                          <span className="px-1.5 py-0.5 rounded bg-white/10 text-[9px] font-mono text-zinc-300">
                              Pg. {pageNum}
                          </span>
                      )}
                      <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-[9px] font-bold text-zinc-400 uppercase">
                          {s.type}
                      </span>
                  </div>
                </div>
                {s.pdf_url && (
                  <a href={s.pdf_url} target="_blank" rel="noreferrer" className="mt-2 flex items-center gap-1 text-[10px] font-medium text-indigo-400 hover:underline">
                    <FileText className="h-3 w-3" /> Open PDF
                  </a>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  );
}

function TypewriterMessage({ content }: { content: string }) {
  const [displayedContent, setDisplayedContent] = useState("");
  
  useEffect(() => {
    let index = 0;
    const timer = setInterval(() => {
      setDisplayedContent((prev) => content.slice(0, index + 3)); 
      index += 3;
      if (index >= content.length) {
         clearInterval(timer);
         setDisplayedContent(content); 
      }
    }, 10); 

    return () => clearInterval(timer);
  }, [content]);

  return <FormattedMessage content={displayedContent} isUser={false} />;
}

function FormattedMessage({ content, isUser }: { content: string; isUser: boolean }) {
  let clean = isUser ? content : content.replace(/\[LOCAL\s+\d+(?:,\s*LOCAL\s+\d+)*\]/g, "");
  
  if (!isUser) {
     clean = clean.replace(/([^\n])\s(\*|\-)\s/g, "$1\n$2 ");
  }

  const lines = clean.split("\n");
  
  const parseLine = (text: string) => {
     const parts = text.split(/(\[(?:[^\]]+)\]\((?:https?:\/\/[^\)]+)\)|https?:\/\/[^\s)]+|\*\*[^\*]+\*\*|`[^`]+`)/g);
     return parts.map((part, i) => {
        if (part.startsWith("[") && part.includes("](") && part.endsWith(")")) {
           const m = part.match(/\[([^\]]+)\]\(([^)]+)\)/);
           if(m) return <a key={i} href={m[2]} target="_blank" className="text-indigo-400 hover:underline">{m[1]}</a>;
        }
        if (part.match(/^https?:\/\/[^\s)]+$/)) return <a key={i} href={part} target="_blank" className="text-indigo-400 hover:underline inline-flex items-center gap-0.5"><LinkIcon className="h-3 w-3"/>Link</a>;
        if (part.startsWith("**")) return <strong key={i} className="font-bold text-white">{part.slice(2,-2)}</strong>;
        if (part.startsWith("`")) return <code key={i} className="bg-white/10 px-1 rounded font-mono text-pink-400">{part.slice(1,-1)}</code>;
        return part;
     });
  }

  return (
     <div className="space-y-3 break-words">
        {lines.map((line, i) => {
           const trimmed = line.trim();
           if(!trimmed) return <div key={i} className="h-1" />;
           if(trimmed.startsWith("### ")) return <h3 key={i} className="text-base font-bold text-indigo-300 mt-4">{parseLine(trimmed.slice(4))}</h3>;
           if(trimmed.startsWith("## ")) return <h2 key={i} className="text-lg font-bold text-white mt-6 border-b border-white/10 pb-2">{parseLine(trimmed.slice(3))}</h2>;
           if(trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
              return <div key={i} className="flex gap-2 ml-1 text-zinc-300"><div className="mt-2 h-1.5 w-1.5 rounded-full bg-zinc-500 shrink-0"/><div>{parseLine(trimmed.slice(2))}</div></div>;
           }
           return <p key={i} className="text-zinc-300">{parseLine(trimmed)}</p>;
        })}
     </div>
  );
}

function ArxivSearchBox({ isSearching, onSearch }: any) {
  const [val, setVal] = useState("");
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSearch(val); }} className="relative">
      <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
      <input 
        className="w-full bg-black/20 border border-white/5 rounded-xl py-2 pl-9 pr-4 text-xs text-zinc-200 outline-none focus:ring-1 ring-indigo-500/50 transition placeholder:text-zinc-600"
        placeholder="Search papers (e.g. 'Attention is All You Need')..."
        value={val} onChange={e => setVal(e.target.value)}
      />
      {isSearching && <div className="absolute right-3 top-2.5"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>}
    </form>
  )
}

function ArxivCard({ paper, isLoading, isFailed, onImport }: any) {
   return (
      <div className="bg-white/5 border border-white/5 rounded-xl p-3 shadow-sm hover:border-indigo-500/30 transition group">
         <div className="flex justify-between items-start gap-2">
            <h4 className="text-xs font-bold text-zinc-200 leading-tight line-clamp-2">{paper.title}</h4>
            <span className="text-[9px] font-mono text-zinc-500 shrink-0">{paper.published?.slice(0,4)}</span>
         </div>
         <p className="text-[10px] text-zinc-500 mt-2 line-clamp-2 leading-relaxed">{paper.summary}</p>
         
         <div className="flex items-center gap-2 mt-3">
            <button 
               onClick={() => onImport(paper.arxiv_id, paper.title, paper.pdf_url)}
               disabled={isLoading || isFailed}
               className={`flex-1 flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-[10px] font-bold transition disabled:opacity-50 disabled:cursor-not-allowed
                  ${isFailed 
                     ? "bg-red-500/10 text-red-400 border border-red-500/20"
                     : "bg-indigo-600 hover:bg-indigo-500 text-white"
                  }`}
            >
               {isLoading ? (
                  <>
                     <Loader2 className="h-3 w-3 animate-spin" /> 
                     <span>Importing...</span>
                  </>
               ) : isFailed ? (
                  <>
                     <AlertCircle className="h-3 w-3" /> 
                     <span>Failed</span>
                  </>
               ) : (
                  <>
                     <Plus className="h-3 w-3" /> 
                     <span>Add to Project</span>
                  </>
               )}
            </button>
            
            {paper.pdf_url && (
               <a 
                  href={paper.pdf_url} 
                  target="_blank" 
                  className="p-1.5 text-zinc-400 hover:text-indigo-400 hover:bg-white/5 border border-white/5 rounded-lg transition"
                  title="Open PDF"
               >
                  <ExternalLink className="h-3 w-3" />
               </a>
            )}
         </div>
      </div>
   )
}