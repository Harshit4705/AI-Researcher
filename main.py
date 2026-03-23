from __future__ import annotations

import asyncio
import re
import logging
import tempfile
import json
import textwrap
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from read_pdf import read_pdf_tool
from vector_tools import (
    upsert_project_paper_chunks,
    remove_paper_from_project,
    list_project_papers,
    load_all_paper_metadata,
    get_all_project_chunks,
)
from research_tool import search_research_papers_tool as search_arxiv_tool, search_arxiv
from ai_researcher import APP, get_llm
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ARXIV_IDENTIFIER_PATTERN = r'(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d+(?:v\d+)?)'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="AI Research Assistant",
    description="Upload PDFs, import papers, and chat about them with full memory",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.gettempdir()) / "ai_researcher_downloads"
TEMP_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# In-memory chat history store
# Key  : session_id (or project_id as fallback)
# Value: list of {"role": "user"|"assistant", "content": str}
# ─────────────────────────────────────────────

_chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_HISTORY_TURNS = 10   # keep last 10 user+assistant exchanges per session


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    project_id: str
    question: str
    session_id: Optional[str] = None   # optional: isolate histories per browser tab/user


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class ArxivSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5


class ArxivSearchResponse(BaseModel):
    papers: List[Dict[str, Any]]


class ImportArxivRequest(BaseModel):
    project_id: str
    arxiv_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    summary: Optional[str] = None
    published: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    abs_url: Optional[str] = None
    pdf_url: Optional[str] = None


class ImportArxivResponse(BaseModel):
    status: str
    paper: Dict[str, Any]
    message: str


class IngestPdfResponse(BaseModel):
    status: str
    paper_id: str
    message: str


class RemovePaperRequest(BaseModel):
    project_id: str
    paper_id: str


class RemovePaperResponse(BaseModel):
    status: str
    message: str


class ProjectInfoResponse(BaseModel):
    project_id: str
    description: str


class ProjectStatsResponse(BaseModel):
    project_id: str
    paper_count: int
    chunk_count: int
    papers_with_pdf: int
    papers_with_links: int
    source_counts: Dict[str, int]


class GenerateBriefRequest(BaseModel):
    project_id: str


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def download_arxiv_pdf(arxiv_id: str) -> Optional[str]:
    base_id = arxiv_id.split("v")[0]
    pdf_url = f"https://arxiv.org/pdf/{base_id}"
    try:
        logger.info("Downloading arXiv PDF: %s", pdf_url)
        response = requests.get(pdf_url, timeout=30, headers={"User-Agent": "ai-researcher/3.0"})
        response.raise_for_status()
        pdf_path = TEMP_DIR / f"{base_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return str(pdf_path)
    except Exception as exc:
        logger.error("Failed to download arXiv PDF %s: %s", arxiv_id, exc)
        return None


def cleanup_temp_file(file_path: str) -> None:
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to clean up temp file %s: %s", file_path, exc)


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    return text.replace("[DONE]", "").replace("[SOURCES]", "").strip()


def _best_source_url(source: Dict[str, Any]) -> str:
    return source.get("pdf_url") or source.get("abs_url") or source.get("link") or ""


def _extract_arxiv_id(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None

    patterns = [
        rf'arxiv\.org/(?:abs|pdf)/(?P<id>{ARXIV_IDENTIFIER_PATTERN})',
        rf'10\.48550/arxiv\.(?P<id>{ARXIV_IDENTIFIER_PATTERN})',
        rf'arxiv[:/](?P<id>{ARXIV_IDENTIFIER_PATTERN})',
        rf'^(?P<id>{ARXIV_IDENTIFIER_PATTERN})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            candidate = match.group("id")
            return candidate[:-4] if candidate.lower().endswith(".pdf") else candidate
    return None


def _normalize_import_identifier(raw_id: str) -> str:
    return _extract_arxiv_id(raw_id) or raw_id.strip()


def _lookup_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    base_id = (_extract_arxiv_id(arxiv_id) or arxiv_id).split("v")[0]
    papers = search_arxiv(
        {
            "author": None,
            "topic": None,
            "year": None,
            "arxiv_id": base_id,
            "source_hint": "arxiv",
            "raw": base_id,
        },
        max_results=3,
    )
    for paper in papers:
        payload = paper.to_dict()
        if (payload.get("source_id") or "").split("v")[0] == base_id:
            return payload
    return None


def _fallback_import_paper(request: ImportArxivRequest, raw_id: str) -> Dict[str, Any]:
    source = request.source or ("arxiv" if _extract_arxiv_id(raw_id) else "web")
    source_id = request.source_id or raw_id
    abs_url = request.abs_url or (f"https://arxiv.org/abs/{raw_id}" if source == "arxiv" else "")
    pdf_url = request.pdf_url
    return {
        "title": request.title or source_id,
        "authors": request.authors or [],
        "summary": request.summary or "",
        "published": request.published or "",
        "source": source,
        "source_id": source_id,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
    }


def _build_history_content(answer: str, sources: List[Dict[str, Any]]) -> str:
    history_content = answer
    if sources:
        link_lines = []
        for s in sources:
            title = s.get("title") or "Untitled"
            url = _best_source_url(s)
            if url:
                link_lines.append("  - " + title + ": " + url)
        if link_lines:
            history_content += "\n\n[Paper Links]\n" + "\n".join(link_lines)
    return history_content


async def _run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def _get_history_key(request: ChatRequest) -> str:
    """Use session_id if provided, otherwise scope to project_id."""
    return request.session_id or request.project_id


def _trim_history(key: str) -> None:
    """Keep only the last MAX_HISTORY_TURNS exchanges (2 messages each)."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_chat_histories[key]) > max_msgs:
        _chat_histories[key] = _chat_histories[key][-max_msgs:]


def _normalize_source_label(source: Optional[str], paper: Dict[str, Any]) -> str:
    raw = (source or "").strip().lower()
    if raw in {"arxiv", "semantic_scholar", "pubmed", "openalex", "pdf_upload"}:
        return raw
    if paper.get("arxiv_id"):
        return "arxiv"
    if raw.endswith(".pdf") or "ai_researcher_downloads" in raw or ":" in raw or "\\" in raw or "/" in raw:
        return "pdf_upload"
    return "pdf_upload" if paper.get("pdf_url") or not paper.get("abs_url") else "web"


def _build_project_stats(project_id: str) -> ProjectStatsResponse:
    sidecar = load_all_paper_metadata(project_id)
    chunks = get_all_project_chunks(project_id)
    registry = []
    faiss_ids = list_project_papers.invoke({"project_id": project_id})
    for pid in faiss_ids:
        registry.append(sidecar.get(pid, {"paper_id": pid, "title": pid}))

    source_counts: Dict[str, int] = defaultdict(int)
    for paper in registry:
        label = _normalize_source_label(paper.get("source"), paper)
        source_counts[label] += 1

    papers_with_pdf = sum(1 for paper in registry if paper.get("pdf_url"))
    papers_with_links = sum(1 for paper in registry if _best_source_url(paper))

    return ProjectStatsResponse(
        project_id=project_id,
        paper_count=len(registry),
        chunk_count=len(chunks),
        papers_with_pdf=papers_with_pdf,
        papers_with_links=papers_with_links,
        source_counts=dict(sorted(source_counts.items())),
    )


def _prepare_brief_inputs(chunks: List[Dict[str, Any]], max_papers: int = 6, max_chars_per_paper: int = 3200) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        paper_id = metadata.get("paper_id") or "unknown"
        title = metadata.get("title") or paper_id
        content = (chunk.get("content") or "").strip()
        if not content:
            continue

        entry = grouped.setdefault(
            paper_id,
            {"paper_id": paper_id, "title": title, "parts": [], "chars": 0},
        )
        remaining = max_chars_per_paper - entry["chars"]
        if remaining <= 0:
            continue
        snippet = content[:remaining]
        entry["parts"].append(snippet)
        entry["chars"] += len(snippet)

    papers = []
    for paper in grouped.values():
        paper_text = "\n\n".join(paper["parts"]).strip()
        if paper_text:
            papers.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "content": paper_text,
            })

    papers.sort(key=lambda item: len(item["content"]), reverse=True)
    return papers[:max_papers]


async def _summarize_paper_for_brief(paper: Dict[str, str]) -> str:
    prompt = (
        "Summarize this research paper for a later synthesis.\n"
        "Return concise markdown with exactly these sections:\n"
        "### Overview\n### Methods\n### Findings\n### Limitations\n\n"
        "Use only information present in the text.\n\n"
        f"Paper title: {paper['title']}\n\n"
        f"Paper text:\n{paper['content']}"
    )
    llm = get_llm("brief_map", max_tokens=320)
    response = await llm.ainvoke([
        SystemMessage(content="You produce concise, grounded paper digests."),
        HumanMessage(content=prompt),
    ])
    summary = (response.content or "").strip()
    if summary:
        return f"## {paper['title']}\n\n{summary}"

    fallback = textwrap.shorten(paper["content"], width=900, placeholder="...")
    return f"## {paper['title']}\n\n### Overview\n{fallback}"


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "AI Research Assistant API v3.0",
        "endpoints": {
            "chat":           "POST /chat",
            "chat_stream":    "POST /chat-stream",
            "paper_search":   "POST /paper-search",
            "arxiv_import":   "POST /arxiv-import",
            "ingest_pdf":     "POST /ingest-pdf",
            "remove_paper":   "POST /remove-paper",
            "list_papers":    "GET  /projects/{project_id}/papers",
            "clear_history":  "POST /projects/{project_id}/clear-history",
            "docs":           "/docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Chat (non-streaming)
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.project_id or not request.question:
        raise HTTPException(status_code=400, detail="project_id and question are required")

    key = _get_history_key(request)

    # Add user message to history BEFORE invoking (so graph sees prior turns)
    _chat_histories[key].append({"role": "user", "content": request.question})
    _trim_history(key)

    # Pass history EXCLUDING the current message (everything before the last entry)
    prior_history = list(_chat_histories[key][:-1])

    try:
        state = await _run_blocking(
            APP.invoke,
            {
                "project_id":   request.project_id,
                "question":     request.question,
                "chat_history": prior_history,
            }
        )
        answer = _clean_answer(state.get("answer", "No answer generated"))

        # Save assistant reply with source links for follow-up context
        sources_for_history = state.get("sources", [])
        history_content = _build_history_content(answer, sources_for_history)
        _chat_histories[key].append({"role": "assistant", "content": history_content})
        _trim_history(key)

        return ChatResponse(answer=answer, sources=state.get("sources", []))

    except Exception as exc:
        # Remove the user message we added if the call failed
        if _chat_histories[key] and _chat_histories[key][-1]["role"] == "user":
            _chat_histories[key].pop()
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}")


# ─────────────────────────────────────────────
# Chat (streaming)
# ─────────────────────────────────────────────

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    if not request.project_id or not request.question:
        raise HTTPException(status_code=400, detail="project_id and question are required")

    key = _get_history_key(request)

    _chat_histories[key].append({"role": "user", "content": request.question})
    _trim_history(key)
    prior_history = list(_chat_histories[key][:-1])

    async def generate():
        try:
            state = await _run_blocking(
                APP.invoke,
                {
                    "project_id":   request.project_id,
                    "question":     request.question,
                    "chat_history": prior_history,
                },
            )
            answer  = _clean_answer(state.get("answer", ""))
            sources = state.get("sources", [])

            # Save assistant reply with source links for follow-up context
            history_content = _build_history_content(answer, sources)
            _chat_histories[key].append({"role": "assistant", "content": history_content})
            _trim_history(key)

            # Stream answer in chunks
            chunk_size = 15
            for i in range(0, len(answer), chunk_size):
                yield f"data: {json.dumps(answer[i:i+chunk_size])}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: [SOURCES]{json.dumps(sources)}\n\n"

            # Stream follow-up suggestions
            suggestions = state.get("suggestions", [])
            if suggestions:
                yield f"data: [SUGGESTIONS]{json.dumps(suggestions)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            # Rollback user message on failure
            if _chat_histories[key] and _chat_histories[key][-1]["role"] == "user":
                _chat_histories[key].pop()
            logger.error("Chat stream error: %s", exc)
            yield f"data: [ERROR]{str(exc)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# AI Research Brief Generator
# ─────────────────────────────────────────────

@app.post("/generate-brief")
async def generate_brief(request: GenerateBriefRequest):
    if not request.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    async def generate():
        try:
            chunks = await _run_blocking(get_all_project_chunks, request.project_id)
            if not chunks:
                yield "data: [ERROR]No papers found in this project. Please add papers first.\n\n"
                yield "data: [DONE]\n\n"
                return

            papers = _prepare_brief_inputs(chunks)
            if not papers:
                yield "data: [ERROR]No usable paper content was found in this project.\n\n"
                yield "data: [DONE]\n\n"
                return

            paper_summaries: List[str] = []
            for paper in papers:
                try:
                    paper_summaries.append(await _summarize_paper_for_brief(paper))
                except Exception as exc:
                    logger.warning("Paper-level brief summary failed for %s: %s", paper["paper_id"], exc)
                    fallback = textwrap.shorten(paper["content"], width=900, placeholder="...")
                    paper_summaries.append(f"## {paper['title']}\n\n### Overview\n{fallback}")

            combined_summaries = "\n\n".join(paper_summaries)[:7000]
            system_prompt = (
                "You are an expert AI Research Assistant. Synthesize the provided paper summaries into a grounded research brief.\n\n"
                "Structure the output strictly in markdown with these sections:\n"
                "## Executive Summary\n"
                "## Key Themes\n"
                "## Methodology Analysis\n"
                "## Research Gaps\n"
                "## Future Directions\n\n"
                "Only use information from the provided summaries. Keep the brief professional and specific."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Paper summaries:\n{combined_summaries}\n\nGenerate the research brief."),
            ]

            llm = get_llm("brief_final", max_tokens=700)
            async for chunk in llm.astream(messages):
                content = chunk.content
                if content:
                    chunk_size = 15
                    for i in range(0, len(content), chunk_size):
                        yield f"data: {json.dumps(content[i:i+chunk_size])}\n\n"
                        await asyncio.sleep(0.01)

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("Brief generation error: %s", exc)
            yield f"data: [ERROR]{str(exc)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# Clear chat history  (for "New Chat" button)
# ─────────────────────────────────────────────

@app.post("/projects/{project_id}/clear-history")
async def clear_history(
    project_id: str,
    session_id: Optional[str] = Query(default=None),
):
    key = session_id or project_id
    _chat_histories.pop(key, None)
    return {"status": "ok", "cleared_key": key}


# ─────────────────────────────────────────────
# Paper search  (multi-source: arXiv + Semantic Scholar + PubMed)
# ─────────────────────────────────────────────

@app.post("/paper-search", response_model=ArxivSearchResponse)
async def paper_search(request: ArxivSearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        results = await _run_blocking(
            search_arxiv_tool.invoke,
            {"query": request.query.strip(), "max_results": request.max_results or 5},
        )
        return ArxivSearchResponse(papers=results)
    except Exception as exc:
        logger.error("Paper search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Paper search failed: {str(exc)}")


# Keep old endpoint alive for frontend backward compatibility
@app.post("/arxiv-search", response_model=ArxivSearchResponse)
async def arxiv_search(request: ArxivSearchRequest):
    return await paper_search(request)


# ─────────────────────────────────────────────
# PDF ingestion
# ─────────────────────────────────────────────

@app.post("/ingest-pdf", response_model=IngestPdfResponse)
async def ingest_pdf(project_id: str, file: UploadFile = File(...)):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    temp_file_path = TEMP_DIR / file.filename
    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        chunks = await _run_blocking(
            read_pdf_tool.invoke,
            {
                "file_path":     str(temp_file_path),
                "chunk_size":    1200,
                "chunk_overlap": 200,
            }
        )

        paper_id = Path(file.filename).stem

        for ch in chunks:
            md = ch.get("metadata") or {}
            md["title"] = file.filename
            md["paper_id"] = paper_id
            md["source"] = "pdf_upload"
            md["is_metadata_only"] = False
            md["chunk_count"] = len(chunks)
            ch["metadata"] = md

        msg = await _run_blocking(
            upsert_project_paper_chunks.invoke,
            {
                "project_id": project_id,
                "paper_id":   paper_id,
                "chunks":     chunks,
            }
        )

        return IngestPdfResponse(status="success", paper_id=paper_id, message=msg)

    except Exception as exc:
        logger.error("PDF ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(exc)}")
    finally:
        cleanup_temp_file(str(temp_file_path))


# ─────────────────────────────────────────────
# arXiv import
# ─────────────────────────────────────────────

@app.post("/arxiv-import", response_model=ImportArxivResponse)
async def import_arxiv_paper(request: ImportArxivRequest, background_tasks: BackgroundTasks):
    if not request.project_id or not request.arxiv_id:
        raise HTTPException(status_code=400, detail="project_id and arxiv_id are required")

    try:
        raw_id = _normalize_import_identifier(request.arxiv_id)

        # ── Detect ID type ────────────────────────────────────────────────────
        is_arxiv_id = bool(_extract_arxiv_id(raw_id))
        is_s2_id    = bool(re.match(r'^[0-9a-f]{40}$', raw_id))
        is_pmid     = bool(re.match(r'^\d{6,9}$', raw_id))
        is_doi      = bool(re.match(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$', raw_id))
        is_openalex = bool(re.match(r'^W\d{5,15}$', raw_id, re.IGNORECASE))

        paper    = None
        pdf_path = None

        if is_arxiv_id:
            # ── arXiv paper: search by ID, download PDF ───────────────────────
            paper = await _run_blocking(_lookup_arxiv_paper, raw_id)
            if not paper and request.title:
                paper = _fallback_import_paper(request, raw_id)
                paper["source"] = "arxiv"
                paper["source_id"] = raw_id
                paper["abs_url"] = request.abs_url or f"https://arxiv.org/abs/{raw_id}"
                paper["pdf_url"] = request.pdf_url or f"https://arxiv.org/pdf/{raw_id}"
            if not paper:
                raise HTTPException(status_code=404, detail=f"arXiv paper {raw_id} not found")

            pdf_path = await _run_blocking(download_arxiv_pdf, raw_id)
            if not pdf_path:
                raise HTTPException(status_code=500, detail=f"Failed to download PDF for {raw_id}")

        elif is_s2_id or is_pmid or is_doi or is_openalex:
            # ── Non-arXiv paper (Semantic Scholar / PubMed / OpenAlex / DOI) ────
            results = await _run_blocking(
                search_arxiv_tool.invoke,
                {"query": raw_id, "max_results": 5}
            )
            paper = next(
                (p for p in results if (p.get("source_id") or "") == raw_id),
                None,
            )

            if not paper:
                fallback = _fallback_import_paper(request, raw_id)
                if not fallback.get("abs_url"):
                    fallback["abs_url"] = f"https://doi.org/{raw_id}" if is_doi else (
                        f"https://openalex.org/{raw_id}" if is_openalex else (
                            f"https://www.semanticscholar.org/paper/{raw_id}" if is_s2_id
                            else f"https://pubmed.ncbi.nlm.nih.gov/{raw_id}/"
                        )
                    )
                paper = fallback

            # Try to get a PDF — check open access URL
            if paper.get("pdf_url"):
                try:
                    resp = await _run_blocking(
                        requests.get,
                        paper["pdf_url"], timeout=20,
                        headers={"User-Agent": "ai-researcher/3.0"},
                        allow_redirects=True,
                    )
                    if resp.status_code == 200 and b"%PDF" in resp.content[:1024]:
                        pdf_path = str(TEMP_DIR / f"{raw_id}.pdf")
                        with open(pdf_path, "wb") as f:
                            f.write(resp.content)
                except Exception as exc:
                    logger.warning("Could not download open-access PDF for %s: %s", raw_id, exc)
                    pdf_path = None

            if not pdf_path:
                # ── BUG FIX: Index a metadata stub into FAISS so the paper ──
                # ── actually exists in the registry and bot can reference it ──
                logger.info("No PDF for %s — indexing metadata stub into FAISS", raw_id)
                paper_id = raw_id

                stub_content = "\n".join(filter(None, [
                    f"Title: {paper.get('title', raw_id)}",
                    f"Authors: {', '.join(paper.get('authors') or [])}",
                    f"Abstract: {paper.get('summary') or 'No abstract available.'}",
                    f"Published: {paper.get('published', '')}",
                    f"Abstract URL: {paper.get('abs_url', '')}",
                ]))

                stub_chunks = [{
                    "content": stub_content,
                    "page":    None,
                    "source":  paper.get("abs_url", ""),
                    "metadata": {
                        "paper_id": paper_id,
                        "title":    paper.get("title", raw_id),
                        "abs_url":  paper.get("abs_url"),
                        "pdf_url":  paper.get("pdf_url"),
                        "arxiv_id": paper.get("source_id"),
                        "source":   paper.get("source"),
                        "is_metadata_only": True,
                        "chunk_count": 1,
                    },
                }]

                await _run_blocking(
                    upsert_project_paper_chunks.invoke,
                    {
                        "project_id": request.project_id,
                        "paper_id":   paper_id,
                        "chunks":     stub_chunks,
                    }
                )

                return ImportArxivResponse(
                    status="metadata_only",
                    paper=paper,
                    message=(
                        f"Paper '{paper.get('title', raw_id)}' added to your library (metadata only). "
                        f"Full PDF is not freely available. You can view it at: "
                        f"{paper.get('abs_url', '')}"
                    ),
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized paper ID format: {raw_id!r}. "
                       "Expected arXiv ID, Semantic Scholar ID, PubMed ID, DOI, or OpenAlex ID."
            )

        # ── Ingest PDF into FAISS ─────────────────────────────────────────────
        chunks = await _run_blocking(
            read_pdf_tool.invoke,
            {
                "file_path":     pdf_path,
                "chunk_size":    1200,
                "chunk_overlap": 200,
            }
        )

        paper_id = raw_id.split("v")[0] if is_arxiv_id else raw_id

        for ch in chunks:
            md = ch.get("metadata") or {}
            md["arxiv_id"] = paper.get("source_id") or paper.get("arxiv_id")
            md["pdf_url"]  = paper.get("pdf_url")
            md["abs_url"]  = paper.get("abs_url")
            md["title"]    = paper.get("title")
            md["paper_id"] = paper_id
            md["is_metadata_only"] = False
            md["chunk_count"] = len(chunks)
            ch["metadata"] = md

        msg = await _run_blocking(
            upsert_project_paper_chunks.invoke,
            {
                "project_id": request.project_id,
                "paper_id":   paper_id,
                "chunks":     chunks,
            }
        )

        if pdf_path:
            background_tasks.add_task(cleanup_temp_file, pdf_path)

        return ImportArxivResponse(status="success", paper=paper, message=msg)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Import error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(exc)}")


# ─────────────────────────────────────────────
# Project info
# ─────────────────────────────────────────────

@app.get("/projects/{project_id}", response_model=ProjectInfoResponse)
async def get_project_info(project_id: str):
    return ProjectInfoResponse(
        project_id=project_id,
        description=f"Research project '{project_id}'",
    )


@app.get("/projects/{project_id}/stats", response_model=ProjectStatsResponse)
async def get_project_stats(project_id: str):
    try:
        return await _run_blocking(_build_project_stats, project_id)
    except Exception as exc:
        logger.error("Project stats error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to build project stats: {str(exc)}")


@app.get("/projects/{project_id}/papers")
async def get_project_papers(project_id: str):
    try:
        faiss_ids = await _run_blocking(list_project_papers.invoke, {"project_id": project_id})
        sidecar = await _run_blocking(load_all_paper_metadata, project_id)
        registry = []
        for pid in faiss_ids:
            if pid in sidecar:
                registry.append(sidecar[pid])
            else:
                registry.append({"paper_id": pid, "title": pid, "source": "pdf_upload"})
        return {"project_id": project_id, "papers": registry}
    except Exception as exc:
        logger.error("List papers error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list papers: {str(exc)}")

@app.post("/remove-paper", response_model=RemovePaperResponse)
async def remove_paper(request: RemovePaperRequest):
    if not request.project_id or not request.paper_id:
        raise HTTPException(status_code=400, detail="project_id and paper_id are required")
    try:
        msg = await _run_blocking(
            remove_paper_from_project.invoke,
            {"project_id": request.project_id, "paper_id": request.paper_id}
        )
        return RemovePaperResponse(status="success", message=msg)
    except Exception as exc:
        logger.error("Remove paper error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to remove paper: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
