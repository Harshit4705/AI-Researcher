from __future__ import annotations

import asyncio
import os
import re
import logging
import tempfile
import json
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from read_pdf import read_pdf_tool
from vector_tools import (
    upsert_project_paper_chunks,
    remove_paper_from_project,
    list_project_papers,
    load_all_paper_metadata,
    delete_paper_metadata,
)
from arxiv_tool import search_arxiv_tool
from ai_researcher import APP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="AI Research Assistant",
    description="Upload PDFs, import arXiv papers, and chat about them",
    version="2.0.0",
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
# Pydantic models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    project_id: str
    question: str

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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def download_arxiv_pdf(arxiv_id: str) -> Optional[str]:
    base_id = arxiv_id.split("v")[0]
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
    try:
        logger.info("Downloading arXiv PDF: %s", pdf_url)
        response = requests.get(pdf_url, timeout=30, headers={"User-Agent": "ai-researcher/2.0"})
        response.raise_for_status()
        pdf_path = TEMP_DIR / f"{base_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return str(pdf_path)
    except Exception as exc:
        logger.error("Failed to download arXiv PDF %s: %s", arxiv_id, exc)
        return None

def cleanup_temp_file(file_path: str):
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to clean up temp file %s: %s", file_path, exc)

def _clean_answer(text: str) -> str:
    """Strip [DONE] and other SSE artifacts from final answer."""
    return text.replace("[DONE]", "").replace("[SOURCES]", "").strip()

async def _run_blocking(fn, *args, **kwargs):
    """Run a blocking function in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "AI Research Assistant API v2.0",
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat-stream",
            "arxiv_search": "POST /arxiv-search",
            "arxiv_import": "POST /arxiv-import",
            "ingest_pdf": "POST /ingest-pdf",
            "remove_paper": "POST /remove-paper",
            "list_papers": "GET /projects/{project_id}/papers",
            "docs": "/docs",
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
    try:
        state = APP.invoke({
            "project_id": request.project_id,
            "question": request.question,
        })
        answer = _clean_answer(state.get("answer", "No answer generated"))
        return ChatResponse(answer=answer, sources=state.get("sources", []))
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}")


# ─────────────────────────────────────────────
# Chat (streaming)
# ─────────────────────────────────────────────

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    if not request.project_id or not request.question:
        raise HTTPException(status_code=400, detail="project_id and question are required")

    async def generate():
        try:
            state = APP.invoke({
                "project_id": request.project_id,
                "question": request.question,
            })
            answer = _clean_answer(state.get("answer", ""))
            sources = state.get("sources", [])

            # Stream chunk by chunk to preserve spaces and newlines
            chunk_size = 15
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: [SOURCES]{json.dumps(sources)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Chat stream error: %s", exc)
            yield f"data: [ERROR]{str(exc)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# arXiv search
# ─────────────────────────────────────────────

@app.post("/arxiv-search", response_model=ArxivSearchResponse)
async def arxiv_search(request: ArxivSearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="query is required")

    q = request.query.strip()
    max_results = request.max_results or 5

    try:
        id_match = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", q)
        if id_match:
            base_id = q.split("v")[0]
            # Run blocking HTTP call in thread pool so we don't block the event loop
            results = await _run_blocking(
                search_arxiv_tool.invoke, {"query": base_id, "max_results": max_results * 2}
            )
            filtered = [p for p in results if (p.get("arxiv_id") or "").split("v")[0] == base_id]
            return ArxivSearchResponse(papers=filtered[:max_results])

        is_title_search = len(q.split()) > 3
        search_query = f'"{q}"' if is_title_search and not q.startswith('"') else q

        # Run blocking HTTP call in thread pool so we don't block the event loop
        results = await _run_blocking(
            search_arxiv_tool.invoke, {"query": search_query, "max_results": max_results}
        )
        return ArxivSearchResponse(papers=results)

    except Exception as exc:
        logger.error("arXiv search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"arXiv search failed: {str(exc)}")


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

        chunks = read_pdf_tool.invoke({
            "file_path": str(temp_file_path),
            "chunk_size": 1200,
            "chunk_overlap": 200,
        })

        paper_id = Path(file.filename).stem

        for ch in chunks:
            md = ch.get("metadata") or {}
            md["title"] = file.filename
            md["paper_id"] = paper_id
            ch["metadata"] = md

        msg = upsert_project_paper_chunks.invoke({
            "project_id": project_id,
            "paper_id": paper_id,
            "chunks": chunks,
        })

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
        raw_id = request.arxiv_id.strip()
        base_id = raw_id.split("v")[0]

        # Run blocking HTTP call in thread pool
        results = await _run_blocking(
            search_arxiv_tool.invoke, {"query": base_id, "max_results": 10}
        )
        matched = next(
            (p for p in results if (p.get("arxiv_id") or "").split("v")[0] == base_id),
            None,
        )

        if not matched:
            raise HTTPException(status_code=404, detail=f"arXiv paper {request.arxiv_id} not found")

        paper = matched
        pdf_path = download_arxiv_pdf(raw_id)
        if not pdf_path:
            raise HTTPException(status_code=500, detail=f"Failed to download PDF for {request.arxiv_id}")

        chunks = read_pdf_tool.invoke({
            "file_path": pdf_path,
            "chunk_size": 1200,
            "chunk_overlap": 200,
        })

        for ch in chunks:
            md = ch.get("metadata") or {}
            md["arxiv_id"] = paper.get("arxiv_id")
            md["pdf_url"] = paper.get("pdf_url")
            md["title"] = paper.get("title")
            md["paper_id"] = base_id
            ch["metadata"] = md

        msg = upsert_project_paper_chunks.invoke({
            "project_id": request.project_id,
            "paper_id": base_id,
            "chunks": chunks,
        })

        background_tasks.add_task(cleanup_temp_file, pdf_path)

        return ImportArxivResponse(status="success", paper=paper, message=msg)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("arXiv import error: %s", exc)
        raise HTTPException(status_code=500, detail=f"arXiv import failed: {str(exc)}")


# ─────────────────────────────────────────────
# Remove paper
# ─────────────────────────────────────────────

@app.post("/remove-paper", response_model=RemovePaperResponse)
async def remove_paper(request: RemovePaperRequest):
    if not request.project_id or not request.paper_id:
        raise HTTPException(status_code=400, detail="project_id and paper_id are required")

    logger.info("Removing paper %s from project %s", request.paper_id, request.project_id)
    try:
        msg = remove_paper_from_project.invoke({
            "project_id": request.project_id,
            "paper_id": request.paper_id,
        })
        delete_paper_metadata(request.project_id, request.paper_id)
        return RemovePaperResponse(status="success", message=msg)
    except Exception as exc:
        logger.error("Remove paper error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to remove paper: {str(exc)}")


# ─────────────────────────────────────────────
# List papers
# ─────────────────────────────────────────────

@app.get("/projects/{project_id}/papers")
async def list_papers(project_id: str):
    sidecar_meta = load_all_paper_metadata(project_id)

    try:
        faiss_ids = set(list_project_papers.invoke({"project_id": project_id}))
    except Exception:
        faiss_ids = set()

    papers = [meta for pid, meta in sidecar_meta.items() if pid in faiss_ids]

    stale = set(sidecar_meta.keys()) - faiss_ids
    for stale_id in stale:
        logger.info("Auto-cleaning stale sidecar entry: %s", stale_id)
        delete_paper_metadata(project_id, stale_id)

    return {"project_id": project_id, "count": len(papers), "papers": papers}


# ─────────────────────────────────────────────
# Project info
# ─────────────────────────────────────────────

@app.get("/projects/{project_id}", response_model=ProjectInfoResponse)
async def get_project_info(project_id: str):
    return ProjectInfoResponse(
        project_id=project_id,
        description=f"Research project '{project_id}'",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
