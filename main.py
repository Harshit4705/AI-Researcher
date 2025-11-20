# main.py

from __future__ import annotations

import os
import re
import logging
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from read_pdf import read_pdf_tool
from vector_tools import upsert_project_paper_chunks
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
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.gettempdir()) / "ai_researcher_downloads"
TEMP_DIR.mkdir(exist_ok=True)


# =====================================================================
# Pydantic models
# =====================================================================

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


class ProjectInfoResponse(BaseModel):
    project_id: str
    description: str


# =====================================================================
# Helper functions
# =====================================================================

def download_arxiv_pdf(arxiv_id: str) -> Optional[str]:
    """
    Download arXiv PDF by ID and save to temp directory.
    """
    base_id = arxiv_id.split("v")[0]
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

    try:
        logger.info("Downloading arXiv PDF: %s (url=%s)", arxiv_id, pdf_url)
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        pdf_path = TEMP_DIR / f"{base_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        logger.info("Downloaded PDF to: %s", pdf_path)
        return str(pdf_path)
    except Exception as exc:
        logger.error("Failed to download arXiv PDF %s: %s", arxiv_id, exc)
        return None


def cleanup_temp_file(file_path: str):
    try:
        Path(file_path).unlink(missing_ok=True)
        logger.debug("Cleaned up temp file: %s", file_path)
    except Exception as exc:
        logger.warning("Failed to clean up temp file %s: %s", file_path, exc)


def _normalize_text(s: str) -> str:
    """Lowercase and strip punctuation for simple similarity checks."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


# =====================================================================
# Health & info
# =====================================================================

@app.get("/")
async def root():
    return {
        "message": "AI Research Assistant API",
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat-stream",
            "arxiv_search": "POST /arxiv-search",
            "arxiv_import": "POST /arxiv-import",
            "ingest_pdf": "POST /ingest-pdf",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# =====================================================================
# Chat endpoints
# =====================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.project_id or not request.question:
        raise HTTPException(status_code=400, detail="project_id and question are required")

    logger.info("Chat request - project: %s", request.project_id)

    try:
        state = APP.invoke(
            {
                "project_id": request.project_id,
                "question": request.question,
            }
        )
        return ChatResponse(
            answer=state.get("answer", "No answer generated"),
            sources=state.get("sources", []),
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(exc)}")


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    if not request.project_id or not request.question:
        raise HTTPException(status_code=400, detail="project_id and question are required")

    logger.info("Chat stream request - project: %s", request.project_id)

    async def generate():
        try:
            state = APP.invoke(
                {
                    "project_id": request.project_id,
                    "question": request.question,
                }
            )
            answer = state.get("answer", "")
            sources = state.get("sources", [])

            for word in answer.split():
                yield f"data: {word} \n\n"

            yield f"data: [SOURCES]{json.dumps(sources)}\n\n"
        except Exception as exc:
            logger.error("Chat stream error: %s", exc)
            yield f"data: [ERROR]{str(exc)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =====================================================================
# arXiv search
# =====================================================================

@app.post("/arxiv-search", response_model=ArxivSearchResponse)
async def arxiv_search(request: ArxivSearchRequest):
    """
    Search arXiv with improved handling for full titles and IDs.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="query is required")

    q = request.query.strip()
    max_results = request.max_results or 5
    logger.info("arXiv search: %r", q)

    try:
        # 1. Check if query is an arXiv ID (e.g. 2511.13720)
        id_match = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", q)
        if id_match:
            base_id = q.split("v")[0]
            results = search_arxiv_tool.invoke(
                {"query": base_id, "max_results": max_results * 2}
            )
            filtered = []
            for p in results:
                pid = (p.get("arxiv_id") or "").split("v")[0]
                if pid == base_id:
                    filtered.append(p)
            return ArxivSearchResponse(papers=filtered[:max_results])

        # 2. Check if query looks like a full title (long string with quotes/spaces)
        is_title_search = len(q.split()) > 3
        
        search_query = q
        if is_title_search and not (q.startswith('"') and q.endswith('"')):
             search_query = f'"{q}"'

        results = search_arxiv_tool.invoke(
            {"query": search_query, "max_results": max_results}
        )
        
        return ArxivSearchResponse(papers=results)

    except Exception as exc:
        logger.error("arXiv search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"arXiv search failed: {str(exc)}")


# =====================================================================
# PDF ingestion
# =====================================================================

@app.post("/ingest-pdf", response_model=IngestPdfResponse)
async def ingest_pdf(
    project_id: str,
    file: UploadFile = File(...),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    logger.info("Ingesting PDF %s for project %s", file.filename, project_id)

    temp_file_path = TEMP_DIR / file.filename

    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        chunks = read_pdf_tool.invoke(
            {
                "file_path": str(temp_file_path),
                "chunk_size": 800,
                "chunk_overlap": 100,
            }
        )
        logger.info("Created %d chunks from PDF", len(chunks))

        paper_id = Path(file.filename).stem

        msg = upsert_project_paper_chunks.invoke(
            {
                "project_id": project_id,
                "paper_id": paper_id,
                "chunks": chunks,
            }
        )

        return IngestPdfResponse(
            status="success",
            paper_id=paper_id,
            message=msg,
        )
    except Exception as exc:
        logger.error("PDF ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(exc)}")
    finally:
        # cleanup_temp_file(str(temp_file_path))
        pass


# =====================================================================
# arXiv import
# =====================================================================

@app.post("/arxiv-import", response_model=ImportArxivResponse)
async def import_arxiv_paper(
    request: ImportArxivRequest,
    background_tasks: BackgroundTasks,
):
    if not request.project_id or not request.arxiv_id:
        raise HTTPException(status_code=400, detail="project_id and arxiv_id are required")

    logger.info(
        "Importing arXiv paper %s into project %s",
        request.arxiv_id,
        request.project_id,
    )

    try:
        raw_id = request.arxiv_id.strip()
        base_id = raw_id.split("v")[0]

        # Find metadata first
        results = search_arxiv_tool.invoke(
            {"query": base_id, "max_results": 10}
        )

        matched = None
        for p in results:
            pid = (p.get("arxiv_id") or "").split("v")[0]
            if pid == base_id:
                matched = p
                break

        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"arXiv paper {request.arxiv_id} not found",
            )

        paper = matched
        
        # Download PDF
        pdf_path = download_arxiv_pdf(raw_id)
        if not pdf_path:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download PDF for arXiv ID {request.arxiv_id}",
            )

        # Chunk PDF
        chunks = read_pdf_tool.invoke(
            {
                "file_path": pdf_path,
                "chunk_size": 800,
                "chunk_overlap": 100,
            }
        )
        
        # Attach metadata (Title + URL) to every chunk for Chat links
        for ch in chunks:
            md = ch.get("metadata") or {}
            md["arxiv_id"] = paper.get("arxiv_id")
            md["pdf_url"] = paper.get("pdf_url")
            md["title"] = paper.get("title")
            ch["metadata"] = md

        # Upsert to FAISS
        msg = upsert_project_paper_chunks.invoke(
            {
                "project_id": request.project_id,
                "paper_id": base_id,
                "chunks": chunks,
            }
        )
        
        background_tasks.add_task(cleanup_temp_file, pdf_path)

        return ImportArxivResponse(
            status="success",
            paper=paper,
            message=msg,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("arXiv import error: %s", exc)
        raise HTTPException(status_code=500, detail=f"arXiv import failed: {str(exc)}")


# =====================================================================
# Project info
# =====================================================================

@app.get("/projects/{project_id}", response_model=ProjectInfoResponse)
async def get_project_info(project_id: str):
    return ProjectInfoResponse(
        project_id=project_id,
        description=f"Research project '{project_id}'",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
