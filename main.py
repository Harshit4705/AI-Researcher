from __future__ import annotations

import asyncio
import re
import logging
import tempfile
import json
import textwrap
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from read_pdf import read_pdf_tool
from vector_tools import (
    upsert_project_paper_chunks,
    remove_paper_from_project,
    list_project_papers,
    load_all_paper_metadata,
    save_paper_metadata,
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
PAPER_STORAGE_DIR = Path("paper_store")
PAPER_STORAGE_DIR.mkdir(exist_ok=True)

GEMINI_NOTEBOOK_MODEL = "gemini-2.5-pro"
GEMINI_MAX_TOKENS_ANALYSIS = 8192
GEMINI_MAX_TOKENS_PLAN = 8192
GEMINI_MAX_TOKENS_GENERATE = 65536
GEMINI_MAX_TOKENS_REVIEW = 65536
GEMINI_RETRY_DELAYS = [5, 15, 30]

RESEARCH_LAB_SYSTEM_PROMPT = (
    "You are a senior research engineer building trustworthy research companion artifacts "
    "from academic papers. Produce grounded, educational, CPU-runnable materials that clearly "
    "separate paper-faithful content from simplifications. Prefer transparent assumptions, "
    "evidence-aware summaries, and reproducibility guidance over flashy claims."
)

RESEARCH_LAB_ANALYSIS_PROMPT = """Study the attached research paper PDF and return ONLY valid JSON.

Expected schema:
{
  "paper_title": "full title",
  "author_line": "authors as a single line",
  "field": "primary field",
  "elevator_summary": "2-3 sentence plain English summary",
  "core_question": "what question the paper answers",
  "main_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"],
  "method_blueprint": [
    {
      "name": "component or stage name",
      "role": "why it matters",
      "details": ["key detail 1", "key detail 2"]
    }
  ],
  "datasets": ["dataset or benchmark names"],
  "metrics": ["metric names"],
  "equations": ["important equations or formulas"],
  "implementation_hazards": ["details that make faithful reproduction hard"],
  "evidence_anchors": [
    {
      "claim": "important claim",
      "paper_support": "where or how the paper supports it"
    }
  ]
}

Be conservative: if a detail is unclear, say so instead of inventing it."""

RESEARCH_LAB_PLAN_PROMPT_TEMPLATE = """You are designing a custom Research Lab Pack for a paper.
Return ONLY valid JSON.

Generation goal: {generation_goal}
Compute profile: {compute_profile}
Include study questions: {include_study_questions}
Include reproducibility checklist: {include_reproducibility_checklist}
Include risk notes: {include_risk_notes}

Paper analysis:
```json
{analysis_json}
```

Expected schema:
{{
  "artifact_title": "title for the lab pack",
  "reader_positioning": "who this pack is best for",
  "implementation_strategy": "how to keep the code faithful but runnable",
  "dataset_strategy": "real dataset use or synthetic fallback",
  "execution_notes": ["important runtime notes"],
  "notebook_sections": [
    {{
      "heading": "section heading",
      "goal": "why this section exists",
      "cell_mix": ["markdown", "code"]
    }}
  ],
  "deliverables": ["notebook", "study guide", "reproducibility checklist"],
  "artifact_summary": ["summary bullet 1", "summary bullet 2", "summary bullet 3"]
}}"""

RESEARCH_LAB_GENERATE_PROMPT_TEMPLATE = """Create a Research Lab Pack for this paper and return ONLY valid JSON.

Generation goal: {generation_goal}
Compute profile: {compute_profile}
Include study questions: {include_study_questions}
Include reproducibility checklist: {include_reproducibility_checklist}
Include risk notes: {include_risk_notes}

Paper analysis:
```json
{analysis_json}
```

Plan:
```json
{plan_json}
```

Expected schema:
{{
  "title": "artifact title",
  "dependencies": ["python package names"],
  "artifact_summary": ["summary bullet 1", "summary bullet 2", "summary bullet 3"],
  "study_questions": ["question 1", "question 2"],
  "reproducibility_checklist": ["check 1", "check 2"],
  "risk_notes": ["risk 1", "risk 2"],
  "cells": [
    {{
      "cell_type": "markdown" | "code",
      "source": "complete cell content"
    }}
  ]
}}

Requirements:
- The notebook must run on CPU.
- Use real PyTorch code when modeling is needed.
- Keep cells compact and educational.
- Mark any approximation explicitly.
- Avoid placeholders in important code paths."""

RESEARCH_LAB_REVIEW_PROMPT_TEMPLATE = """You are reviewing a generated Research Lab Pack for validity and clarity.
Return ONLY valid JSON using the exact same schema as the input.

Fix issues such as:
- invalid JSON
- repeated cells
- misleading claims
- code that is not CPU-friendly
- missing explanation around approximations

Generated pack:
```json
{draft_json}
```"""

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


class GenerateNotebookResponse(BaseModel):
    status: str
    paper_id: str
    title: str
    file_name: str
    notebook: Dict[str, Any]
    notebook_json: str
    preview_markdown: str
    dependencies: List[str]
    source_url: Optional[str] = None
    generated_with_model: str
    colab_ready: bool
    artifact_summary: List[str] = []
    study_questions: List[str] = []
    reproducibility_checklist: List[str] = []
    risk_notes: List[str] = []
    generation_goal: str = "teaching"
    compute_profile: str = "balanced"


class GenerateNotebookRequest(BaseModel):
    api_key: str
    model: Optional[str] = GEMINI_NOTEBOOK_MODEL
    generation_goal: str = "teaching"
    compute_profile: str = "balanced"
    include_study_questions: bool = True
    include_reproducibility_checklist: bool = True
    include_risk_notes: bool = True


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


def _project_paper_dir(project_id: str) -> Path:
    safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", project_id).strip("_") or "project"
    path = PAPER_STORAGE_DIR / safe_project
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paper_storage_path(project_id: str, paper_id: str, suffix: str = ".pdf") -> Path:
    safe_paper = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("_") or "paper"
    return _project_paper_dir(project_id) / f"{safe_paper}{suffix}"


def _persist_pdf_file(project_id: str, paper_id: str, source_path: str) -> str:
    target = _paper_storage_path(project_id, paper_id)
    target.write_bytes(Path(source_path).read_bytes())
    return str(target)


def _persist_pdf_bytes(project_id: str, paper_id: str, pdf_bytes: bytes) -> str:
    target = _paper_storage_path(project_id, paper_id)
    target.write_bytes(pdf_bytes)
    return str(target)


def _delete_persisted_pdf(file_path: Optional[str]) -> None:
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Failed to delete persisted PDF %s: %s", file_path, exc)


def _download_pdf_bytes(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=45,
        headers={"User-Agent": "ai-researcher/3.0"},
        allow_redirects=True,
    )
    response.raise_for_status()
    data = response.content
    if b"%PDF" not in data[:1024]:
        raise ValueError("Downloaded file is not a valid PDF.")
    return data


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


def _chunk_sort_key(chunk: Dict[str, Any]) -> tuple[int, int]:
    metadata = chunk.get("metadata") or {}
    page = metadata.get("page_number") or metadata.get("page") or chunk.get("page") or 0
    try:
        page_num = int(page)
    except Exception:
        page_num = 0
    content_len = len((chunk.get("content") or "").strip())
    return (page_num, -content_len)


def _extract_equation_candidates(text: str, limit: int = 8) -> List[str]:
    candidates: List[str] = []
    patterns = [
        r"\$\$(.+?)\$\$",
        r"\$(.+?)\$",
        r"\\\[(.+?)\\\]",
        r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.DOTALL):
            cleaned = re.sub(r"\s+", " ", match).strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
            if len(candidates) >= limit:
                return candidates

    for line in text.splitlines():
        clean_line = re.sub(r"\s+", " ", line).strip()
        if len(clean_line) < 12 or len(clean_line) > 160:
            continue
        if "=" in clean_line and any(token in clean_line for token in ("sum", "softmax", "loss", "exp", "sqrt", "W", "Q", "K", "V", "lambda", "theta")):
            if clean_line not in candidates:
                candidates.append(clean_line)
        if len(candidates) >= limit:
            break

    return candidates[:limit]


def _normalize_dependency_name(package: str) -> Optional[str]:
    pkg = (package or "").strip().lower()
    if not pkg:
        return None

    pkg = pkg.replace("_", "-")
    alias_map = {
        "sklearn": "scikit-learn",
        "pil": "pillow",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
    }
    return alias_map.get(pkg, pkg)


def _detect_dependencies_from_code(*blocks: str) -> List[str]:
    packages: List[str] = []
    seen = set()
    stdlib = {
        "os", "sys", "math", "json", "time", "pathlib", "typing", "random",
        "collections", "itertools", "functools", "dataclasses", "statistics",
    }

    for block in blocks:
        for match in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)", block or "", flags=re.MULTILINE):
            root = match.split(".", 1)[0]
            normalized = _normalize_dependency_name(root)
            if normalized and normalized not in stdlib and normalized not in seen:
                seen.add(normalized)
                packages.append(normalized)

    return packages


def _clean_model_block(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:python|json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _prepare_notebook_context(project_id: str, paper_id: str, max_chars: int = 12000) -> Dict[str, Any]:
    sidecar = load_all_paper_metadata(project_id)
    paper = sidecar.get(paper_id) or {"paper_id": paper_id, "title": paper_id}
    chunks = [
        chunk for chunk in get_all_project_chunks(project_id)
        if ((chunk.get("metadata") or {}).get("paper_id") or paper_id) == paper_id
    ]
    if not chunks:
        raise ValueError(f"Paper '{paper_id}' was not found in this project.")

    is_metadata_only = bool(paper.get("is_metadata_only")) or all(
        bool((chunk.get("metadata") or {}).get("is_metadata_only")) for chunk in chunks
    )
    if is_metadata_only:
        raise ValueError("Notebook generation requires a full-text paper in your library, not a metadata-only entry.")

    ordered_chunks = sorted(chunks, key=_chunk_sort_key)
    content_parts: List[str] = []
    chars_used = 0
    for chunk in ordered_chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        remaining = max_chars - chars_used
        if remaining <= 0:
            break
        snippet = content[:remaining]
        content_parts.append(snippet)
        chars_used += len(snippet)

    paper_text = "\n\n".join(content_parts).strip()
    if not paper_text:
        raise ValueError(f"Paper '{paper.get('title', paper_id)}' does not have usable text content.")

    return {
        "paper_id": paper_id,
        "title": paper.get("title") or paper_id,
        "authors": paper.get("authors") or [],
        "published": paper.get("published") or "",
        "source_url": paper.get("pdf_url") or paper.get("abs_url") or "",
        "local_pdf_path": paper.get("local_pdf_path") or "",
        "paper_text": paper_text,
        "equation_candidates": _extract_equation_candidates(paper_text),
        "chunk_count": len(chunks),
    }


def _get_gemini_api_key(api_key: Optional[str]) -> str:
    key = (api_key or "").strip()
    if key:
        return key
    raise ValueError("Gemini API key is required to generate notebooks.")


def _call_gemini(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: List[Any],
    max_tokens: int,
) -> str:
    client = genai.Client(api_key=_get_gemini_api_key(api_key))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.7,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=config,
    )
    text = getattr(response, "text", "") or ""
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text


def _call_gemini_with_retry(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: List[Any],
    max_tokens: int,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(len(GEMINI_RETRY_DELAYS) + 1):
        try:
            return _call_gemini(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if any(keyword in message for keyword in ("api key not valid", "api_key_invalid", "invalid_argument", "permission_denied", "unauthenticated")):
                raise ValueError("Invalid Gemini API key. Please check it and try again.") from exc
            if attempt >= len(GEMINI_RETRY_DELAYS) or not any(keyword in message for keyword in ("429", "rate", "500", "503", "overloaded", "unavailable", "deadline")):
                raise
            time.sleep(GEMINI_RETRY_DELAYS[attempt])
    raise RuntimeError(f"Gemini notebook generation failed: {last_error}")


def _parse_llm_json(raw_text: str, step_name: str, *, api_key: str, model: str) -> Any:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        repair_prompt = (
            f"The following text was supposed to be valid JSON but has a syntax error.\n\n"
            f"Step: {step_name}\n"
            f"Error: {exc}\n\n"
            f"Text:\n{text[:12000]}\n\n"
            "Return ONLY corrected valid JSON and nothing else."
        )
        repaired = _call_gemini_with_retry(
            api_key=api_key,
            model=model,
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=[repair_prompt],
            max_tokens=max(4096, min(len(text), 12000)),
        ).strip()
        if repaired.startswith("```"):
            first_newline = repaired.find("\n")
            if first_newline != -1:
                repaired = repaired[first_newline + 1 :]
            if repaired.endswith("```"):
                repaired = repaired[:-3]
            repaired = repaired.strip()
        return json.loads(repaired)


def _load_paper_pdf_bytes(project_id: str, paper_id: str, context: Dict[str, Any]) -> bytes:
    local_pdf_path = (context.get("local_pdf_path") or "").strip()
    if local_pdf_path and Path(local_pdf_path).exists():
        return Path(local_pdf_path).read_bytes()

    sidecar = load_all_paper_metadata(project_id)
    paper = sidecar.get(paper_id) or {}
    source_url = (paper.get("pdf_url") or paper.get("abs_url") or context.get("source_url") or "").strip()
    if not source_url:
        raise ValueError(
            "This paper does not have an accessible PDF file yet. Re-upload the PDF or import a paper with a downloadable PDF before generating a notebook."
        )

    pdf_bytes = _download_pdf_bytes(source_url)
    persisted_path = _persist_pdf_bytes(project_id, paper_id, pdf_bytes)
    paper["local_pdf_path"] = persisted_path
    if paper:
        sidecar.update({paper_id: paper})
        save_payload = {
            "paper_id": paper.get("paper_id") or paper_id,
            "title": paper.get("title") or paper_id,
            "arxiv_id": paper.get("arxiv_id"),
            "pdf_url": paper.get("pdf_url"),
            "local_pdf_path": persisted_path,
            "abs_url": paper.get("abs_url"),
            "source": paper.get("source"),
            "is_metadata_only": bool(paper.get("is_metadata_only")),
            "chunk_count": int(paper.get("chunk_count") or 0),
        }
        save_paper_metadata(project_id, paper_id, save_payload)
    return pdf_bytes


def _gemini_cells_to_preview(cells: List[Dict[str, Any]], max_cells: int = 8) -> str:
    preview_parts: List[str] = []
    for cell in cells[:max_cells]:
        source = (cell.get("source") or "").strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            preview_parts.append(source)
        else:
            preview_parts.append(f"```python\n{source}\n```")
    return "\n\n".join(preview_parts).strip()


def _build_notebook_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    notebook_cells: List[Dict[str, Any]] = []
    for cell in cells:
        cell_type = cell.get("cell_type") or "markdown"
        source = cell.get("source") or ""
        if cell_type == "code":
            notebook_cells.append(_code_cell(source))
        else:
            notebook_cells.append(_markdown_cell(source))
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {
                "provenance": [],
                "include_colab_link": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _normalize_string_list(values: Any, *, fallback: Optional[List[str]] = None, limit: int = 8) -> List[str]:
    items = values if isinstance(values, list) else fallback or []
    cleaned: List[str] = []
    for item in items[:limit]:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _append_lab_pack_cells(
    cells: List[Dict[str, Any]],
    *,
    study_questions: List[str],
    reproducibility_checklist: List[str],
    risk_notes: List[str],
) -> List[Dict[str, Any]]:
    enriched = list(cells)
    if study_questions:
        enriched.append(
            {
                "cell_type": "markdown",
                "source": "## Study Questions\n\n" + "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(study_questions)),
            }
        )
    if reproducibility_checklist:
        enriched.append(
            {
                "cell_type": "markdown",
                "source": "## Reproducibility Checklist\n\n" + "\n".join(f"- [ ] {item}" for item in reproducibility_checklist),
            }
        )
    if risk_notes:
        enriched.append(
            {
                "cell_type": "markdown",
                "source": "## Risk And Assumption Notes\n\n" + "\n".join(f"- {item}" for item in risk_notes),
            }
        )
    return enriched


def _build_lab_pack_preview(
    *,
    title: str,
    artifact_summary: List[str],
    study_questions: List[str],
    reproducibility_checklist: List[str],
    risk_notes: List[str],
    cells: List[Dict[str, Any]],
) -> str:
    sections: List[str] = [f"# {title}"]
    if artifact_summary:
        sections.append("## Artifact Summary\n\n" + "\n".join(f"- {item}" for item in artifact_summary))
    if study_questions:
        sections.append("## Study Questions\n\n" + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(study_questions)))
    if reproducibility_checklist:
        sections.append("## Reproducibility Checklist\n\n" + "\n".join(f"- [ ] {item}" for item in reproducibility_checklist))
    if risk_notes:
        sections.append("## Risk Notes\n\n" + "\n".join(f"- {item}" for item in risk_notes))
    cell_preview = _gemini_cells_to_preview(cells)
    if cell_preview:
        sections.append("## Notebook Preview\n\n" + cell_preview)
    return "\n\n".join(section for section in sections if section).strip()


def _run_gemini_notebook_pipeline(
    *,
    project_id: str,
    paper_id: str,
    context: Dict[str, Any],
    api_key: str,
    model: str,
    generation_goal: str,
    compute_profile: str,
    include_study_questions: bool,
    include_reproducibility_checklist: bool,
    include_risk_notes: bool,
) -> tuple[Dict[str, Any], str, List[str], str, str, str, List[str], List[str], List[str], List[str]]:
    pdf_bytes = _load_paper_pdf_bytes(project_id, paper_id, context)
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    analysis_raw = _call_gemini_with_retry(
        api_key=api_key,
        model=model,
        system_prompt=RESEARCH_LAB_SYSTEM_PROMPT,
        user_content=[pdf_part, RESEARCH_LAB_ANALYSIS_PROMPT],
        max_tokens=GEMINI_MAX_TOKENS_ANALYSIS,
    )
    analysis = _parse_llm_json(analysis_raw, "research_lab_analysis", api_key=api_key, model=model)

    plan_prompt = RESEARCH_LAB_PLAN_PROMPT_TEMPLATE.format(
        generation_goal=generation_goal,
        compute_profile=compute_profile,
        include_study_questions=str(include_study_questions).lower(),
        include_reproducibility_checklist=str(include_reproducibility_checklist).lower(),
        include_risk_notes=str(include_risk_notes).lower(),
        analysis_json=json.dumps(analysis, indent=2, ensure_ascii=True),
    )
    plan_raw = _call_gemini_with_retry(
        api_key=api_key,
        model=model,
        system_prompt=RESEARCH_LAB_SYSTEM_PROMPT,
        user_content=[pdf_part, plan_prompt],
        max_tokens=GEMINI_MAX_TOKENS_PLAN,
    )
    plan = _parse_llm_json(plan_raw, "research_lab_plan", api_key=api_key, model=model)

    generate_prompt = RESEARCH_LAB_GENERATE_PROMPT_TEMPLATE.format(
        generation_goal=generation_goal,
        compute_profile=compute_profile,
        include_study_questions=str(include_study_questions).lower(),
        include_reproducibility_checklist=str(include_reproducibility_checklist).lower(),
        include_risk_notes=str(include_risk_notes).lower(),
        analysis_json=json.dumps(analysis, indent=2, ensure_ascii=True),
        plan_json=json.dumps(plan, indent=2, ensure_ascii=True),
    )
    draft_raw = _call_gemini_with_retry(
        api_key=api_key,
        model=model,
        system_prompt=RESEARCH_LAB_SYSTEM_PROMPT,
        user_content=[pdf_part, generate_prompt],
        max_tokens=GEMINI_MAX_TOKENS_GENERATE,
    )
    draft = _parse_llm_json(draft_raw, "research_lab_draft", api_key=api_key, model=model)
    if not isinstance(draft, dict):
        raise ValueError("Gemini did not return a valid Research Lab Pack payload.")

    review_prompt = RESEARCH_LAB_REVIEW_PROMPT_TEMPLATE.format(
        draft_json=json.dumps(draft, indent=2, ensure_ascii=True)
    )
    reviewed_raw = _call_gemini_with_retry(
        api_key=api_key,
        model=model,
        system_prompt=RESEARCH_LAB_SYSTEM_PROMPT,
        user_content=[review_prompt],
        max_tokens=GEMINI_MAX_TOKENS_REVIEW,
    )
    reviewed = _parse_llm_json(reviewed_raw, "research_lab_review", api_key=api_key, model=model)
    if not isinstance(reviewed, dict):
        reviewed = draft

    raw_cells = reviewed.get("cells") if isinstance(reviewed, dict) else None
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("Gemini did not return a valid notebook cell list.")

    artifact_summary = _normalize_string_list(
        reviewed.get("artifact_summary"),
        fallback=_normalize_string_list((plan or {}).get("artifact_summary"), limit=4),
        limit=6,
    )
    study_questions = _normalize_string_list(
        reviewed.get("study_questions"),
        fallback=["What is the paper's main claim?", "Which part of the notebook is the biggest approximation?"],
        limit=6,
    ) if include_study_questions else []
    reproducibility_checklist = _normalize_string_list(
        reviewed.get("reproducibility_checklist"),
        fallback=_normalize_string_list((analysis or {}).get("implementation_hazards"), limit=6),
        limit=8,
    ) if include_reproducibility_checklist else []
    risk_notes = _normalize_string_list(
        reviewed.get("risk_notes"),
        fallback=["Treat the generated code as an educational reconstruction, not a verified reproduction."],
        limit=8,
    ) if include_risk_notes else []

    enriched_cells = _append_lab_pack_cells(
        raw_cells,
        study_questions=study_questions,
        reproducibility_checklist=reproducibility_checklist,
        risk_notes=risk_notes,
    )

    notebook = _build_notebook_from_cells(enriched_cells)
    notebook["metadata"]["paper_metadata"] = {
        "paper_id": paper_id,
        "source_url": context.get("source_url") or "",
        "generated_with_model": model,
        "generation_goal": generation_goal,
        "compute_profile": compute_profile,
    }
    notebook_json = json.dumps(notebook, indent=2)
    title = str(
        (reviewed.get("title") if isinstance(reviewed, dict) else None)
        or (analysis.get("paper_title") if isinstance(analysis, dict) else None)
        or context["title"]
    )
    preview_markdown = _build_lab_pack_preview(
        title=title,
        artifact_summary=artifact_summary,
        study_questions=study_questions,
        reproducibility_checklist=reproducibility_checklist,
        risk_notes=risk_notes,
        cells=enriched_cells,
    )
    dependencies = _detect_dependencies_from_code(
        *[
            str(cell.get("source") or "")
            for cell in enriched_cells
            if (cell.get("cell_type") or "") == "code"
        ]
    )
    dependencies = list(
        dict.fromkeys(
            [
                *(_normalize_string_list((reviewed or {}).get("dependencies"), limit=12)),
                *dependencies,
            ]
        )
    )
    file_name = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', title).strip('_') or paper_id}.ipynb"
    return (
        notebook,
        notebook_json,
        dependencies,
        preview_markdown,
        file_name,
        title,
        artifact_summary,
        study_questions,
        reproducibility_checklist,
        risk_notes,
    )


def _markdown_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


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
            "paper_notebook": "POST /projects/{project_id}/papers/{paper_id}/generate-notebook",
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


@app.post("/projects/{project_id}/papers/{paper_id}/generate-notebook", response_model=GenerateNotebookResponse)
async def generate_paper_notebook(project_id: str, paper_id: str, request: GenerateNotebookRequest):
    if not project_id or not paper_id:
        raise HTTPException(status_code=400, detail="project_id and paper_id are required")
    if not request.api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key is required to generate a notebook.")

    try:
        context = await _run_blocking(_prepare_notebook_context, project_id, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Notebook context error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to prepare notebook context: {str(exc)}") from exc

    try:
        notebook, notebook_json, dependencies, preview_markdown, file_name, title, artifact_summary, study_questions, reproducibility_checklist, risk_notes = await _run_blocking(
            _run_gemini_notebook_pipeline,
            project_id=project_id,
            paper_id=paper_id,
            context=context,
            api_key=request.api_key.strip(),
            model=(request.model or GEMINI_NOTEBOOK_MODEL).strip() or GEMINI_NOTEBOOK_MODEL,
            generation_goal=request.generation_goal.strip() or "teaching",
            compute_profile=request.compute_profile.strip() or "balanced",
            include_study_questions=request.include_study_questions,
            include_reproducibility_checklist=request.include_reproducibility_checklist,
            include_risk_notes=request.include_risk_notes,
        )
        return GenerateNotebookResponse(
            status="success",
            paper_id=paper_id,
            title=title,
            file_name=file_name,
            notebook=notebook,
            notebook_json=notebook_json,
            preview_markdown=preview_markdown,
            dependencies=dependencies,
            source_url=context.get("source_url") or None,
            generated_with_model=(request.model or GEMINI_NOTEBOOK_MODEL).strip() or GEMINI_NOTEBOOK_MODEL,
            colab_ready=True,
            artifact_summary=artifact_summary,
            study_questions=study_questions,
            reproducibility_checklist=reproducibility_checklist,
            risk_notes=risk_notes,
            generation_goal=request.generation_goal.strip() or "teaching",
            compute_profile=request.compute_profile.strip() or "balanced",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Notebook generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Notebook generation failed: {str(exc)}") from exc


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

        paper_id = Path(file.filename).stem
        persisted_pdf_path = _persist_pdf_bytes(project_id, paper_id, contents)

        chunks = await _run_blocking(
            read_pdf_tool.invoke,
            {
                "file_path":     str(temp_file_path),
                "chunk_size":    1200,
                "chunk_overlap": 200,
            }
        )

        for ch in chunks:
            md = ch.get("metadata") or {}
            md["title"] = file.filename
            md["paper_id"] = paper_id
            md["source"] = "pdf_upload"
            md["pdf_url"] = None
            md["local_pdf_path"] = persisted_pdf_path
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
        persisted_pdf_path = None

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
        persisted_pdf_path = _persist_pdf_file(
            request.project_id,
            raw_id.split("v")[0] if is_arxiv_id else raw_id,
            pdf_path,
        )

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
            md["local_pdf_path"] = persisted_pdf_path
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
        sidecar = await _run_blocking(load_all_paper_metadata, request.project_id)
        paper_meta = sidecar.get(request.paper_id) or {}
        msg = await _run_blocking(
            remove_paper_from_project.invoke,
            {"project_id": request.project_id, "paper_id": request.paper_id}
        )
        _delete_persisted_pdf(paper_meta.get("local_pdf_path"))
        return RemovePaperResponse(status="success", message=msg)
    except Exception as exc:
        logger.error("Remove paper error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to remove paper: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

