from __future__ import annotations

import os
import json
import shutil
import logging
from typing import List, Dict, Any, Optional

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_indexes")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# ─────────────────────────────────────────────
# Lazy loaders
# ─────────────────────────────────────────────

_EMBEDDINGS = None
_RERANKER = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _EMBEDDINGS


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model...")
            _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Reranker loaded.")
        except Exception as exc:
            logger.warning("Reranker not available: %s. Skipping reranking.", exc)
            _RERANKER = False  # Mark as unavailable
    return _RERANKER if _RERANKER is not False else None


# ─────────────────────────────────────────────
# Reranking
# ─────────────────────────────────────────────

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Rerank chunks using a cross-encoder for higher relevance precision.
    Falls back to original order if reranker is unavailable.
    """
    if not chunks:
        return chunks

    reranker = get_reranker()
    if reranker is None:
        logger.info("Reranker unavailable — returning top %d chunks as-is", top_k)
        return chunks[:top_k]

    try:
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_k]]
        logger.info("Reranked %d → %d chunks", len(chunks), len(result))
        return result
    except Exception as exc:
        logger.warning("Reranking failed: %s", exc)
        return chunks[:top_k]


# ─────────────────────────────────────────────
# FAISS helpers
# ─────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _project_index_path(project_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in project_id)
    return os.path.join(FAISS_INDEX_DIR, f"project_{safe_id}")


def _load_faiss_for_project(project_id: str) -> Optional[FAISS]:
    index_path = _project_index_path(project_id)
    if not os.path.isdir(index_path):
        return None
    try:
        vs = FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception as exc:
        logger.warning("Failed to load FAISS index for project %s: %s", project_id, exc)
        return None


def _save_faiss_for_project(project_id: str, vs: FAISS) -> None:
    index_path = _project_index_path(project_id)
    _ensure_dir(index_path)
    vs.save_local(index_path)


def _chunks_to_documents(
    project_id: str,
    paper_id: str,
    chunks: List[Dict[str, Any]],
) -> List[Document]:
    docs: List[Document] = []
    for ch in chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        md = dict(ch.get("metadata") or {})
        md.update({
            "project_id": str(project_id),
            "paper_id": str(paper_id),
            "page": ch.get("page"),
            "source": ch.get("source"),
        })
        docs.append(Document(page_content=content, metadata=md))
    return docs


# ─────────────────────────────────────────────
# Paper metadata JSON sidecar
# ─────────────────────────────────────────────

def save_paper_metadata(project_id: str, paper_id: str, metadata: dict) -> None:
    """Save paper metadata to a JSON sidecar file for fast registry lookups."""
    index_path = _project_index_path(project_id)
    _ensure_dir(index_path)
    meta_file = os.path.join(index_path, "papers.json")

    existing: Dict[str, Any] = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing[paper_id] = metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata for paper '%s' in project '%s'", paper_id, project_id)


def load_all_paper_metadata(project_id: str) -> Dict[str, Any]:
    """Load all paper metadata from the JSON sidecar file."""
    index_path = _project_index_path(project_id)
    meta_file = os.path.join(index_path, "papers.json")
    if not os.path.exists(meta_file):
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not load paper metadata: %s", exc)
        return {}


def delete_paper_metadata(project_id: str, paper_id: str) -> None:
    """Remove a paper's entry from the metadata sidecar."""
    index_path = _project_index_path(project_id)
    meta_file = os.path.join(index_path, "papers.json")
    if not os.path.exists(meta_file):
        return
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.pop(paper_id, None)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Could not delete metadata for paper %s: %s", paper_id, exc)


# ─────────────────────────────────────────────
# LangChain tools
# ─────────────────────────────────────────────

@tool("upsert_project_paper_chunks")
def upsert_project_paper_chunks(
    project_id: str,
    paper_id: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Index paper chunks into FAISS. Removes old chunks for same paper
    first to prevent duplicates on re-import.
    """
    if not chunks:
        return f"No chunks provided for paper '{paper_id}'."

    new_docs = _chunks_to_documents(project_id, paper_id, chunks)
    if not new_docs:
        return f"All chunks were empty for paper '{paper_id}'."

    vs = _load_faiss_for_project(project_id)

    if vs is not None:
        existing_docs = list(vs.docstore._dict.values())
        filtered = [d for d in existing_docs if d.metadata.get("paper_id") != paper_id]
        all_docs = filtered + new_docs
        vs = FAISS.from_documents(all_docs, get_embeddings())
    else:
        vs = FAISS.from_documents(new_docs, get_embeddings())

    _save_faiss_for_project(project_id, vs)

    # Save title/arxiv metadata to sidecar from first chunk
    if new_docs:
        first_md = new_docs[0].metadata
        save_paper_metadata(project_id, paper_id, {
            "paper_id": paper_id,
            "title": first_md.get("title") or paper_id,
            "arxiv_id": first_md.get("arxiv_id"),
            "pdf_url": first_md.get("pdf_url"),
            "source": first_md.get("source"),
        })

    return f"Indexed {len(new_docs)} chunks for paper '{paper_id}' in project '{project_id}'."


@tool("query_project_papers")
def query_project_papers(
    project_id: str,
    query: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks across ALL papers in a project."""
    if not query.strip():
        raise ValueError("query must be non-empty")

    top_k = max(1, min(top_k, 30))
    vs = _load_faiss_for_project(project_id)
    if vs is None:
        return []

    docs = vs.similarity_search(query, k=top_k)
    return [{"content": doc.page_content, "metadata": dict(doc.metadata or {})} for doc in docs]


@tool("query_specific_paper")
def query_specific_paper(
    project_id: str,
    paper_id: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from ONE specific paper only.
    Essential for multi-paper projects.
    """
    if not query.strip():
        raise ValueError("query must be non-empty")

    top_k = max(1, min(top_k, 20))
    vs = _load_faiss_for_project(project_id)
    if vs is None:
        return []

    # Fetch wider pool then filter by paper_id (FAISS has no native metadata filter)
    fetch_k = min(top_k * 15, 150)
    all_docs = vs.similarity_search(query, k=fetch_k)
    filtered = [d for d in all_docs if d.metadata.get("paper_id") == paper_id]

    return [
        {"content": doc.page_content, "metadata": dict(doc.metadata or {})}
        for doc in filtered[:top_k]
    ]


@tool("remove_paper_from_project")
def remove_paper_from_project(project_id: str, paper_id: str) -> str:
    """Remove all chunks of a specific paper and rebuild the index."""
    vs = _load_faiss_for_project(project_id)
    if vs is None:
        return f"No index found for project '{project_id}'."

    all_docs = list(vs.docstore._dict.values())
    remaining = [d for d in all_docs if d.metadata.get("paper_id") != paper_id]

    if len(remaining) == len(all_docs):
        return f"Paper '{paper_id}' not found in project '{project_id}'."

    removed = len(all_docs) - len(remaining)
    index_path = _project_index_path(project_id)

    if not remaining:
        shutil.rmtree(index_path, ignore_errors=True)
        delete_paper_metadata(project_id, paper_id)
        return f"Removed '{paper_id}' ({removed} chunks). Index is now empty."

    new_vs = FAISS.from_documents(remaining, get_embeddings())
    _save_faiss_for_project(project_id, new_vs)
    delete_paper_metadata(project_id, paper_id)

    return f"Removed '{paper_id}' ({removed} chunks). {len(remaining)} chunks remain."


@tool("list_project_papers")
def list_project_papers(project_id: str) -> List[str]:
    """List all unique paper IDs currently indexed in a project."""
    vs = _load_faiss_for_project(project_id)
    if vs is None:
        return []
    all_docs = list(vs.docstore._dict.values())
    return sorted({d.metadata.get("paper_id") for d in all_docs if d.metadata.get("paper_id")})
