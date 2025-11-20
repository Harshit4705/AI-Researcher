# vector_tools.py

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings  # from langchain-huggingface [web:167]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_indexes")

# Free local embedding model (runs via sentence-transformers, no API key). [web:166][web:157]
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)

logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # recommended for BGE. [web:166]
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _project_index_path(project_id: str) -> str:
    return os.path.join(FAISS_INDEX_DIR, f"project_{project_id}")


def _load_faiss_for_project(project_id: str) -> FAISS | None:
    """
    Load FAISS index for a project if it exists; otherwise return None. [web:37][web:38]
    """
    index_path = _project_index_path(project_id)
    if not os.path.isdir(index_path):
        return None

    try:
        vs = FAISS.load_local(
            index_path,
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )
        logger.debug("Loaded existing FAISS index at %s", index_path)
        return vs
    except Exception as exc:
        logger.warning("Failed to load FAISS index for project %s: %s", project_id, exc)
        return None


def _save_faiss_for_project(project_id: str, vs: FAISS) -> None:
    """
    Persist FAISS index for a project to disk. [web:37][web:38]
    """
    index_path = _project_index_path(project_id)
    _ensure_dir(index_path)
    vs.save_local(index_path)
    logger.debug("Saved FAISS index for project %s to %s", project_id, index_path)


def _chunks_to_documents(
    project_id: str,
    paper_id: str,
    chunks: List[Dict[str, Any]],
) -> List[Document]:
    """
    Convert chunk dicts (from read_pdf_tool) into LangChain Documents. [web:35][web:153]
    """
    docs: List[Document] = []
    for ch in chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue

        md = dict(ch.get("metadata") or {})
        md.update(
            {
                "project_id": str(project_id),
                "paper_id": str(paper_id),
                "page": ch.get("page"),
                "source": ch.get("source"),
            }
        )
        docs.append(Document(page_content=content, metadata=md))
    return docs


# ---------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------

@tool("upsert_project_paper_chunks")
def upsert_project_paper_chunks(
    project_id: str,
    paper_id: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Index / update text chunks for a given project and paper in a FAISS vector store.

    Args:
        project_id: ID of the research project.
        paper_id: ID of the paper within that project.
        chunks: List of dicts from read_pdf_tool:
            - content (str)
            - page (int, optional)
            - source (str)
            - metadata (dict)

    Returns:
        Status message.
    """
    if not chunks:
        return f"No chunks provided for paper {paper_id} in project {project_id}."

    docs = _chunks_to_documents(project_id, paper_id, chunks)

    vs = _load_faiss_for_project(project_id)
    if vs is None:
        logger.info("Creating new FAISS index for project %s", project_id)
        vs = FAISS.from_documents(docs, EMBEDDINGS)  # builds index with local BGE embeddings. [web:38][web:166]
    else:
        logger.info("Adding documents to existing FAISS index for project %s", project_id)
        vs.add_documents(docs)

    _save_faiss_for_project(project_id, vs)

    msg = f"Indexed {len(docs)} chunks for paper {paper_id} in project {project_id}."
    logger.info(msg)
    return msg


@tool("query_project_papers")
def query_project_papers(
    project_id: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a question from a project's papers.

    Args:
        project_id: ID of the project.
        query: Natural language query.
        top_k: Number of chunks to return.

    Returns:
        List of dicts:
        - content: text
        - metadata: includes project_id, paper_id, page, source
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    vs = _load_faiss_for_project(project_id)
    if vs is None:
        logger.warning("No FAISS index found for project %s", project_id)
        return []

    docs = vs.similarity_search(query, k=top_k)  # FAISS + BGE embeddings for similarity search. [web:37][web:166]

    results: List[Dict[str, Any]] = []
    for doc in docs:
        results.append(
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
        )
    logger.info(
        "Retrieved %d chunks for project %s with query %r",
        len(results),
        project_id,
        query,
    )
    return results


# ---------------------------------------------------------------------
# Simple CLI test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Manual test in uv project:

    1. Run `uv add ...` as shown in README / above.
    2. Place 'sample.pdf' in project root.
    3. Run: `uv run python vector_tools.py`
    """
    import json
    from read_pdf import _read_pdf_impl

    logging.basicConfig(level=logging.INFO)

    project_id = "demo_project"
    paper_id = "sample_paper"
    pdf_path = "sample.pdf"

    if not os.path.exists(pdf_path):
        print(f"Place a test PDF at: {pdf_path}")
        raise SystemExit(1)

    print(f"Reading PDF: {pdf_path}")
    chunks = _read_pdf_impl(pdf_path, chunk_size=800, chunk_overlap=100)
    print(f"Got {len(chunks)} chunks; indexing into FAISS with local BGE embeddings...")

    msg = upsert_project_paper_chunks.invoke(
        {
            "project_id": project_id,
            "paper_id": paper_id,
            "chunks": chunks,
        }
    )
    print(msg)

    question = "What is the main contribution of this paper?"
    print(f"\nQuerying with: {question!r}")
    hits = query_project_papers.invoke(
        {
            "project_id": project_id,
            "query": question,
            "top_k": 3,
        }
    )

    print("\nTop chunks:")
    for i, hit in enumerate(hits, start=1):
        print("-" * 80)
        print(f"Hit {i}")
        print("Metadata:", json.dumps(hit["metadata"], indent=2))
        print(hit["content"][:400], "...")
