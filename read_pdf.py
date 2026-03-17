from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Improved defaults — bigger chunks with more overlap
# preserves more context per chunk and respects paragraph boundaries
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# Separators ordered by priority — paragraph > sentence > word
# This ensures chunks break at natural boundaries not mid-sentence
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


@dataclass
class PDFChunk:
    content: str
    page: Optional[int]
    source: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_pdf_impl(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf file, got: {file_path}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got: {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d); clamping to chunk_size // 5",
            chunk_overlap, chunk_size,
        )
        chunk_overlap = chunk_size // 5

    logger.info("Loading PDF: %s", file_path)

    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF '{file_path}': {exc}") from exc

    if not documents:
        logger.warning("PyPDFLoader returned no pages for: %s", file_path)
        return []

    # Improved splitter — respects paragraph/sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = splitter.split_documents(documents)

    chunks: List[PDFChunk] = []
    for doc in split_docs:
        text = (doc.page_content or "").strip()
        if not text or len(text) < 50:  # Skip tiny meaningless chunks
            continue

        md = dict(doc.metadata or {})
        page = md.get("page", md.get("page_number"))
        source = md.get("source", file_path)

        chunk = PDFChunk(content=text, page=page, source=source, metadata=md)
        chunks.append(chunk)

    logger.info("Created %d chunks from %s", len(chunks), file_path)
    return [c.to_dict() for c in chunks]


@tool("read_pdf")
def read_pdf_tool(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Read a local PDF and return overlapping text chunks.

    Args:
        file_path: Path to the PDF on disk.
        chunk_size: Max characters per chunk (default 1200).
        chunk_overlap: Overlap between chunks (default 200).

    Returns:
        List of dicts with: content, page, source, metadata.
    """
    return _read_pdf_impl(file_path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
