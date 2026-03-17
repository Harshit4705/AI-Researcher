from __future__ import annotations

import time
import logging
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET

import requests
from langchain_core.tools import tool

ARXIV_API_URL = "https://export.arxiv.org/api/query"
MIN_DELAY_BETWEEN_CALLS = 1.0
MAX_RESULTS_LIMIT = 50
DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
BACKOFF_FACTOR = 1.5

_last_request_ts: float = 0.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    updated: str
    pdf_url: Optional[str]
    primary_category: Optional[str]
    categories: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _respect_rate_limit() -> None:
    global _last_request_ts
    now = time.time()
    elapsed = now - _last_request_ts
    if elapsed < MIN_DELAY_BETWEEN_CALLS:
        wait_for = MIN_DELAY_BETWEEN_CALLS - elapsed
        logger.debug("Sleeping %.2f seconds to respect arXiv rate limit", wait_for)
        time.sleep(wait_for)
    _last_request_ts = time.time()


def _request_with_backoff(params: Dict[str, Any]) -> str:
    attempt = 0
    while True:
        attempt += 1
        _respect_rate_limit()
        try:
            resp = requests.get(
                ARXIV_API_URL,
                params=params,
                timeout=(10, REQUEST_TIMEOUT),  # (connect_timeout, read_timeout)
                headers={"User-Agent": "ai-researcher/0.1 (mailto:youremail@example.com)"},
            )
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"arXiv request failed after {attempt} attempts: {exc}") from exc
            sleep_for = BACKOFF_FACTOR ** (attempt - 1)
            logger.warning("Request error on attempt %d: %s; retrying in %.1fs", attempt, exc, sleep_for)
            time.sleep(sleep_for)
            continue

        if resp.status_code == 200:
            return resp.text

        if attempt >= MAX_RETRIES:
            raise RuntimeError(f"arXiv API returned status {resp.status_code} after {attempt} attempts.")
        sleep_for = BACKOFF_FACTOR ** (attempt - 1)
        logger.warning("arXiv API status %d on attempt %d; retrying in %.1fs", resp.status_code, attempt, sleep_for)
        time.sleep(sleep_for)


def _parse_arxiv_feed(feed_xml: str) -> List[ArxivPaper]:
    # FIX: Handle malformed XML gracefully
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError as exc:
        logger.error("Failed to parse arXiv XML response: %s", exc)
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers: List[ArxivPaper] = []
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        arxiv_id = raw_id.split("/")[-1] if raw_id else ""

        title = entry.findtext("atom:title", default="", namespaces=ns)
        title = " ".join(title.split())

        summary = entry.findtext("atom:summary", default="", namespaces=ns)
        summary = textwrap.shorten(" ".join(summary.split()), width=2000, placeholder="...")

        authors: List[str] = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns).strip()
            if name:
                authors.append(name)

        published = entry.findtext("atom:published", default="", namespaces=ns).strip()
        updated = entry.findtext("atom:updated", default=published, namespaces=ns).strip()

        # FIX: Also track abs_url as fallback for pdf_url
        pdf_url: Optional[str] = None
        abs_url: Optional[str] = None
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            title_attr = link.attrib.get("title", "")
            link_type = link.attrib.get("type", "")
            if title_attr == "pdf" or link_type == "application/pdf":
                pdf_url = href
            elif rel == "alternate" or title_attr == "abs":
                abs_url = href

        # FIX: Derive pdf_url from abs_url or arxiv_id if not found
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        primary_cat_elem = entry.find("arxiv:primary_category", ns)
        primary_category = primary_cat_elem.attrib.get("term") if primary_cat_elem is not None else None

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.attrib.get("term")
            if term:
                categories.append(term)

        papers.append(ArxivPaper(
            arxiv_id=arxiv_id, title=title, summary=summary, authors=authors,
            published=published, updated=updated, pdf_url=pdf_url,
            primary_category=primary_category, categories=categories,
        ))

    return papers


def _search_arxiv_impl(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    feed_xml = _request_with_backoff(params)
    papers = _parse_arxiv_feed(feed_xml)
    return [p.to_dict() for p in papers]


@tool("search_arxiv")
def search_arxiv_tool(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Search arXiv for recent papers about a topic.

    Args:
        query: Natural language query.
        max_results: Maximum number of papers to return (1-50).

    Returns:
        List of dicts with: arxiv_id, title, summary, authors,
        published, updated, pdf_url, primary_category, categories.
    """
    return _search_arxiv_impl(query=query, max_results=max_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "large language models"
    print(f"Querying arXiv for: {q!r}")
    results = _search_arxiv_impl(q, max_results=3)
    for i, paper in enumerate(results, start=1):
        print("-" * 80)
        print(f"[{i}] {paper['title']} ({paper['published']})")
        print(f"    arxiv_id: {paper['arxiv_id']}")
        print(f"    pdf_url:  {paper['pdf_url']}")
        print(f"    authors:  {', '.join(paper['authors'])}")
        print()
        print(textwrap.shorten(paper['summary'], width=400, placeholder='...'))
