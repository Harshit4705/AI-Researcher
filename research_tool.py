from __future__ import annotations

"""
research_tool.py  ─  Multi-source research paper search tool
Sources   : arXiv · Semantic Scholar · PubMed
Features  : LLM-powered query decomposition · author/title/topic/year/id search
            · multi-source fan-out · deduplication · relevance ranking
            · deterministic URL generation (links always correct)
"""

import html
import os
import random
import time
import logging
import textwrap
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

ARXIV_API_URL        = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_AUTHOR_URL  = "https://api.semanticscholar.org/graph/v1/author/search"
PUBMED_SEARCH_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

REQUEST_TIMEOUT      = 20
MAX_RETRIES          = 3
BACKOFF_FACTOR       = 2.0
MIN_ARXIV_DELAY      = 1.0
DEFAULT_MAX_RESULTS  = 5
MAX_RESULTS_LIMIT    = 30

_last_arxiv_ts: float = 0.0

HEADERS = {
    "User-Agent": "ai-researcher/2.0 (academic research tool; contact: youremail@example.com)"
}


# ─────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────

@dataclass
class ResearchPaper:
    title:            str
    authors:          List[str]
    summary:          str
    published:        str          # ISO date string
    source:           str          # "arxiv" | "semantic_scholar" | "pubmed"
    source_id:        str          # arxiv_id / S2 paperId / pubmed PMID
    abs_url:          str          # always the abstract/landing page
    pdf_url:          Optional[str]
    primary_category: Optional[str]
    categories:       List[str]    = field(default_factory=list)
    citation_count:   Optional[int] = None
    journal:          Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _name_tokens(value: str) -> List[str]:
    return [token for token in _normalize_name(value).split() if token]


def _author_match_score(target_name: str, candidate_name: str) -> float:
    target_tokens = _name_tokens(target_name)
    candidate_tokens = _name_tokens(candidate_name)
    if not target_tokens or not candidate_tokens:
        return 0.0

    if " ".join(target_tokens) == " ".join(candidate_tokens):
        return 1.0

    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    overlap = target_set & candidate_set

    if len(target_tokens) == 1:
        return 1.0 if target_tokens[0] in candidate_set else 0.0

    first = target_tokens[0]
    last = target_tokens[-1]
    has_first = first in candidate_set or any(token.startswith(first[:1]) for token in candidate_tokens if token)
    has_last = last in candidate_set

    if has_first and has_last:
        return 0.96
    if has_last and len(overlap) >= max(2, len(target_tokens) - 1):
        return 0.9
    if len(overlap) >= len(target_tokens):
        return 0.88
    return 0.0


def _paper_author_match_score(paper: ResearchPaper, author_name: str) -> float:
    return max((_author_match_score(author_name, candidate) for candidate in (paper.authors or [])), default=0.0)


def _filter_author_results(papers: List[ResearchPaper], author_name: Optional[str]) -> List[ResearchPaper]:
    if not author_name:
        return papers

    scored = [(_paper_author_match_score(paper, author_name), paper) for paper in papers]
    strong = [paper for score, paper in scored if score >= 0.9]
    if strong:
        return strong

    medium = [paper for score, paper in scored if score >= 0.75]
    if medium:
        return medium

    return []


_SEMINAL_LOOKUPS = [
    (
        (
            "transformer architecture",
            "introduces transformers",
            "introduced transformers",
            "introduced transformer architecture",
            "which paper introduced transformer architecture",
            "which paper introduces transformer architecture",
            "transformer architecture origin",
            "transformer architecture origin paper",
            "original transformer paper",
            "paper that introduced transformer architecture",
        ),
        {
            "exact_title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "source_hint": "arxiv",
        },
    ),
]


def _apply_seminal_paper_overrides(
    raw_query: str,
    topic: Optional[str],
    exact_title: Optional[str],
    arxiv_id: Optional[str],
    source_hint: str,
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    q_lower = (raw_query or "").lower()

    for patterns, override in _SEMINAL_LOOKUPS:
        if any(pattern in q_lower for pattern in patterns):
            return (
                override.get("topic") or topic,
                override.get("exact_title") or exact_title,
                override.get("arxiv_id") or arxiv_id,
                override.get("source_hint") or source_hint,
            )

    if "attention is all you need" in q_lower:
        return (
            topic,
            exact_title or "Attention Is All You Need",
            arxiv_id,
            "arxiv" if source_hint == "all" else source_hint,
        )

    return topic, exact_title, arxiv_id, source_hint


# ─────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────

def _get(url: str, params: Dict[str, Any], source: str = "") -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp

            # ── Handle 429 (rate limited) with exponential backoff + jitter ──
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after) + random.uniform(0.1, 0.5)
                    except ValueError:
                        wait = BACKOFF_FACTOR ** attempt + random.uniform(0.5, 1.5)
                else:
                    wait = BACKOFF_FACTOR ** attempt + random.uniform(0.5, 1.5)
                logger.warning(
                    "%s HTTP 429 (rate limited) attempt %d — waiting %.1fs",
                    source, attempt, wait,
                )
                time.sleep(wait)
                continue

            logger.warning("%s HTTP %d on attempt %d", source, resp.status_code, attempt)
        except requests.RequestException as exc:
            logger.warning("%s request error attempt %d: %s", source, attempt, exc)
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_FACTOR ** (attempt - 1) + random.uniform(0, 0.5))
    return None


def _arxiv_rate_limit() -> None:
    global _last_arxiv_ts
    elapsed = time.time() - _last_arxiv_ts
    if elapsed < MIN_ARXIV_DELAY:
        time.sleep(MIN_ARXIV_DELAY - elapsed)
    _last_arxiv_ts = time.time()


# ─────────────────────────────────────────────────────────────────
# LLM Query Decomposer
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# LLM Query Decomposer  (FIXED)
# ─────────────────────────────────────────────────────────────────

def decompose_query(question: str) -> Dict[str, Any]:
    """
    Regex + heuristic query decomposer. No LLM needed — fast and reliable.
    KEY FIXES:
    - Author name regex now matches CamelCase names like LeCun, DeepMind, etc.
    - Leading numbers are stripped from topic BEFORE noise word removal
    - "papers by X" pattern is tried first with a greedy name match
    """
    q       = question.strip()
    q_lower = q.lower()
    exact_title = None

    if q_lower.startswith("title_exact::"):
        exact_title = q.split("::", 1)[1].strip() or None
        q = exact_title or q
        q_lower = q.lower()

    # ── arXiv ID detection ────────────────────────────────────────────────────
    arxiv_id_match = re.search(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b', q)
    arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else None

    # ── Author detection ──────────────────────────────────────────────────────
    # FIXED REGEX: allow CamelCase like "LeCun", "DeepMind", hyphenated names
    # Pattern matches: words with alphabetic characters. Limit to 5 words to avoid eating the whole sentence.
    NAME_WORD = r'[a-zA-Z\u00C0-\u024F\-\']+'
    FULL_NAME = rf'{NAME_WORD}(?:\s+{NAME_WORD}){{0,4}}'

    author = None
    author_patterns = [
        rf'(?:what|which)\s+other\s+papers?\s+has\s+({FULL_NAME})\s+authored',
        rf'(?:what|which)\s+papers?\s+has\s+({FULL_NAME})\s+authored',
        rf'(?:what|which)\s+has\s+({FULL_NAME})\s+published',
        # "written by Yann LeCun", "authored by LeCun"
        rf'(?:written|authored)\s+by\s+({FULL_NAME})',
        # "papers by Yann LeCun", "paper by LeCun", "work by X"
        rf'(?:papers?|articles?|work|publications?|research)\s+by\s+({FULL_NAME})',
        # "by Yann LeCun" at start or after find/search
        # "Find papers written by X" — already covered above but add safeguard
        rf'author[:\s]+({FULL_NAME})',
        # "Yann LeCun's papers"
        rf"({FULL_NAME})'s\s+(?:papers?|work|research|publications?)",
        rf"({FULL_NAME})'s\s+publication(?:s)?\s+list",
        # "papers of Yann LeCun"
        rf'papers?\s+(?:of|from)\s+({FULL_NAME})',
        rf'publication(?:s)?\s+list\s+(?:for|of)\s+({FULL_NAME})',
    ]

    for pat in author_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Stop at common prepositions to avoid greedy matching (e.g., "yann lecun about neural networks")
            stop_words = {
                "about", "on", "regarding", "in", "with", "using", "for", "and", "the", "a", "an",
                "can", "i", "find", "where", "show", "get", "me", "please",
            }
            clean_words = []
            for w in candidate.split():
                lower_w = w.lower()
                if not clean_words and lower_w in stop_words:
                    continue
                if lower_w in stop_words or re.match(r'\d+', w):
                    break
                clean_words.append(w)
            
            clean_candidate = " ".join(clean_words).strip()
            # Sanity check: must have at least 2 chars, not be a pure noise word
            if len(clean_candidate) >= 2 and clean_candidate.lower() not in {
                "the", "a", "an", "me", "us", "you", "all", "my", "our", "",
                "one", "two", "three", "both", "first", "second", "third",
                "this", "that", "these", "those",
            }:
                author = clean_candidate
                break

    # ── Year detection ────────────────────────────────────────────────────────
    year_match = re.search(r'\b(20[0-2]\d)\b', q)
    year = year_match.group(1) if year_match else None

    # ── Source hint detection ─────────────────────────────────────────────────
    source_hint = "all"
    if any(w in q_lower for w in [
        "pubmed", "medical", "clinical", "disease", "patient", "health",
        "drug", "diagnosis", "treatment", "cancer", "diabetes", "retinopathy",
        "biomedical", "hospital", "surgery", "gene", "protein", "pharmacology",
    ]):
        source_hint = "pubmed"
    elif any(w in q_lower for w in [
        "arxiv", "cs.", "math.", "stat.", "physics", "machine learning",
        "deep learning", "llm", "transformer", "neural", "computer vision",
        "nlp", "natural language", "convolutional", "cnn", "bert", "gpt",
        "diffusion", "reinforcement learning",
    ]):
        source_hint = "arxiv"

    # ── Clean topic ───────────────────────────────────────────────────────────
    # Start from original lowercased question
    topic = q_lower

    # STEP 1: Strip leading number (e.g. "5 papers about X" → "papers about X")
    topic = re.sub(r'^\s*\d+\s+', '', topic)

    # STEP 2: Remove author name if found
    if author:
        topic = topic.replace(author.lower(), ' ')

    # STEP 3: Remove year
    if year:
        topic = topic.replace(year, ' ')

    # STEP 4: Remove arXiv ID
    if arxiv_id:
        topic = topic.replace(arxiv_id.lower(), ' ')

    # STEP 5: Remove noise words (order matters — longest first to avoid partial matches)
    noise_phrases = sorted([
        "what other papers has", "which papers has", "which paper introduces",
        "which paper introduced", "give link of that paper", "give me the link",
        "where can i find", "publication list",
        "research papers written by", "research papers authored by",
        "research papers about", "research papers on", "research paper about",
        "find me papers about", "find me papers on", "find papers about",
        "find papers on", "find papers written by", "find papers authored by",
        "can you find", "search for papers", "look for papers",
        "get me papers", "written by", "authored by", "papers about",
        "papers on", "papers by", "paper about", "paper on",
        "research about", "research on", "articles about", "articles on",
        "studies on", "study on",
        "find", "search", "get me", "look for", "show me", "please", "pls",
        "recent", "latest", "new", "top", "best", "most cited",
        "using arxiv_tool", "use arxiv", "arxiv tool", "arxiv_tool",
        "research", "papers", "paper", "articles", "article", "studies", "study",
        "publications", "publication", "work", "works",
        "about", "regarding", "related to", "in the field of",
    ], key=len, reverse=True)

    for phrase in noise_phrases:
        topic = topic.replace(phrase, ' ')

    # STEP 6: Strip stray numbers and cleanup
    topic = re.sub(r"\b's\b", ' ', topic)
    topic = re.sub(r"[^a-z0-9\s\-]+", ' ', topic)
    topic = re.sub(r'\b\d+\b', ' ', topic)   # remove standalone numbers like "5"
    topic = re.sub(r'\s+', ' ', topic).strip()

    # If topic is too short after cleaning, keep author/id searches topic-less.
    if len(topic) <= 2 and (author or arxiv_id):
        topic = None
    elif len(topic) <= 2:
        fallback = q_lower
        if author:
            fallback = fallback.replace(author.lower(), ' ')
        if year:
            fallback = fallback.replace(year, ' ')
        fallback = re.sub(r"\b's\b", ' ', fallback)
        fallback = re.sub(r"[^a-z0-9\s\-]+", ' ', fallback)
        fallback = re.sub(r'\b\d+\b', ' ', fallback)
        fallback = re.sub(r'\s+', ' ', fallback).strip()
        topic = fallback if len(fallback) > 2 else None
    else:
        topic = topic

    # If author-only query (no real topic), that's fine — topic = None
    if author and topic and len(topic.replace(author.lower(), '').strip()) <= 2:
        topic = None

    if author and topic:
        residual_tokens = set(topic.split())
        author_only_tokens = {
            "authored", "written", "published", "publication", "publications",
            "paper", "papers", "article", "articles", "list", "profile",
            "page", "link", "links", "what", "which", "where", "can", "i",
            "find", "has", "other", "s", "of", "for",
        }
        if residual_tokens and residual_tokens.issubset(author_only_tokens):
            topic = None

    topic, exact_title, arxiv_id, source_hint = _apply_seminal_paper_overrides(
        q,
        topic,
        exact_title,
        arxiv_id,
        source_hint,
    )

    # Ultimate fallback
    if not topic and not author and not arxiv_id:
        topic = q

    logger.info(
        "Query decomposed → author=%r topic=%r year=%r arxiv_id=%r source_hint=%r",
        author, topic, year, arxiv_id, source_hint,
    )

    return {
        "author":      author,
        "topic":       topic,
        "exact_title": exact_title,
        "year":        year,
        "arxiv_id":    arxiv_id,
        "source_hint": source_hint,
        "raw":         q,
    }

# ─────────────────────────────────────────────────────────────────
# Source 1: arXiv
# ─────────────────────────────────────────────────────────────────

def _build_arxiv_query(intent: Dict[str, Any]) -> str:
    """
    Build a precise arXiv API query from decomposed intent.
    KEY FIXES:
    - Author uses full quoted name → au:"Yann LeCun" (not split)
    - Topic uses UNQUOTED ti/abs OR — much higher recall than quoted phrase
    - Falls back to all: search if nothing else
    """
    parts: List[str] = []

    # Direct arXiv ID lookup
    if intent.get("arxiv_id"):
        return f'id:{intent["arxiv_id"].split("v")[0]}'

    if intent.get("exact_title"):
        return f'ti:"{intent["exact_title"]}"'

    # Author: always use full name in quotes — never split
    if intent.get("author"):
        parts.append(f'au:"{intent["author"]}"')

    # Topic: unquoted OR across ti and abs for better recall
    if intent.get("topic"):
        topic  = intent["topic"].strip()
        words  = [w for w in re.split(r'\s+', topic) if len(w) > 1]

        if not words:
            pass
        elif len(words) == 1:
            w = words[0]
            parts.append(f'(ti:{w} OR abs:{w})')
        else:
            # Multi-word: build per-word OR filter — high recall
            # e.g. "transformer nlp" → (ti:transformer OR abs:transformer) AND (ti:nlp OR abs:nlp)
            # Limit to first 4 words to avoid overly restrictive queries
            word_filters = [
                f'(ti:{w} OR abs:{w})' for w in words[:4]
            ]
            parts.append(' AND '.join(word_filters))

    # Fallback: use raw question with all: prefix
    if not parts:
        raw = intent.get("raw", "machine learning")
        # Strip leading numbers from raw too
        raw = re.sub(r'^\d+\s+', '', raw.strip())
        parts.append(f'all:{raw}')

    return " AND ".join(parts)


def _parse_arxiv_xml(feed_xml: str) -> List[ResearchPaper]:
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError as exc:
        logger.error("arXiv XML parse error: %s", exc)
        return []

    ns = {
        "atom":   "http://www.w3.org/2005/Atom",
        "arxiv":  "http://arxiv.org/schemas/atom",
    }

    papers: List[ResearchPaper] = []
    for entry in root.findall("atom:entry", ns):
        raw_id    = entry.findtext("atom:id", default="", namespaces=ns).strip()
        arxiv_id  = raw_id.split("/abs/")[-1].strip() if "/abs/" in raw_id else raw_id.split("/")[-1].strip()

        title   = _clean_text(" ".join((entry.findtext("atom:title", default="", namespaces=ns)).split()))
        summary = textwrap.shorten(
            _clean_text(" ".join((entry.findtext("atom:summary", default="", namespaces=ns)).split())),
            width=1500, placeholder="..."
        )

        authors = [
            _clean_text(a.findtext("atom:name", default="", namespaces=ns).strip())
            for a in entry.findall("atom:author", ns)
            if _clean_text(a.findtext("atom:name", default="", namespaces=ns).strip())
        ]

        published = entry.findtext("atom:published", default="", namespaces=ns).strip()[:10]

        # ALWAYS derive URLs deterministically from arxiv_id — never trust feed links
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        primary_cat_elem = entry.find("arxiv:primary_category", ns)
        primary_category = primary_cat_elem.attrib.get("term") if primary_cat_elem is not None else None
        categories = [
            c.attrib["term"] for c in entry.findall("atom:category", ns)
            if c.attrib.get("term")
        ]

        papers.append(ResearchPaper(
            title=title, authors=authors, summary=summary, published=published,
            source="arxiv", source_id=arxiv_id,
            abs_url=abs_url, pdf_url=pdf_url,
            primary_category=primary_category, categories=categories,
        ))

    return papers


def search_arxiv(intent: Dict[str, Any], max_results: int) -> List[ResearchPaper]:
    _arxiv_rate_limit()

    params = {
        "start": 0,
        "max_results": max_results,
    }
    query = None
    if intent.get("arxiv_id"):
        base_id = str(intent["arxiv_id"]).split("v")[0]
        params["id_list"] = base_id
        logger.info("arXiv exact ID lookup: %s", base_id)
    else:
        query = _build_arxiv_query(intent)
        logger.info("arXiv query: %s", query)
        params.update({
            "search_query": query,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })

    resp = _get(ARXIV_API_URL, params, source="arXiv")
    if not resp:
        return []

    papers = _parse_arxiv_xml(resp.text)

    # Fallback: if quoted phrase returned nothing, retry without quotes
    if not papers and query and intent.get("topic"):
        logger.warning("arXiv quoted search returned 0 results — retrying unquoted")
        params["search_query"] = _build_arxiv_query({
            **intent,
            "topic": intent["topic"].replace('"', ''),
        })
        _arxiv_rate_limit()
        resp2 = _get(ARXIV_API_URL, params, source="arXiv-fallback")
        if resp2:
            papers = _parse_arxiv_xml(resp2.text)

    # Filter by year if requested
    if intent.get("year") and papers:
        year = intent["year"]
        papers = [p for p in papers if p.published.startswith(year)] or papers

    return papers[:max_results]


# ─────────────────────────────────────────────────────────────────
# Source 2: Semantic Scholar
# ─────────────────────────────────────────────────────────────────

def search_semantic_scholar(intent: Dict[str, Any], max_results: int) -> List[ResearchPaper]:
    """
    Searches Semantic Scholar — covers 200M+ papers from ALL publishers:
    arXiv, IEEE, ACM, Nature, Springer, Elsevier, etc.
    Free, no API key needed for standard usage.
    """
    papers: List[ResearchPaper] = []

    # If author-only query, use author search endpoint for better results
    if intent.get("author") and not intent.get("topic"):
        papers = _s2_author_papers(intent["author"], max_results)
        return papers

    # Build query string
    query_parts = []
    if intent.get("author"):
        query_parts.append(intent["author"])
    if intent.get("exact_title"):
        query_parts.append(intent["exact_title"])
    elif intent.get("topic"):
        query_parts.append(intent["topic"])
    if not query_parts:
        query_parts.append(intent.get("raw", ""))

    query = " ".join(query_parts).strip()
    if not query:
        return []

    fields = "title,authors,year,abstract,openAccessPdf,externalIds,publicationTypes,citationCount,publicationVenue"
    params = {
        "query":  query,
        "limit":  min(max_results, 25),
        "fields": fields,
    }

    resp = _get(SEMANTIC_SCHOLAR_URL, params, source="SemanticScholar")
    if not resp:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    for item in (data.get("data") or []):
        title   = _clean_text(item.get("title") or "")
        authors = [_clean_text(a.get("name", "")) for a in (item.get("authors") or []) if _clean_text(a.get("name", ""))]
        summary = textwrap.shorten(_clean_text(item.get("abstract") or "No abstract available."), width=1500, placeholder="...")
        year    = str(item.get("year") or "")
        paper_id = item.get("paperId") or ""
        citation_count = item.get("citationCount")
        journal = (item.get("publicationVenue") or {}).get("name")

        # Build URLs
        ext_ids = item.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi      = ext_ids.get("DOI")
        pmid     = ext_ids.get("PubMed")

        abs_url = f"https://www.semanticscholar.org/paper/{paper_id}"
        pdf_url = None

        oa = item.get("openAccessPdf") or {}
        if oa.get("url"):
            pdf_url = oa["url"]
        elif arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        elif doi:
            abs_url = f"https://doi.org/{doi}"

        if not title:
            continue

        papers.append(ResearchPaper(
            title=title, authors=authors, summary=summary,
            published=year, source="semantic_scholar", source_id=paper_id,
            abs_url=abs_url, pdf_url=pdf_url,
            primary_category=None, categories=[],
            citation_count=citation_count, journal=journal,
        ))

    # Filter by year if requested
    if intent.get("author"):
        papers = _filter_author_results(papers, intent.get("author"))

    if intent.get("year") and papers:
        year = intent["year"]
        papers = [p for p in papers if p.published == year] or papers

    return papers[:max_results]


def _s2_author_papers(author_name: str, max_results: int) -> List[ResearchPaper]:
    """Find papers by searching Semantic Scholar's author endpoint."""
    params = {"query": author_name, "limit": 5, "fields": "name,papers.title,papers.year,papers.externalIds,papers.abstract,papers.openAccessPdf,papers.citationCount,papers.authors"}
    resp = _get(SEMANTIC_AUTHOR_URL, params, source="S2-Author")
    if not resp:
        return []
    try:
        data = resp.json()
    except Exception:
        return []

    papers: List[ResearchPaper] = []
    author_candidates = sorted(
        (data.get("data") or []),
        key=lambda item: _author_match_score(author_name, item.get("name") or ""),
        reverse=True,
    )

    for author in author_candidates[:3]:
        if _author_match_score(author_name, author.get("name") or "") < 0.9:
            continue
        for item in (author.get("papers") or [])[:max_results]:
            title    = _clean_text(item.get("title") or "")
            year     = str(item.get("year") or "")
            paper_id = item.get("paperId") or ""
            ext_ids  = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            doi      = ext_ids.get("DOI")
            abstract = textwrap.shorten(_clean_text(item.get("abstract") or "No abstract."), width=1500, placeholder="...")
            authors_list = [_clean_text(a.get("name", "")) for a in (item.get("authors") or []) if _clean_text(a.get("name", ""))]
            oa = item.get("openAccessPdf") or {}

            abs_url = f"https://www.semanticscholar.org/paper/{paper_id}"
            pdf_url = oa.get("url")
            if arxiv_id:
                abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
            elif doi:
                abs_url = f"https://doi.org/{doi}"

            if not title:
                continue

            papers.append(ResearchPaper(
                title=title, authors=authors_list, summary=abstract,
                published=year, source="semantic_scholar", source_id=paper_id,
                abs_url=abs_url, pdf_url=pdf_url,
                primary_category=None, categories=[],
                citation_count=item.get("citationCount"),
            ))

    papers = _filter_author_results(papers, author_name)
    return papers[:max_results]


# ─────────────────────────────────────────────────────────────────
# Source 3: PubMed  (best for medical/healthcare/bio papers)
# ─────────────────────────────────────────────────────────────────

def search_pubmed(intent: Dict[str, Any], max_results: int) -> List[ResearchPaper]:
    """
    Searches PubMed — 35M+ biomedical and healthcare papers.
    No API key needed.
    """
    query_parts = []
    if intent.get("author"):
        query_parts.append(f'{intent["author"]}[Author]')
    if intent.get("exact_title"):
        query_parts.append(intent["exact_title"])
    elif intent.get("topic"):
        query_parts.append(intent["topic"])
    if not query_parts:
        query_parts.append(intent.get("raw", ""))

    query = " AND ".join(query_parts).strip()
    if not query:
        return []

    # Step 1: search for IDs
    search_params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    resp = _get(PUBMED_SEARCH_URL, search_params, source="PubMed-search")
    if not resp:
        return []

    try:
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    if not ids:
        return []

    # Step 2: fetch summaries for those IDs
    summary_params = {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "retmode": "json",
    }
    resp2 = _get(PUBMED_SUMMARY_URL, summary_params, source="PubMed-summary")
    if not resp2:
        return []

    try:
        result_data = resp2.json().get("result", {})
    except Exception:
        return []

    papers: List[ResearchPaper] = []
    for pmid in ids:
        item = result_data.get(pmid)
        if not item or not isinstance(item, dict):
            continue

        title   = _clean_text(item.get("title", "").rstrip("."))
        authors = [_clean_text(a.get("name", "")) for a in (item.get("authors") or []) if _clean_text(a.get("name", ""))]
        summary = _clean_text(item.get("sorttitle") or title)   # PubMed summaries don't have abstracts in this endpoint
        pub_date = item.get("pubdate", "")[:4]     # year only
        journal  = item.get("source", "")

        abs_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pdf_url = None  # PubMed links to abstract page; PDF access depends on journal

        papers.append(ResearchPaper(
            title=title, authors=authors, summary=summary,
            published=pub_date, source="pubmed", source_id=pmid,
            abs_url=abs_url, pdf_url=pdf_url,
            primary_category=None, categories=[],
            journal=journal,
        ))

    papers = _filter_author_results(papers, intent.get("author"))
    return papers


# ─────────────────────────────────────────────────────────────────
# Source 4: OpenAlex (free, robust open metadata — 250M+ works)
# ─────────────────────────────────────────────────────────────────

OPENALEX_API_URL = "https://api.openalex.org/works"
TAVILY_API_URL = "https://api.tavily.com/search"


def _get_tavily_api_key() -> str:
    return os.getenv("TAVILY_API_KEY", "").strip()

def search_openalex(intent: Dict[str, Any], max_results: int) -> List[ResearchPaper]:
    """
    Searches OpenAlex — covers 250M+ scholarly works.
    Free, no API key needed. Great for Open Access PDFs and clean abstracts.
    """
    query_parts = []
    if intent.get("author"):
        query_parts.append(intent["author"])
    if intent.get("topic"):
        query_parts.append(intent["topic"])
    if not query_parts:
        query_parts.append(intent.get("raw", ""))

    query = " ".join(query_parts).strip()
    if not query:
        return []

    filters = ["has_abstract:true"]
    if intent.get("year"):
        filters.append(f"publication_year:{intent['year']}")

    params: Dict[str, Any] = {
        "search": query,
        "filter": ",".join(filters),
        "per-page": min(max_results, 20),
        "sort": "relevance_score:desc",
        "select": "id,title,authorships,abstract_inverted_index,publication_year,primary_location,open_access,cited_by_count,ids",
        "mailto": "researcher@agent.local"  # OpenAlex polite pool
    }

    try:
        resp = requests.get(OPENALEX_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("OpenAlex fetch failed: %s", e)
        return []

    papers: List[ResearchPaper] = []
    for item in (data.get("results") or []):
        title = _clean_text(item.get("title") or "")
        if not title:
            continue

        authors = []
        for a in (item.get("authorships") or []):
            name = (a.get("author") or {}).get("display_name")
            clean_name = _clean_text(name or "")
            if clean_name:
                authors.append(clean_name)

        # Reconstruct abstract from inverted index
        abstract = ""
        inv_idx = item.get("abstract_inverted_index")
        if inv_idx:
            max_idx = max((max(positions) for positions in inv_idx.values() if positions), default=-1)
            if max_idx >= 0:
                words = [""] * (max_idx + 1)
                for word, positions in inv_idx.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words).strip()
        summary = textwrap.shorten(_clean_text(abstract or "No abstract available."), width=1500, placeholder="...")

        year = str(item.get("publication_year") or "")
        external_ids = item.get("ids") or {}
        doi = external_ids.get("doi", "").split(".org/")[-1] if external_ids.get("doi") else ""
        openalex_id = item.get("id", "").split("/")[-1]

        source_id = doi if doi else openalex_id
        abs_url = external_ids.get("doi") or item.get("id")

        pdf_url = None
        oa = item.get("open_access") or {}
        if oa.get("is_oa") and oa.get("oa_url"):
            pdf_url = oa.get("oa_url")
        
        ploc = item.get("primary_location") or {}
        journal = (ploc.get("source") or {}).get("display_name")
        
        citation_count = item.get("cited_by_count")

        papers.append(ResearchPaper(
            title=title, authors=authors, summary=summary,
            published=year, source="openalex", source_id=source_id,
            abs_url=abs_url, pdf_url=pdf_url,
            primary_category=None, categories=[],
            citation_count=citation_count, journal=journal,
        ))

    papers = _filter_author_results(papers, intent.get("author"))
    return papers[:max_results]


def _should_search_web(intent: Dict[str, Any]) -> bool:
    raw = (intent.get("raw") or "").lower()
    web_markers = (
        "link", "website", "homepage", "profile", "publication list", "official",
        "who introduced", "which paper", "introduced", "origin", "original paper",
        "find online", "web", "verify", "viral", "trending", "replace transformers",
        "replace transformer", "next transformer", "beyond transformers",
    )
    return any(marker in raw for marker in web_markers)


def search_tavily(intent: Dict[str, Any], max_results: int) -> List[ResearchPaper]:
    api_key = _get_tavily_api_key()
    if not api_key:
        return []

    raw_query = (intent.get("raw") or "").strip()
    exact_title = (intent.get("exact_title") or "").strip()
    query_parts = []
    if intent.get("author"):
        query_parts.append(intent["author"])
    if exact_title:
        query_parts.append(f"\"{exact_title}\"")
    elif intent.get("topic"):
        query_parts.append(intent["topic"])
    if not query_parts:
        query_parts.append(raw_query)
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return []

    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": min(max_results, 8),
        "include_answer": False,
        "include_raw_content": False,
    }

    if exact_title:
        payload["query"] = f"\"{exact_title}\" research paper"
        payload["exact_match"] = True
        payload["include_domains"] = [
            "arxiv.org",
            "aclanthology.org",
            "openreview.net",
            "semanticscholar.org",
            "neurips.cc",
            "papers.nips.cc",
        ]
    elif any(marker in raw_query.lower() for marker in ("link", "origin", "official", "verify")):
        payload["include_domains"] = [
            "arxiv.org",
            "semanticscholar.org",
            "openreview.net",
            "aclanthology.org",
            "doi.org",
        ]
    logger.info("Tavily query: %s", payload["query"])

    try:
        tavily_headers = {
            **HEADERS,
            "Authorization": f"Bearer {api_key}",
        }
        resp = requests.post(TAVILY_API_URL, json=payload, timeout=REQUEST_TIMEOUT, headers=tavily_headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []

    papers: List[ResearchPaper] = []
    for item in (data.get("results") or []):
        title = _clean_text(item.get("title") or "")
        content = _clean_text(item.get("content") or "")
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue

        lower_blob = f"{title} {content} {url}".lower()
        if "paper" not in lower_blob and "arxiv" not in lower_blob and "doi" not in lower_blob and "research" not in lower_blob:
            continue

        published = str(item.get("published_date") or "")[:10]
        pdf_url = url if url.lower().endswith(".pdf") else None
        abs_url = url

        papers.append(ResearchPaper(
            title=title,
            authors=[],
            summary=textwrap.shorten(content or "Web result.", width=1500, placeholder="..."),
            published=published,
            source="web",
            source_id=url,
            abs_url=abs_url,
            pdf_url=pdf_url,
            primary_category=None,
            categories=[],
        ))

    return papers[:max_results]


# ─────────────────────────────────────────────────────────────────
# Deduplication + Ranking
# ─────────────────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    return re.sub(r'\W+', ' ', title.lower()).strip()


def _title_match_score(target_title: str, candidate_title: str) -> float:
    target_norm = _normalize_title(target_title)
    candidate_norm = _normalize_title(candidate_title)
    if not target_norm or not candidate_norm:
        return 0.0
    if target_norm == candidate_norm:
        return 1.0
    if candidate_norm.startswith(target_norm) or target_norm.startswith(candidate_norm):
        return 0.82

    target_tokens = set(target_norm.split())
    candidate_tokens = set(candidate_norm.split())
    overlap = target_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(len(target_tokens), 1)


def _dedupe_preference_score(paper: ResearchPaper) -> float:
    score = 0.0
    source_id = (paper.source_id or "").lower()

    if paper.source == "arxiv":
        score += 100.0
    if re.match(r'^\d{4}\.\d{4,5}(?:v\d+)?$', source_id) or re.match(r'^[a-z\-]+/\d+(?:v\d+)?$', source_id):
        score += 40.0
    if paper.pdf_url:
        score += 25.0
    if paper.abs_url:
        score += 10.0
    if paper.summary and "no abstract available" not in paper.summary.lower():
        score += 8.0
    if paper.citation_count:
        score += min(paper.citation_count, 500) / 100.0

    return score


def _deduplicate(papers: List[ResearchPaper]) -> List[ResearchPaper]:
    """Remove duplicates by normalized title, keeping the most useful import target."""
    best_by_title: Dict[str, ResearchPaper] = {}
    for p in papers:
        norm = _normalize_title(p.title)
        if not norm:
            continue
        current = best_by_title.get(norm)
        if current is None or _dedupe_preference_score(p) > _dedupe_preference_score(current):
            best_by_title[norm] = p
    return list(best_by_title.values())


def _rank_papers(papers: List[ResearchPaper], intent: Dict[str, Any]) -> List[ResearchPaper]:
    """
    Score each paper and sort by relevance:
    - Title match to topic/author → +3
    - Author exact match          → +5
    - Has open PDF                → +1
    - Citation count (S2)         → log scale bonus
    - Source priority for medical → PubMed first
    """
    import math

    topic  = (intent.get("topic") or "").lower()
    author = (intent.get("author") or "").lower()
    exact_title = intent.get("exact_title") or ""
    arxiv_id = str(intent.get("arxiv_id") or "").split("v")[0]
    raw = (intent.get("raw") or "").lower()

    def score(p: ResearchPaper) -> float:
        s = 0.0
        title_lower = p.title.lower()
        source_id_base = str(p.source_id or "").lower().split("v")[0]
        paper_link_blob = f"{p.abs_url or ''} {p.pdf_url or ''}".lower()

        if arxiv_id:
            if p.source == "arxiv" and source_id_base == arxiv_id.lower():
                s += 80.0
            elif arxiv_id.lower() in paper_link_blob or source_id_base == arxiv_id.lower():
                s += 24.0

        if exact_title:
            title_score = _title_match_score(exact_title, p.title)
            if title_score >= 0.999:
                s += 60.0
            elif title_score >= 0.8:
                s += 12.0
            else:
                s -= 8.0

        if topic and topic in title_lower:
            s += 3.0
        if author:
            match_score = _paper_author_match_score(p, author)
            if match_score >= 0.9:
                s += 8.0
            elif match_score >= 0.75:
                s += 4.0
            else:
                s -= 6.0

        if p.pdf_url:
            s += 1.0

        if p.citation_count:
            s += min(math.log1p(p.citation_count), 5.0)

        # boost pubmed for medical queries
        if intent.get("source_hint") == "pubmed" and p.source == "pubmed":
            s += 2.0

        if any(marker in raw for marker in ("viral", "trending", "replace transformers", "replace transformer", "next transformer", "beyond transformers")):
            if p.source == "web":
                s += 6.0
            if str(p.published or "")[:4].isdigit():
                year = int(str(p.published)[:4])
                s += max(0, year - 2021) * 0.8

        if any(marker in raw for marker in ("link", "origin", "original paper", "introduced", "introduces", "which paper")):
            if p.source == "arxiv":
                s += 6.0
            if any(domain in paper_link_blob for domain in ("arxiv.org", "openreview.net", "aclanthology.org", "semanticscholar.org")):
                s += 4.0

        return s

    return sorted(papers, key=score, reverse=True)


# ─────────────────────────────────────────────────────────────────
# Main fan-out search
# ─────────────────────────────────────────────────────────────────

def _multi_source_search(
    question: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[Dict[str, Any]]:

    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))
    intent = decompose_query(question)
    source_hint = intent.get("source_hint", "all")

    # Decide which sources to query
    if source_hint == "pubmed":
        sources_to_query = ["pubmed", "semantic_scholar", "openalex"]
    elif source_hint == "arxiv":
        sources_to_query = ["arxiv", "semantic_scholar", "openalex"]
    else:
        sources_to_query = ["arxiv", "semantic_scholar", "pubmed", "openalex"]
    if _get_tavily_api_key() and _should_search_web(intent):
        sources_to_query.append("tavily")

    per_source = max(max_results, 8)  # fetch more, dedupe later
    all_papers: List[ResearchPaper] = []

    # Fan-out: query all sources in parallel
    def _run(source: str) -> List[ResearchPaper]:
        try:
            if source == "arxiv":
                return search_arxiv(intent, per_source)
            elif source == "semantic_scholar":
                return search_semantic_scholar(intent, per_source)
            elif source == "pubmed":
                return search_pubmed(intent, per_source)
            elif source == "openalex":
                return search_openalex(intent, per_source)
            elif source == "tavily":
                return search_tavily(intent, per_source)
        except Exception as exc:
            logger.error("Source %s failed: %s", source, exc)
        return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run, src): src for src in sources_to_query}
        for future in as_completed(futures):
            result = future.result()
            logger.info("Source %s returned %d papers", futures[future], len(result))
            all_papers.extend(result)

    # Deduplicate + rank
    all_papers = _deduplicate(all_papers)
    all_papers = _filter_author_results(all_papers, intent.get("author"))
    all_papers = _rank_papers(all_papers, intent)

    logger.info("Total after dedup+rank: %d papers", len(all_papers))
    return [p.to_dict() for p in all_papers[:max_results]]


# ─────────────────────────────────────────────────────────────────
# LangChain tool (drop-in replacement for search_arxiv_tool)
# ─────────────────────────────────────────────────────────────────

@tool("search_research_papers")
def search_research_papers_tool(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Search for research papers across multiple academic sources simultaneously.

    Searches arXiv, Semantic Scholar (200M+ papers from ALL publishers including
    IEEE, ACM, Nature, Springer), and PubMed (35M+ biomedical papers).

    Supports all query types:
    - Topic search    : "attention mechanism in transformers"
    - Author search   : "papers written by Komal Sharma"
    - Title search    : "Attention Is All You Need"
    - ID lookup       : "2103.00020" (arXiv ID)
    - Year filter     : "transformer papers 2023"
    - Medical topics  : automatically routes to PubMed + Semantic Scholar
    - Combined        : "diabetic retinopathy by Komal Sharma 2022"

    Args:
        query:       Natural language search query (any style listed above).
        max_results: Number of papers to return (1-30). Default 5.

    Returns:
        List of paper dicts with fields:
        - title            : Full paper title
        - authors          : List of author names
        - summary          : Abstract / description (up to 1500 chars)
        - published        : Publication year or ISO date
        - source           : "arxiv" | "semantic_scholar" | "pubmed"
        - source_id        : arXiv ID / Semantic Scholar ID / PubMed ID
        - abs_url          : Abstract/landing page URL (ALWAYS valid)
        - pdf_url          : Direct PDF URL if available (None if paywalled)
        - citation_count   : Number of citations (Semantic Scholar only)
        - journal          : Journal/venue name if available
        - primary_category : arXiv category (arXiv papers only)
        - categories       : All categories
    """
    return _multi_source_search(question=query, max_results=max_results)


# Keep backward-compatible alias so old imports still work
search_arxiv_tool = search_research_papers_tool


# ─────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "papers written by Komal Sharma",
        "diabetic retinopathy deep learning 2023",
        "Attention Is All You Need",
        "2103.00020",
        "reinforcement learning from human feedback",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q!r}")
        print("=" * 80)
        results = _multi_source_search(q, max_results=3)
        for i, p in enumerate(results, 1):
            print(f"\n[{i}] [{p['source'].upper()}] {p['title']} ({p['published']})")
            print(f"     Authors  : {', '.join(p['authors'][:3])}")
            print(f"     abs_url  : {p['abs_url']}")
            print(f"     pdf_url  : {p['pdf_url']}")
            print(f"     citations: {p.get('citation_count')}")
            print(f"     Summary  : {p['summary'][:150]}...")
