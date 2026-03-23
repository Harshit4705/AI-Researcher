from __future__ import annotations

import os
import re
import logging
from typing import List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from vector_tools import (
    query_project_papers,
    query_specific_paper,
    list_project_papers,
    rerank_chunks,
    load_all_paper_metadata,
    delete_paper_metadata,
)
from research_tool import decompose_query, search_arxiv_tool

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def get_llm(intent: str = "chat", max_tokens: int = 2048) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_tokens=max_tokens)
    # CRITICAL: Prevent model from ever emitting tool call JSON.
    # Without this, the model spontaneously calls "arxiv.run" from training memory
    # when users mention "arxiv tool", causing Groq 400: tool_use_failed.
    return llm.bind(tool_choice="none")

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    project_id: str
    question: str
    paper_registry: List[Dict[str, Any]]
    intent: str
    target_paper_ids: List[str]
    expanded_queries: List[str]
    local_hits: List[Dict[str, Any]]
    arxiv_recommendations: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    suggestions: List[str]
    chat_history: List[Dict[str, str]]
    comparison_clarification: bool

# ─────────────────────────────────────────────
# Node 1: Build paper registry
# ─────────────────────────────────────────────

def build_paper_registry(state: ResearchState) -> ResearchState:
    project_id = state["project_id"]

    try:
        faiss_ids = set(list_project_papers.invoke({"project_id": project_id}))
    except Exception as exc:
        logger.warning("Could not list FAISS papers: %s", exc)
        faiss_ids = set()

    sidecar = load_all_paper_metadata(project_id)

    registry = []
    for pid in faiss_ids:
        if pid in sidecar:
            registry.append(sidecar[pid])
        else:
            try:
                hits = query_specific_paper.invoke({
                    "project_id": project_id,
                    "paper_id": pid,
                    "query": "title abstract introduction",
                    "top_k": 1,
                })
                md = (hits[0].get("metadata") or {}) if hits else {}
                registry.append({
                    "paper_id": pid,
                    "title": md.get("title") or pid,
                    "arxiv_id": md.get("arxiv_id"),
                    "pdf_url": md.get("pdf_url"),
                    "abs_url": md.get("abs_url"),
                    "source": md.get("source"),
                    "is_metadata_only": bool(md.get("is_metadata_only")),
                    "chunk_count": int(md.get("chunk_count") or (1 if hits else 0)),
                })
            except Exception:
                registry.append({
                    "paper_id": pid,
                    "title": pid,
                    "arxiv_id": None,
                    "pdf_url": None,
                    "abs_url": None,
                    "source": None,
                    "is_metadata_only": False,
                    "chunk_count": 0,
                })

    for stale_id in set(sidecar.keys()) - faiss_ids:
        logger.info("Removing stale sidecar entry: %s", stale_id)
        delete_paper_metadata(project_id, stale_id)

    state["paper_registry"] = registry
    logger.info("Registry: %s", [r["paper_id"] for r in registry])
    return state


def _paper_has_full_text(paper: Dict[str, Any]) -> bool:
    if paper.get("is_metadata_only"):
        return False
    chunk_count = int(paper.get("chunk_count") or 0)
    if chunk_count > 1:
        return True
    source = (paper.get("source") or "").lower()
    return source == "pdf_upload"


def _should_target_all_local_papers(question: str, registry: List[Dict[str, Any]]) -> bool:
    q_lower = question.lower()
    if not registry:
        return False
    summary_markers = ("summarize", "summary", "summaries", "explain", "overview")
    scope_markers = ("both paper", "both papers", "all papers", "all the papers", "one by one", "each paper")
    return any(marker in q_lower for marker in summary_markers) and any(marker in q_lower for marker in scope_markers)


def _resolve_compare_targets(question: str, registry: List[Dict[str, Any]], explicit_ids: List[str]) -> tuple[List[str], bool]:
    if explicit_ids:
        return explicit_ids, False

    if len(registry) < 2:
        return [], False

    q_lower = question.lower()
    full_text_ids = [paper["paper_id"] for paper in registry if _paper_has_full_text(paper)]

    if len(registry) == 2:
        return [paper["paper_id"] for paper in registry], False

    if len(full_text_ids) == 2 and any(token in q_lower for token in ("both", "two", "compare", "difference")):
        return full_text_ids, False

    if any(token in q_lower for token in ("all papers", "all the papers", "all of them")):
        preferred = full_text_ids or [paper["paper_id"] for paper in registry]
        return preferred[:4], False

    return [], True

# ─────────────────────────────────────────────
# Node 2: Classify intent + detect target paper
# ─────────────────────────────────────────────

def classify_intent(state: ResearchState) -> ResearchState:
    question = state["question"]
    q_lower = question.lower()
    registry = state.get("paper_registry") or []
    state["comparison_clarification"] = False

    inventory_kws = [
        "what papers do i have", "which papers do i have", "list my papers",
        "show my papers", "papers in my library", "papers in this project",
        "papers in my project", "what have i uploaded", "what have i imported",
        "what is in my library", "what's in my library", "show uploaded papers",
        "show imported papers", "list uploaded papers", "list imported papers",
    ]
    compare_kws = [
        "compare", "difference between", "contrast", "versus", "vs",
        "both papers", "between the two", "better paper",
        "written better", "writing quality",
    ]
    similar_kws = [
        "similar", "recommend", "related paper", "find more", "find papers",
        "search for", "like this", "other paper", "more papers", "suggest",
        "attention mechanism", "find me", "look for", "papers about", "papers on",
        "can you find", "find the paper", "find paper", "get the paper",
        "get me the", "show me the", "fetch the paper", "where is the paper",
        "introduction paper", "what paper", "which paper introduced", "which paper introduces",
        "publication list", "publications by", "papers by", "authored by",
        "what other papers", "other papers has", "author profile",
    ]

    parsed_query = decompose_query(question)

    if any(k in q_lower for k in compare_kws):
        intent = "compare"
    elif any(k in q_lower for k in inventory_kws):
        intent = "list_library"
    elif any(k in q_lower for k in similar_kws):
        intent = "search_similar"
    elif _should_attempt_external_search(question, parsed_query, registry):
        intent = "search_similar"
    elif parsed_query.get("author") or parsed_query.get("arxiv_id"):
        intent = "search_similar"
    else:
        intent = "chat"

    state["intent"] = intent
    logger.info("Intent: %s", intent)

    target_ids: List[str] = []
    for paper in registry:
        pid = (paper.get("paper_id") or "").lower()
        title = (paper.get("title") or "").lower()
        if pid and pid in q_lower:
            target_ids.append(paper["paper_id"])
        else:
            title_words = [w for w in re.split(r"\W+", title) if len(w) > 4]
            if title_words and sum(1 for w in title_words if w in q_lower) >= 2:
                target_ids.append(paper["paper_id"])

    if intent == "compare" and not target_ids and len(registry) >= 2:
        target_ids, needs_clarification = _resolve_compare_targets(question, registry, target_ids)
        state["comparison_clarification"] = needs_clarification
    elif intent != "compare" and not target_ids and _should_target_all_local_papers(question, registry):
        target_ids = [paper["paper_id"] for paper in registry if _paper_has_full_text(paper)] or [paper["paper_id"] for paper in registry]

    state["target_paper_ids"] = list(dict.fromkeys(target_ids))
    logger.info("Target paper IDs: %s", state["target_paper_ids"])
    return state

# ─────────────────────────────────────────────
# Node 3: Query expansion
# ─────────────────────────────────────────────

def expand_query(state: ResearchState) -> ResearchState:
    question = state["question"]
    intent = state.get("intent", "chat")

    if intent in {"search_similar", "list_library"}:
        state["expanded_queries"] = [question]
        return state

    try:
        llm = get_llm("chat", max_tokens=150)
        msg = (
            "Generate 2 alternative phrasings of this question to improve document search recall.\n"
            "Return ONLY the 2 queries, one per line. No numbering, no explanation.\n\n"
            "Original question: " + question
        )
        result = llm.invoke([HumanMessage(content=msg)]).content
        alternatives = [q.strip() for q in result.strip().split("\n") if q.strip()][:2]
        state["expanded_queries"] = [question] + alternatives
        logger.info("Expanded queries: %s", state["expanded_queries"])
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        state["expanded_queries"] = [question]

    return state

# ─────────────────────────────────────────────
# Node 4: Retrieve local chunks
# ─────────────────────────────────────────────

def retrieve_local(state: ResearchState) -> ResearchState:
    project_id = state["project_id"]
    question = state["question"]
    expanded_queries = state.get("expanded_queries") or [question]
    intent = state.get("intent", "chat")
    target_ids = state.get("target_paper_ids") or []
    registry = state.get("paper_registry") or []
    all_hits: List[Dict[str, Any]] = []
    seen_contents: set = set()

    if intent == "compare" and state.get("comparison_clarification"):
        state["local_hits"] = []
        return state

    def _dedupe(hits: List[Dict]) -> List[Dict]:
        result = []
        for h in hits:
            content = (h.get("content") or "").strip()[:100]
            if content not in seen_contents:
                seen_contents.add(content)
                result.append(h)
        return result

    try:
        if intent == "compare":
            papers_to_query = target_ids if target_ids else [r["paper_id"] for r in registry]
            per_paper = max(5, 12 // max(len(papers_to_query), 1))
            for pid in papers_to_query:
                for query in expanded_queries:
                    hits = query_specific_paper.invoke({
                        "project_id": project_id,
                        "paper_id": pid,
                        "query": query,
                        "top_k": per_paper,
                    })
                    all_hits.extend(_dedupe(hits))
        elif target_ids:
            for pid in target_ids:
                for query in expanded_queries:
                    hits = query_specific_paper.invoke({
                        "project_id": project_id,
                        "paper_id": pid,
                        "query": query,
                        "top_k": 8,
                    })
                    all_hits.extend(_dedupe(hits))
        else:
            for query in expanded_queries:
                hits = query_project_papers.invoke({
                    "project_id": project_id,
                    "query": query,
                    "top_k": 12,
                })
                all_hits.extend(_dedupe(hits))
    except Exception as exc:
        logger.warning("retrieve_local failed: %s", exc)
        all_hits = []

    if all_hits:
        all_hits = rerank_chunks(question, all_hits, top_k=15)

    state["local_hits"] = all_hits
    logger.info("Retrieved %d chunks after reranking", len(all_hits))
    return state

# ─────────────────────────────────────────────
# Node 5: Search similar on arXiv
# ─────────────────────────────────────────────

# Phrases indicating the user is referring to their own uploaded/library papers
_SELF_REF_PATTERNS = [
    "the paper i uploaded", "paper i uploaded", "my paper", "uploaded paper",
    "this paper", "that paper", "like this", "like that",
    "similar papers", "similar to", "papers like", "related to my",
    "based on my", "related papers", "related research",
]

_SEMINAL_QUERY_REWRITES = [
    (
        (
            "transformer architecture",
            "introduces transformers",
            "introduced transformers",
            "introduced transformer architecture",
            "transformer architecture origin",
            "transformer architecture origin paper",
            "origin paper",
            "original transformer paper",
            "transformer paper link",
        ),
        "Attention Is All You Need",
    ),
]

_CANONICAL_ARXIV_IDS = {
    "Attention Is All You Need": "1706.03762",
}


def _is_self_referencing_query(question: str) -> bool:
    """Check if the user is referring to papers already in their library."""
    q_lower = question.lower()
    return any(pat in q_lower for pat in _SELF_REF_PATTERNS)


def _rewrite_external_paper_query(question: str) -> Optional[str]:
    q_lower = question.lower()
    for patterns, rewritten in _SEMINAL_QUERY_REWRITES:
        if any(pattern in q_lower for pattern in patterns):
            return rewritten
    if "transformer" in q_lower and "architecture" in q_lower and "paper" in q_lower:
        return "Attention Is All You Need"
    return None


def _build_query_from_registry(
    registry: List[Dict[str, Any]],
    local_hits: List[Dict[str, Any]],
) -> str:
    """Build a meaningful search query from paper titles and content in the library."""
    parts: List[str] = []

    for r in registry:
        title = r.get("title") or ""
        if title and title != r.get("paper_id"):
            parts.append(title)

    for hit in local_hits[:3]:
        content = (hit.get("content") or "")[:200]
        if content:
            parts.append(content)

    combined = " ".join(parts)
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "it", "its", "we", "our", "they", "their", "has", "have",
        "had", "not", "can", "will", "do", "does", "did", "as", "if", "than",
        "also", "more", "which", "each", "paper", "research", "study", "using",
        "used", "use", "based", "proposed", "results", "show", "method",
    }
    words = re.findall(r'[a-zA-Z]{3,}', combined.lower())
    keywords = []
    seen = set()
    for w in words:
        if w not in stop and w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= 8:
            break

    query = " ".join(keywords)
    logger.info("Built query from registry: %r", query)
    return query or "machine learning"


def search_similar_papers(state: ResearchState) -> ResearchState:
    question = state["question"]
    registry = state.get("paper_registry") or []
    parsed_query = decompose_query(question)
    rewritten_query = _rewrite_external_paper_query(question)

    # Detect self-referencing queries ("find similar to my paper")
    is_self_ref = _is_self_referencing_query(question)

    if rewritten_query:
        clean_query = question.strip()
        logger.info(
            "Using rewritten paper lookup focus: %r via original query %r",
            rewritten_query,
            clean_query,
        )
    elif is_self_ref and registry:
        temp_state = dict(state)
        temp_state["expanded_queries"] = ["abstract introduction methodology key contributions"]
        temp_state = retrieve_local(temp_state)
        local_hits = temp_state.get("local_hits") or []
        clean_query = _build_query_from_registry(registry, local_hits)
        logger.info("Self-ref query detected, using registry-based query: %r", clean_query)
    else:
        if parsed_query.get("arxiv_id"):
            clean_query = str(parsed_query["arxiv_id"])
        elif parsed_query.get("author") and not parsed_query.get("topic"):
            clean_query = f"papers by {parsed_query['author']}"
        else:
            clean_query = question.strip()
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        logger.info("Using direct search query: %r", clean_query)

    # Try to extract number of papers from question
    match = re.search(r'\b([1-9]|[1-4][0-9]|50)\b', question)
    max_results = int(match.group(1)) if match else 5

    # If query still too short, enrich with registry titles
    if len(clean_query.split()) < 3 and registry:
        title_terms = " ".join(
            r["title"] for r in registry
            if r.get("title") and r["title"] != r.get("paper_id")
        )
        if title_terms:
            clean_query = (clean_query + " " + title_terms).strip()[:150]

    logger.info("Final search query: %r  max_results=%d", clean_query, max_results)
    try:
        results = search_arxiv_tool.invoke({"query": clean_query, "max_results": max_results})
        logger.info("Search returned %d results", len(results))
    except Exception as exc:
        logger.error("arXiv search failed: %s", exc)
        results = []

    state["arxiv_recommendations"] = results
    state = retrieve_local(state)
    return state


def _should_attempt_external_search(
    question: str,
    parsed_query: Dict[str, Any] | None = None,
    registry: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    q_lower = (question or "").lower()
    parsed = parsed_query or decompose_query(question)
    registry = registry or []

    local_task_markers = (
        "summarize", "summary", "summaries", "explain", "compare", "difference",
        "contrast", "one by one", "both papers", "two papers", "all papers",
        "in my library", "in this project", "uploaded", "my papers",
    )
    if registry and any(marker in q_lower for marker in local_task_markers):
        return False

    if parsed.get("author") or parsed.get("arxiv_id"):
        return True

    if re.search(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", q_lower):
        return True

    search_nouns = (
        "paper", "papers", "publication", "publications", "article",
        "articles", "study", "studies", "author", "authors", "link",
    )
    search_verbs = (
        "find", "search", "look for", "show me", "get me", "which",
        "what", "where", "who wrote", "written by", "authored by",
        "published", "introduces", "introduced", "link", "original",
        "origin", "tell me", "replace", "viral", "trending",
    )
    if not any(noun in q_lower for noun in search_nouns):
        return False

    if any(verb in q_lower for verb in search_verbs):
        return True

    if parsed.get("source_hint") == "arxiv" and any(term in q_lower for term in ("transformer", "architecture", "paper")):
        return True

    return False

# ─────────────────────────────────────────────
# Build context + sources
# ─────────────────────────────────────────────

def _build_context(state: ResearchState) -> str:
    lines = []
    registry = state.get("paper_registry") or []
    local_hits = state.get("local_hits") or []
    recs = state.get("arxiv_recommendations") or []

    if registry:
        lines.append("=== PAPERS IN THIS PROJECT ===")
        for i, r in enumerate(registry, 1):
            pdf_link = r.get("pdf_url") or ""
            abs_link = r.get("abs_url") or ""
            title = r.get("title") or r["paper_id"]
            link_str = (" | PDF: " + pdf_link if pdf_link else "") + \
                       (" | Abstract: " + abs_link if abs_link and abs_link != pdf_link else "")
            lines.append(
                " Paper " + str(i) + ": ID=[" + r["paper_id"] + "] | Title: " + title + link_str
            )
    lines.append("")

    if local_hits:
        lines.append("=== PAPER CONTENT (Retrieved Chunks) ===")
        grouped: Dict[str, List[Dict]] = {}
        for hit in local_hits:
            md = hit.get("metadata") or {}
            pid = md.get("paper_id", "unknown")
            grouped.setdefault(pid, []).append(hit)

        for pid, hits in grouped.items():
            title = next((r.get("title") or pid for r in registry if r["paper_id"] == pid), pid)
            lines.append("\n--- Paper [" + pid + "]: " + title + " ---")
            for j, hit in enumerate(hits, 1):
                md = hit.get("metadata") or {}
                page = md.get("page")
                content = (hit.get("content") or "").strip()
                page_str = " (page " + str(page) + ")" if page is not None else ""
                lines.append("  Chunk " + str(j) + page_str + ":\n  " + content + "\n")

    if recs:
        lines.append("=== SIMILAR PAPERS FROM ARXIV ===")
        for i, p in enumerate(recs, 1):
            authors = ", ".join(p.get("authors") or [])
            summary = (p.get("summary") or "")[:500]
            title = p.get("title") or "Untitled"
            pdf_link = p.get("pdf_url") or p.get("abs_url") or ""
            abs_link = p.get("abs_url") or ""
            # Pre-format as markdown link so LLM copies verbatim
            title_link = "[" + title + "](" + pdf_link + ")" if pdf_link else title
            lines.append(
                "[ARXIV " + str(i) + "] " + title_link + "\n"
                "  Authors: " + authors + "\n"
                "  Published: " + (p.get("published") or "N/A") + "\n"
                "  Abstract Page: " + abs_link + "\n"
                "  Summary: " + summary + "...\n"
            )

    if not lines:
        return "No papers found in this project. Please upload or import a paper first."

    return "\n".join(lines)


def _build_sources(
    state: ResearchState,
    include_registry_only: bool = False,
    include_registry_background: bool = False,
) -> List[Dict[str, Any]]:
    sources = []
    registry = state.get("paper_registry") or []
    seen_ids: set = set()  # deduplicate sources by paper_id

    if not include_registry_only:
        for hit in (state.get("local_hits") or []):
            md = hit.get("metadata") or {}
            pid = md.get("paper_id", "unknown")

            # Deduplicate — one source entry per paper, not per chunk
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Prefer metadata from registry sidecar (more complete) over chunk metadata
            reg_entry = next((r for r in registry if r.get("paper_id") == pid), {})

            title = (
                md.get("title")
                or reg_entry.get("title")
                or pid
            )
            pdf_url = md.get("pdf_url") or reg_entry.get("pdf_url") or ""
            abs_url = md.get("abs_url") or reg_entry.get("abs_url") or ""

            sources.append({
                "id":       "[" + pid + "]",
                "type":     "local",
                "title":    title,
                "pdf_url":  pdf_url,
                "abs_url":  abs_url,
                "link":     pdf_url or abs_url,
                "page":     md.get("page"),
                "metadata": md,
            })

    if include_registry_only or include_registry_background:
        for entry in registry:
            pid = entry.get("paper_id") or "unknown"
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            pdf_url = entry.get("pdf_url") or ""
            abs_url = entry.get("abs_url") or ""
            sources.append({
                "id": "[" + pid + "]",
                "type": "library",
                "title": entry.get("title") or pid,
                "pdf_url": pdf_url,
                "abs_url": abs_url,
                "link": pdf_url or abs_url,
                "metadata": {
                    "paper_id": pid,
                    "source": entry.get("source"),
                    "is_metadata_only": entry.get("is_metadata_only"),
                },
            })

    for idx, p in enumerate(state.get("arxiv_recommendations") or [], start=1):
        pdf_url = p.get("pdf_url") or ""
        abs_url = p.get("abs_url") or ""
        sources.append({
            "id":      "ARXIV " + str(idx),
            "type":    "arxiv",
            "title":   p.get("title") or "Untitled",
            "pdf_url": pdf_url,
            "abs_url": abs_url,            # ← Bug #5 fix
            "link":    pdf_url or abs_url, # convenience: best available link
        })

    return sources


def _is_smalltalk(question: str) -> bool:
    q = question.strip().lower()
    research_markers = (
        "paper", "papers", "research", "arxiv", "transformer", "model",
        "architecture", "find", "search", "link", "author", "replace",
        "viral", "trending", "publication",
    )
    if any(marker in q for marker in research_markers):
        return False
    if re.match(r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)\b", q):
        return True
    if any(phrase in q for phrase in ("i am ", "i'm ", "myself ", "my name is ", "how are you", "who are you", "what can you do")):
        return True
    return q in {
        "hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "help",
    }


def _best_paper_link(paper: Dict[str, Any]) -> str:
    return paper.get("pdf_url") or paper.get("abs_url") or ""


def _default_suggestions(state: ResearchState) -> List[str]:
    registry = state.get("paper_registry") or []
    if registry:
        first_title = registry[0].get("title") or registry[0].get("paper_id") or "the first paper"
        return [
            f"Summarize {first_title}",
            "Compare the papers in my library",
            "Find related papers to my library",
            "Generate a research brief",
        ]
    return [
        "Find papers about transformers",
        "Find papers written by Yann LeCun",
        "Compare two papers I upload",
        "Generate a research brief",
    ]


def _search_clarification_suggestions(state: ResearchState) -> List[str]:
    parsed = decompose_query(state.get("question") or "")
    author = parsed.get("author")
    if author and not parsed.get("topic"):
        return [
            f"Find papers by {author}",
            f"Find {author} papers from 2024",
            f"Find {author} papers about transformers",
            f"Show highly cited papers by {author}",
        ]
    return [
        "Find papers on transformers from 2024",
        "Find papers by Yann LeCun",
        "Find papers titled Attention Is All You Need",
        "Find papers on diabetic retinopathy",
    ]


def _search_result_suggestions(state: ResearchState) -> List[str]:
    question = (state.get("question") or "").strip()
    parsed = decompose_query(question)
    author = parsed.get("author")
    recs = state.get("arxiv_recommendations") or []

    if author:
        return [
            f"Show more papers by {author}",
            f"Find {author} papers from 2024",
            f"Find most cited papers by {author}",
            f"Find {author} papers on transformers",
        ]

    if recs:
        top_title = recs[0].get("title") or "that paper"
        return [
            f"Summarize {top_title}",
            f"Find papers similar to {top_title}",
            f"Compare {top_title} with a paper in my library",
            "Show more papers on this topic",
        ]

    return _default_suggestions(state)


def _best_paper_summary(paper: Dict[str, Any], max_len: int = 240) -> str:
    summary = re.sub(r"\s+", " ", (paper.get("summary") or "").strip())
    if not summary:
        return "No summary available."
    if len(summary) <= max_len:
        return summary
    return summary[: max_len - 3].rstrip() + "..."


def _paper_markdown_link(paper: Dict[str, Any]) -> str:
    title = paper.get("title") or "Untitled"
    url = paper.get("pdf_url") or paper.get("abs_url") or ""
    return f"[{title}]({url})" if url else title


def _is_single_paper_lookup(question: str) -> bool:
    q = question.lower()
    patterns = [
        "which paper introduces", "which paper introduced", "what paper introduced",
        "what paper introduces", "give link of that paper", "give me the link",
        "which paper invented", "who introduced transformers",
    ]
    return any(pattern in q for pattern in patterns)


def _build_search_results_answer(state: ResearchState) -> str:
    question = state.get("question") or ""
    recs = state.get("arxiv_recommendations") or []
    parsed = decompose_query(question)
    author = parsed.get("author")

    if not recs:
        return _build_search_clarification_answer(state)

    if _is_single_paper_lookup(question):
        top = recs[0]
        title_link = _paper_markdown_link(top)
        authors = ", ".join(top.get("authors") or [])
        summary = _best_paper_summary(top, max_len=320)
        lines = [
            "## Best Match",
            "",
            f"The best match is {title_link}.",
        ]
        if authors:
            lines.append("")
            lines.append(f"Authors: {authors}")
        if top.get("published"):
            lines.append(f"Published: {top.get('published')}")
        lines.extend([
            "",
            summary,
        ])
        if len(recs) > 1:
            lines.extend([
                "",
                "### Other Good Matches",
                "",
            ])
            for idx, paper in enumerate(recs[1:4], start=1):
                lines.append(f"{idx}. {_paper_markdown_link(paper)}")
        return "\n".join(lines)

    heading = "## Search Results"
    intro = f"I found **{len(recs)}** paper(s) that match your request."
    if author:
        intro = f"I found **{len(recs)}** paper(s) for **{author}**."

    lines = [heading, "", intro, ""]
    for idx, paper in enumerate(recs, start=1):
        authors = ", ".join(paper.get("authors") or [])
        source = (paper.get("source") or "research").replace("_", " ")
        summary = _best_paper_summary(paper)
        lines.append(f"{idx}. {_paper_markdown_link(paper)}")
        meta = []
        if authors:
            meta.append(f"Authors: {authors}")
        if paper.get("published"):
            meta.append(f"Published: {paper.get('published')}")
        meta.append(f"Source: {source}")
        lines.append("   " + " | ".join(meta))
        lines.append(f"   {summary}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_library_inventory_answer(state: ResearchState) -> str:
    registry = state.get("paper_registry") or []
    if not registry:
        return (
            "## Your Library\n\n"
            "Your library is empty right now.\n\n"
            "Upload a PDF or import a paper from search, then I can summarize it, compare papers, or find related work."
        )

    lines = [
        "## Your Library",
        "",
        f"You currently have **{len(registry)}** paper(s) in this project:",
        "",
    ]
    for idx, paper in enumerate(registry, start=1):
        title = paper.get("title") or paper.get("paper_id") or "Untitled"
        link = _best_paper_link(paper)
        title_text = f"[{title}]({link})" if link else title
        source = paper.get("source") or ("arxiv" if paper.get("arxiv_id") else "pdf")
        lines.append(f"{idx}. {title_text}")
        lines.append(f"   ID: `{paper.get('paper_id') or 'unknown'}`")
        lines.append(f"   Source: `{source}`")
    lines.extend([
        "",
        "You can ask me to summarize one of these papers, compare them, or find similar work.",
    ])
    return "\n".join(lines)


def _build_insufficient_evidence_answer(state: ResearchState) -> str:
    registry = state.get("paper_registry") or []
    if registry:
        titles = [
            f"- {paper.get('title') or paper.get('paper_id') or 'Untitled'}"
            for paper in registry[:5]
        ]
        library_hint = "\n".join(titles)
        return (
            "I don't have enough information to answer this from the papers currently available.\n\n"
            "Papers already in your library:\n"
            f"{library_hint}\n\n"
            "Try asking about one of those papers directly, import related papers, or ask me to search for papers on this topic."
        )
    return (
        "I don't have enough information to answer this.\n\n"
        "Upload a PDF, import a paper into the project, or ask me to search for papers on the topic first."
    )


def _build_search_clarification_answer(state: ResearchState) -> str:
    question = state.get("question") or ""
    parsed = decompose_query(question)
    author = parsed.get("author")
    topic = (parsed.get("topic") or "").strip().lower()
    weak_tokens = {
        "where", "can", "i", "find", "publication", "publications", "list",
        "paper", "papers", "author", "authored", "s",
    }
    meaningful_tokens = [
        token for token in re.sub(r"[^a-z0-9\s]", " ", topic).split()
        if token not in weak_tokens
    ]

    if author and not meaningful_tokens:
        return (
            f"I couldn't find a strong match yet for **{author}**.\n\n"
            "Can you clarify which of these you want?\n\n"
            f"1. Papers authored by {author}\n"
            f"2. A publication/profile page for {author}\n"
            f"3. Papers by {author} on a specific topic or year\n\n"
            "If you want, reply with a topic or year and I'll refine the search."
        )

    return (
        "I couldn't find a strong match yet.\n\n"
        "Can you clarify it with an author name, paper title, topic, or year? "
        "That will help me search more reliably."
    )


def _build_compare_clarification_answer(state: ResearchState) -> str:
    registry = state.get("paper_registry") or []
    if not registry:
        return "I need at least two papers in your library before I can compare them."

    lines = [
        "I found more than two papers in your library, so I need you to specify which ones to compare.",
        "",
        "Available papers:",
        "",
    ]
    for idx, paper in enumerate(registry[:8], start=1):
        title = paper.get("title") or paper.get("paper_id") or "Untitled"
        suffix = " (metadata only)" if paper.get("is_metadata_only") else ""
        lines.append(f"{idx}. {title}{suffix}")
    lines.extend([
        "",
        "Try asking with the paper titles, or say `compare all papers` if you want a broad comparison.",
    ])
    return "\n".join(lines)

# ─────────────────────────────────────────────
# Node 6: Generate answer
# ─────────────────────────────────────────────

def generate_answer(state: ResearchState) -> ResearchState:
    question = state["question"]
    intent = state.get("intent", "chat")
    registry = state.get("paper_registry") or []
    local_hits = state.get("local_hits") or []
    recs = state.get("arxiv_recommendations") or []

    if intent == "list_library":
        state["answer"] = _build_library_inventory_answer(state)
        state["sources"] = _build_sources(state, include_registry_only=True)
        state["suggestions"] = _default_suggestions(state)
        return state

    if intent == "compare" and state.get("comparison_clarification"):
        state["answer"] = _build_compare_clarification_answer(state)
        state["sources"] = _build_sources(state, include_registry_only=True)
        state["suggestions"] = _default_suggestions(state)
        return state

    if intent == "search_similar" and not local_hits and not recs:
        state["answer"] = _build_search_clarification_answer(state)
        state["sources"] = _build_sources(state, include_registry_background=bool(registry))
        state["suggestions"] = _search_clarification_suggestions(state)
        return state

    if intent == "search_similar" and recs:
        state["answer"] = _build_search_results_answer(state)
        state["sources"] = _build_sources(state)
        state["suggestions"] = _search_result_suggestions(state)
        return state

    if not local_hits and not recs and not _is_smalltalk(question):
        state["answer"] = _build_insufficient_evidence_answer(state)
        state["sources"] = _build_sources(state, include_registry_background=bool(registry))
        state["suggestions"] = _default_suggestions(state)
        return state

    context = _build_context(state)

    # Build a lookup: paper_id -> full title for use in prompt
    id_to_title = {r["paper_id"]: (r.get("title") or r["paper_id"]) for r in registry}

    system_prompt = (
        "You are an expert AI research assistant — like a knowledgeable professor who explains papers clearly and thoroughly.\n\n"

        "=== FORMATTING RULES (MUST FOLLOW) ===\n"
        "1. ALWAYS use the paper's FULL TITLE as a heading, NEVER the arxiv ID code like [2603.15530].\n"
        "   - Correct: ## DUET: Disaggregated Hybrid Mamba-Transformer LLMs\n"
        "   - Wrong:   ## [2603.15530] or ## Paper [2603.15530]\n"
        "2. Use proper markdown: ## for paper titles, ### for subsections, **bold** for key terms, bullet points for lists.\n"
        "3. For summaries/explanations: use sections with ### headings like ### Overview, ### Key Contributions, ### Methodology, ### Results.\n"
        "4. For comparisons: use a markdown table with columns for each paper.\n"
        "5. For similar paper recommendations: use numbered list with title as markdown link, authors, and 2-sentence description.\n"
        "6. STRICT REQUIREMENT: MUST insert blank lines (double newlines) before and after EVERY heading (##, ###), paragraph, and list.\n\n"

        "=== ANSWER QUALITY RULES ===\n"
        "7. Give DETAILED, thorough answers. If asked to summarize, write at least 4-6 sentences per section.\n"
        "8. If asked to explain page by page, dedicate a ### heading per page with detailed explanation.\n"
        "9. If asked about math/methodology, explain the concepts clearly with examples where helpful.\n"
        "10. For casual greetings like 'hi', 'hello' — respond warmly and briefly mention what you can help with (summarize papers, find similar papers, compare, explain concepts, etc.).\n"
        "11. For 'what papers should I download' or vague questions without paper context — suggest the user upload a paper or search arXiv.\n\n"

        "=== STRICT RULES ===\n"
        "12. NEVER use the arxiv ID code as a heading or paper reference in the final answer. Always use the paper title.\n"
        "13. NEVER mention the words 'context', 'chunks', 'LOCAL', or internal system details.\n"
        "14. NEVER repeat the question back to the user.\n"
        "15. When reporting the number of papers found, count BOTH 'PAPERS IN THIS PROJECT' and 'SIMILAR PAPERS FROM ARXIV'.\n"
        "16. If the answer cannot be found in the provided context at all, ONLY THEN say: 'I don't have enough information to answer this.' Otherwise, do your best to answer.\n"
        "17. You have NO tools or function-calling ability. NEVER output tool call JSON. Answer in plain text/markdown ONLY.\n"
        "18. For paper links, COPY the markdown links EXACTLY as provided in the context (e.g. [Title](url)). NEVER swap or rearrange titles and URLs.\n"
        "19. Strip [DONE] from your output — never include it.\n"
        "20. CRITICAL: Each paper title is ALREADY paired with its correct URL in the context. ALWAYS use the EXACT title-URL pair. NEVER mix a title from one paper with a URL from another.\n"
    )

    # Inject id->title map into human prompt so LLM can resolve IDs to titles
    title_map_str = "\n".join(
        "  [" + pid + "] = " + title
        for pid, title in id_to_title.items()
    )

    if intent == "compare":
        system_prompt += (
            "\n\n=== COMPARISON MODE ACTIVE ===\n"
            "The user request involves comparing multiple papers. You MUST output a structured Markdown table to compare them directly.\n"
            "Include columns such as: Paper Title, Methodology/Approach, Key Findings/Results, Datasets, and Limitations (adapt as needed for the query).\n"
            "Provide detailed, precise information in the table cells. You may include introductory or concluding paragraphs around the table."
        )

    chat_history = state.get("chat_history") or []
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-8:]:  # last 4 turns (user+assistant pairs)
            role = (msg.get("role") or "").upper()
            content = (msg.get("content") or "").strip()[:600]
            if role and content:
                history_lines.append(f"{role}: {content}")
        if history_lines:
            history_str = (
                "=== CONVERSATION HISTORY (use this for context/follow-up questions) ===\n"
                + "\n\n".join(history_lines)
                + "\n\n"
            )

    human_prompt = (
        "PAPER TITLE LOOKUP (use these titles in your answer headings, NEVER the raw IDs):\n"
        + (title_map_str if title_map_str else " (no papers yet)") + "\n\n"
        + history_str   # ← INJECT HISTORY HERE
        + "Context:\n" + context + "\n\n"
        "User Question: " + question + "\n\n"
        "Answer:"
    )

    try:
        llm = get_llm(intent, max_tokens=2048)
        answer = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]).content

        answer = answer.replace("[DONE]", "").strip()
        answer = re.sub(r"(?i)<br\s*/?>", "\n", answer)

        # Clean leaked chain-of-thought
        bad_markers = ["Step 1:", "## Step", "### Step", "# Question"]
        if any(marker in answer for marker in bad_markers):
            lower = answer.lower()
            idx = lower.rfind("final answer")
            if idx != -1:
                answer = answer[idx + len("final answer"):].lstrip(":").strip()
            else:
                clean_lines = [
                    line for line in answer.split("\n")
                    if not re.match(r"^Step \d+", line.strip())
                ]
                answer = "\n".join(clean_lines).strip()

    except Exception as exc:
        logger.error("generate_answer failed: %s", exc)
        answer = "I encountered an error generating the answer. Please try again."

    state["answer"] = answer
    state["sources"] = _build_sources(state)

    # ── Generate follow-up suggestions ──
    try:
        suggestion_prompt = (
            "Based on the following question and answer, generate exactly 4 short follow-up questions "
            "the user might want to ask next. Each question should be specific, concise (under 12 words), "
            "and explore a different angle of the topic.\n\n"
            "Original question: " + (state.get("question") or "") + "\n\n"
            "Answer summary: " + (answer or "")[:500] + "\n\n"
            "Return ONLY a JSON array of 4 strings, nothing else. Example:\n"
            '[\"What are the limitations?\", \"How does it compare to X?\", \"What datasets were used?\", \"Can you explain the methodology?\"]'
        )
        suggestion_resp = get_llm("suggestions", max_tokens=256).invoke([HumanMessage(content=suggestion_prompt)])
        raw_suggestions = (suggestion_resp.content or "").strip()
        import json as _json
        array_match = re.search(r"\[[\s\S]*\]", raw_suggestions)
        payload = array_match.group(0) if array_match else raw_suggestions
        parsed = _json.loads(payload)
        state["suggestions"] = [str(item) for item in parsed][:4]
    except Exception as exc:
        logger.warning("Failed to generate suggestions: %s", exc)
        state["suggestions"] = _default_suggestions(state)

    return state

# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

def route_intent(state: ResearchState) -> Literal["search_similar_papers", "retrieve_local", "generate_answer"]:
    if state.get("intent") == "search_similar":
        return "search_similar_papers"
    if state.get("intent") == "list_library":
        return "generate_answer"
    if _should_attempt_external_search(
        state.get("question", ""),
        registry=state.get("paper_registry") or [],
    ):
        return "search_similar_papers"
    return "retrieve_local"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("build_paper_registry", build_paper_registry)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("expand_query", expand_query)
    graph.add_node("retrieve_local", retrieve_local)
    graph.add_node("search_similar_papers", search_similar_papers)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "build_paper_registry")
    graph.add_edge("build_paper_registry", "classify_intent")
    graph.add_edge("classify_intent", "expand_query")

    graph.add_conditional_edges(
        "expand_query",
        route_intent,
        {
            "search_similar_papers": "search_similar_papers",
            "retrieve_local": "retrieve_local",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("search_similar_papers", "generate_answer")
    graph.add_edge("retrieve_local", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


APP = build_graph()

