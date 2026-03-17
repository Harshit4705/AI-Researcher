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
from arxiv_tool import search_arxiv_tool

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
                })
            except Exception:
                registry.append({"paper_id": pid, "title": pid, "arxiv_id": None, "pdf_url": None})

    for stale_id in set(sidecar.keys()) - faiss_ids:
        logger.info("Removing stale sidecar entry: %s", stale_id)
        delete_paper_metadata(project_id, stale_id)

    state["paper_registry"] = registry
    logger.info("Registry: %s", [r["paper_id"] for r in registry])
    return state

# ─────────────────────────────────────────────
# Node 2: Classify intent + detect target paper
# ─────────────────────────────────────────────

def classify_intent(state: ResearchState) -> ResearchState:
    question = state["question"]
    q_lower = question.lower()
    registry = state.get("paper_registry") or []

    compare_kws = [
        "compare", "difference between", "contrast", "versus", "vs",
        "both papers", "which paper", "between the two", "better paper",
        "written better", "writing quality",
    ]
    similar_kws = [
        "similar", "recommend", "related paper", "find more", "find papers",
        "search for", "like this", "other paper", "more papers", "suggest",
        "attention mechanism", "find me", "look for", "papers about", "papers on",
    ]

    if any(k in q_lower for k in compare_kws):
        intent = "compare"
    elif any(k in q_lower for k in similar_kws):
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
        target_ids = [r["paper_id"] for r in registry]

    state["target_paper_ids"] = list(dict.fromkeys(target_ids))
    logger.info("Target paper IDs: %s", state["target_paper_ids"])
    return state

# ─────────────────────────────────────────────
# Node 3: Query expansion
# ─────────────────────────────────────────────

def expand_query(state: ResearchState) -> ResearchState:
    question = state["question"]
    intent = state.get("intent", "chat")

    if intent == "search_similar":
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

def search_similar_papers(state: ResearchState) -> ResearchState:
    question = state["question"]
    registry = state.get("paper_registry") or []

    # Use LLM to extract a clean search query
    try:
        llm = get_llm("chat", max_tokens=100)
        prompt = (
            "Extract the core research topic from this user request for an arXiv search.\n"
            "Return ONLY the search keywords (e.g. 'reinforcement learning', 'transformers', 'mamba').\n"
            "Do not include phrases like 'find me papers' or numbers.\n"
            f"Request: {question}"
        )
        clean_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        clean_query = clean_query.strip("'\"").replace("\n", " ")
    except Exception as exc:
        logger.warning("Failed to extract clean query: %s", exc)
        clean_query = question

    # Try to extract number of papers from question
    match = re.search(r'\b([1-9]|[1-4][0-9]|50)\b', question)
    max_results = int(match.group(1)) if match else 5

    # If the user didn't specify any topic but just said "find similar papers", use registry titles
    title_terms = ""
    if len(clean_query.split()) < 2 and registry:
        title_terms = " ".join(
            r["title"] for r in registry
            if r.get("title") and r["title"] != r.get("paper_id")
        )

    search_query = (clean_query + " " + title_terms).strip()[:100]

    logger.info("arXiv search query: %r, max_results: %d", search_query, max_results)
    try:
        results = search_arxiv_tool.invoke({"query": search_query, "max_results": max_results})
    except Exception as exc:
        logger.error("arXiv search failed: %s", exc)
        results = []

    state["arxiv_recommendations"] = results
    state = retrieve_local(state)
    return state

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
            pdf_link = r.get("pdf_url", "")
            title = r.get("title") or r["paper_id"]
            lines.append(
                "  Paper " + str(i) + ": ID=[" + r["paper_id"] + "] | Title: " + title +
                (" | PDF: " + pdf_link if pdf_link else "")
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
            lines.append(
                "[ARXIV " + str(i) + "] " + p.get("title", "N/A") + "\n"
                "  Authors: " + authors + "\n"
                "  Published: " + p.get("published", "N/A") + "\n"
                "  Link: " + p.get("pdf_url", "N/A") + "\n"
                "  Summary: " + summary + "...\n"
            )

    if not lines:
        return "No papers found in this project. Please upload or import a paper first."

    return "\n".join(lines)


def _build_sources(state: ResearchState) -> List[Dict[str, Any]]:
    sources = []
    registry = state.get("paper_registry") or []

    for idx, hit in enumerate(state.get("local_hits") or [], start=1):
        md = hit.get("metadata") or {}
        pid = md.get("paper_id", "unknown")
        title = md.get("title") or next(
            (r.get("title") or pid for r in registry if r["paper_id"] == pid), pid
        )
        sources.append({
            "id": "[" + pid + "]",
            "type": "local",
            "title": title,
            "pdf_url": md.get("pdf_url"),
            "page": md.get("page"),
            "metadata": md,
        })

    for idx, p in enumerate(state.get("arxiv_recommendations") or [], start=1):
        sources.append({
            "id": "ARXIV " + str(idx),
            "type": "arxiv",
            "title": p.get("title"),
            "pdf_url": p.get("pdf_url"),
        })

    return sources

# ─────────────────────────────────────────────
# Node 6: Generate answer
# ─────────────────────────────────────────────

def generate_answer(state: ResearchState) -> ResearchState:
    question = state["question"]
    context = _build_context(state)
    intent = state.get("intent", "chat")
    registry = state.get("paper_registry") or []

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
        "19. You have NO tools or function-calling ability. NEVER output tool call JSON. Answer in plain text/markdown ONLY.\n"
        "17. For PDF links, always format as a clickable markdown link: [View PDF](url).\n"
        "18. Strip [DONE] from your output — never include it.\n"
    )

    # Inject id->title map into human prompt so LLM can resolve IDs to titles
    title_map_str = "\n".join(
        "  [" + pid + "] = " + title
        for pid, title in id_to_title.items()
    )

    human_prompt = (
        "PAPER TITLE LOOKUP (use these titles in your answer headings, NEVER the raw IDs):\n"
        + (title_map_str if title_map_str else "  (no papers yet)") + "\n\n"
        "Context:\n" + context + "\n\n"
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
    return state

# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

def route_intent(state: ResearchState) -> Literal["search_similar_papers", "retrieve_local"]:
    if state.get("intent") == "search_similar":
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
        },
    )
    graph.add_edge("search_similar_papers", "generate_answer")
    graph.add_edge("retrieve_local", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


APP = build_graph()
