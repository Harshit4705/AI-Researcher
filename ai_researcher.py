# ai_researcher.py

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from vector_tools import query_project_papers
from arxiv_tool import search_arxiv_tool

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------

def get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
    )


LLM = get_llm()
OUTPUT_PARSER = StrOutputParser()


# ---------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    project_id: str
    question: str

    # Internal state
    intent: str  # "chat" or "search_similar"
    local_hits: List[Dict[str, Any]]
    arxiv_recommendations: List[Dict[str, Any]]  # New: external recs

    # Output
    answer: str
    sources: List[Dict[str, Any]]


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def classify_intent(state: ResearchState) -> ResearchState:
    """
    Decide if the user just wants to chat about local papers OR wants to find similar external papers.
    """
    question = state["question"]
    
    # Simple keyword heuristic (faster/cheaper than LLM call)
    # If user asks for "similar", "recommend", "related papers", "find more", etc.
    keywords = ["similar", "recommend", "related paper", "find more", "search for", "like this", "other paper"]
    q_lower = question.lower()
    
    if any(k in q_lower for k in keywords):
        state["intent"] = "search_similar"
    else:
        state["intent"] = "chat"
        
    logger.info("Classified intent: %s", state["intent"])
    return state


def retrieve_local(state: ResearchState) -> ResearchState:
    """Retrieve local chunks."""
    project_id = state["project_id"]
    question = state["question"]
    
    try:
        hits = query_project_papers.invoke({
            "project_id": project_id,
            "query": question,
            "top_k": 6,
        })
    except Exception:
        hits = []
        
    state["local_hits"] = hits
    return state


def search_similar_papers(state: ResearchState) -> ResearchState:
    """
    If intent is 'search_similar', use the User's question + local context to find external arXiv papers.
    """
    question = state["question"]
    
    # If we have local hits, use their titles to find related work; 
    # otherwise just use the question.
    local_hits = state.get("local_hits") or []
    
    search_query = question
    
    # Improve search query by extracting key terms from local hits if possible?
    # For simplicity, let's just append "research paper" to the user query if generic.
    
    logger.info("Searching arXiv for similar papers: %s", search_query)
    
    try:
        # Search arXiv
        results = search_arxiv_tool.invoke({
            "query": search_query,
            "max_results": 5
        })
    except Exception as exc:
        logger.error("arXiv recommendation search failed: %s", exc)
        results = []

    state["arxiv_recommendations"] = results
    return state


def _build_context(state: ResearchState) -> str:
    lines = []
    
    # 1. Local papers
    local_hits = state.get("local_hits") or []
    if local_hits:
        lines.append("=== UPLOADED / IMPORTED PAPERS (Primary Context) ===")
        for i, hit in enumerate(local_hits, start=1):
            md = hit.get("metadata", {})
            title = md.get("title") or md.get("source")
            content = (hit.get("content") or "").strip()
            lines.append(f"[LOCAL {i}] Title: {title}\n{content}\n")

    # 2. External recommendations (only if present)
    recs = state.get("arxiv_recommendations") or []
    if recs:
        lines.append("=== EXTERNAL ARXIV SEARCH RESULTS (For Recommendation) ===")
        for i, p in enumerate(recs, start=1):
            lines.append(
                f"[ARXIV {i}] Title: {p.get('title')}\n"
                f"Authors: {', '.join(p.get('authors', []))}\n"
                f"Published: {p.get('published')}\n"
                f"Link: {p.get('pdf_url')}\n"  # Providing link to LLM
                f"Summary: {p.get('summary')[:400]}...\n"
            )

    return "\n".join(lines)


def generate_answer(state: ResearchState) -> ResearchState:
    question = state["question"]
    context = _build_context(state)
    
    system_prompt = (
        "You are a helpful research assistant.\n"
        "Your goal is to answer the user's question using the provided context.\n\n"
        "RULES:\n"
        "1. If the user asks about the content of uploaded papers, use the [LOCAL] context.\n"
        "2. If the user asks for similar papers, recommendations, or related work, use the [ARXIV] context.\n"
        "   - Explicitly mention the paper titles and authors.\n"
        "   - **ALWAYS provide the PDF link** for any arXiv paper you recommend.\n"
        "   - Format links as: (Link: http://...)\n"
        "3. Do not make up papers or links.\n"
        "4. Be direct and helpful. No filler phrases like 'I checked the database'.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

    chain = prompt | LLM | OUTPUT_PARSER

    try:
        answer = chain.invoke({"question": question, "context": context})
    except Exception as exc:
        answer = "I encountered an error while generating the response."

    state["answer"] = answer
    
    # Build sources list for UI
    sources = []
    
    # Local sources
    for idx, hit in enumerate(state.get("local_hits") or [], start=1):
        md = hit.get("metadata") or {}
        sources.append({
            "id": f"LOCAL {idx}",
            "type": "local",
            "title": md.get("title") or md.get("source"),
            "pdf_url": md.get("pdf_url")
        })
        
    # Arxiv sources (if used)
    for idx, p in enumerate(state.get("arxiv_recommendations") or [], start=1):
        sources.append({
            "id": f"ARXIV {idx}",
            "type": "arxiv",
            "title": p.get("title"),
            "pdf_url": p.get("pdf_url")
        })

    state["sources"] = sources
    return state


# ---------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------

def route_intent(state: ResearchState) -> Literal["search_similar_papers", "generate_answer"]:
    if state["intent"] == "search_similar":
        return "search_similar_papers"
    return "generate_answer"

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve_local", retrieve_local)
    graph.add_node("search_similar_papers", search_similar_papers)
    graph.add_node("generate_answer", generate_answer)

    # Flow
    graph.add_edge(START, "retrieve_local")
    graph.add_edge("retrieve_local", "classify")
    
    # Conditional routing
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "search_similar_papers": "search_similar_papers",
            "generate_answer": "generate_answer"
        }
    )
    
    graph.add_edge("search_similar_papers", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


APP = build_graph()
