from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")

import ai_researcher
from ai_researcher import _build_context, _is_smalltalk, _rewrite_external_paper_query, _should_attempt_external_search, classify_intent, generate_answer, get_model_name


class AiResearcherTests(unittest.TestCase):
    def test_build_context_lists_every_paper_in_registry(self) -> None:
        state = {
            "paper_registry": [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "pdf_url": "https://example.com/p1.pdf",
                    "abs_url": "https://example.com/p1",
                },
                {
                    "paper_id": "p2",
                    "title": "Paper Two",
                    "pdf_url": "https://example.com/p2.pdf",
                    "abs_url": "https://example.com/p2",
                },
            ],
            "local_hits": [],
            "arxiv_recommendations": [],
        }

        context = _build_context(state)

        self.assertIn("Paper 1: ID=[p1] | Title: Paper One", context)
        self.assertIn("Paper 2: ID=[p2] | Title: Paper Two", context)

    def test_generate_answer_lists_library_without_calling_llm(self) -> None:
        state = {
            "intent": "list_library",
            "question": "What papers do I have?",
            "paper_registry": [
                {"paper_id": "p1", "title": "Paper One", "pdf_url": "https://example.com/p1.pdf", "source": "pdf_upload"},
                {"paper_id": "p2", "title": "Paper Two", "abs_url": "https://example.com/p2", "source": "arxiv"},
            ],
            "local_hits": [],
            "arxiv_recommendations": [],
        }

        with patch("ai_researcher.get_llm", side_effect=AssertionError("LLM should not be called")):
            result = generate_answer(state)

        self.assertIn("## Your Library", result["answer"])
        self.assertIn("Paper One", result["answer"])
        self.assertIn("Paper Two", result["answer"])
        self.assertEqual(len(result["suggestions"]), 4)
        self.assertEqual(result["sources"][0]["type"], "library")

    def test_generate_answer_returns_fallback_without_evidence(self) -> None:
        state = {
            "intent": "chat",
            "question": "What are the results of this method?",
            "paper_registry": [{"paper_id": "p1", "title": "Paper One", "source": "pdf_upload"}],
            "local_hits": [],
            "arxiv_recommendations": [],
        }

        with patch("ai_researcher.get_llm", side_effect=AssertionError("LLM should not be called")):
            result = generate_answer(state)

        self.assertIn("I don't have enough information to answer this", result["answer"])
        self.assertIn("Paper One", result["answer"])
        self.assertEqual(len(result["suggestions"]), 4)

    def test_classify_intent_handles_publication_list_queries(self) -> None:
        state = {"question": "Where can I find Komal Sharma's publication list?", "paper_registry": []}
        result = classify_intent(state)
        self.assertEqual(result["intent"], "search_similar")

    def test_classify_intent_routes_transformer_origin_query_to_search(self) -> None:
        state = {
            "question": "which paper introduces transformers give link of that paper",
            "paper_registry": [{"paper_id": "p1", "title": "Local One"}, {"paper_id": "p2", "title": "Local Two"}],
        }
        result = classify_intent(state)
        self.assertEqual(result["intent"], "search_similar")

    def test_external_search_fallback_detects_publication_queries(self) -> None:
        self.assertTrue(_should_attempt_external_search("What has Yann LeCun published?"))

    def test_external_search_fallback_detects_paper_link_queries(self) -> None:
        self.assertTrue(_should_attempt_external_search("transformer architecture paper link"))

    def test_external_search_fallback_stays_local_for_summary_requests(self) -> None:
        self.assertFalse(
            _should_attempt_external_search(
                "summaries both paper one by one",
                registry=[{"paper_id": "p1"}, {"paper_id": "p2"}],
            )
        )

    def test_rewrite_external_paper_query_handles_transformer_origin(self) -> None:
        self.assertEqual(
            _rewrite_external_paper_query("which paper introduces transformers"),
            "Attention Is All You Need",
        )
        self.assertEqual(
            _rewrite_external_paper_query("transformer architecture origin paper"),
            "Attention Is All You Need",
        )

    def test_rewritten_query_uses_exact_title_prefix(self) -> None:
        state = {
            "question": "which paper introduced transformer architecture give its link",
            "project_id": "demo",
            "paper_registry": [],
        }

        captured = {}

        def fake_invoke(payload):
            captured.update(payload)
            return []

        with patch("ai_researcher.search_arxiv_tool", new=SimpleNamespace(invoke=fake_invoke)), \
             patch("ai_researcher.retrieve_local", side_effect=lambda current: {**current, "local_hits": []}):
            result = ai_researcher.search_similar_papers(state)  # type: ignore[name-defined]

        self.assertEqual(result["arxiv_recommendations"], [])
        self.assertEqual(captured["query"], "which paper introduced transformer architecture give its link")

    def test_smalltalk_detects_self_introduction(self) -> None:
        self.assertTrue(_is_smalltalk("hi myself harshit"))
        self.assertFalse(_is_smalltalk("hi tell me the paper that is being viral that it will replace transformers"))

    def test_get_model_name_stays_on_default_groq_model(self) -> None:
        self.assertEqual(get_model_name("chat"), "openai/gpt-oss-20b")
        self.assertEqual(get_model_name("notebook_codegen"), "openai/gpt-oss-20b")

    def test_classify_intent_keeps_compare_queries_as_compare(self) -> None:
        state = {
            "question": "compare the two papers in my library",
            "paper_registry": [{"paper_id": "p1", "title": "Local One"}, {"paper_id": "p2", "title": "Local Two"}],
        }
        result = classify_intent(state)
        self.assertEqual(result["intent"], "compare")

    def test_generate_answer_clarifies_ambiguous_search_without_results(self) -> None:
        state = {
            "intent": "search_similar",
            "question": "Where can I find Komal Sharma's publication list?",
            "paper_registry": [],
            "local_hits": [],
            "arxiv_recommendations": [],
        }

        with patch("ai_researcher.get_llm", side_effect=AssertionError("LLM should not be called")):
            result = generate_answer(state)

        self.assertIn("Can you clarify", result["answer"])
        self.assertIn("Komal Sharma", result["answer"])

    def test_generate_answer_builds_deterministic_search_results(self) -> None:
        state = {
            "intent": "search_similar",
            "question": "find 2 papers written by Yann LeCun",
            "paper_registry": [],
            "local_hits": [],
            "arxiv_recommendations": [
                {
                    "title": "Paper One",
                    "authors": ["Yann LeCun"],
                    "published": "2024",
                    "source": "arxiv",
                    "abs_url": "https://example.com/p1",
                    "pdf_url": "https://example.com/p1.pdf",
                    "summary": "First summary",
                },
                {
                    "title": "Paper Two",
                    "authors": ["Yann LeCun"],
                    "published": "2023",
                    "source": "semantic_scholar",
                    "abs_url": "https://example.com/p2",
                    "summary": "Second summary",
                },
            ],
        }

        with patch("ai_researcher.get_llm", side_effect=AssertionError("LLM should not be called")):
            result = generate_answer(state)

        self.assertIn("## Search Results", result["answer"])
        self.assertIn("[Paper One](https://example.com/p1.pdf)", result["answer"])
        self.assertIn("Yann LeCun", result["answer"])

    def test_generate_answer_compare_clarifies_when_library_has_many_candidates(self) -> None:
        state = {
            "intent": "compare",
            "question": "compare papers in my library",
            "comparison_clarification": True,
            "paper_registry": [
                {"paper_id": "p1", "title": "Paper One"},
                {"paper_id": "p2", "title": "Paper Two", "is_metadata_only": True},
                {"paper_id": "p3", "title": "Paper Three"},
            ],
            "local_hits": [],
            "arxiv_recommendations": [],
        }

        with patch("ai_researcher.get_llm", side_effect=AssertionError("LLM should not be called")):
            result = generate_answer(state)

        self.assertIn("specify which ones to compare", result["answer"])
        self.assertEqual(result["sources"][0]["type"], "library")

    def test_generate_answer_only_surfaces_used_local_sources(self) -> None:
        state = {
            "intent": "chat",
            "question": "Summarize this paper",
            "paper_registry": [
                {"paper_id": "p1", "title": "Paper One", "pdf_url": "https://example.com/p1.pdf"},
                {"paper_id": "p2", "title": "Paper Two", "pdf_url": "https://example.com/p2.pdf"},
            ],
            "local_hits": [
                {"content": "Useful content", "metadata": {"paper_id": "p1", "title": "Paper One", "pdf_url": "https://example.com/p1.pdf"}}
            ],
            "arxiv_recommendations": [],
        }

        with patch("ai_researcher.get_llm") as get_llm:
            get_llm.return_value.invoke.return_value.content = "Answer"
            result = generate_answer(state)

        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["title"], "Paper One")


if __name__ == "__main__":
    unittest.main()
