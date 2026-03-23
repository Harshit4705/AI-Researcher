from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-key")

import main
from research_tool import ResearchPaper


class MainApiTests(unittest.TestCase):
    def setUp(self) -> None:
        main._chat_histories.clear()
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        main._chat_histories.clear()

    def test_chat_history_uses_abs_url_when_pdf_missing(self) -> None:
        payloads = []

        def fake_invoke(payload):
            payloads.append(payload)
            if len(payloads) == 1:
                return {
                    "answer": "First answer",
                    "sources": [{"title": "Paper One", "abs_url": "https://example.com/paper-one"}],
                }
            return {"answer": "Second answer", "sources": []}

        with patch.object(main.APP, "invoke", side_effect=fake_invoke):
            first = self.client.post("/chat", json={"project_id": "proj", "question": "Hi", "session_id": "sess"})
            second = self.client.post("/chat", json={"project_id": "proj", "question": "Follow up", "session_id": "sess"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("https://example.com/paper-one", payloads[1]["chat_history"][-1]["content"])

    def test_project_stats_endpoint_summarizes_library(self) -> None:
        with patch.object(main, "list_project_papers", SimpleNamespace(invoke=lambda _: ["p1", "p2"])), \
             patch.object(main, "load_all_paper_metadata", return_value={
                 "p1": {"paper_id": "p1", "title": "Paper One", "source": "arxiv", "pdf_url": "https://example.com/p1.pdf"},
                 "p2": {"paper_id": "p2", "title": "Paper Two", "source": "pdf_upload", "abs_url": "https://example.com/p2"},
             }), \
             patch.object(main, "get_all_project_chunks", return_value=[{"content": "a"}, {"content": "b"}, {"content": "c"}]):
            response = self.client.get("/projects/demo/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["paper_count"], 2)
        self.assertEqual(data["chunk_count"], 3)
        self.assertEqual(data["papers_with_pdf"], 1)
        self.assertEqual(data["papers_with_links"], 2)
        self.assertEqual(data["source_counts"]["arxiv"], 1)
        self.assertEqual(data["source_counts"]["pdf_upload"], 1)

    def test_normalize_import_identifier_converts_arxiv_doi(self) -> None:
        self.assertEqual(
            main._normalize_import_identifier("10.48550/arXiv.1706.03762"),
            "1706.03762",
        )

    def test_prepare_brief_inputs_limits_context_per_paper(self) -> None:
        long_text = "A" * 5000
        chunks = [
            {"content": long_text, "metadata": {"paper_id": "p1", "title": "Paper One"}},
            {"content": long_text, "metadata": {"paper_id": "p2", "title": "Paper Two"}},
        ]

        prepared = main._prepare_brief_inputs(chunks, max_papers=1, max_chars_per_paper=1200)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["title"], "Paper One")
        self.assertLessEqual(len(prepared[0]["content"]), 1200)

    def test_lookup_arxiv_paper_returns_exact_identifier_match(self) -> None:
        with patch.object(main, "search_arxiv", return_value=[
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                summary="Transformer paper",
                published="2017-06-12",
                source="arxiv",
                source_id="1706.03762v1",
                abs_url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                primary_category="cs.CL",
            )
        ]):
            paper = main._lookup_arxiv_paper("1706.03762")

        self.assertIsNotNone(paper)
        self.assertEqual(paper["source_id"], "1706.03762v1")
        self.assertEqual(paper["title"], "Attention Is All You Need")

    def test_fallback_import_paper_preserves_selected_metadata(self) -> None:
        request = main.ImportArxivRequest(
            project_id="demo",
            arxiv_id="opaque-id",
            title="Selected Title",
            authors=["Komal Sharma"],
            summary="Selected summary",
            published="2024",
            source="semantic_scholar",
            source_id="opaque-id",
            abs_url="https://example.com/paper",
        )

        paper = main._fallback_import_paper(request, "opaque-id")

        self.assertEqual(paper["title"], "Selected Title")
        self.assertEqual(paper["authors"], ["Komal Sharma"])
        self.assertEqual(paper["abs_url"], "https://example.com/paper")


if __name__ == "__main__":
    unittest.main()
