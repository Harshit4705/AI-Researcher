from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research_tool import ResearchPaper, _deduplicate, _filter_author_results, _rank_papers, decompose_query, search_arxiv, search_tavily


class ResearchToolTests(unittest.TestCase):
    def test_decompose_query_handles_authored_pattern(self) -> None:
        parsed = decompose_query("What other papers has Komal Sharma authored?")
        self.assertEqual(parsed["author"], "Komal Sharma")
        self.assertIsNone(parsed["topic"])

    def test_decompose_query_handles_published_pattern(self) -> None:
        parsed = decompose_query("What has Yann LeCun published?")
        self.assertEqual(parsed["author"], "Yann LeCun")
        self.assertIsNone(parsed["topic"])

    def test_decompose_query_treats_publication_list_as_author_only(self) -> None:
        parsed = decompose_query("Where can I find Komal Sharma's publication list?")
        self.assertEqual(parsed["author"], "Komal Sharma")
        self.assertIsNone(parsed["topic"])

    def test_decompose_query_does_not_treat_one_by_one_as_author(self) -> None:
        parsed = decompose_query("summaries both paper one by one")
        self.assertIsNone(parsed["author"])

    def test_decompose_query_applies_transformer_origin_override(self) -> None:
        parsed = decompose_query("which paper introduced transformer architecture give its link")
        self.assertEqual(parsed["exact_title"], "Attention Is All You Need")
        self.assertEqual(parsed["arxiv_id"], "1706.03762")
        self.assertEqual(parsed["source_hint"], "arxiv")

    def test_deduplicate_prefers_arxiv_import_target(self) -> None:
        papers = [
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["A"],
                summary="Short abstract",
                published="2017",
                source="openalex",
                source_id="10.48550/arXiv.1706.03762",
                abs_url="https://doi.org/10.48550/arXiv.1706.03762",
                pdf_url=None,
                primary_category=None,
            ),
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["A"],
                summary="Full abstract",
                published="2017",
                source="arxiv",
                source_id="1706.03762",
                abs_url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                primary_category="cs.CL",
            ),
        ]

        deduped = _deduplicate(papers)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].source, "arxiv")
        self.assertEqual(deduped[0].source_id, "1706.03762")

    def test_search_arxiv_uses_id_list_for_exact_identifier(self) -> None:
        captured = {}

        def fake_get(url, params, source=""):
            captured.update(params)
            return SimpleNamespace(text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")

        with patch("research_tool._arxiv_rate_limit"), \
             patch("research_tool._get", side_effect=fake_get), \
             patch("research_tool._parse_arxiv_xml", return_value=[]):
            search_arxiv({"arxiv_id": "1706.03762", "topic": None}, 5)

        self.assertEqual(captured["id_list"], "1706.03762")
        self.assertNotIn("search_query", captured)

    def test_filter_author_results_drops_similar_but_wrong_names(self) -> None:
        papers = [
            ResearchPaper(
                title="Correct Paper",
                authors=["Komal Sharma", "A"],
                summary="A",
                published="2024",
                source="arxiv",
                source_id="1",
                abs_url="https://example.com/1",
                pdf_url=None,
                primary_category=None,
            ),
            ResearchPaper(
                title="Wrong Paper",
                authors=["Komal Tyagi", "B"],
                summary="B",
                published="2024",
                source="openalex",
                source_id="2",
                abs_url="https://example.com/2",
                pdf_url=None,
                primary_category=None,
            ),
        ]

        filtered = _filter_author_results(papers, "Komal Sharma")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Correct Paper")

    def test_search_tavily_maps_web_results_to_papers(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Attention Is All You Need - arXiv",
                    "content": "Research paper introducing the Transformer architecture.",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "published_date": "2017-06-12",
                }
            ]
        }

        with patch("research_tool.os.getenv", return_value="test-key"), \
             patch("research_tool.requests.post", return_value=SimpleNamespace(
                 raise_for_status=lambda: None,
                 json=lambda: payload,
             )):
            results = search_tavily({"raw": "transformer architecture paper link"}, 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "web")
        self.assertEqual(results[0].abs_url, "https://arxiv.org/abs/1706.03762")

    def test_search_tavily_uses_exact_match_for_exact_title(self) -> None:
        captured = {}
        payload = {"results": []}

        def fake_post(url, json, timeout, headers):
            captured.update(json)
            captured["_headers"] = headers
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: payload,
            )

        with patch("research_tool.os.getenv", return_value="test-key"), \
             patch("research_tool.requests.post", side_effect=fake_post):
            search_tavily(
                {
                    "raw": "which paper introduced transformer architecture give its link",
                    "exact_title": "Attention Is All You Need",
                },
                5,
            )

        self.assertTrue(captured["exact_match"])
        self.assertIn("arxiv.org", captured["include_domains"])
        self.assertIn("Attention Is All You Need", captured["query"])
        self.assertEqual(captured["_headers"]["Authorization"], "Bearer test-key")

    def test_rank_papers_boosts_web_results_for_trending_queries(self) -> None:
        papers = [
            ResearchPaper(
                title="Recent model replacing transformers",
                authors=[],
                summary="Web verification result",
                published="2026-01-01",
                source="web",
                source_id="https://example.com/recent",
                abs_url="https://example.com/recent",
                pdf_url=None,
                primary_category=None,
            ),
            ResearchPaper(
                title="Older transformer variant",
                authors=["A"],
                summary="Academic metadata",
                published="2022",
                source="openalex",
                source_id="1",
                abs_url="https://example.com/old",
                pdf_url=None,
                primary_category=None,
            ),
        ]

        ranked = _rank_papers(papers, {"raw": "viral paper that will replace transformers"})
        self.assertEqual(ranked[0].source, "web")

    def test_rank_papers_prefers_exact_title_match(self) -> None:
        papers = [
            ResearchPaper(
                title="Attention is all you need: utilizing attention in AI-enabled drug discovery",
                authors=["A"],
                summary="A",
                published="2023",
                source="openalex",
                source_id="1",
                abs_url="https://example.com/1",
                pdf_url=None,
                primary_category=None,
            ),
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                summary="Transformer paper",
                published="2017",
                source="arxiv",
                source_id="1706.03762",
                abs_url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                primary_category="cs.CL",
            ),
        ]

        ranked = _rank_papers(papers, {"exact_title": "Attention Is All You Need", "topic": "attention is all you need"})

        self.assertEqual(ranked[0].title, "Attention Is All You Need")

    def test_rank_papers_prefers_exact_arxiv_identifier(self) -> None:
        papers = [
            ResearchPaper(
                title="Attention is all you need: utilizing attention in AI-enabled drug discovery",
                authors=["A"],
                summary="A",
                published="2023",
                source="openalex",
                source_id="1",
                abs_url="https://doi.org/10.1000/example",
                pdf_url=None,
                primary_category=None,
            ),
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                summary="Transformer paper",
                published="2017",
                source="arxiv",
                source_id="1706.03762",
                abs_url="https://arxiv.org/abs/1706.03762",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                primary_category="cs.CL",
            ),
        ]

        ranked = _rank_papers(
            papers,
            {
                "arxiv_id": "1706.03762",
                "exact_title": "Attention Is All You Need",
                "raw": "which paper introduced transformer architecture give its link",
            },
        )

        self.assertEqual(ranked[0].source, "arxiv")


if __name__ == "__main__":
    unittest.main()
