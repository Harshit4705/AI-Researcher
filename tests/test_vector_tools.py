from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import vector_tools
from vector_tools import LocalHashEmbeddings, get_embeddings


class VectorToolsTests(unittest.TestCase):
    def tearDown(self) -> None:
        vector_tools._EMBEDDINGS = None

    def test_get_embeddings_falls_back_when_hf_loader_fails(self) -> None:
        with patch("vector_tools.HuggingFaceEmbeddings", side_effect=NotImplementedError("meta tensor")):
            embeddings = get_embeddings()

        self.assertIsInstance(embeddings, LocalHashEmbeddings)
        self.assertEqual(len(embeddings.embed_query("transformer paper")), 384)


if __name__ == "__main__":
    unittest.main()
