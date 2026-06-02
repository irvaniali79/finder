import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import DocumentRetriever


class TestDocumentRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = DocumentRetriever.__new__(DocumentRetriever)
        self.retriever.docs_path = "docs/"
        self.retriever.embed_model = MagicMock()
        self.retriever.index = MagicMock()
        self.retriever.chunks = [MagicMock(text="chunk one"), MagicMock(text="chunk two"), MagicMock(text="chunk three")]

    def test_load_documents(self):
        with patch("retrieval.SimpleDirectoryReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["doc1", "doc2"]
            self.retriever.docs_path = "tests/fixtures"
            docs = self.retriever.load_documents()
            self.assertEqual(docs, ["doc1", "doc2"])
            MockReader.assert_called_with("tests/fixtures", required_exts=[".md"])

    def test_chunk_documents(self):
        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes.return_value = ["node1", "node2"]
            docs = ["doc1"]
            nodes = self.retriever.chunk_documents(docs)
            self.assertEqual(nodes, ["node1", "node2"])
            self.assertEqual(self.retriever.chunks, ["node1", "node2"])

    def test_create_embeddings(self):
        import types
        node = MagicMock()
        node.text = "hello world"
        self.retriever.embed_model.get_text_embedding.return_value = [0.1, 0.2]
        embeddings = self.retriever.create_embeddings([node])
        self.retriever.embed_model.get_text_embedding.assert_called_with("hello world")
        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_build_index(self):
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
        import faiss
        self.retriever.build_index(embeddings)
        self.assertIsInstance(self.retriever.index, faiss.IndexFlatL2)
        self.assertEqual(self.retriever.index.ntotal, 2)

    def test_retrieve_returns_top_k_chunks(self):
        self.retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        self.retriever.index = MagicMock()
        self.retriever.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        results = self.retriever.retrieve("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "chunk one")
        self.assertEqual(results[1][0], "chunk two")

    def test_setup_runs_full_pipeline(self):
        retriever = DocumentRetriever.__new__(DocumentRetriever)
        retriever.docs_path = "tests/fixtures"
        retriever.embed_model = MagicMock()
        retriever.index = MagicMock()
        retriever.chunks = []

        with patch.object(retriever, "load_documents", return_value=["d1"]) as load_mock, \
             patch.object(retriever, "chunk_documents", return_value=["n1"]) as chunk_mock, \
             patch.object(retriever, "create_embeddings", return_value=np.array([[0.5, 0.5]], dtype="float32")) as emb_mock, \
             patch.object(retriever, "build_index") as build_mock:
            retriever.setup()
            load_mock.assert_called_once()
            chunk_mock.assert_called_once_with(["d1"])
            build_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
