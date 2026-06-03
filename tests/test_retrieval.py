import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import DocumentRetriever


class TestDocumentRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = DocumentRetriever.__new__(DocumentRetriever)
        self.retriever.docs_path = Path("docs/")
        self.retriever.embed_model = MagicMock()
        self.retriever.index = MagicMock()
        self.retriever.chunks = [MagicMock(text="chunk one"), MagicMock(text="chunk two"), MagicMock(text="chunk three")]

    def test_load_documents_md(self):
        with patch("retrieval.TxtReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["md-doc"]
            with patch.object(self.retriever.docs_path, "exists", return_value=True), \
                 patch.object(self.retriever.docs_path, "iterdir") as iterdir_mock:
                file_mock = MagicMock()
                file_mock.is_file.return_value = True
                file_mock.suffix.lower.return_value = ".md"
                iterdir_mock.return_value = [file_mock]
                docs = self.retriever.load_documents()
                self.assertEqual(docs, ["md-doc"])
                mock_reader.load_data.assert_called_once_with(file_mock)

    def test_load_documents_txt(self):
        with patch("retrieval.TxtReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["txt-doc"]
            with patch.object(self.retriever.docs_path, "exists", return_value=True), \
                 patch.object(self.retriever.docs_path, "iterdir") as iterdir_mock:
                file_mock = MagicMock()
                file_mock.is_file.return_value = True
                file_mock.suffix.lower.return_value = ".txt"
                iterdir_mock.return_value = [file_mock]
                docs = self.retriever.load_documents()
                self.assertEqual(docs, ["txt-doc"])
                mock_reader.load_data.assert_called_once_with(file_mock)

    def test_load_documents_pdf(self):
        with patch("retrieval.PDFReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["pdf-doc"]
            with patch.object(self.retriever.docs_path, "exists", return_value=True), \
                 patch.object(self.retriever.docs_path, "iterdir") as iterdir_mock:
                file_mock = MagicMock()
                file_mock.is_file.return_value = True
                file_mock.suffix.lower.return_value = ".pdf"
                iterdir_mock.return_value = [file_mock]
                docs = self.retriever.load_documents()
                self.assertEqual(docs, ["pdf-doc"])
                mock_reader.load_data.assert_called_once_with(file_mock)

    def test_load_documents_empty_dir(self):
        with patch.object(self.retriever.docs_path, "exists", return_value=True), \
             patch.object(self.retriever.docs_path, "iterdir", return_value=iter([])):
            docs = self.retriever.load_documents()
            self.assertEqual(docs, [])

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
        retriever.docs_path = self.retriever.docs_path
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
