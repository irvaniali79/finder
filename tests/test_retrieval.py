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
        self.retriever.docs_path = MagicMock(spec=Path)
        self.retriever.docs_path.exists.return_value = True
        self.retriever.embed_model = MagicMock()
        self.retriever.index = MagicMock()
        self.retriever.chunks = [MagicMock(text="chunk one"), MagicMock(text="chunk two"), MagicMock(text="chunk three")]

    def test_load_documents_md(self):
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix.lower.return_value = ".md"
        mock_file.name = "test.md"
        mock_file.read_text.return_value = "# md content"
        with patch.object(self.retriever.docs_path, "iterdir", return_value=[mock_file]):
            docs = self.retriever.load_documents()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].text, "# md content")
            self.assertEqual(docs[0].metadata["file_name"], "test.md")

    def test_load_documents_txt(self):
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = True
        mock_file.suffix.lower.return_value = ".txt"
        mock_file.name = "test.txt"
        mock_file.read_text.return_value = "plain text content"
        with patch.object(self.retriever.docs_path, "iterdir", return_value=[mock_file]):
            docs = self.retriever.load_documents()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].text, "plain text content")
            self.assertEqual(docs[0].metadata["file_name"], "test.txt")

    def test_load_documents_pdf(self):
        with patch("retrieval.PDFReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["pdf-doc"]
            mock_file = MagicMock(spec=Path)
            mock_file.is_file.return_value = True
            mock_file.suffix.lower.return_value = ".pdf"
            with patch.object(self.retriever.docs_path, "iterdir", return_value=[mock_file]):
                docs = self.retriever.load_documents()
                self.assertEqual(docs, ["pdf-doc"])
                mock_reader.load_data.assert_called_once_with(mock_file)

    def test_load_documents_empty_dir(self):
        with patch("retrieval.Path") as MockPath:
            mock_path_instance = MockPath.return_value
            mock_path_instance.iterdir.return_value = iter([])
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
