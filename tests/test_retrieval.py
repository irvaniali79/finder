import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import DocumentRetriever


class TestDocumentRetriever(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmpdir) / "cache"
        self.docs_dir = Path(self.tmpdir) / "docs"
        self.docs_dir.mkdir()
        self.retriever = DocumentRetriever.__new__(DocumentRetriever)
        self.retriever.docs_path = self.docs_dir
        self.retriever.cache_dir = self.cache_dir
        self.retriever.embed_model = MagicMock()
        self.retriever.index = None
        self.retriever.chunks = []
        self.retriever.state = {}

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, content="hello"):
        path = self.docs_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_documents_md(self):
        self._write("test.md", "# md content")
        docs = self.retriever.load_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "# md content")
        self.assertEqual(docs[0].metadata["file_name"], "test.md")

    def test_load_documents_txt(self):
        self._write("test.txt", "plain text content")
        docs = self.retriever.load_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "plain text content")
        self.assertEqual(docs[0].metadata["file_name"], "test.txt")

    def test_load_documents_pdf(self):
        with patch("retrieval.PDFReader") as MockReader:
            mock_reader = MockReader.return_value
            mock_reader.load_data.return_value = ["pdf-doc"]
            path = self.docs_dir / "file.pdf"
            path.write_bytes(b"%PDF-1.4")
            docs = self.retriever.load_documents()
            self.assertEqual(docs, ["pdf-doc"])
            mock_reader.load_data.assert_called_once_with(path)

    def test_load_documents_empty_dir(self):
        docs = self.retriever.load_documents()
        self.assertEqual(docs, [])

    def test_compute_file_fingerprint_changes_with_content(self):
        p1 = self._write("a.md", "alpha")
        fp1 = DocumentRetriever.compute_file_fingerprint(p1)
        p1.write_text("alpha changed", encoding="utf-8")
        fp2 = DocumentRetriever.compute_file_fingerprint(p1)
        self.assertNotEqual(fp1, fp2)

    def test_compute_file_fingerprint_stable(self):
        p1 = self._write("a.md", "alpha")
        self.assertEqual(
            DocumentRetriever.compute_file_fingerprint(p1),
            DocumentRetriever.compute_file_fingerprint(p1),
        )

    def test_chunk_documents_returns_dicts(self):
        mock_splitter = MagicMock()
        node1 = MagicMock()
        node1.text = "n1 text"
        node1.metadata = {"file_name": "a.md"}
        node2 = MagicMock()
        node2.text = "n2 text"
        node2.metadata = {"file_name": "a.md"}
        mock_splitter.get_nodes_from_documents.return_value = [node1, node2]
        with patch("retrieval.SentenceSplitter", return_value=mock_splitter):
            docs = ["d1"]
            chunks = self.retriever.chunk_documents(docs)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["text"], "n1 text")
            self.assertEqual(chunks[0]["metadata"], {"file_name": "a.md"})
            self.assertEqual(self.retriever.chunks, chunks)

    def test_chunk_documents_empty_input(self):
        result = self.retriever.chunk_documents([])
        self.assertEqual(result, [])

    def test_create_embeddings(self):
        self.retriever.embed_model.get_text_embedding.return_value = [0.1, 0.2]
        embeddings = self.retriever.create_embeddings([{"text": "hello", "metadata": {}}])
        self.retriever.embed_model.get_text_embedding.assert_called_with("hello")
        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_build_index(self):
        import faiss
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
        self.retriever.build_index(embeddings)
        self.assertIsInstance(self.retriever.index, faiss.IndexFlatL2)
        self.assertEqual(self.retriever.index.ntotal, 2)

    def test_build_index_appends_when_existing(self):
        import faiss
        self.retriever.index = faiss.IndexFlatL2(2)
        self.retriever.index.add(np.array([[0.1, 0.2]], dtype="float32"))
        new = np.array([[0.3, 0.4]], dtype="float32")
        self.retriever.build_index(new)
        self.assertEqual(self.retriever.index.ntotal, 2)

    def test_retrieve_returns_top_k_chunks(self):
        self.retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        self.retriever.chunks = [
            {"text": "chunk one", "metadata": {"file_name": "a.md"}},
            {"text": "chunk two", "metadata": {"file_name": "a.md"}},
        ]
        self.retriever.index = MagicMock()
        self.retriever.index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]]),
        )
        results = self.retriever.retrieve("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "chunk one")
        self.assertEqual(results[1][0], "chunk two")

    def test_retrieve_empty_index(self):
        self.assertEqual(self.retriever.retrieve("query"), [])

    def test_setup_first_run_builds_and_persists(self):
        self._write("a.md", "alpha content")
        self._write("b.md", "beta content")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            node_a = MagicMock()
            node_a.text = "alpha content"
            node_a.metadata = {"file_name": "a.md"}
            node_b = MagicMock()
            node_b.text = "beta content"
            node_b.metadata = {"file_name": "b.md"}
            splitter.get_nodes_from_documents.return_value = [node_a, node_b]
            self.retriever.setup()

        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.index.ntotal, 2)
        self.assertEqual(len(self.retriever.chunks), 2)
        self.assertIn("a.md", self.retriever.state)
        self.assertIn("b.md", self.retriever.state)
        self.assertTrue((self.cache_dir / "ingestion_state.json").exists())
        self.assertTrue((self.cache_dir / "faiss.index").exists())
        self.assertTrue((self.cache_dir / "chunks.pkl").exists())

    def test_setup_second_run_skips_unchanged_files(self):
        self._write("a.md", "alpha content")
        self._write("b.md", "beta content")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            node_a = MagicMock()
            node_a.text = "alpha content"
            node_a.metadata = {"file_name": "a.md"}
            node_b = MagicMock()
            node_b.text = "beta content"
            node_b.metadata = {"file_name": "b.md"}
            splitter.get_nodes_from_documents.return_value = [node_a, node_b]
            self.retriever.setup()

            splitter.get_nodes_from_documents.reset_mock()

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunks = []
            second.state = {}
            second.setup()

        self.assertEqual(splitter.get_nodes_from_documents.call_count, 0)
        self.assertEqual(second.embed_model.get_text_embedding.call_count, 0)
        self.assertEqual(second.index.ntotal, 2)
        self.assertEqual(len(second.chunks), 2)

    def test_setup_processes_only_changed_file(self):
        self._write("a.md", "alpha")
        self._write("b.md", "beta")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            node_a = MagicMock()
            node_a.text = "alpha"
            node_a.metadata = {"file_name": "a.md"}
            node_b = MagicMock()
            node_b.text = "beta"
            node_b.metadata = {"file_name": "b.md"}
            splitter.get_nodes_from_documents.return_value = [node_a, node_b]
            self.retriever.setup()

            self._write("b.md", "beta updated content")
            node_b2 = MagicMock()
            node_b2.text = "beta updated content"
            node_b2.metadata = {"file_name": "b.md"}
            splitter.get_nodes_from_documents.return_value = [node_b2]

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunks = []
            second.state = {}
            second.setup()

        self.assertEqual(second.index.ntotal, 3)
        self.assertEqual(len(second.chunks), 3)
        self.assertEqual(second.chunks[2]["text"], "beta updated content")

    def test_setup_handles_new_file(self):
        self._write("a.md", "alpha")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            node_a = MagicMock()
            node_a.text = "alpha"
            node_a.metadata = {"file_name": "a.md"}
            splitter.get_nodes_from_documents.return_value = [node_a]
            self.retriever.setup()

            self._write("c.md", "gamma new")
            node_c = MagicMock()
            node_c.text = "gamma new"
            node_c.metadata = {"file_name": "c.md"}
            splitter.get_nodes_from_documents.return_value = [node_c]

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunks = []
            second.state = {}
            second.setup()

        self.assertEqual(second.index.ntotal, 2)
        self.assertEqual(len(second.chunks), 2)

    def test_clear_cache_removes_persisted_files(self):
        self._write("a.md", "alpha")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed
        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            node_a = MagicMock()
            node_a.text = "alpha"
            node_a.metadata = {"file_name": "a.md"}
            splitter.get_nodes_from_documents.return_value = [node_a]
            self.retriever.setup()

        self.assertTrue(self.retriever._state_path().exists())
        self.retriever.clear_cache()
        self.assertFalse(self.retriever._state_path().exists())
        self.assertFalse(self.retriever._index_path().exists())
        self.assertFalse(self.retriever._chunks_path().exists())


if __name__ == "__main__":
    unittest.main()
