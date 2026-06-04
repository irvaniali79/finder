import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import ChunkStore, DocumentRetriever


class TestChunkStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "chunks.db"

    def tearDown(self):
        if hasattr(self, "store") and self.store is not None:
            self.store.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_returns_sequential_ids(self):
        self.store = ChunkStore(self.db_path)
        ids = self.store.add([
            {"text": "a", "metadata": {"file_name": "f1.md"}},
            {"text": "b", "metadata": {"file_name": "f1.md"}},
        ])
        self.assertEqual(ids, [0, 1])
        self.assertEqual(self.store.count(), 2)

    def test_add_empty_returns_empty(self):
        self.store = ChunkStore(self.db_path)
        self.assertEqual(self.store.add([]), [])
        self.assertEqual(self.store.count(), 0)

    def test_get_returns_chunk(self):
        self.store = ChunkStore(self.db_path)
        ids = self.store.add([{"text": "hello", "metadata": {"file_name": "x.md"}}])
        chunk = self.store.get(ids[0])
        self.assertEqual(chunk["text"], "hello")
        self.assertEqual(chunk["file_name"], "x.md")

    def test_get_missing_returns_none(self):
        self.store = ChunkStore(self.db_path)
        self.assertIsNone(self.store.get(999))

    def test_get_many_preserves_order(self):
        self.store = ChunkStore(self.db_path)
        ids = self.store.add([
            {"text": "zero", "metadata": {"file_name": "a.md"}},
            {"text": "one", "metadata": {"file_name": "a.md"}},
            {"text": "two", "metadata": {"file_name": "a.md"}},
        ])
        result = self.store.get_many([ids[2], ids[0], ids[1]])
        self.assertEqual([c["text"] for c in result], ["two", "zero", "one"])

    def test_get_many_handles_missing_ids(self):
        self.store = ChunkStore(self.db_path)
        ids = self.store.add([{"text": "x", "metadata": {"file_name": "a.md"}}])
        result = self.store.get_many([ids[0], 999])
        self.assertEqual(result[0]["text"], "x")
        self.assertIsNone(result[1])

    def test_get_many_empty(self):
        self.store = ChunkStore(self.db_path)
        self.assertEqual(self.store.get_many([]), [])

    def test_clear_resets_ids(self):
        self.store = ChunkStore(self.db_path)
        self.store.add([{"text": "x", "metadata": {"file_name": "a.md"}}])
        self.store.clear()
        self.assertEqual(self.store.count(), 0)
        ids = self.store.add([{"text": "y", "metadata": {"file_name": "a.md"}}])
        self.assertEqual(ids, [0])

    def test_persists_across_instances(self):
        self.store = ChunkStore(self.db_path)
        self.store.add([
            {"text": "alpha", "metadata": {"file_name": "a.md"}},
            {"text": "beta", "metadata": {"file_name": "b.md"}},
        ])
        self.store.close()
        self.store = None

        store2 = ChunkStore(self.db_path)
        self.assertEqual(store2.count(), 2)
        self.assertEqual(store2.get(0)["text"], "alpha")
        store2.close()
        self.store = None


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
        self.retriever.chunk_store = None
        self.retriever.state = {}

    def tearDown(self):
        if self.retriever.chunk_store is not None:
            self.retriever.chunk_store.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, content="hello"):
        path = self.docs_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def _make_fake_node(self, text, file_name):
        node = MagicMock()
        node.text = text
        node.metadata = {"file_name": file_name}
        return node

    def _mock_store(self, chunks_data):
        store = MagicMock()
        store.count.return_value = len(chunks_data)
        store.get_many.side_effect = lambda ids: [
            chunks_data[i] if 0 <= i < len(chunks_data) else None for i in ids
        ]
        store.add.side_effect = lambda cs: list(range(len(cs)))
        return store

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
        node1 = self._make_fake_node("n1 text", "a.md")
        node2 = self._make_fake_node("n2 text", "a.md")
        mock_splitter.get_nodes_from_documents.return_value = [node1, node2]
        with patch("retrieval.SentenceSplitter", return_value=mock_splitter):
            chunks = self.retriever.chunk_documents(["d1"])
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["text"], "n1 text")
            self.assertEqual(chunks[0]["metadata"], {"file_name": "a.md"})

    def test_chunk_documents_empty_input(self):
        self.assertEqual(self.retriever.chunk_documents([]), [])

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
        self.retriever.chunk_store = self._mock_store([
            {"text": "chunk one", "file_name": "a.md"},
            {"text": "chunk two", "file_name": "a.md"},
        ])
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
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha content", "a.md"),
                self._make_fake_node("beta content", "b.md"),
            ]
            self.retriever.setup()

        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.index.ntotal, 2)
        self.assertIsNotNone(self.retriever.chunk_store)
        self.assertEqual(self.retriever.chunk_store.count(), 2)
        self.assertIn("a.md", self.retriever.state)
        self.assertIn("b.md", self.retriever.state)
        self.assertTrue((self.cache_dir / "ingestion_state.json").exists())
        self.assertTrue((self.cache_dir / "faiss.index").exists())
        self.assertTrue((self.cache_dir / "chunks.db").exists())

    def test_setup_second_run_skips_unchanged_files(self):
        self._write("a.md", "alpha content")
        self._write("b.md", "beta content")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha content", "a.md"),
                self._make_fake_node("beta content", "b.md"),
            ]
            self.retriever.setup()
            splitter.get_nodes_from_documents.reset_mock()

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunk_store = None
            second.state = {}
            second.setup()

        self.assertEqual(splitter.get_nodes_from_documents.call_count, 0)
        self.assertEqual(second.embed_model.get_text_embedding.call_count, 0)
        self.assertEqual(second.index.ntotal, 2)
        self.assertEqual(second.chunk_store.count(), 2)

    def test_setup_processes_only_changed_file(self):
        self._write("a.md", "alpha")
        self._write("b.md", "beta")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha", "a.md"),
                self._make_fake_node("beta", "b.md"),
            ]
            self.retriever.setup()

            self._write("b.md", "beta updated content")
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("beta updated content", "b.md")
            ]

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunk_store = None
            second.state = {}
            second.setup()

        self.assertEqual(second.index.ntotal, 3)
        self.assertEqual(second.chunk_store.count(), 3)
        self.assertEqual(second.chunk_store.get(2)["text"], "beta updated content")

    def test_setup_handles_new_file(self):
        self._write("a.md", "alpha")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed

        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha", "a.md")
            ]
            self.retriever.setup()

            self._write("c.md", "gamma new")
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("gamma new", "c.md")
            ]

            second = DocumentRetriever.__new__(DocumentRetriever)
            second.docs_path = self.docs_dir
            second.cache_dir = self.cache_dir
            second.embed_model = MagicMock()
            second.embed_model.get_text_embedding.side_effect = fake_embed
            second.index = None
            second.chunk_store = None
            second.state = {}
            second.setup()

        self.assertEqual(second.index.ntotal, 2)
        self.assertEqual(second.chunk_store.count(), 2)

    def test_setup_rebuilds_when_index_and_store_mismatch(self):
        self._write("a.md", "alpha")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed
        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha", "a.md")
            ]
            self.retriever.setup()

        import sqlite3
        conn = sqlite3.connect(str(self.cache_dir / "chunks.db"))
        conn.execute("DELETE FROM chunks WHERE id = 0")
        conn.commit()
        conn.close()

        second = DocumentRetriever.__new__(DocumentRetriever)
        second.docs_path = self.docs_dir
        second.cache_dir = self.cache_dir
        second.embed_model = MagicMock()
        second.embed_model.get_text_embedding.side_effect = fake_embed
        second.index = None
        second.chunk_store = None
        second.state = {}
        with patch("retrieval.SentenceSplitter") as MockSplitter2:
            splitter2 = MockSplitter2.return_value
            splitter2.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha", "a.md")
            ]
            second.setup()

        self.assertEqual(second.index.ntotal, 1)
        self.assertEqual(second.chunk_store.count(), 1)

    def test_clear_cache_removes_persisted_files(self):
        self._write("a.md", "alpha")

        def fake_embed(text):
            return [float(len(text)), 1.0]

        self.retriever.embed_model.get_text_embedding.side_effect = fake_embed
        with patch("retrieval.SentenceSplitter") as MockSplitter:
            splitter = MockSplitter.return_value
            splitter.get_nodes_from_documents.return_value = [
                self._make_fake_node("alpha", "a.md")
            ]
            self.retriever.setup()

        self.assertTrue(self.retriever._state_path().exists())
        self.assertTrue(self.retriever._chunk_db_path().exists())
        self.assertTrue(self.retriever._index_path().exists())

        self.retriever.chunk_store.close()
        self.retriever.clear_cache()

        self.assertFalse(self.retriever._state_path().exists())
        self.assertFalse(self.retriever._index_path().exists())
        self.assertFalse(self.retriever._chunk_db_path().exists())

    def test_chunks_not_held_in_ram_list(self):
        self.assertFalse(hasattr(self.retriever, "chunks") and isinstance(self.retriever.chunks, list))


if __name__ == "__main__":
    unittest.main()
