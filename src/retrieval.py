import json
import sqlite3
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import numpy as np

from src.ingestion import (
    SUPPORTED_SUFFIXES,
    chunk_documents as _chunk_documents,
    compute_file_fingerprint as _compute_file_fingerprint,
    list_supported_files as _list_supported_files,
    load_documents as _load_documents,
    read_file_as_documents as _read_file_as_documents,
)
from src.storage import ChunkStore


class DocumentRetriever:
    def __init__(
        self,
        docs_path="docs/",
        embed_model_name="all-MiniLM-L6-v2",
        cache_dir=".cache",
    ):
        self.docs_path = Path(docs_path)
        self.cache_dir = Path(cache_dir)
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.index = None
        self.chunk_store = None
        self.state = {}

    def _ensure_cache_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self):
        return self.cache_dir / "ingestion_state.json"

    def _index_path(self):
        return self.cache_dir / "faiss.index"

    def _chunk_db_path(self):
        return self.cache_dir / "chunks.db"

    def _open_chunk_store(self):
        if self.chunk_store is None:
            self.chunk_store = ChunkStore(self._chunk_db_path())
        return self.chunk_store

    @staticmethod
    def compute_file_fingerprint(path):
        return _compute_file_fingerprint(path)

    def _list_supported_files(self):
        return _list_supported_files(self.docs_path)

    def _read_file_as_documents(self, path):
        return _read_file_as_documents(path)

    def load_documents(self):
        return _load_documents(self.docs_path)

    def _load_state(self):
        path = self._state_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {}

    def _save_state(self):
        path = self._state_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)

    def _persist_index(self):
        if self.index is None:
            return
        faiss.write_index(self.index, str(self._index_path()))

    def _restore_index(self):
        index_path = self._index_path()
        if not index_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(index_path))
            self._open_chunk_store()
            if self.chunk_store.count() != self.index.ntotal:
                self.index = None
                self.chunk_store.clear()
                self.chunk_store = None
                return False
            return True
        except Exception:
            self.index = None
            if self.chunk_store is not None:
                self.chunk_store.clear()
                self.chunk_store = None
            return False

    def clear_cache(self):
        self.state = {}
        if self.chunk_store is not None:
            try:
                self.chunk_store.clear()
            except sqlite3.ProgrammingError:
                pass
            self.chunk_store = None
        self.index = None
        for path in (self._state_path(), self._index_path(), self._chunk_db_path()):
            if path.exists():
                path.unlink()
        legacy_pkl = self.cache_dir / "chunks.pkl"
        if legacy_pkl.exists():
            legacy_pkl.unlink()

    def chunk_documents(self, documents):
        return _chunk_documents(documents)

    def create_embeddings(self, chunks):
        embeddings = [self.embed_model.get_text_embedding(c["text"]) for c in chunks]
        return np.array(embeddings).astype("float32")

    def build_index(self, embeddings):
        if embeddings.size == 0:
            return
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        if self.index is None or self.chunk_store is None or self.chunk_store.count() == 0:
            return []
        query_emb = self.embed_model.get_text_embedding(query)
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        ids = [int(i) for i in indices[0]]
        chunks = self.chunk_store.get_many(ids)
        results = []
        for chunk, dist in zip(chunks, distances[0]):
            if chunk is not None:
                metadata = {
                    "file_name": chunk.get("file_name", ""),
                    "tags": chunk.get("tags", []) or [],
                    "summary": chunk.get("summary", "") or "",
                    "importance": float(chunk.get("importance", 0.0) or 0.0),
                }
                results.append((chunk["text"], float(dist), metadata))
        return results

    def _build_full_index(self):
        store = self._open_chunk_store()
        store.clear()
        self._next_id = 0

        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        ids = store.add(chunks)
        embeddings = self.create_embeddings(chunks)
        self.index = None
        self.build_index(embeddings)

        file_chunks = {}
        for chunk in chunks:
            name = chunk["metadata"].get("file_name")
            file_chunks[name] = file_chunks.get(name, 0) + 1

        new_state = {}
        for path in self._list_supported_files():
            new_state[path.name] = {
                "fingerprint": self.compute_file_fingerprint(path),
                "chunk_count": file_chunks.get(path.name, 0),
            }
        self.state = new_state

    def _append_chunks_for_files(self, files):
        store = self._open_chunk_store()
        documents = []
        for path in files:
            documents.extend(self._read_file_as_documents(path))
        if not documents:
            return
        new_chunks = self.chunk_documents(documents)

        store.add(new_chunks)
        embeddings = self.create_embeddings(new_chunks)
        self.build_index(embeddings)

        counts = {}
        for chunk in new_chunks:
            name = chunk["metadata"].get("file_name")
            counts[name] = counts.get(name, 0) + 1
        for path in files:
            self.state[path.name] = {
                "fingerprint": self.compute_file_fingerprint(path),
                "chunk_count": counts.get(path.name, 0),
            }

    def setup(self):
        self._ensure_cache_dir()
        self._load_state()
        restored = self._restore_index()

        files = self._list_supported_files()
        current_names = {p.name for p in files}
        known_names = set(self.state.keys())

        if not restored or (known_names - current_names):
            self._build_full_index()
        else:
            changed = []
            for path in files:
                fp = self.compute_file_fingerprint(path)
                prev = self.state.get(path.name, {})
                if prev.get("fingerprint") != fp:
                    changed.append(path)
            if changed:
                self._append_chunks_for_files(changed)

        self._save_state()
        self._persist_index()
        return self
