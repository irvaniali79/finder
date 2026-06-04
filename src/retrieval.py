import hashlib
import json
import pickle
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
import faiss
import numpy as np


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


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
        self.chunks = []
        self.state = {}

    def _ensure_cache_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self):
        return self.cache_dir / "ingestion_state.json"

    def _index_path(self):
        return self.cache_dir / "faiss.index"

    def _chunks_path(self):
        return self.cache_dir / "chunks.pkl"

    @staticmethod
    def compute_file_fingerprint(path):
        stat = path.stat()
        h = hashlib.sha256()
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        h.update(path.read_bytes())
        return h.hexdigest()

    def _list_supported_files(self):
        if not self.docs_path.exists():
            return []
        files = []
        for path in sorted(self.docs_path.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
        return files

    def _read_file_as_documents(self, path):
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            return [Document(text=text, metadata={"file_name": path.name})]
        if suffix == ".pdf":
            return list(PDFReader().load_data(path))
        return []

    def load_documents(self):
        documents = []
        for path in self._list_supported_files():
            documents.extend(self._read_file_as_documents(path))
        return documents

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
        with open(self._chunks_path(), "wb") as f:
            pickle.dump(self.chunks, f)

    def _restore_index(self):
        index_path = self._index_path()
        chunks_path = self._chunks_path()
        if not (index_path.exists() and chunks_path.exists()):
            return False
        try:
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        except Exception:
            self.index = None
            self.chunks = []
            return False

    def clear_cache(self):
        self.state = {}
        self.chunks = []
        self.index = None
        for path in (self._state_path(), self._index_path(), self._chunks_path()):
            if path.exists():
                path.unlink()

    def chunk_documents(self, documents):
        if not documents:
            return []
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        chunk_dicts = []
        for node in nodes:
            metadata = dict(node.metadata) if node.metadata else {}
            chunk_dicts.append({"text": node.text, "metadata": metadata})
        self.chunks = chunk_dicts
        return chunk_dicts

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
        if self.index is None or not self.chunks:
            return []
        query_emb = self.embed_model.get_text_embedding(query)
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx]["text"], float(distances[0][i])))
        return results

    def _build_full_index(self):
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
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
        documents = []
        for path in files:
            documents.extend(self._read_file_as_documents(path))
        if not documents:
            return
        new_chunks = []
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        for node in nodes:
            metadata = dict(node.metadata) if node.metadata else {}
            new_chunks.append({"text": node.text, "metadata": metadata})

        embeddings = self.create_embeddings(new_chunks)
        self.build_index(embeddings)
        self.chunks.extend(new_chunks)

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
