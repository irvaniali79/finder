import hashlib
import json
import re
import sqlite3
from pathlib import Path
from llama_index.core import Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
import faiss
import numpy as np

from src.preprocessor import extract_tags


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_WEIGHT = 0.5
_LENGTH_WEIGHT = 0.5
_MAX_IMPORTANCE = 1.0


def extract_chunk_metadata(text, file_name=""):
    if not text:
        return {"tags": [], "summary": "", "importance": 0.0, "file_name": file_name or ""}
    tags = extract_tags(text)
    first_sentence = _SENTENCE_SPLIT_RE.split(text.strip(), maxsplit=1)[0].strip()
    summary = first_sentence[:200]
    word_count = len(re.findall(r"\b\w+\b", text))
    length_score = min(1.0, word_count / 100.0)
    tag_score = min(1.0, len(tags) / 5.0)
    importance = min(_MAX_IMPORTANCE, _LENGTH_WEIGHT * length_score + _TAG_WEIGHT * tag_score)
    return {
        "tags": tags,
        "summary": summary,
        "importance": float(importance),
        "file_name": file_name or "",
    }


class ChunkStore:
    METADATA_COLUMNS = ("tags", "summary", "importance", "file_name")

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                file_name TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name)"
        )
        self._migrate_add_metadata_columns()
        self._conn.commit()
        self._next_id = int(
            self._conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM chunks").fetchone()[0]
        )

    def _migrate_add_metadata_columns(self):
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "tags" not in existing:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN tags TEXT DEFAULT '[]'")
        if "summary" not in existing:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN summary TEXT DEFAULT ''")
        if "importance" not in existing:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN importance REAL DEFAULT 0.0")

    @staticmethod
    def _encode_metadata(chunk):
        metadata = chunk.get("metadata", {}) or {}
        return {
            "text": chunk["text"],
            "file_name": metadata.get("file_name", ""),
            "tags": json.dumps(metadata.get("tags", [])),
            "summary": metadata.get("summary", ""),
            "importance": float(metadata.get("importance", 0.0)),
        }

    def add(self, chunks):
        if not chunks:
            return []
        encoded = [self._encode_metadata(c) for c in chunks]
        rows = [
            (
                self._next_id + i,
                e["text"],
                e["file_name"],
                e["tags"],
                e["summary"],
                e["importance"],
            )
            for i, e in enumerate(encoded)
        ]
        self._conn.executemany(
            "INSERT INTO chunks (id, text, file_name, tags, summary, importance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        ids = [r[0] for r in rows]
        self._next_id += len(rows)
        self._conn.commit()
        return ids

    def _row_to_chunk(self, row):
        if row is None:
            return None
        chunk_id, text, file_name, tags_json, summary, importance = row
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (TypeError, ValueError):
            tags = []
        return {
            "text": text,
            "file_name": file_name,
            "tags": tags,
            "summary": summary or "",
            "importance": float(importance) if importance is not None else 0.0,
        }

    def get(self, id):
        row = self._conn.execute(
            "SELECT id, text, file_name, tags, summary, importance "
            "FROM chunks WHERE id = ?",
            (id,),
        ).fetchone()
        return self._row_to_chunk(row)

    def get_many(self, ids):
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, text, file_name, tags, summary, importance "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        id_to_chunk = {r[0]: self._row_to_chunk(r) for r in rows}
        return [id_to_chunk.get(i) for i in ids]

    def count(self):
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def clear(self):
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self._next_id = 0

    def close(self):
        self._conn.close()


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
        if not documents:
            return []
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        chunk_dicts = []
        for node in nodes:
            metadata = dict(node.metadata) if node.metadata else {}
            file_name = metadata.get("file_name", "")
            extracted = extract_chunk_metadata(node.text, file_name)
            merged = {**metadata, **extracted}
            chunk_dicts.append({"text": node.text, "metadata": merged})
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
                results.append((chunk["text"], float(dist)))
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
        new_chunks = []
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        for node in nodes:
            metadata = dict(node.metadata) if node.metadata else {}
            file_name = metadata.get("file_name", "")
            extracted = extract_chunk_metadata(node.text, file_name)
            merged = {**metadata, **extracted}
            new_chunks.append({"text": node.text, "metadata": merged})

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
