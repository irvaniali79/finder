import json
import sqlite3
from pathlib import Path


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
