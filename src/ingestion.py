import hashlib
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.readers.file import PDFReader

from src.query import extract_tags


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


def compute_file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    h.update(path.read_bytes())
    return h.hexdigest()


def list_supported_files(docs_path):
    docs_path = Path(docs_path)
    if not docs_path.exists():
        return []
    files = []
    for path in sorted(docs_path.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def read_file_as_documents(path):
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [Document(text=text, metadata={"file_name": path.name})]
    if suffix == ".pdf":
        return list(PDFReader().load_data(path))
    return []


def load_documents(docs_path):
    documents = []
    for path in list_supported_files(docs_path):
        documents.extend(read_file_as_documents(path))
    return documents


def chunk_documents(documents):
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
