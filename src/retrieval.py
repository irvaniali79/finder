import os
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
import faiss
import numpy as np


class DocumentRetriever:
    def __init__(self, docs_path="docs/", embed_model_name="all-MiniLM-L6-v2"):
        self.docs_path = Path(docs_path)
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.index = None
        self.chunks = []

    def load_documents(self):
        documents = []
        if not self.docs_path.exists():
            return documents
        for path in sorted(self.docs_path.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".md", ".txt"}:
                text = path.read_text(encoding="utf-8")
                documents.append(Document(text=text, metadata={"file_name": path.name}))
            elif suffix == ".pdf":
                documents.extend(PDFReader().load_data(path))
        return documents

    def chunk_documents(self, documents):
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        self.chunks = nodes
        return nodes

    def create_embeddings(self, nodes):
        embeddings = [self.embed_model.get_text_embedding(node.text) for node in nodes]
        return np.array(embeddings).astype("float32")

    def build_index(self, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        query_emb = self.embed_model.get_text_embedding(query)
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.chunks):
                results.append((self.chunks[idx].text, float(distances[0][i])))
        return results

    def setup(self):
        documents = self.load_documents()
        nodes = self.chunk_documents(documents)
        embeddings = self.create_embeddings(nodes)
        self.build_index(embeddings)
        return self
