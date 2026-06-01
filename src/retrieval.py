import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import numpy as np
from typing import List, Tuple

class DocumentRetriever:
    def __init__(self, docs_path: str = "docs/", embed_model_name: str = "all-MiniLM-L6-v2"):
        self.docs_path = docs_path
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.index = None
        self.chunks = []
        
    def load_documents(self):
        reader = SimpleDirectoryReader(self.docs_path, required_exts=[".md"])
        documents = reader.load_data()
        return documents
    
    def chunk_documents(self, documents) -> List:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes(documents)
        self.chunks = nodes
        return nodes
    
    def create_embeddings(self, nodes) -> np.ndarray:
        embeddings = []
        for node in nodes:
            emb = self.embed_model.get_text_embedding(node.text)
            embeddings.append(emb)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_emb = self.embed_model.get_text_embedding(query)
        query_vector = np.array([query_emb]).astype('float32')
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx].text, float(distances[0][i])))
        return results
    
    def setup(self):
        documents = self.load_documents()
        nodes = self.chunk_documents(documents)
        embeddings = self.create_embeddings(nodes)
        self.build_index(embeddings)
        return self