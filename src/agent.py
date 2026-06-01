from llama_index.llms.ollama import Ollama
from typing import List, Tuple

class QAAgent:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.llm = Ollama(model=model_name, base_url=base_url, request_timeout=120.0)
    
    def generate_prompt(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        context_text = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        prompt = f"""You are a helpful company knowledge assistant. Answer the question based ONLY on the provided context.
        
Context:
{context_text}

Question: {query}

Answer concisely and directly, based only on the information in the context above."""
        return prompt
    
    def answer(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        prompt = self.generate_prompt(query, context_chunks)
        response = self.llm.complete(prompt)
        return response.text