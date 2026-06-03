import os
from llama_index.llms.ollama import Ollama


class QAAgent:
    def __init__(self, model_name=None, base_url=None):
        model_name = model_name or os.environ.get("OLLAMA_MODEL", "llama3.2")
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = Ollama(model=model_name, base_url=base_url, request_timeout=120.0)
    
    def generate_prompt(self, query, context_chunks):
        context_text = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        prompt = """You are a helpful company knowledge assistant. Answer the question based ONLY on the provided context.

Context:
{context_text}

Question: {query}

Answer concisely and directly, based only on the information in the context above.""".format(context_text=context_text, query=query)
        return prompt
    
    def answer(self, query, context_chunks):
        prompt = self.generate_prompt(query, context_chunks)
        response = self.llm.complete(prompt)
        return response.text