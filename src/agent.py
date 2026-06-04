import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class QAAgent:
    def __init__(self, model_name=None, api_key=None, api_base=None):
        self.model_name = model_name or os.environ.get("OPENROUTER_MODEL", "openrouter/free")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set.")
        self.api_base = api_base or os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1/chat/completions")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Company Knowledge Bot"
        })
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
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
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = self.session.post(self.api_base, json=payload, timeout=120.0)
        response.raise_for_status()
        
        try:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to parse LLM response: {response.text}") from e