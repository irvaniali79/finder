# Agent Context
See SUBAGENT_PROTOCOL.md for rules on when and how to delegate feature tasks to subagents.


## Project Overview
Mini Company Knowledge Bot - A RAG-based Q&A system for internal company documentation.

## Current State
- Core implementation complete: retrieval.py, agent.py, main.py
- Company documents: FAQ.md, leave_policy.md, product_guide.md
- Ready for testing with Ollama

## Tech Stack
- LlamaIndex for document processing (SimpleDirectoryReader, SentenceSplitter)
- FAISS for vector storage (IndexFlatL2)
- Ollama (llama3.2) for LLM
- HuggingFace embeddings (all-MiniLM-L6-v2)

## Usage
```bash
pip install -r requirements.txt
ollama serve  # Ensure Ollama is running
python src/main.py
```

## Key Files Structure
- `docs/` - Company documents (FAQ, leave policy, product guide)
- `src/retrieval.py` - Document loading, chunking, embedding, similarity search
- `src/agent.py` - Prompt construction, LLM call, answer generation  
- `src/main.py` - CLI entry point