# Agent Context


## Project Overview
Mini Company Knowledge Bot - A RAG-based Q&A system for internal company documentation.

## Current State
- Core implementation complete: retrieval.py, agent.py, main.py
- Company documents: FAQ.md, leave_policy.md, product_guide.md
- Evaluation-related work: tests/test_evals.py exists; full LLM-dependent evaluation requires Ollama.
- Remaining work: None. All features complete. Follow task execution protocol from AGENTS.md and subagent protocol from SUBAGENT_PROTOCOL.md.

## Tech Decisions
See `/TECHDECISIONS.md` for the chosen RAG architecture, query flow, and evaluation strategy.

## How to test
```bash
pytest
pytest tests/test_evals.py
```

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