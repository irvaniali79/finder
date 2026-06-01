# Agent Context

## Project Overview
Mini Company Knowledge Bot - A RAG-based Q&A system for internal company documentation.

## Current State
- Project initialized, ready for implementation
- No code written yet, no commits made

## Tech Stack
- LlamaIndex for document processing
- FAISS for vector storage
- Ollama + Sentence-Transformers for LLM/embeddings

## Key Files Structure
- `docs/` - Company documents (FAQ, leave policy, product guide)
- `src/retrieval.py` - Document loading, chunking, embedding, similarity search
- `src/agent.py` - Prompt construction, LLM call, answer generation
- `src/main.py` - CLI entry point