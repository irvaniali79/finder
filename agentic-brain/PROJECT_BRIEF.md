# Mini Company Knowledge Bot - Project Brief

## What We Built
A RAG-based Q&A system that answers questions using company documents. The system provides a CLI interface for users to query internal knowledge base (FAQ, leave policy, product guide).

## MVP Scope
- `docs/`: 3 plain-text files (FAQ, Leave Policy, Product Guide)
- `src/`: Basic RAG implementation using LlamaIndex + FAISS + OpenRouter
- `main.py`: CLI entry point for Q&A
- `agentic-brain/`: Project documentation and evaluation

## Architecture
- Document Ingestion: LlamaIndex with SimpleDirectoryReader and SentenceSplitter
- Vector Database: FAISS for local similarity search
- LLM & Embeddings: OpenRouter API + Sentence-Transformers
- Entry Point: Python CLI application