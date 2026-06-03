# Agent Context

## Project Overview
Mini Company Knowledge Bot - a RAG-based CLI Q&A system backed by the docs/ knowledge base.

## Current State
- Retrieval core is functional: `src/retrieval.py`, `src/agent.py`, `src/main.py`
- Document ingestion flow supports `.txt` and `.pdf` files directly under `docs/`
- Test coverage updated in `tests/test_retrieval.py`
- Agentic-brain docs updated to reflect the ingestion change
- Remaining work: none pending user-requested changes

## Notable Constraints
- Local development tests require installed packages (`pytest`, `numpy`, `faiss-cpu`, etc.)
