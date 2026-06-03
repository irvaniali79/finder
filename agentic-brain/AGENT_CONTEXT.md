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
- Ollama endpoint is configurable via `OLLAMA_BASE_URL` and `OLLAMA_MODEL` env vars (defaults to `http://localhost:11434` and `llama3.2`)
- Note: 4 pre-existing test failures in retrieval/eval suites due to outdated mocks from prior ingestion flow refactor; unrelated to this feature
