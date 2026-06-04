# Agent Context

## Project Overview
Mini Company Knowledge Bot - a RAG-based CLI Q&A system backed by the docs/ knowledge base.

## Current State
- Retrieval core is functional: `src/retrieval.py`, `src/agent.py`, `src/main.py`
- Document ingestion flow supports `.txt`, `.md`, and `.pdf` files directly under `docs/`
- OpenRouter is used as the LLM provider instead of Ollama
- Ingestion is incremental: `DocumentRetriever.setup()` fingerprints each file in `docs/`, persists the FAISS index, chunks, and per-file state to `.cache/`, and only re-processes files that are new or changed on subsequent runs
- `.cache/` is git-ignored; use `DocumentRetriever.clear_cache()` to force a full rebuild
- Test coverage updated in `tests/test_retrieval.py` and `tests/test_agent.py`

## Notable Constraints
- Local development tests require installed packages (`pytest`, `numpy`, `faiss-cpu`, etc.)
- OpenRouter endpoint is configurable via `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_API_BASE` env vars (defaults to `openrouter/free` model and `https://openrouter.ai/api/v1/chat/completions`)
- Required env vars are documented in `.env` at the repo root: `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_API_BASE`
- 4 pre-existing test failures in retrieval/eval suites due to outdated mocks from prior ingestion flow refactor; unrelated to current features
