# Agent Context

## Project Overview
Mini Company Knowledge Bot - a RAG-based CLI Q&A system backed by the docs/ knowledge base.

## Current State
- Retrieval core is functional: `src/retrieval.py`, `src/agent.py`, `src/main.py`, `src/pipeline.py`, `src/preprocessor.py`
- Document ingestion flow supports `.txt`, `.md`, and `.pdf` files directly under `docs/`
- OpenRouter is used as the LLM provider instead of Ollama
- Ingestion is incremental: `DocumentRetriever.setup()` fingerprints each file in `docs/`, persists the FAISS index, chunks, and per-file state to `.cache/`, and only re-processes files that are new or changed on subsequent runs
- Chunk text is stored in a SQLite DB (`.cache/chunks.db`) via the `ChunkStore` class, not in RAM — keeps the process usable on large corpora
- `Pipeline` class in `src/pipeline.py` implements the DESIGN.md ablation harness with variants V0–V5; V0 is functional, V4 and V5 now run end-to-end via the two-stage rerank path, V1–V3 still raise `NotImplementedError` and will be filled in by F13
- `Pipeline.retrieve_top_n`, `Pipeline.rerank` (static, takes optional `importance_fn`), and `Pipeline.retrieve_top_k` provide the two-stage retrieval: vector search for top-N (default 20) → heuristic rerank (similarity + importance + file diversity penalty) → top-K (default 5)
- Query preprocessing (`src/preprocessor.py`) provides `extract_tags`, `expand_query` (≤3 variants), `detect_question_intent`, and `combine_query_embeddings` (centroid of variants); `Pipeline._build_query_embedding` uses expansion when the `query_expansion` feature is enabled
- `.cache/` is git-ignored; use `DocumentRetriever.clear_cache()` to force a full rebuild
- Test coverage in `tests/test_retrieval.py`, `tests/test_agent.py`, `tests/test_main.py`, `tests/test_evals.py`, `tests/test_pipeline.py`

## Notable Constraints
- Local development tests require installed packages (`pytest`, `numpy`, `faiss-cpu`, etc.)
- OpenRouter endpoint is configurable via `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_API_BASE` env vars (defaults to `openrouter/free` model and `https://openrouter.ai/api/v1/chat/completions`)
- Required env vars are documented in `.env` at the repo root: `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_API_BASE`
- 4 pre-existing test failures in retrieval/eval suites due to outdated mocks from prior ingestion flow refactor; unrelated to current features
