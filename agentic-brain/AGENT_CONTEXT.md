# Agent Context

## Project Overview
Mini Company Knowledge Bot - a RAG-based CLI Q&A system backed by the docs/ knowledge base.

## Current State
- Retrieval core is functional: `src/retrieval.py`, `src/agent.py`, `src/main.py`, `src/pipeline.py`, `src/preprocessor.py`, `src/storage.py`
- Document ingestion flow supports `.txt`, `.md`, and `.pdf` files directly under `docs/`
- OpenRouter is used as the LLM provider instead of Ollama
- Ingestion is incremental: `DocumentRetriever.setup()` fingerprints each file in `docs/`, persists the FAISS index, chunks, and per-file state to `.cache/`, and only re-processes files that are new or changed on subsequent runs
- Chunk text is stored in a SQLite DB (`.cache/chunks.db`) via the `ChunkStore` class in `src/storage.py`; each chunk now has `tags`, `summary`, `importance` metadata extracted at ingestion time
- `Pipeline` variants V0–V5 now run end-to-end:
  - V0: pure vector search
  - V1: vector search + tag filtering (drops candidates with zero tag overlap)
  - V2: V1 + summary-match boost (reduces distance when summary contains query words)
  - V3: V2 + query expansion (centroid of 2-3 variants)
  - V4: V3 + two-stage rerank (similarity + importance + diversity penalty)
  - V5: V4 + per-chunk importance weighting from metadata
- `Pipeline.retrieve_top_n`, `Pipeline.rerank`, and `Pipeline.retrieve_top_k` provide the two-stage retrieval path
- `DocumentRetriever.retrieve()`, `Pipeline.retrieve()`, and `Pipeline.retrieve_top_k()` return 3-tuples: `(text, distance, metadata)`, which preserves ingestion-time metadata (`file_name`, `summary`, `tags`, `importance`) and propagates it all the way to `QAAgent` for prompt formatting in production.
- Query preprocessing (`src/preprocessor.py`) provides `extract_tags`, `expand_query` (≤3 variants), `detect_question_intent`, and `combine_query_embeddings`
- `.cache/` is git-ignored; use `DocumentRetriever.clear_cache()` to force a full rebuild
- Test coverage in `tests/test_retrieval.py`, `tests/test_agent.py`, `tests/test_main.py`, `tests/test_evals.py`, `tests/test_pipeline.py` (103 tests passing)

## Pending Integration Tasks
- **Feature 18 Sub-task B**: Extract Ingestion into `src/ingestion.py` — move `load_documents`, `_read_file_as_documents`, `_list_supported_files`, `chunk_documents`, `extract_chunk_metadata`, `SUPPORTED_SUFFIXES`, and `compute_file_fingerprint` from `retrieval.py`.
- **Feature 18 Sub-task C**: Extract Reranking + Filtering into `src/reranking.py` — move `_query_word_set`, `_summary_match_count`, `_filter_by_tags`, `_boost_by_summary`, and `Pipeline.rerank` from `pipeline.py`.
- **Feature 18 Sub-task D**: Rename `src/preprocessor.py` → `src/query.py` and `src/agent.py` → `src/generation.py`, updating all imports.
- **Feature 18 Sub-task E**: Remove `sys.path.insert(0, ...)` hacks from tests; ensure all tests import from canonical domain paths; run full test suite and verify CLI still works.

## Notable Constraints
- Local development tests require the `.venv` at the repo root (Python 3.12) with `pytest`, `numpy`, `faiss-cpu`, `llama-index`, etc. installed
- OpenRouter endpoint is configurable via `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_API_BASE` env vars (defaults to `openrouter/free` model and `https://openrouter.ai/api/v1/chat/completions`)
- Required env vars are documented in `.env` at the repo root: `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_API_BASE`
