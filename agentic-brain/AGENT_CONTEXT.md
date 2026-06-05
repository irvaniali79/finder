# Agent Context

## Project Overview
Mini Company Knowledge Bot — a RAG-based CLI Q&A system backed by the
`docs/` knowledge base. Includes a V0–V5 ablation harness for measuring
the contribution of each retrieval feature in isolation.

## Current State
- **Ingestion** (`src/ingestion.py`): file loading, `SentenceSplitter`
  chunking, per-chunk metadata extraction (`tags`, `summary`,
  `importance`, `file_name`), SHA-256 content fingerprinting
- **Storage** (`src/storage.py`): `ChunkStore` (SQLite at
  `.cache/chunks.db`) with a `tags` / `summary` / `importance` schema
  and an idempotent `ALTER TABLE` migration
- **Retrieval** (`src/retrieval.py`): `DocumentRetriever` orchestrates
  ingestion, FAISS index persistence (`.cache/faiss.index`), ingestion
  state (`.cache/ingestion_state.json`), and search. `setup()` is
  incremental — it fingerprints each file under `docs/`, restores the
  index from cache, and only re-processes files that are new or
  changed
- **Query Preprocessing** (`src/query.py`): `extract_tags`,
  `expand_query` (≤ 3 variants), `detect_question_intent`,
  `combine_query_embeddings`
- **Re-ranking & Filtering** (`src/reranking.py`): `query_word_set`,
  `summary_match_count`, `filter_by_tags`, `boost_by_summary`,
  `rerank` (similarity + importance + per-file diversity penalty)
- **Pipeline** (`src/pipeline.py`): `Pipeline` + `VARIANT_FEATURES`
  registry + `build_pipeline()` factory. Variants are monotonic
  (V(n+1).features ⊇ V(n).features). `Pipeline.retrieve()` returns
  3-tuples `(text, distance, metadata)`. `Pipeline.retrieve_top_n`,
  `rerank`, and `retrieve_top_k` provide the two-stage retrieval
  path
- **Generation** (`src/generation.py`): `QAAgent` builds the prompt (with
  source + summary headers) and calls the OpenRouter API with
  exponential-backoff retry
- **CLI** (`src/main.py`): interactive loop using `build_pipeline("V5")`
  + `QAAgent`
- `.cache/` is git-ignored; use `DocumentRetriever.clear_cache()` (or
  delete `.cache/`) to force a full rebuild
- 121 tests passing across `tests/test_retrieval.py`,
  `test_ingestion.py`, `test_agent.py`, `test_main.py`,
  `test_pipeline.py`, `test_reranking.py`, `test_evals.py`

## Pipeline Variants
| Variant | Features | Notes |
| --- | --- | --- |
| V0 | (none) | pure vector search |
| V1 | `tag_filtering` | drops candidates with no query-tag overlap |
| V2 | + `summary_embedding` | reduces distance on summary word matches |
| V3 | + `query_expansion` | centroid of up to 3 query variants |
| V4 | + `rerank` | two-stage sim + importance + diversity |
| V5 | + `importance` | per-chunk importance weight from metadata |

The CLI uses V5. Pass a different variant to `build_pipeline()` in
Python.

## Pending Tasks
- **Feature 18 Sub-task D**: Rename `src/preprocessor.py` → `src/query.py`
  and `src/agent.py` → `src/generation.py`, updating all imports.
- **Feature 18 Sub-task E**: Remove `sys.path.insert(0, ...)` hacks from
  tests; ensure all tests import from canonical domain paths; run the
  full test suite and verify the CLI still works.

## Notable Constraints
- Local development uses the `.venv` at the repo root (Python 3.12) with
  `pytest`, `numpy`, `faiss-cpu`, `llama-index`, `pypdf`,
  `sentence-transformers`, `python-dotenv` installed
- OpenRouter endpoint is configurable via `OPENROUTER_API_KEY`,
  `OPENROUTER_MODEL`, and `OPENROUTER_API_BASE` env vars
  (defaults: `openrouter/free` model, `https://openrouter.ai/api/v1/chat/completions`)
- Required env vars are documented in `.env` at the repo root:
  `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`,
  `OPENROUTER_API_BASE`
- `extract_tags` and `extract_chunk_metadata` are deterministic
  heuristics (no LLM call). The DESIGN.md flow calls for an LLM-based
  tagger; a real LLM tagger can be dropped in behind the same function
  signature without changing callers
- `rerank` is heuristic (no cross-encoder). A cross-encoder can replace
  it later behind the same `rerank` signature
