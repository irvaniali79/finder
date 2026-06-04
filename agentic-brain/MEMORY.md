# Project Memory

## Initial Setup Decisions
- Chose LlamaIndex for document processing due to its simplicity and integration capabilities
- Selected FAISS for vector storage as it's fast, local, and has minimal dependencies
- Used OpenRouter API for LLM capabilities (model: openrouter/free by default)
- Selected HuggingFace all-MiniLM-L6-v2 for embeddings due to good balance of performance and size
- Implemented a CLI interface for simplicity and ease of testing

## Initial Implementation Notes
- Document loading originally used SimpleDirectoryReader for .md files in docs/
- Chunking uses sentence splitting with 512 chunk size and 50 overlap
- Embedding generation uses the HuggingFace model through LlamaIndex
- FAISS index uses IndexFlatL2 for exact similarity search
- Prompt engineering emphasizes answering only from provided context
- CLI includes continuous loop with exit/quit commands

## Ingestion Flow Extension
- Added support for .txt files alongside .md
- Added support for .pdf files
- Switched to file-by-file ingestion under docs/ using LlamaIndex TxtReader and PDFReader
- This expands the knowledge source without changing chunking, embedding, or retrieval behavior

## OpenRouter Migration
- Migrated from Ollama to OpenRouter API for LLM capabilities
- Ollama-related dependencies removed from requirements.txt
- Default model changed from `llama3.2` to `openrouter/free`
- Environment variables updated to `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_API_BASE`
- Constructor args in `QAAgent` maintain precedence over env vars; env vars override hardcoded defaults
- This change simplifies deployment by removing the local Ollama dependency requirement

## Incremental Ingestion (Feature 8)
- Added per-file fingerprinting (SHA-256 of size + mtime_ns + bytes) to detect new/changed files
- Persisted state, FAISS index, and chunks to `.cache/` (ignored by git)
- `setup()` now does incremental loading: restores the index from cache, then chunks + embeds + appends only files whose fingerprint has changed or that are new
- When a previously known file is no longer present, the index is rebuilt from scratch (safe fallback for deletions)
- `chunks` is now a list of `{"text", "metadata"}` dicts instead of raw LlamaIndex nodes, so it is safely picklable
- `clear_cache()` was added to wipe the persisted state, index, and chunks when a full rebuild is required
- The CLI was not changed; the new caching kicks in automatically on every `setup()` call

## Disk-Backed Chunk Store (Feature 9)
- Replaced the in-RAM `self.chunks` list + `chunks.pkl` with a SQLite database at `.cache/chunks.db`
- New `ChunkStore` class wraps a single SQLite connection; chunks are inserted in bulk and queried by id
- FAISS-internal indices map directly to `chunks.id` (0-based, sequential, assigned on insert)
- `retrieve()` now does a single `SELECT ... WHERE id IN (...)` after FAISS returns the nearest ids — no chunk text is kept in a Python list, so RAM usage scales with the FAISS index only, not the corpus size
- `setup()` reopens the SQLite DB on each run; on restore it cross-checks `chunk_store.count()` against `index.ntotal` and forces a full rebuild if they disagree
- Legacy `chunks.pkl` is removed by `clear_cache()` for hygiene
- Trade-off: changed files still append to the index (no in-place update); repeated edits to the same file will accumulate chunk versions. For an MVP this is acceptable, and `clear_cache()` is the supported way to compact

## Pipeline Variants Harness (Feature 10)
- New `src/pipeline.py` with a `Pipeline` class and a `VARIANT_FEATURES` registry matching DESIGN.md's V0–V5
- V0 = pure vector search (delegates to `DocumentRetriever.retrieve`)
- V1–V5 raise `NotImplementedError` and point to the feature that fills them in (F11, F12, F13) — this is the ablation harness only
- `build_pipeline(variant, ...)` is a convenience factory that creates the retriever and runs `setup()`
- Variants are monotonic: V(n+1).features ⊇ V(n).features (enforced by test)
- 11 unit tests in `tests/test_pipeline.py` cover the registry, monotonicity, defaults, and V0 delegation

## Query Preprocessing (Feature 11)
- New `src/preprocessor.py` with `extract_tags()`, `expand_query()`, `detect_question_intent()`, `combine_query_embeddings()`
- `extract_tags()` is a stopword-based keyword extractor (handles hyphens, underscores, lowercasing)
- `expand_query()` returns up to 3 variants: original, keywords-only, and `<intent> <keywords>` (e.g. `what is vacation policy`)
- `combine_query_embeddings()` takes the centroid of the variant embeddings for FAISS search
- `Pipeline` now has `_build_query_embedding()` that switches between single-embedding and centroid-of-variants based on the `query_expansion` feature flag
- V3 still raises `NotImplementedError` because its other features (`tag_filtering`, `summary_embedding`) are F13; the expansion logic is exercised via direct pipeline tests
- Trade-off: keyword extraction is heuristic (no LLM call). The DESIGN.md says "using small local LLM" but for the MVP this keeps preprocessing deterministic and free; an LLM-based extractor can be added later behind the same `extract_tags` signature

## Two-Stage Retrieval with Re-Ranking (Feature 12)
- `Pipeline` gained three new methods: `retrieve_top_n`, `rerank`, and `retrieve_top_k`
- `retrieve_top_n(query, n=20)` returns a list of candidate dicts `{"text", "distance", "file_name"}` (richer than the `(text, distance)` tuple form so the reranker has more signal)
- `Pipeline.rerank(candidates, query, importance_fn=None)` is a static, side-effect-free function. It min-max normalizes FAISS L2 distance into a [0,1] similarity, then computes `score = 0.7 * sim + 0.3 * importance − 0.1 * (file_seen_count − 1)`. The per-file penalty is an MMR-style diversity signal that prevents top‑K from being dominated by chunks of one document
- `importance_fn` is optional; F13 will hook it up once chunks carry an importance score. The default of 0.0 means rerank degenerates to a similarity+diversity reorder, which is still a meaningful second stage
- `retrieve_top_k(query, k=5)` chains the two: top-N → rerank → top-K, returns the same `(text, distance)` tuple shape the rest of the system already expects
- `Pipeline.retrieve()` now dispatches to the rerank path whenever `"rerank"` is in the variant features, so V4 and V5 actually run end-to-end (the F13 metadata features are passive until F13 lands — they don't block the rerank path)
- V0 still uses the original `_search`; V1/V2/V3 still raise `NotImplementedError` pointing at F13
- 19 new unit tests in `tests/test_pipeline.py` (4 retrieval-stage tests, 6 rerank tests, 4 retrieve_top_k tests, plus updated V4/V5 tests)
- All 83 tests in the suite pass
- Trade-off: rerank is heuristic (no cross-encoder, no LLM judge). It's cheap, deterministic, and gives us a working ablation hook. A real cross-encoder can be dropped in behind the same `rerank` signature later

## Per-Chunk Metadata Extraction (Feature 13)
- Added `extract_chunk_metadata(text, file_name)` in `src/retrieval.py` that extracts:
  - `tags`: reuses stopword-based keyword extraction from preprocessor
  - `summary`: first sentence (up to 200 chars) of chunk text
  - `importance`: heuristic combining word count and tag count (clamped to [0, 1])
- Extended `ChunkStore` schema with `tags` (TEXT JSON), `summary` (TEXT), `importance` (REAL). Migration via `ALTER TABLE` ensures backward compatibility with existing DBs.
- Updated `ChunkStore.add()` to accept and persist these new metadata fields; `get()`/`get_many()` return them.
- Modified `chunk_documents()` and `_append_chunks_for_files()` to call `extract_chunk_metadata` and merge into chunk metadata.
- Updated `src/agent.py`'s `generate_prompt()` to accept optional 3-tuple `(text, distance, metadata)` and render file_name + summary into the prompt when present.
- Implemented V1/V2/V3 in `src/pipeline.py`:
  - V1: `_filter_by_tags` drops candidates with zero tag overlap when `tag_filtering` active
  - V2: `_boost_by_summary` subtracts a small distance boost when summary matches query terms
  - V3: V2 + query expansion (already implemented in F11)
- V4/V5 now consume importance scores from the metadata for the rerank weighting.
- Trade-off: metadata extraction uses deterministic heuristics (no LLM). Matches the F11 choice to keep preprocessing free and deterministic. LLM-based tag extraction/summarization can be added later behind the same function signatures.
- Need to add proper error handling for edge cases
- Should add validation for empty queries
- Could improve chunking strategy based on document structure
- Consider purging old chunk versions for a file when it changes (or rebuilding the index)
- Wire RAGAS/DeepEval ablation runner once F13 lands
- Optional: replace stopword tag extractor with an LLM-based tagger when not on a tight budget
- Optional: replace heuristic rerank with a cross-encoder once one is available locally

## Propagate Chunk Metadata Through Production Path (Feature 14)
- Modified `DocumentRetriever.retrieve()` to return 3-tuples: `(text, distance, metadata)`.
- Modified `Pipeline.retrieve()` and `Pipeline.retrieve_top_k()` to also return 3-tuples: `(text, distance, metadata)`. This ensures that whichever retriever path is used, metadata is preserved and passed to the QA Agent.
- Updated `tests/test_retrieval.py` and `tests/test_pipeline.py` to assert the 3-tuple shape and metadata contents.
- Updated `tests/test_evals.py` to unpack the 3-tuple retrieved chunks correctly.
- All 103 tests in the suite pass successfully.

## Environment Loading (Bootstrapping)
- `src/main.py` now calls `load_dotenv()` at startup so `.env` values are loaded automatically
- `python-dotenv` was added to `requirements.txt` because it was missing

## Feature 18 — Sub-task A: Extract ChunkStore into src/storage.py
- New module `src/storage.py` owns the `ChunkStore` class and its private helpers (`_encode_metadata`, `_row_to_chunk`, `_migrate_add_metadata_columns`)
- `src/retrieval.py` now imports `ChunkStore` from `src.storage` (no functional changes to `DocumentRetriever`)
- `tests/test_retrieval.py` imports `ChunkStore` from `src.storage` while keeping `DocumentRetriever` / `extract_chunk_metadata` from `src.retrieval`
- `sqlite3` is still imported in `retrieval.py` because `clear_cache()` catches `sqlite3.ProgrammingError`
- All 103 tests in the suite still pass; no regressions

## Feature 18 — Sub-task B: Extract Ingestion into src/ingestion.py
- New module `src/ingestion.py` owns the file → chunk pipeline: `SUPPORTED_SUFFIXES`, `extract_chunk_metadata`, `compute_file_fingerprint`, `list_supported_files`, `read_file_as_documents`, `load_documents`, `chunk_documents` (all standalone, no `self`)
- `src/retrieval.py` keeps thin wrapper methods on `DocumentRetriever` (`load_documents`, `chunk_documents`, `compute_file_fingerprint`, `_list_supported_files`, `_read_file_as_documents`) that delegate to `src.ingestion` for backward compat with existing tests and the `_build_full_index`/`_append_chunks_for_files` internals
- `tests/test_retrieval.py` imports `extract_chunk_metadata` from `src.ingestion` and now patches `src.ingestion.SentenceSplitter` and `src.ingestion.PDFReader` (not `retrieval.*`) because `retrieval.py` resolves the symbols through `src.ingestion`; patching the wrong module is silently a no-op since the test bootstrap (`sys.path.insert(0, 'src')`) creates a separate top-level `ingestion` module
- All 103 tests still pass; CLI (`python -m src.main`) still loads and reads from `.cache/`
- No functional change to the public API; this is purely a domain split
