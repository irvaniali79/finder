# Mini Company Knowledge Bot — Project Brief

## What We Built
A RAG-based Q&A system that answers questions using company documents. The
system provides a CLI for users to query an internal knowledge base
(FAQ, leave policy, product guide) and includes a V0–V5 ablation harness
that lets us measure the contribution of each retrieval feature in
isolation.

## MVP Scope
- `docs/`: 3 plain-text files (FAQ, Leave Policy, Product Guide); supports
  additional `.txt` and `.pdf` files dropped in at runtime
- `src/`: Domain-split RAG implementation
  - `retrieval.py` — `DocumentRetriever` orchestration
  - `ingestion.py` — file loading, chunking, metadata extraction, fingerprinting
  - `storage.py` — `ChunkStore` (SQLite)
  - `query.py` — query preprocessing
  - `reranking.py` — tag filtering, summary boost, re-ranking
  - `pipeline.py` — `Pipeline` + `VARIANT_FEATURES` + `build_pipeline()`
  - `generation.py` — `QAAgent` (OpenRouter)
  - `main.py` — CLI entry point
- `tests/`: unit + integration tests across every module
- `agentic-brain/`: project documentation and evaluation

## Architecture
- Document Ingestion: LlamaIndex `SentenceSplitter` (512 chars, 50 overlap),
  per-chunk metadata (tags, summary, importance) extracted deterministically
- Vector Database: FAISS `IndexFlatL2` (L2 distance, exact search)
- Chunk Storage: SQLite (`.cache/chunks.db`) via `ChunkStore`
- LLM: OpenRouter API (configurable model + base URL, automatic retry)
- Embeddings: HuggingFace `all-MiniLM-L6-v2` via
  `llama-index-embeddings-huggingface`
- Pipeline: monotonic V0–V5 ablation variants (vector → tag filter →
  summary boost → query expansion → rerank → importance weighting)
- Entry Point: Python CLI (`python -m src.main`)
