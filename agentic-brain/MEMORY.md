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

## Pending Improvements
- Need to add proper error handling for edge cases
- Should add validation for empty queries
- Could improve chunking strategy based on document structure

## Environment Loading (Bootstrapping)
- `src/main.py` now calls `load_dotenv()` at startup so `.env` values are loaded automatically
- `python-dotenv` was added to `requirements.txt` because it was missing
