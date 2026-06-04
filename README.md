# Mini Company Knowledge Bot

A RAG-based Q&A CLI that answers questions using documents in `docs/`. Built with LlamaIndex, FAISS, Sentence-Transformers, and OpenRouter.

## Prerequisites

- Python 3.10+
- OpenRouter API key (or any OpenAI-compatible API)
- HuggingFace token (for embedding model download)

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

Required:
- `OPENROUTER_API_KEY` — your OpenRouter API key
- `HF_TOKEN` — your HuggingFace access token (for embedding model)

Optional:
- `OPENROUTER_MODEL` — model name (default: `openrouter/free`)
- `OPENROUTER_API_BASE` — API base URL (default: `https://openrouter.ai/api/v1/chat/completions`)

## Run

```bash
.venv/bin/python -m src.main
```

The first run will ingest all supported documents under `docs/` (`.md`, `.txt`, `.pdf`), chunk them, generate embeddings, and build a FAISS index saved to `.cache/`. Subsequent runs only re-process changed files using content fingerprinting.

To force a full rebuild, call `DocumentRetriever.clear_cache()` from Python.

## Architecture

- **Document Ingestion**: LlamaIndex `SimpleDirectoryReader` / direct file reads + `SentenceSplitter` (512 chars, 50 overlap)
- **Chunk Storage**: SQLite (`.cache/chunks.db`) via `ChunkStore`
- **Vector Storage**: FAISS `IndexFlatL2` (`.cache/faiss.index`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` via `llama-index-embeddings-huggingface`
- **LLM**: OpenRouter API with automatic retry
- **Entry Point**: `src/main.py` — interactive CLI

## Documents

- `docs/FAQ.md`
- `docs/leave_policy.md`
- `docs/product_guide.md`

`.pdf`, `.txt`, and `.md` files in `docs/` are ingested automatically.

## Testing

```bash
pytest
```

Test files: `tests/test_retrieval.py`, `tests/test_agent.py`, `tests/test_main.py`, `tests/test_evals.py`.

LLM-dependent evaluation tests are skipped by default. To run them:

```bash
FINDER_RUN_LLM_EVALS=1 pytest tests/test_evals.py
```

## Evaluation

Five golden questions are defined in `agentic-brain/EVALS.md` with expected answers. Run the CLI and verify answers match the expected outputs. Document any discrepancies in `agentic-brain/MEMORY.md` and update the system without changing the expected answers.
