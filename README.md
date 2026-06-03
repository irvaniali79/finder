# Company Knowledge Bot

A RAG-based Q&A system using company documents.

## Setup

```bash
pip install -r requirements.txt
```

Ensure Ollama is installed and running:
```bash
ollama pull llama3.2
ollama serve
```

**Environment variables** (optional):
- `OLLAMA_BASE_URL` — Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` — Model name (default: `llama3.2`)

These can be set to point at a third-party Ollama-compatible endpoint.

## Run

```bash
./.venv/bin/python src/main.py
```

## Architecture

- **Document Ingestion**: LlamaIndex SimpleDirectoryReader + SentenceSplitter
- **Vector Storage**: FAISS (IndexFlatL2)
- **LLM**: Ollama (llama3.2)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)

## Documents

- `docs/FAQ.md` - Frequently asked questions
- `docs/leave_policy.md` - Employee leave policies
- `docs/product_guide.md` - ProCRM product guide

Ingestion also reads plain `.txt` and `.pdf` files directly from `docs/` during setup. Unknown file types are skipped.

## Testing

Run the test suite with:

```bash
pytest
```

Included tests: `tests/test_retrieval.py`, `tests/test_agent.py`, `tests/test_main.py`.

LLM-dependent evaluation tests are skipped by default. To run them, set `FINDER_RUN_LLM_EVALS=1`:

```bash
FINDER_RUN_LLM_EVALS=1 pytest tests/test_evals.py
```