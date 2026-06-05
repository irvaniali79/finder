# Mini Company Knowledge Bot

A RAG-based Q&A CLI that answers questions using documents in `docs/`. Built with
LlamaIndex, FAISS, Sentence-Transformers, and OpenRouter. The retrieval pipeline
is structured as an ablation harness (variants V0–V5) that adds tag filtering,
summary boosting, query expansion, and two-stage re-ranking on top of pure
vector search.

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

`src/main.py` calls `load_dotenv()` at startup, so a `.env` file at the repo
root is picked up automatically.

## Run

```bash
python -m src.main
```

The first run ingests all supported documents under `docs/` (`.md`, `.txt`,
`.pdf`), chunks them, generates embeddings, and builds a FAISS index plus
a SQLite chunk store in `.cache/`. Subsequent runs only re-process files
whose content fingerprint has changed.

To force a full rebuild, call `DocumentRetriever.clear_cache()` from Python,
or delete the `.cache/` directory.

## Architecture

### Code Layout (`src/`)

| Module | Responsibility |
| --- | --- |
| `main.py` | CLI entry point — builds a `V5` pipeline and a `QAAgent`, runs the interactive loop |
| `retrieval.py` | `DocumentRetriever` — orchestrates ingestion, FAISS index management, persistence, search |
| `ingestion.py` | File loading (`.md`/`.txt`/`.pdf`), chunking with `SentenceSplitter`, per-chunk metadata extraction, content fingerprinting |
| `storage.py` | `ChunkStore` — SQLite-backed chunk storage (text + tags + summary + importance) |
| `preprocessor.py` | Query preprocessing — stopword-based `extract_tags`, `expand_query` (≤3 variants), `detect_question_intent`, `combine_query_embeddings` |
| `reranking.py` | Tag filtering, summary-match distance boost, two-stage re-ranking (similarity + importance + diversity penalty) |
| `pipeline.py` | `Pipeline` orchestration + `VARIANT_FEATURES` registry + `build_pipeline()` factory |
| `agent.py` | `QAAgent` — prompt construction and OpenRouter API call with retry |

### Pipeline Variants (Ablation Harness)

`Pipeline` runs a monotonic set of features. Each successive variant adds
exactly one more capability, which makes it possible to ablate components
individually.

| Variant | Features |
| --- | --- |
| `V0` | pure vector search |
| `V1` | `V0` + tag filtering (drops candidates with zero query-tag overlap) |
| `V2` | `V1` + summary embedding (reduces distance when chunk summary matches query words) |
| `V3` | `V2` + query expansion (centroid of up to 3 query variants) |
| `V4` | `V3` + two-stage re-rank (similarity + importance + diversity penalty) |
| `V5` | `V4` + per-chunk importance weighting from metadata |

The CLI uses `V5` by default. Pass a different variant to `build_pipeline()`
in Python.

### Two-Stage Retrieval

- **Stage 1** — `Pipeline.retrieve_top_n(query, n=20)`: encodes the query
  (centroid of variants when `query_expansion` is active), runs a FAISS
  top-N search, and pulls the full chunk + metadata from `ChunkStore`.
- **Stage 2** — `rerank(candidates, query, importance_fn=...)`: min-max
  normalizes L2 distance into similarity, then scores each candidate as
  `0.7 * sim + 0.3 * importance − 0.1 * (file_seen_count − 1)`. The
  per-file penalty is an MMR-style diversity signal.
- `Pipeline.retrieve()` dispatches to the rerank path whenever `rerank` is
  in the active feature set; otherwise it returns the filtered / boosted
  top-K from stage 1.

### Chunk Metadata

Each chunk carries four metadata fields extracted at ingestion time:

- `tags` — stopword-filtered keywords from the chunk text
- `summary` — first sentence of the chunk (≤ 200 chars)
- `importance` — heuristic in `[0, 1]` combining word count and tag density
- `file_name` — source document filename

Metadata flows from `ChunkStore` → `Pipeline` → `QAAgent.generate_prompt()`,
which renders `[source: …]` and `summary: …` headers into the context block.

### Persistence

| Artifact | Location | Format |
| --- | --- | --- |
| FAISS index | `.cache/faiss.index` | `IndexFlatL2` (L2 distance) |
| Chunks | `.cache/chunks.db` | SQLite (one row per chunk) |
| Ingestion state | `.cache/ingestion_state.json` | per-file SHA-256 fingerprint + chunk count |

`.cache/` is git-ignored. `clear_cache()` wipes all three artifacts plus
the legacy `chunks.pkl` (if present).

## Documents

- `docs/FAQ.md`
- `docs/leave_policy.md`
- `docs/product_guide.md`

Any `.pdf`, `.txt`, or `.md` file dropped into `docs/` is ingested
automatically on the next run.

## Testing

```bash
pytest
```

Test files:

- `tests/test_retrieval.py` — `DocumentRetriever`, `ChunkStore`, ingestion
  helpers (file loading, chunking, metadata extraction)
- `tests/test_agent.py` — `QAAgent` prompt construction (no network)
- `tests/test_main.py` — CLI integration
- `tests/test_pipeline.py` — `Pipeline` variants, two-stage retrieval, rerank
- `tests/test_reranking.py` — standalone reranking / filtering helpers
- `tests/test_evals.py` — golden questions from `agentic-brain/EVALS.md`

LLM-dependent evaluation tests are skipped by default. To run them:

```bash
FINDER_RUN_LLM_EVALS=1 pytest tests/test_evals.py
```

## Evaluation

Five golden questions with expected answers live in `agentic-brain/EVALS.md`.
Run the CLI and verify that answers match the expected outputs. Document any
discrepancies in `agentic-brain/MEMORY.md` and update the system **without**
changing the expected answers.
