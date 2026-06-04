# Project Tasks

## Feature 1: Document Ingestion and Chunking
- **Dependencies**: None
- **Affected files**: src/retrieval.py
- **Key functions**: load_documents(), chunk_documents()
- **Tests**: tests/test_retrieval.py
- [x] Task complete

## Feature 2: Vector Embedding and FAISS Index
- **Dependencies**: Feature 1
- **Affected files**: src/retrieval.py
- **Key functions**: create_embeddings(), build_index(), retrieve()
- **Tests**: tests/test_retrieval.py
- [x] Task complete

## Feature 3: QA Agent with LLM Integration
- **Dependencies**: None
- **Affected files**: src/agent.py
- **Key functions**: generate_prompt(), answer()
- **Tests**: tests/test_agent.py
- [x] Task complete

## Feature 4: CLI Entry Point
- **Dependencies**: Feature 1, Feature 2, Feature 3
- **Affected files**: src/main.py
- **Key functions**: main()
- **Tests**: tests/test_main.py
- [x] Task complete

 ## Feature 5: Evaluation Suite and Integration Tests
 - **Dependencies**: Feature 1, Feature 2, Feature 3, Feature 4
 - **Affected files**: agentic-brain/EVALS.md, tests/test_evals.py
 - **Key functions**: N/A (QA pairs + integration test)
 - **Tests**: tests/test_evals.py
 - [x] Task complete
 
 ## Feature 6: Multi-format Document Ingestion
 - **Dependencies**: Feature 1
 - **Affected files**: src/retrieval.py, tests/test_retrieval.py
 - **Key functions**: load_documents()
 - **Tests**: tests/test_retrieval.py
 - [x] Task complete

## Feature 7: Configurable OpenRouter Endpoint via Environment Variables
- **Dependencies**: Feature 3
- **Affected files**: src/agent.py, tests/test_agent.py, README.md
- **Key functions**: QAAgent.__init__()
- **Tests**: tests/test_agent.py
- [x] Task complete

## Feature 8: Incremental Document Ingestion with Persistent Index
- **Dependencies**: Feature 1, Feature 2
- **Affected files**: src/retrieval.py, tests/test_retrieval.py, agentic-brain/MEMORY.md, agentic-brain/AGENT_CONTEXT.md
- **Key functions**: load_documents(), setup(), compute_file_fingerprint(), _save_state(), _load_state(), clear_cache()
- **Tests**: tests/test_retrieval.py
- [x] Task complete

## Feature 9: Disk-Backed Chunk Store for Large Documents
- **Dependencies**: Feature 8
- **Affected files**: src/retrieval.py, tests/test_retrieval.py, agentic-brain/MEMORY.md, agentic-brain/AGENT_CONTEXT.md
- **Key functions**: ChunkStore.__init__(), ChunkStore.add(), ChunkStore.get(), ChunkStore.get_many(), DocumentRetriever.setup(), DocumentRetriever.retrieve()
- **Tests**: tests/test_retrieval.py
- [x] Task complete

## Feature 10: Pipeline Variants V0–V5 (Ablation Harness)
- **Dependencies**: Feature 9
- **Affected files**: src/retrieval.py, src/pipeline.py, tests/test_pipeline.py, agentic-brain/MEMORY.md, agentic-brain/AGENT_CONTEXT.md
- **Key functions**: build_pipeline(variant), Pipeline.retrieve()
- **Tests**: tests/test_pipeline.py
- [x] Task complete

## Feature 11: Query Preprocessing (Tags + 2–3 Rewrites)
- **Dependencies**: Feature 10
- **Affected files**: src/preprocessor.py, src/pipeline.py, tests/test_pipeline.py, agentic-brain/MEMORY.md, agentic-brain/AGENT_CONTEXT.md
- **Key functions**: extract_tags(), expand_query(), detect_question_intent(), combine_query_embeddings(), Pipeline._build_query_embedding()
- **Tests**: tests/test_pipeline.py
- [x] Task complete

## Feature 12: Two-Stage Retrieval with Re-Ranking
- **Dependencies**: Feature 11
- **Affected files**: src/pipeline.py, tests/test_pipeline.py
- **Key functions**: retrieve_top_n(), rerank(), retrieve_top_k()
- **Tests**: tests/test_pipeline.py
- [x] Task complete

## Feature 13: Per-Chunk Metadata Extraction
- **Dependencies**: Feature 12
- **Affected files**: src/retrieval.py, src/agent.py, src/pipeline.py, tests/test_retrieval.py, tests/test_agent.py, tests/test_pipeline.py
- **Key functions**: extract_chunk_metadata(), ChunkStore.add(), _filter_by_tags(), _boost_by_summary()
- **Tests**: tests/test_retrieval.py, tests/test_agent.py, tests/test_pipeline.py
- [x] Task complete

## Feature 14: Propagate Chunk Metadata Through Production Path
- **Dependencies**: Feature 13
- **Affected files**: src/retrieval.py, src/main.py, tests/test_retrieval.py, tests/test_main.py
- **Key functions**: DocumentRetriever.retrieve(), main()
- **Tests**: tests/test_retrieval.py, tests/test_main.py
- [x] Task complete

## Feature 15: Wire Pipeline into main.py
- **Dependencies**: Feature 14
- **Affected files**: src/main.py, tests/test_main.py
- **Key functions**: main(), build_pipeline(), Pipeline.retrieve()
- **Tests**: tests/test_main.py
- **Details**: `main.py` currently calls `DocumentRetriever.retrieve()` directly, bypassing the advanced retrieval features. Update to use `build_pipeline(variant="V5")` to enable tag filtering, summary boost, query expansion, and reranking in the CLI.
- [ ] Task complete

## Feature 16: Unify Import Conventions Across src/
- **Dependencies**: None
- **Affected files**: src/pipeline.py, tests/test_pipeline.py
- **Key functions**: N/A (module-level imports)
- **Tests**: tests/test_pipeline.py
- **Details**: `pipeline.py` uses bare imports (`from retrieval import ...`, `from preprocessor import ...`), while `main.py` and `retrieval.py` use the `src.` prefix. Change `pipeline.py` to use `from src.retrieval import ...` to match the project convention and prevent `ModuleNotFoundError` when imported from the project root.
- [ ] Task complete

## Feature 17: De-duplicate Chunking Logic in retrieval.py
- **Dependencies**: None
- **Affected files**: src/retrieval.py, tests/test_retrieval.py
- **Key functions**: chunk_documents(), _append_chunks_for_files()
- **Tests**: tests/test_retrieval.py
- **Details**: `chunk_documents()` and `_append_chunks_for_files()` contain identical chunking logic (instantiating `SentenceSplitter`, iterating `get_nodes_from_documents`, calling `extract_chunk_metadata`). Refactor `_append_chunks_for_files()` to reuse `chunk_documents()`.
- [ ] Task complete