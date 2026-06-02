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
- [ ] Task complete
