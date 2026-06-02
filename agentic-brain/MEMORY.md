# Project Memory

## Initial Setup Decisions
- Chose LlamaIndex for document processing due to its simplicity and integration capabilities
- Selected FAISS for vector storage as it's fast, local, and has minimal dependencies
- Used Ollama with Llama 3.2 for LLM capabilities
- Selected HuggingFace all-MiniLM-L6-v2 for embeddings due to good balance of performance and size
- Implemented a CLI interface for simplicity and ease of testing

## Initial Implementation Notes
- Document loading works with .md files from the docs/ directory
- Chunking uses sentence splitting with 512 chunk size and 50 overlap
- Embedding generation uses the HuggingFace model through LlamaIndex
- FAISS index uses IndexFlatL2 for exact similarity search
- Prompt engineering emphasizes answering only from provided context
- CLI includes continuous loop with exit/quit commands

## Pending Improvements
- Need to add proper error handling for edge cases
- Should add validation for empty queries
- Could improve chunking strategy based on document structure
- Need to implement evaluation suite as per requirements