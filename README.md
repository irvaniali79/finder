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

## Run

```bash
python src/main.py
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