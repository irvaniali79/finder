1. Ingestion Flow (LlamaIndex / RAGFlow)
text
Upload .txt or .pdf file
        ↓
[LlamaIndex] Load documents (SimpleDirectoryReader)
        ↓
[LlamaIndex] Semantic Chunking
(SentenceSplitter with paragraph boundaries + overlap)
        ↓
For each chunk:
        ↓
[Ollama / local LLM] Extract metadata:
  - tags, keywords, summary, FAQs, importance score
        ↓
[LlamaIndex] Generate embedding (Sentence-Transformers via HuggingFaceEmbedding)
        ↓
[ChromaDB / Qdrant] Store chunk + metadata + embedding
        ↓
(Optional) Save raw chunk to /data

2. Query & Answer Flow (LlamaIndex / RAGFlow)
text
User question
        ↓
[LlamaIndex] Preprocess query:
  - extract tags (using small local LLM)
  - query expansion (rewrite into 2–3 variants)
        ↓
Generate embedding for each expanded query
        ↓
[ChromaDB] Vector similarity search (top‑20 chunks)
        ↓
Filter results by extracted tags (metadata filtering)
        ↓
[LlamaIndex] Re‑rank using:
  - similarity score + importance score (weighted sum)
        ↓
Select top‑K chunks (e.g., K=5)
        ↓
Build prompt: question + chunks (text, summary, FAQs)
        ↓
[Ollama] Call LLM to generate final answer
        ↓
Return answer to user

3. Evaluation Flow (RAGAS / DeepEval)
text
Build Golden Test Set:
  - 20–30 real user questions
  - manually map each to relevant chunk IDs
        ↓
[RAGAS] For each pipeline variant (ablation):
        ↓
Run questions through retrieval system → get top‑10 chunk IDs
        ↓
[RAGAS] Compute metrics automatically:
  - Recall@10
  - NDCG@10
  - Faithfulness, Answer Relevancy (optional)
        ↓
Compare variants:
  V0 = pure vector search (no metadata)
  V1 = + tag filtering
  V2 = + summary embedding
  V3 = + query expansion
  V4 = + re‑ranking (similarity + importance)
  V5 = full system (all features)
        ↓
Identify highest‑lift components
        ↓
Store results in a dashboard (e.g., a CSV/Excel log)
        ↓
Repeat after every major pipeline change