# Mini Company Knowledge Bot – Agent Rules (for Kilo)

## 🎯 Project Goal
Build a simple RAG-based Q&A system that answers questions using documents in `docs/`. The primary deliverable is a working CLI or API, but the real quality metric is clean code architecture, perfect agentic-brain documentation, and meaningful Git history.

## 📁 Mandatory Repository Structure
- `docs/`: at least 3 plain-text files (FAQ, leave policy, product guide)
- `src/`:
  - `retrieval.py` – document loading, chunking, embedding, similarity search
  - `agent.py` – prompt construction, LLM call, answer generation
  - `main.py` or `app.py` – entry point (CLI or FastAPI)
- `tests/` (optional)
- `agentic-brain/`:
  - `PROJECT_BRIEF.md` – what we built, MVP scope
  - `AGENT_CONTEXT.md` – comprehensive context for a future AI agent resuming work
  - `MEMORY.md` – decisions, mistakes, learnings, pivots
  - `TASKS.md` – completed and remaining tasks (checkboxes)
  - `EVALS.md` – at least 5 test questions with expected answers
- `requirements.txt`
- `README.md` – run instructions, architecture overview

## 🤖 Agent Workflow – CRITICAL RULES

### Task Execution Protocol (MUST FOLLOW)
- **You are strictly single-task.** Work on ONLY ONE unchecked task from `TASKS.md` at a time. 
- When you finish that task:
  1. Mark it as `[x]` in `TASKS.md` immediately.
  2. Save the file.
  3. **STOP completely.** Do not read the next task. Do not execute the next task. Output a short summary of what you did and wait for the user's explicit "continue" or next instruction.
- If there are multiple unchecked tasks, **never** proceed to the next one without explicit user instruction. This is a hard rule with zero exceptions.

### Overall Sequence
1. **Before any code**, create `agentic-brain/PROJECT_BRIEF.md` with MVP scope and `TASKS.md` with a step-by-step task list. Commit.
2. **Work iteratively, one logical step at a time.** After each step, commit with a clear message like "Add document loading and chunking".
3. **Update `AGENT_CONTEXT.md`** after any significant change so that a new AI agent would know exactly what the project is and where to continue.
4. **Record every important decision, mistake, or direction change in `MEMORY.md` immediately.** Don't wait.
5. Once core RAG works, generate 5 QA pairs (based on the docs) and save them in `EVALS.md`. Then test the system against them. Document the actual outputs and fix any mismatches.
6. Write a clean `README.md` with run instructions (dependencies, how to start).
7. Ensure Git history has at least 5 meaningful commits. Use branches if you like (feature branches), but at minimum a clear linear history.

## 💻 Code Style
- Don't comment anything
- Keep it simple – no over-engineering.
- Do not add features outside the specification.

## 📚 Stack
- **Document Ingestion**: LlamaIndex (`SimpleDirectoryReader`, `SentenceSplitter`). For a lighter alternative, consider TinyRag.
- **Vector Database**: FAISS (fast, local, minimal dependencies).
- **LLM & Embeddings**: Ollama (for Llama 3) + Sentence-Transformers.
- **Evaluation Framework**: DeepEval (works with Pytest, built for local evaluation).

## 🔍 Quality & Self-Evaluation
- After implementing retrieval, manually verify that relevant chunks are returned for a few sample queries.
- If LLM answers are poor, modify prompts or retrieval and document the change in MEMORY.
- Avoid AI slop: only build what's requested.


## business requirement document : ./SSOT.md (source of truth) 