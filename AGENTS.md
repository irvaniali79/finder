# Mini Company Knowledge Bot – Agent Rules (for Kilo)

## 🎯 Project Goal
Build a simple RAG-based Q&A system that answers questions using documents in `docs/`. The primary deliverable is a working CLI or API, but the real quality metric is clean code architecture, perfect agentic-brain documentation, and meaningful Git history.

## 📄 Business Requirement Document
The single source of truth is `./SSOT.md`. All features, tests, and evaluations must conform to it.
## Tech requirements: `./DESIGN.md`

## 📁 Mandatory Repository Structure
- `docs/`: at least 3 plain-text files (FAQ, leave policy, product guide)
- `src/`:
  - `retrieval.py` – document loading, chunking, embedding, similarity search
  - `agent.py` – prompt construction, LLM call, answer generation
  - `main.py` or `app.py` – entry point (CLI or FastAPI)
- `tests/` – unit and integration tests (required, not optional)
- `agentic-brain/`:
  - `PROJECT_BRIEF.md` – what we built, MVP scope
  - `AGENT_CONTEXT.md` – comprehensive context for a future AI agent resuming work
  - `MEMORY.md` – decisions, mistakes, learnings, pivots
  - `TASKS.md` – completed and remaining tasks (checkboxes)
  - `EVALS.md` – at least 5 test questions with expected answers
- `requirements.txt`
- `README.md` – run instructions, architecture overview

## 🤖 Agent Workflow – CRITICAL RULES

### Task List Format (`TASKS.md`)
Every task must be written as a feature-level deliverable with the following fields:
```markdown
## Feature <N>: <Short Description>
- **Dependencies**: <List of feature IDs or "None">
- **Affected files**: <file paths, e.g., src/retrieval.py>
- **Key functions**: <function names, e.g., load_documents(), split_documents()>
- **Tests**: <test file path, e.g., tests/test_retrieval.py>
- [ ] Task complete
```

### Task Execution Protocol
- **Single-task focus**: Work on ONLY ONE unchecked task at a time.
- **Dependency check**: Before starting a task, verify all its dependencies are marked `[x]`. If not, **halt** and tell the user which tasks must be completed first.
- **Completion steps**:
  0. read `GITWORKFLOW.md`
  1. Write the code and the required tests for that feature.
  2. Run the tests. If any fail, follow the **Debug & Test Integrity Rule** (see below).
  3. Mark the task as `[x]` in `agentic-brain/TASKS.md` only when **all its tests pass**.
  4.1 Save the file.
  4.2. update `agentic-brain` if necessery 
  5. **STOP completely.** Do not read or start the next task. Output a short summary and wait for the user’s explicit "continue" or next instruction.
- **No skipping**: Never proceed to the next task without explicit user instruction, even if there are multiple unchecked tasks. This is a hard rule with zero exceptions.

### Subagent Protocol (for Complex Features)
A feature task that is large and multi-step may be delegated to a subagent to keep the main agent’s context lean.
- **Trigger**: Use a subagent only for feature‑sized tasks, never for micro‑steps like “create a single file”.
- **Context isolation**: Provide the subagent with only the task description, the affected files list, the SSOT, and any relevant existing code/docs. Do **not** include unrelated task history.
- **Output**: The subagent must return the modified files and a short summary. The main agent then integrates the changes, runs the tests, and follows the standard completion protocol.
- **Stopping rule**: Even when a subagent is used, the main agent still stops after marking the task complete.

### Debug & Test Integrity Rule
- **Never change a test just to make it pass.** The source of truth is `SSOT.md`. If a test fails, the system (code, retrieval, prompt) is wrong, not the test.
- **Debugging process**: When a test fails:
  1. Confirm the test is correctly derived from the SSOT and the agent’s behavior specification.
  2. If the test is correct, fix the system.
  3. If the fix is unclear, **halt** and ask the user for help. Provide the test, expected output, and actual output.
- **Debug file**: For additional debugging strategies, consult `agentic-brain/DEBUG.md` if it exists.
- **Test writing**: Every feature must include unit/integration tests in `tests/`. Tests must be traceable to the SSOT and to the functions listed in the task’s “Key functions” field.

### Overall Sequence
1. **Before any code**, create `agentic-brain/PROJECT_BRIEF.md` with MVP scope and `agentic-brain/TASKS.md` with a step-by-step task list (using the format above). Commit.
2. **Work iteratively, one feature task at a time.** After each step, commit with a clear message like "Add document loading and chunking".
3. **Update `agentic-brain/AGENT_CONTEXT.md`** after any significant change so that a new AI agent would know exactly what the project is and where to continue.
4. **Record every important decision, mistake, or direction change in `agentic-brain/MEMORY.md` immediately.** Don’t wait.
5. Once core RAG works, generate 5 QA pairs (based strictly on the documents in `docs/`) and save them in `agentic-brain/EVALS.md`. Then test the system against them. Document the actual outputs and fix any mismatches **without altering the expected answers**.
6. Write a clean `README.md` with run instructions (dependencies, how to start).
7. Ensure Git history has at least 5 meaningful commits. Feature branches are encouraged; if using only a linear history, keep commit messages descriptive.

## 💻 Code Style
- Do not add comments.
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
