# Subagent Protocol

## Purpose
To isolate large, feature‑sized tasks so the main agent (Kilo) never accumulates context across domains. This keeps every agent session focused, lean, and reproducible.

## When to Spawn a Subagent
- The task is a **complete feature deliverable** (e.g., “Document ingestion pipeline”, “Q&A agent with CLI”, “Evaluation suite”), **not** a micro‑step like “write a single function”.
- The task touches multiple files or requires iterative debugging that would bloat the main agent’s context.
- If the task can be done in a single trivial action (create a single file with known content), do **not** spawn a subagent – handle it directly.

## Context Isolation
A subagent receives **only** the assets it needs:
- The exact task description from `TASKS.md`.
- The relevant source files (e.g., `docs/`, `src/retrieval.py`, `requirements.txt`).
- The `SSOT.md` (business requirement document) if the feature references business rules.
- This protocol file, so it knows to stop after its work is complete.

The subagent does **not** receive:
- The full Git history or previous conversation log.
- Completed tasks from other features.
- Any memory of past mistakes unless explicitly provided.

## Subagent Inputs & Outputs
**Input (provided by the main agent):**
- A self‑contained task brief that includes acceptance criteria (e.g., “After this feature, `retrieval.py` must return at least 3 relevant chunks for a sample query”).

**Output (returned to the main agent):**
- Modified files (code, tests, docs) ready to commit.
- A concise summary: what was built, any design decisions made, and any follow‑up items that might affect other features.

## Stopping Rule (Hard)
- When the subagent completes the feature, the main agent **immediately** marks the corresponding task as `[x]` in `TASKS.md`, saves the file, and **stops**.
- The main agent does **not** read or start the next task. It outputs a short summary and waits for the user’s “continue” or next instruction.
- This rule applies regardless of whether the feature was built by the main agent directly or delegated to a subagent.

## Example Workflow
1. **Main agent** reads next unchecked task: `[ ] Feature 2: Q&A agent with Ollama`.
2. It identifies this as a feature‑size task and spawns a subagent.
3. **Subagent** receives `TASKS.md` entry, `src/agent.py`, `src/main.py`, `requirements.txt`, and this protocol.
4. Subagent builds the feature, returns `agent.py`, `main.py`, and a summary.
5. Main agent updates `TASKS.md`, commits the changes, prints “Feature 2 complete”, and **halts**.