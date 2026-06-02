# Test Integrity Rule

The Source of Truth (`SSOT.md`) is absolute. If a test fails, the failure indicates that the system does not yet meet the business requirements. Modifying the test to match the system’s current (incorrect) behavior is forbidden under any circumstances.

## Process When a Test Fails
1. Check whether the test itself is correctly derived from `SSOT.md` and the agent’s documented behavior (`AGENT_CONTEXT.md`, prompt template).
2. If the test is correct but the system output is wrong, fix the **system** (code, retrieval, prompt), not the test.
3. If you cannot determine the right fix, **immediately halt** and ask the user for help. Describe the test, the expected output (from SSOT), and the actual output.
4. Do not mark the task as complete until the test passes **without any alteration to the test’s expected answer**.

This rule applies to unit tests, integration tests, and the evaluation QA pairs in `EVALS.md`.