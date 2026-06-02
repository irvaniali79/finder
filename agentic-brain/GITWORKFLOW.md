## SKILL: Strict Git Workflow (Feature / Bugfix)

**Purpose:** Enforce a disciplined branching model where every change lives on a dedicated `feature/` or `bugfix/` branch. The agent **must** follow these rules for any code modification.

### 1. Branch Naming & Creation
- **New work must start from the latest `main` (or `master`).**
- Determine the nature of the task:
  - **Feature:** `feature/<short-description>` or `feature/<issue-id>-<short-description>`
  - **Bugfix:** `bugfix/<short-description>` or `bugfix/<issue-id>-<short-description>`
- **Never commit directly to `main`, `master`, or any protected branch.**
- Before creating the branch, ensure the local `main` is up to date:
  ```bash
  git checkout main
  git pull origin main
  ```
- Create the branch and switch to it:
  ```bash
  git checkout -b feature/awesome-thing
  ```

### 2. Working on the Branch
- Keep changes focused – one logical change per branch.
- Commit early and often with **Conventional Commit** messages:
  - `feat: add user login`
  - `fix: correct calculation error`
  - `refactor: extract helper function`
  - `docs: update README`
  - `test: add unit tests for auth`
- **Prefixes** map to branch types:
  - `feature/*` branches should primarily use `feat:` commits.
  - `bugfix/*` branches should primarily use `fix:` commits.
- Before committing, review the diff:
  ```bash
  git diff --staged
  ```
- Never commit sensitive files (.env, secrets, etc.). Use `.gitignore`.

### 3. Keeping the Branch Updated
- Regularly integrate changes from `main` into your branch to avoid conflicts:
  ```bash
  git fetch origin
  git rebase origin/main
  ```
- Resolve conflicts immediately if they arise. After a successful rebase, force-push your branch only if it has already been pushed (with caution):
  ```bash
  git push --force-with-lease
  ```
  **Never force-push `main` or shared branches.**

### 4. Finishing the Task
- Once the work is complete and tested, push the branch:
  ```bash
  git push origin feature/awesome-thing
  ```
- If the tool has PR/Merge Request capabilities, create one against `main` with a clear description.
- If no PR tooling is available, the agent must **never** merge locally into `main`. Instead, instruct the user to review and merge via the repository UI.
- **Exception**: Only if explicitly instructed by the user and after confirmation, the agent may merge locally using:
  ```bash
  git checkout main
  git pull origin main
  git merge --no-ff feature/awesome-thing
  git push origin main
  ```
  This must be a deliberate, user-approved action.

### 5. Absolute Prohibitions
- ❌ **Never commit on `main`** – not even a "small fix".
- ❌ **Never push to `main` directly.**
- ❌ **Never delete remote branches** unless the user asks to clean up merged branches.
- ❌ **Never use `git push --force` on `main` or shared branches.**
- ❌ **Never skip hooks (--no-verify, --no-gpg-sign) unless the user demands it explicitly.**

