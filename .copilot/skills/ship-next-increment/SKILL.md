---
name: ship-next-increment
description: Advances the llm-batcher proxy (NeverTheSame/llm-batcher) by one roadmap increment end to end. Picks the next undone row from the README roadmap, posts a short plan in chat, gets a design critique, implements the feature with no-network tests, updates README and .env.example, writes an internal deep-dive Markdown explainer, builds a public self-contained HTML visualization, then commits and pushes the public artifacts while keeping the explainer internal. Never labels work "Day N". Never uses em-dashes or en-dashes. Use when the user asks to "ship the next feature", "do the next roadmap step", "what's next / build it", "advance llm-batcher", or names a specific roadmap item to implement.
---

# ship-next-increment: Advance llm-batcher by One Roadmap Increment

End-to-end playbook for taking the `llm-batcher` proxy forward by exactly one
roadmap step, in the same shape as the increments already shipped. The output of
one run is: a tested feature pushed to the public repo, an internal deep-dive
explainer, and a public interactive visualization.

**Repo:** `NeverTheSame/llm-batcher` (public). Default branch, `origin` remote.
**Project:** an OpenAI-compatible FastAPI proxy in front of the Anthropic
Messages API. Python 3.12, venv at `./venv/`.

## Golden Rules (do not skip)

1. **Never label anything "Day N".** Not in code, comments, docstrings, the
   README, the explainer, the visualization, file names, or commit messages.
   Name everything after the feature (for example `BATCHING`, not `DAY2`).
   Older files like `DAY1.md` predate this rule; do not add new ones.
2. **Never use em-dashes or en-dashes** in any file or commit message. Use
   commas, periods, parentheses, colons, or split the sentence. Verify with
   `grep -n "—\|–" <file>` before committing (expect no matches in new files).
3. **Default off, existing tests stay green.** New behavior is opt-in via env
   config so the existing realtime path and translation tests are unchanged.
4. **Tests are deterministic and do no network.** Mock the upstream. No real
   `ANTHROPIC_API_KEY` is available. Bound any time-based test with
   `asyncio.wait_for(..., timeout=5)`.
5. **Design must be interview-defensible.** Prefer the operationally correct
   choice over the flashy one, and be ready to justify it. Get a rubber-duck
   critique on the plan before writing code (Step 3).
6. **Two artifacts, two visibilities.** The Markdown deep-dive is INTERNAL
   (gitignored, never committed). The HTML visualization is PUBLIC (committed).
   This mirrors `BATCHING.md` (internal) and `batching.html` (public).
7. **Commit trailer.** End every commit message with:
   `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
   unless the user says otherwise.

## Repo Layout (where things go)

```
app/main.py              FastAPI app: config env vars, lifespan, endpoints, wiring
app/<feature>.py         New feature module (for example app/batcher.py)
tests/test_<feature>.py  No-network tests for the feature
README.md                "Roadmap" table + a per-feature "## <Feature>" section
.env.example             Documented env vars (new <FEATURE>_* knobs)
<FEATURE>.md             INTERNAL deep-dive explainer (add to .gitignore)
<feature>.html           PUBLIC self-contained visualization (committed)
.gitignore               Lists internal-only docs under its "Internal-only" block
```

## Step 1: Pick the next increment

Read the roadmap table in `README.md` (section `## Roadmap`). Each row is
`| N | description |`; rows already shipped are marked with a check. The next
increment is the first row WITHOUT a check. If the user named a specific item,
use that instead. Restate the chosen increment in your own words and derive a
short, feature-based slug (for example "Cost & latency observatory" gives the
module `app/metrics.py`, files `OBSERVABILITY.md` and `observability.html`).

Do NOT use the row number as a name anywhere in the deliverables.

## Step 2: Post a brief plan in chat

Write a SHORT plan (a few sentences, not a document) covering: what the feature
does, the env knob(s) that gate it, where the code goes, and how it will be
tested without a network. This is the same "post the plan, then build it" rhythm
used for the prior increments. Keep momentum; do not wait for approval unless the
user asked to review first.

## Step 3: Get a design critique (required for non-trivial work)

Call the rubber-duck agent with the plan and the key design choice. Ask it to
surface correctness risks, edge cases, and whether the operationally correct
primitive was chosen. Fold in the findings that prevent bugs; note and skip ones
that add complexity without benefit. This step caught the "use microbatching, not
the offline Batches API" correction on the prior increment.

## Step 4: Implement the feature

- Add `app/<feature>.py` with the core logic. Keep it dependency-light; prefer
  the standard library and what is already in `requirements.txt`.
- Wire it into `app/main.py`:
  - Add `<FEATURE>_*` env constants near the other config (default OFF).
  - If you need startup/shutdown resources, use the existing FastAPI `lifespan`
    asynccontextmanager (do NOT use the deprecated `@app.on_event`).
  - Route through the new code only when its env flag is enabled; otherwise keep
    the existing direct path untouched.
- Honor Golden Rules 1 and 2 in every line you write (no "Day N", no dashes).

## Step 5: Tests (no network)

- Add `tests/test_<feature>.py`. Drive logic directly with `asyncio.run` and
  injected fakes/mocks. Assert the observable behavior and the math.
- Provide a test hook on the feature object if needed for deterministic
  assertions (the batcher exposes an `on_flush(size)` hook for this reason).
- Run the full suite and keep it green:
  ```bash
  ./venv/bin/pytest -q
  ```
  A pre-existing Starlette TestClient httpx deprecation warning is harmless.
- Run two modes so the default-off guarantee is protected: the full suite with
  default env (feature OFF), and the feature tests with its flag enabled (set the
  env var or monkeypatch in the test). Both must pass.

## Step 6: Update README and .env.example

- In `README.md`: mark the roadmap row done (add the check), and add a
  `## <Feature>` section explaining what it does, the env knobs, and the
  tradeoff. No "Day N", no dashes.
- In `.env.example`: add the new `<FEATURE>_*` variables with brief comments and
  safe defaults (feature OFF by default).

## Step 7: Internal deep-dive Markdown (NOT committed)

Write `<FEATURE>.md`: a sophisticated explainer aimed at a technical blog reader.
Cover the problem, the primitive you rejected and why, parallels to real systems
(name them), the relevant theory, a walkthrough of the code, configuration,
testing approach, and honest limitations. Then keep it internal:

```bash
# add the file under the "Internal-only" block in .gitignore
grep -q "^<FEATURE>.md$" .gitignore || \
  printf '%s\n' "<FEATURE>.md" >> .gitignore
```

Verify it is ignored: `git check-ignore <FEATURE>.md` should print the name.
Verify no dashes: `grep -n "—\|–" <FEATURE>.md` should print nothing.

## Step 8: Public HTML visualization (committed)

Write `<feature>.html`: a single self-contained file (inline CSS and JS, no
external build, no network calls) that visualizes WHY the feature matters. Use an
animation or interactive diagram that contrasts the naive approach with the new
one, a short decision/tradeoff section, and concept "chips" that link out to
primary sources for the ideas involved. Match the look and structure of
`batching.html`.

Verify it opens and has no dashes:
```bash
grep -n "—\|–" <feature>.html   # expect nothing
open <feature>.html             # macOS: eyeball it
```

## Step 9: Commit and push (public only)

Stage the public artifacts plus the `.gitignore` change. Do NOT stage the
internal `<FEATURE>.md` (it is ignored, so `git add -A` is safe, but double
check). Then commit and push:

```bash
git add app/ tests/ README.md .env.example <feature>.html .gitignore
git diff --cached --name-only        # GUARD: <FEATURE>.md must NOT appear here
git diff --cached | grep -nP "—|–|Day [0-9]" && echo "STOP: forbidden content staged" || true
git status --short                   # confirm <FEATURE>.md is NOT staged
git commit -m "<concise feature summary>

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

If the internal `<FEATURE>.md`, any em/en dash, or a "Day N" label shows up in
the staged diff, STOP and fix it before committing.

Sanity checks after push:
- `git status` clean.
- `git check-ignore <FEATURE>.md` prints the name (still internal).
- `git ls-files <feature>.html` prints the file (public).

## Step 10: Report back

Summarize for the user: what shipped, the env knob to enable it, test result
(all green), the public HTML that was pushed, and the internal MD that was kept
out of the repo. Offer the next roadmap row as the following increment.

## Quick Reference: standing facts

- Public repo: `NeverTheSame/llm-batcher`. Run tests with `./venv/bin/pytest -q`.
- Existing increments to imitate: `app/batcher.py` (feature), `tests/test_batcher.py`
  (tests), `BATCHING.md` (internal explainer), `batching.html` (public viz).
- The Anthropic Message Batches API is an OFFLINE bulk primitive (slow SLA). Do
  not put it behind the synchronous chat endpoint. Realtime/operational patterns
  belong on the request path; batch/async primitives belong on a separate async
  endpoint if ever added.
- When killing a process, use a literal PID number, not a shell variable.
