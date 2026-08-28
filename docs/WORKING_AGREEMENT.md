# Working agreement

How to work in this repository: the protocol the user set, the environment's
traps, and the commands that actually work here.

---

## 1. The protocol

The user's standing instruction, verbatim:

> "go through coding-agent-prompts, start working, one stage at a time, when one
> stage completes, commit, then wait until i tell you for next stage. also work
> in develop branch"

So:

- **One stage per instruction.** Build it, commit it, report, then **stop**. Do
  not start the next stage because the previous one finished.
- **Work on `develop`.** `main` is the default branch on GitHub and is not
  written to directly.
- **Commit message format:** `stage-NN: <short description>`, then a body
  explaining the decisions and anything measured. Every commit ends with the
  `Co-Authored-By` trailer.
- **Report on completion**, in the chat, naming: files added and changed, tests
  and their results, deviations from the stage prompt, and assumptions made.

Git identity is set repo-locally: **Khizir Farrukh** `<khizirkfc@gmail.com>`.

---

## 2. The contract every stage inherits

`coding-agent-prompts/00_GLOBAL_CONTRACT.json` is binding. The parts that come
up constantly:

- **Stages in ascending order.** No skipping, no partial credit.
- **No scope creep.** Build the stage in front of you, not the one after it.
- **No placeholder logic.** A stub that is genuinely deferred raises
  `NotImplementedError` naming the future stage. A stub that silently returns a
  plausible value is forbidden — that is the confident-wrong-answer failure in
  miniature.
- **Tests are mandatory per stage**, and the stage prompt names the concrete
  cases, boundaries, and failure paths. "Write tests" is never the instruction;
  the cases are enumerated.
- **All thresholds live in `config/thresholds.yaml`.** No magic numbers in
  logic. The model declares every threshold as required with no default, so a
  forgotten one fails at startup.

---

## 3. Environment — the two traps

### 3.1 The virtualenv interpreter is blocked

Windows Smart App Control blocks `.venv\Scripts\python.exe` (os error 4551), so
`uv run`, `make`, and `scripts/dev.ps1` all fail. `uv pip install` still works.

**The workaround, which every command in this project uses:**

```bash
PY="/c/Users/Khizi/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe"
export PYTHONPATH=".venv/Lib/site-packages;src;."
"$PY" -m pytest -q
```

The uv-managed base interpreter is signed and runs; `PYTHONPATH` points at the
venv's packages, the source tree, and the repo root (for `tests.*` imports).

### 3.2 Bash heredocs mangle certain characters

Writing files through `cat > file <<'EOF'` in this environment corrupts
apostrophes inside the body (`unexpected EOF while looking for matching '`), and
silently converts `\x00` escapes into literal NUL bytes, which Python then
refuses to parse.

**Use the `Write` tool for file content.** For scripted edits, a Python heredoc
works — but keep single quotes out of the body, or write the patch script to the
scratchpad with `Write` and run it.

### 3.3 No Docker

120 tests skip with *"Docker is not available for integration tests"*. Every one
of them is a Postgres-backed test: the conformance suite's Postgres parameter,
pgvector search, migrations, constraints, and stage 09's transactional
offset rewrite.

**The in-memory fakes and the conformance suite mean these are not untested
logic — they are untested *bindings*.** They must run in CI before release.
Do not treat a green local run as proof the Postgres path works.

---

## 4. Commands

```bash
PY="/c/Users/Khizi/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe"
export PYTHONPATH=".venv/Lib/site-packages;src;."

"$PY" -m ruff format .            # format
"$PY" -m ruff check . --fix       # lint
"$PY" -m mypy src                 # strict type check, src only
"$PY" -m pytest -q                # full suite with coverage gate
"$PY" -m pytest tests/unit -q --no-cov   # fast loop
```

The full suite takes 75–120 seconds. The coverage gate is 85%; the project sits
at ~93%.

Evaluation scripts, all of which print tables and change nothing:

```bash
"$PY" scripts/evaluate_matching.py [--sweep|--reid|--compare-constraint]
"$PY" scripts/evaluate_pathing.py  [--sweep|--explain SCENARIO]
"$PY" scripts/estimate_camera_offsets.py [--drift]
"$PY" scripts/generate_dataset.py
"$PY" scripts/make_sample_clips.py    # regenerates the committed video fixtures
```

---

## 5. House style

The code in this repository reads a particular way, and matching it matters more
than any individual preference:

- **Docstrings say why, not what.** Every module opens by explaining the failure
  it exists to prevent. Every non-obvious decision carries the reason it was
  made, and often the alternative that was rejected and what went wrong with it.
- **Tests are named `test_<subject>__<expectation>`** and their docstrings
  explain what would break if the assertion failed.
- **Boundaries are tested at their exact values**, not near them.
- **Comments record the bug that motivated the line**, where there was one.
  Several of the sharpest comments in this codebase exist because the first
  version was wrong in a way that looked right.
- Google-style docstrings, `Args:`/`Returns:`/`Raises:`, line length 100, ruff
  rules `E,F,I,N,UP,B,SIM,RUF`, `mypy --strict` on `src/`.

---

## 6. When reality disagrees with the plan

It has, five times so far, and the answer each time was the same: **measure,
then record the disagreement.** See [`DECISIONS.md`](DECISIONS.md).

Do not edit `coding-agent-prompts/` to match what was built. The plan is the
specification; a deviation is a finding, and findings belong in the docs, the
commit message, and the completion report.
