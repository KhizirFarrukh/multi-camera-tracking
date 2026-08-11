# Contributing

This project is built in 20 numbered stages. The process is unusual enough to be
worth reading before you touch anything.

---

## The stage process

Work is defined by [`coding-agent-prompts/`](coding-agent-prompts/): one global
contract plus 20 sequential stage files.

1. **Read [`00_GLOBAL_CONTRACT.json`](coding-agent-prompts/00_GLOBAL_CONTRACT.json) first.**
   It carries the canonical schemas, tech stack, repository layout, and testing
   standards that every stage inherits. Re-read it if you lose context.
2. **Execute stages in ascending order.** Do not start stage N+1 until stage N's
   `exit_criteria` are met and its tests pass.
3. **No scope creep.** Implement exactly what the stage specifies. If a stage
   needs something a later stage builds, stub it behind the interface defined in
   the contract and leave a `TODO` naming the future stage number.
4. **No placeholder logic.** A function must never silently return fake data.
   Stubs raise `NotImplementedError` naming the stage that will implement them.
5. **Tests are part of the stage.** Every stage has a `tests_substage`. The stage
   is not complete until those tests are written and passing.
6. **Report back.** Finish each stage with a completion report: files created or
   modified, tests written, test results, deviations from spec with
   justification, and assumptions made.

### Commits

One commit per stage, minimum:

```
stage-NN: <short description>
```

Work happens on `develop`. `main` holds the bootstrap commit and, later,
released state.

---

## Local setup

```bash
make install        # uv sync --extra dev
make db-up          # Postgres 16 + pgvector via docker compose
make ci             # lint + format check + types + tests
```

Windows without GNU make — `scripts/dev.ps1` mirrors every target exactly:

```powershell
.\scripts\dev.ps1 install
.\scripts\dev.ps1 ci
```

The vision extra (`torch`, `ultralytics`, `paddleocr`) is not required before
stage 11 and is deliberately excluded from CI's fast path. Install it with
`make install-vision` when you get there.

---

## Standards

These are enforced, not suggested. `make ci` is the gate.

| Area | Rule |
|---|---|
| Typing | Full annotations. `mypy --strict` passes on `src/`. |
| Lint | `ruff check` and `ruff format --check` clean. Line length 100. |
| Coverage | ≥ 85% line coverage on `src/multicam_tracker/`. |
| Logging | Structured, via the project logger. No `print` in `src/`. |
| Config | No hardcoded paths, hosts, credentials, or thresholds. |
| Errors | Everything derives from `MulticamTrackerError`. Never raise bare `Exception`. |
| Time | Timezone-aware UTC internally. Naive datetimes rejected at boundaries. |
| Docstrings | Every public function and class: purpose, args, returns, raises. |

---

## Writing tests

**Naming:** `test_<unit_under_test>__<condition>__<expected_outcome>`

```python
def test_settings__nested_delimiter_env_var__populates_nested_section() -> None:
```

**Required case classes** for anything non-trivial:

- happy path
- boundary values — empty, single element, maximum size
- invalid or malformed input
- error propagation and exception types
- idempotency, where the operation claims to be idempotent
- timezone correctness, for anything touching timestamps

**Determinism is mandatory.** Seed all randomness. Inject a `FixedClock` rather
than patching `datetime`. Never sleep for synchronization. A flaky test is
treated as a failing test.

**Unit vs integration:**

| | `tests/unit/` | `tests/integration/` |
|---|---|---|
| Database | never | testcontainers Postgres |
| Network | never | allowed |
| Model weights | never | allowed where the stage requires |
| Speed | milliseconds | seconds |

Shared fixtures go in [`tests/conftest.py`](tests/conftest.py) or
`tests/fixtures/`. Do not duplicate fixture logic across modules.

Markers: `unit`, `integration`, `slow`, `requires_gpu`, `requires_models`.
`--strict-markers` is on, so an unregistered marker is an error.

### Test isolation

`tests/conftest.py` neutralises three channels through which a developer's
machine could change a result: `MCT_` environment variables are cleared, the
working directory moves to a scratch path so a local `.env` is not discovered,
and the settings cache is reset around every test. Build settings through the
`build_settings` fixture rather than calling `Settings()` directly.

---

## Ad-hoc test runs

Coverage is on by default, which means a single-file run reports a failing
coverage gate. That is expected; pass `--no-cov` when iterating:

```bash
uv run pytest tests/unit/test_config.py --no-cov -k thresholds
```

## The coverage gate needs Docker

From stage 03 onward, part of `src/` — the Postgres repository implementations —
can only execute against a real database. A unit-only run therefore *cannot*
reach 85%, and lowering the gate to accommodate that would defeat its purpose.

So the gate is enforced where it can be measured honestly:

| Command | Runs | Coverage gate |
|---|---|---|
| `make ci-quick` | lint, types, unit + in-memory conformance | no |
| `make ci` | everything, including database tests | **yes** |
| CI `quality` job | lint, types, unit + in-memory conformance | no |
| CI `integration` job | everything | **yes** |

Without Docker running, `make ci` will fail on coverage. That failure is
accurate: a third of the persistence layer really is untested on that machine.
Use `make ci-quick` while iterating, and `make ci` before opening a PR.

## The conformance suite

`tests/conformance/` holds one set of behavioural tests that runs against **both**
the in-memory fakes and a real Postgres. Every stage from 04 onward writes its
unit tests against the fakes, and that is only sound if the fakes behave exactly
like the database.

When you add a repository method, add it to the fake and to the conformance
suite in the same change. A fake that quietly differs turns every unit test
above it into a prediction about a system that does not exist.
