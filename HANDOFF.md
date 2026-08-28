# Handoff — read this first

This file exists so that a new agent session, with no memory of the previous
conversation, can pick this project up and continue correctly.

**Current position: stages 00–10 are complete and committed. Stage 11 (vehicle
detection) is next, and is not started.**

---

## What this project is

`multicam-tracker` — a system that reconstructs where a specific vehicle went
across a network of cameras, from video, using licence plates and visual
appearance, and says how confident it is at every step.

It is being built by executing a pre-written 20-stage plan that lives in
[`coding-agent-prompts/`](coding-agent-prompts/). That directory is the
specification. `00_GLOBAL_CONTRACT.json` carries the shared schemas, tech stack,
conventions, and testing standards; each numbered file carries one stage's
tasks, its mandatory tests substage, and its exit criteria.

The plan is not mine and must not be edited to make work easier. Where reality
disagreed with it, the disagreement is recorded as a deviation (see
[`docs/DECISIONS.md`](docs/DECISIONS.md)) rather than resolved by changing the
spec.

---

## The four documents

| file | what it answers |
|---|---|
| **this file** | where am I, what do I do next |
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | what each finished stage delivered, and what it measured |
| [`docs/TODO.md`](docs/TODO.md) | what is left, including debts and gaps that are not stages |
| [`docs/WORKING_AGREEMENT.md`](docs/WORKING_AGREEMENT.md) | how to work here: protocol, environment, commands, traps |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | decisions already made and measured — do not re-litigate these |

Plus [`docs/FEATURE_BACKLOG.md`](docs/FEATURE_BACKLOG.md), which records
capabilities the user asked for that are **not** in the 20-stage plan.

---

## The standing instruction from the user

> "go through coding-agent-prompts, start working, one stage at a time, when one
> stage completes, commit, then wait until i tell you for next stage. also work
> in develop branch"

So: **one stage per instruction, then stop and wait.** Do not run ahead into the
next stage. Each stage ends with a commit on `develop` in the form
`stage-NN: <description>` and a completion report in the chat naming files,
tests, results, deviations, and assumptions.

---

## How to resume, concretely

1. Read [`docs/WORKING_AGREEMENT.md`](docs/WORKING_AGREEMENT.md) — the
   environment has two traps that will waste an hour if you meet them cold.
2. Confirm the tree is clean and the tests pass:

   ```bash
   git status
   git log --oneline -5
   # then the verification command from the working agreement
   ```

3. Read `coding-agent-prompts/11_vehicle_detection.json` in full, plus
   `00_GLOBAL_CONTRACT.json` if you have not.
4. Wait for the user to say to start. When they do, build stage 11 to the
   contract's standard, commit, report, and stop.

---

## The shape of the work so far

Ten stages, ~15,600 lines of source across 97 modules, 83 test modules,
**1,601 tests passing** and 121 skipped (all of them database tests waiting on
Docker), 92.85% coverage, `ruff` and `mypy --strict` clean.

The parts that exist:

```
config, logging, exceptions, clock      stage 01
domain models (Pydantic)                stage 02
Postgres + pgvector, repositories       stage 03
camera topology and travel times        stage 04
synthetic data with ground truth        stage 05
plate matching                          stage 06
appearance re-identification            stage 07
path reconstruction                     stage 08
time synchronization                    stage 09
video ingestion                         stage 10
--------------------------------------------------
vehicle detection                       stage 11  <- next
plate detection + OCR                   stage 12
re-id embedding extraction              stage 13
pipeline orchestration                  stage 14
live processing                         stage 15
API layer                               stage 16
visualization UI                        stage 17
human review workflow                   stage 18
observability and governance            stage 19
system validation and release           stage 20
```

Stages 05–10 were deliberately built **before** any computer vision, so the
matching and pathing engines could be measured against ground truth before real
video introduced uncertainty about what the right answer even is. Stage 11 is
where that changes.

---

## The one thing to understand about this system

Every stage is organised around the same failure: **a confident wrong answer.**

A trajectory that is missing a hop is a nuisance. A trajectory that is complete,
plausible, and wrong is the thing that gets someone accused of being somewhere
they were not. That is why:

- appearance evidence is structurally incapable of outranking a plate match;
- a route whose clock offsets could reorder its own hops is refused outright;
- unmonitored ground is reported as a gap rather than interpolated across;
- ambiguous cases return their alternatives instead of the best guess;
- every trajectory carries a machine-readable account of why each sighting is in
  it, and why the strongest excluded candidates are not.

If you are ever choosing between an answer and an honest refusal, this codebase
chooses the refusal, and says why.
