# Multi-Camera Tracker — Staged Agent Prompts

21 files: one global contract plus 20 sequential build stages, each with a mandatory tests substage.

## How to use

Give the coding agent **`00_GLOBAL_CONTRACT.json` plus one stage file** per session. The contract carries the shared schemas, tech stack, repo layout, and testing standards; the stage file carries that stage's tasks, tests, and exit criteria.

Do not run a stage until the previous stage's `exit_criteria` are met and its tests pass.

## Stage map

| # | Stage | What it produces |
|---|---|---|
| 00 | Global contract | Shared schemas, conventions, standards (not a stage — read first) |
| 01 | Project scaffolding | Repo, config, logging, exceptions, CI, coverage gate |
| 02 | Domain models | Validated Pydantic entities for every contract type |
| 03 | Persistence layer | Postgres + pgvector schema, migrations, repositories, in-memory fakes |
| 04 | Camera topology | Camera graph, travel-time windows, plausibility + reachability queries |
| 05 | Synthetic data generator | Ground-truth datasets with realistic OCR/embedding noise |
| 06 | Plate matching engine | Normalization, fuzzy matching, calibrated confidence, evaluation |
| 07 | Embedding re-id matching | Vector similarity fallback, evidence combination, topology constraints |
| 08 | Path reconstruction | DAG-based optimal trajectory, gaps, alternatives, explanations |
| 09 | Time synchronization | Clock offsets, drift detection, temporal integrity gate |
| 10 | Video ingestion | Source abstraction, sampling, motion prefilter, live streams |
| 11 | Vehicle detection | YOLO detection, single-camera tracking, best-frame ranking |
| 12 | Plate detection + OCR | Plate localization, rectification, multi-frame read fusion, calibration |
| 13 | Re-id embedding extraction | Real embedding model, pooling, quality gating, threshold recalibration |
| 14 | Pipeline orchestration | End-to-end batch pipeline, checkpointing, idempotency, `search_target` |
| 15 | Live processing | Continuous runner, load shedding, watchlist, alerting |
| 16 | API layer | REST endpoints, auth, authorization matrix, audit logging |
| 17 | Visualization UI | Map + timeline with confidence encoded visually |
| 18 | Human review workflow | Adjudication queue, decision propagation, threshold feedback |
| 19 | Observability + governance | Metrics, tracing, silent-failure detection, retention enforcement |
| 20 | System validation + release | Acceptance tests, accuracy measurement, packaging, sizing |

## Structural notes

**Stages 5–8 come before any computer vision.** The generator produces ground truth, so the matching and pathing engines can be measured precisely before real video introduces uncertainty about what the right answer even is. When CV arrives in stages 11–13, the logic consuming it is already validated.

**Every stage's tests substage names concrete cases**, including boundary values and failure paths — not "write tests." Several stages have a defining test the whole stage exists to pass:

- Stage 8: a mid-chain false positive must be excluded from the optimal path
- Stage 9: drift correction on/off must visibly change the reconstructed route
- Stage 14: kill-and-resume must equal an uninterrupted run
- Stage 20: no failure mode may produce a confident wrong trajectory

**Governance requirements are functional requirements** in the contract, not appendices — audit logging, retention, confidence surfacing, and the human review gate are specified alongside the features they constrain, because they are much harder to retrofit than to build in.

## Suggested first session

Give the agent `00_GLOBAL_CONTRACT.json` + `01_project_scaffolding.json`, and ask it to report back with the completion report format specified in the contract's `execution_rules`.
