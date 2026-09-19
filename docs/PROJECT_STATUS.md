# Project status

What each completed stage delivered, what it measured, and where it departed
from the plan. Written so a new session can trust the finished work without
re-reading it.

**Position: stages 00–11 complete. Stage 12 next, not started.**

All commits are on `develop`. Verification at the last commit: `ruff` clean,
`mypy --strict` clean across 109 source files, **1,828 tests passing**, 130
skipped, 93.11% coverage. Of the skips, 121 are Docker-dependent and 9 need
model weights.

---

## Commit history

| commit | stage | date |
|---|---|---|
| `2bf0c9f` | 00 — repository, `.gitignore`, plan committed | 2026-08-11 |
| `596cf87` | 01 — scaffolding, tooling, CI | 2026-08-11 |
| `f602280` | 02 — domain models | 2026-08-11 |
| `a31bdb2` | 03 — schema, migrations, repositories | 2026-08-11 |
| `ff122f6` | 04 — camera topology and travel times | 2026-08-12 |
| `ae741fc` | 05 — synthetic generator and ground truth | 2026-08-18 |
| `51d08b2` | 06 — plate matching engine | 2026-08-22 |
| `dd0e6d8` | 07 — embedding re-identification | 2026-08-22 |
| `63917bb` | docs — requested tracking modes and prediction | 2026-08-23 |
| `2ef6bd5` | 08 — trajectory assembly and path reconstruction | 2026-08-23 |
| `32be503` | 09 — time synchronization and temporal integrity | 2026-08-25 |
| `61ba055` | 10 — video ingestion, sampling, frame sources | 2026-08-27 |
| `fdf7e3a` | docs — handoff pack | 2026-09-18 |
| `b841dde` | 11 — vehicle detection and single-camera tracking | 2026-09-19 |

---

## Stage 01 — Scaffolding

`config.py`, `exceptions.py`, `logging_config.py`, `clock.py`, `pyproject.toml`,
`Makefile`, `scripts/dev.ps1`, `docker-compose.yml`, CI workflow, README and
ARCHITECTURE.

Two decisions that everything since depends on:

- **`ThresholdSettings` fields are all required with no Python defaults.** "No
  magic numbers" becomes a startup assertion: a threshold added to the model but
  forgotten in `config/thresholds.yaml` fails at import, not in production.
- **Nothing calls `datetime.now()` directly.** Code that needs the time takes a
  `Clock`. `FixedClock` is what makes every time-dependent test deterministic.

## Stage 02 — Domain models

`models/{base,enums,camera,sighting,target,match,trajectory,geo,timewindow}.py`.

- `UtcDatetime` truncates to milliseconds **at validation**, so a round trip
  through storage is lossless.
- `created_at` is required rather than defaulted — a `default_factory` would be
  a hidden `datetime.now()` inside the model.
- `Trajectory` enforces strictly ascending sightings, one hop per adjacent pair
  with matching endpoints, and bounds that equal the first and last sighting.

## Stage 03 — Persistence

`db/{folding,orm,session,mappers}.py`, `db/repositories/`, migration `0001`,
in-memory fakes, and a **shared conformance suite** that runs every repository
test twice — once against the fakes, once against Postgres — so a unit test
written against a fake is a truthful prediction about production.

- Hand-written mappers between Pydantic and SQLAlchemy, deliberately: an ORM
  that silently reshapes a domain model is how invariants get lost.
- pgvector with an HNSW index; the planner is asserted to actually use it.

## Stage 04 — Topology

`topology/{graph,loader,plausibility,reachability,coverage,sync}.py`,
`config/topology.yaml`, [`docs/TOPOLOGY.md`](TOPOLOGY.md).

- Travel-time windows are **inclusive at both ends**; a hop's verdict must not
  depend on floating-point equality.
- `plausibility_score` decays exponentially rather than cutting off: an
  implausible transit is unlikely, not impossible, and a hard zero would discard
  a real detour instead of ranking it last.

## Stage 05 — Synthetic data

`synth/{scenario,routes,plate_noise,embeddings,traffic,adversarial,ground_truth,generator}.py`,
six scenario YAMLs, `scripts/generate_dataset.py`,
[`docs/SYNTHETIC_DATA.md`](SYNTHETIC_DATA.md).

Byte-identical determinism from a seed. Six scenarios: `clean`, `realistic`,
`degraded`, `hard_negatives`, `sparse_coverage`, `adversarial`. Every later
stage is measured against these.

Two bugs found and fixed here, both of which would have made later measurements
meaningless: embedding noise was scaled absolutely rather than by `1/√dimension`
(so similarity depended on vector width), and the ground truth was assembled
before injections rather than after (so injected duplicates appeared in one
index and not another).

## Stage 06 — Plate matching

`matching/{normalize,folding,distance,plate_match,scoring,search,conflicts,evaluation}.py`,
`config/confusion_map.yaml`, `scripts/evaluate_matching.py`,
[`docs/PLATE_MATCHING.md`](PLATE_MATCHING.md).

Ambiguity folding (O/0, I/L/1, S/5, B/8, Z/2, G/6) for **comparison only** —
never stored, never displayed. Damerau-Levenshtein with confusion-weighted
substitution costs.

**Measured:** the threshold sweep found a *structural* cliff at weighted
distance 1.0 — precision 1.000 at 0.9 and below, 0.625 at 1.0 and above —
because an in-group substitution costs 0.5 and an arbitrary one costs 1.0.
Intuition would have picked 1.5.

## Stage 07 — Appearance re-identification

`matching/{similarity,aggregation,embedding_search,evidence,references,versioning,reid_evaluation}.py`,
[`docs/REID_MATCHING.md`](REID_MATCHING.md).

| scenario | emb precision | auto-accepted FP | plate failures recovered |
|---|---|---|---|
| clean / realistic / degraded / sparse | 1.000 | 0 | 5/5 |
| hard_negatives | 0.160 | **0** | 1/1 |
| adversarial | 0.267 | **0** | 1/1 |

**Measured:** the contract proposed an auto-accept threshold of 0.92; at 0.92
twenty-one hard-negative decoys are auto-accepted, at 0.95 none are, and recall
is unchanged at 1.000. Topology-constrained retrieval beats unconstrained
(precision 0.100 → 0.160 on hard negatives) at zero cost to recall, and shrinks
the candidate set 27–46% elsewhere.

Visual evidence is structurally incapable of outranking a plate match: the
visual-only ceiling sits below the weakest retained plate score, and that
inequality is now a startup assertion.

## Stage 08 — Path reconstruction

`pathing/{preparation,graph,optimal_path,kbest,gaps,confidence,incremental,movement,explanation,reconstruction,evaluation}.py`,
`scripts/evaluate_pathing.py`, [`docs/PATH_RECONSTRUCTION.md`](PATH_RECONSTRUCTION.md).

Global scoring over whole routes rather than greedy chaining, which is what
makes a mid-chain false positive rejectable. Exact optimum in one linear pass,
because time order is a topological order.

| scenario | exact | P | R | teleports | ambiguous |
|---|---|---|---|---|---|
| clean / realistic / sparse_coverage | ✔ | 1.000 | 1.000 | 0 | no |
| degraded | ✘ | 1.000 | 0.800 | 0 | no |
| hard_negatives | ✘ | 0.444 | 0.800 | **0** | **yes** |
| adversarial | ✘ | 0.385 | 1.000 | **0** | **yes** |

**Measured:** the per-node inclusion bonus is **zero**, against the contract's
expectation. A fixed payment per node is exactly the margin by which a
visual-only candidate gets chained in; at 0.05 the engine strings decoys through
the adversarial route until no competing account survives and reports it as
unambiguous — confidently wrong.

## Stage 09 — Time synchronization

`timesync/{sources,derivation,offsets,estimation,drift,integrity,timezones,watermark,monitoring}.py`,
migration `0002`, `scripts/estimate_camera_offsets.py`,
[`docs/TIME_SYNCHRONIZATION.md`](TIME_SYNCHRONIZATION.md).

The definitive pair, on identical data: with a five-minute drift uncorrected the
route puts the vehicle at `cam_04` before `cam_03` at confidence 0.10 with two
invented gaps; corrected, it is the ground truth at 0.97 with none.

Added `Trajectory.temporal_integrity` (nullable — "unchecked" is a third state,
distinct from "clean") and `requires_recomputation`, with repository methods
`apply_clock_offset` and `flag_for_recomputation`.

## Stage 10 — Video ingestion

`ingest/{frame,protocol,synthetic_source,file_source,live_source,samplers,motion,preprocess,multi_source}.py`,
`scripts/make_sample_clips.py`, `tests/fixtures/local_stream_server.py`, five
committed sample clips, [`docs/INGESTION.md`](INGESTION.md).

`SyntheticVideoSource` is the deliverable that matters most: stages 11–13 can be
tested in CI with no media, no network, and no GPU.

Installed `opencv-python` (already declared in the `vision` extra) and added
`av` to that extra, so the file and live paths are verified against real
decoders and a real loopback socket rather than skipped.

## Stage 11 — Vehicle detection and single-camera tracking

`vision/{detector_protocol,filters,best_frames,track,tracker,thumbnails,sightings,fake_detector,fixture_detector,yolo_detector,instrumentation,factory}.py`,
`scripts/{benchmark_detection,record_detection_fixtures}.py`,
`tests/fixtures/{vision.py,detections/,images/}`, [`docs/DETECTION.md`](DETECTION.md).

**One vehicle pass produces exactly one sighting**, built from every frame it
appeared in. Without that, a car crossing one camera at 5 fps is forty sightings
of forty vehicles as far as everything downstream can tell.

A SORT-style tracker: Kalman prediction plus minimum-cost assignment over IoU.
Both parts earn their place at the same moment — when two vehicles cross,
prediction keeps them apart and optimal assignment stops greedy matching pairing
them the wrong way round. Each is tested directly, the second against a cost
matrix where greedy is provably wrong.

Memory is bounded by construction: a track keeps a few numbers per frame and a
cropped image for only its best few, so a vehicle parked in view for ten minutes
costs kilobytes.

**Not measured.** Every stage 11 threshold is a documented starting point, not a
swept optimum — the only detector available here is synthetic, and sweeping
against it would calibrate the fixture. Stage 13 or 20 must re-derive them.

Four things the tests found that reading the code would not have:

- stage 10's rotation inverse can return -1 at the frame boundary (pixel-centre
  convention against half-open box coordinates); clamped at one pixel, and
  anything larger still fails loudly;
- `sample_clean.mp4` contains **two** passes, because its bar wraps from the
  right edge to the left on frame 18 — and the tracker is right to refuse to
  stitch a teleport into one track;
- `SyntheticVideoSource(moving_rectangle=False)` is not a static scene; its
  background drifts above the motion gate's sensitivity every frame;
- a sighting truncates to milliseconds while a decoder reports microseconds.


---

## What is measured, and where

Every quality claim in this project is a number produced by a committed script
and guarded by a committed baseline:

| area | script | baseline |
|---|---|---|
| plate matching | `scripts/evaluate_matching.py` | `tests/integration/matching/baselines/plate_metrics_baseline.json` |
| appearance re-id | `scripts/evaluate_matching.py --reid` | `tests/integration/matching/baselines/embedding_metrics_baseline.json` |
| path reconstruction | `scripts/evaluate_pathing.py` | `tests/integration/pathing/baselines/pathing_metrics_baseline.json` |
| clock offsets | `scripts/estimate_camera_offsets.py` | (diagnostic; no baseline) |
| detection throughput | `scripts/benchmark_detection.py` | (diagnostic; no baseline) |
| CI detections | `scripts/record_detection_fixtures.py` | `tests/fixtures/detections/sample_clips.json` |

Rewriting a baseline is a deliberate act. A diff there is a claim that quality
changed on purpose.
