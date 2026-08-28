# TODO

What is left. Three kinds of work: the remaining stages, the debts that are not
stages, and the requested features that are outside the plan.

---

## 1. Remaining stages

Each is specified in `coding-agent-prompts/`. **One per instruction, then stop.**

### Next: stage 11 — Vehicle detection

`coding-agent-prompts/11_vehicle_detection.json`. Detect vehicles in sampled
frames and associate detections across consecutive frames into single-camera
tracks, so each vehicle pass produces one track with a best frame rather than
forty near-duplicate sightings.

What is already in place for it:

- `SyntheticVideoSource` produces deterministic frames with a moving rectangle,
  so detection and tracking are testable with no media and no GPU. **The stage
  10 exit criterion was that stages 11–13 need no real media in CI — hold to
  that.**
- `Preprocessor` returns a `CoordinateMapping`; every emitted bbox must go back
  through it into original source-frame coordinates.
- `Sighting.bbox` is validated `[x1, y1, x2, y2]`, non-negative, positive area.
- OpenCV and PyAV are installed. `ultralytics` and `torch` are declared in the
  `vision` extra but **not installed** — stage 11 will need a decision about
  whether to install YOLO weights or keep the detector behind an interface with
  a synthetic implementation for CI.

### Then

| stage | what it produces |
|---|---|
| 12 | plate detection, rectification, multi-frame OCR fusion, calibration |
| 13 | real re-id embedding model, pooling, quality gating, **threshold recalibration** |
| 14 | end-to-end batch pipeline, checkpointing, idempotency, `search_target` |
| 15 | live runner, load shedding, watchlist, alerting |
| 16 | REST API, auth, authorization matrix, audit logging |
| 17 | map + timeline UI with confidence encoded visually |
| 18 | human review queue, decision propagation, threshold feedback |
| 19 | metrics, tracing, silent-failure detection, retention enforcement |
| 20 | acceptance tests, accuracy measurement, packaging, sizing |

Stages with a defining test the whole stage exists to pass:

- **13** — every threshold calibrated in stages 06–08 must be re-swept against
  real embeddings. The synthetic numbers do not transfer.
- **14** — kill-and-resume must equal an uninterrupted run.
- **20** — no failure mode may produce a confident wrong trajectory.

---

## 2. Debts and gaps

Not stages, but real, and each will be someone's surprise if left unrecorded.

### Docker: 120 tests have never run

Every Postgres-backed test skips on this machine. That includes the conformance
suite's Postgres half, pgvector search and HNSW index-usage assertions,
migrations `0001` and `0002`, constraint enforcement, and stage 09's
transactional offset rewrite.

The in-memory fakes plus the conformance suite mean these are untested
*bindings*, not untested logic — but **they must run in CI before any release**,
and `migrations/versions/0002_temporal_integrity.py` has never been applied to a
real database.

### Stage 09 leaves a rough edge in the trajectory read path

Correcting a camera's clock moves its sightings, so a trajectory stored earlier
no longer matches its own recorded bounds and **raises on read**. That is
deliberate — a conclusion whose evidence moved must not be silently re-served —
and `requires_recomputation` is how those routes are found. But nothing yet
*performs* the recomputation. Stage 14 or 18 should close that loop.

### The synthetic thresholds are placeholders

`embedding_auto_accept_min_similarity: 0.95` and everything swept in stages
06–08 were calibrated against stage 05's noise model. Stage 13 must regenerate
`docs/REID_MATCHING.md`'s sweep tables against real embeddings before those
numbers are trusted anywhere near production.

### Keyframe flags are approximate on the OpenCV path

OpenCV does not expose them, so `FileVideoSource` uses a fixed interval there and
`KeyframeOnlySampler` is correspondingly approximate. Exact on the PyAV path.

### Live frames are timestamped on arrival

There is no container to ask, so network latency sits inside the timestamp.
Where a camera exposes its own clock, stage 09's probe is the better source and
stage 15 should wire it in.

---

## 3. Requested features outside the plan

Recorded in full in [`FEATURE_BACKLOG.md`](FEATURE_BACKLOG.md). Summary:

| feature | attaches to | blocked on |
|---|---|---|
| match a target from an uploaded photograph | 13 / 14 / 16 / 17 | real embeddings; **needs its own calibrated thresholds** — a phone photo against a CCTV crop is a different similarity distribution |
| track every vehicle, not one nominated target | 14 | 11–13; also changes the data footprint enough that retention and access control should be revisited **before** it ships |
| operator picks a target from the live feed | 15 / 17 | 11, 15, 17 — cheapest and most reliable of the three modes |
| predict the next camera and arrival window | after 08, used by 15 and 17 | stage 08 (done) + real detections; must never be renderable as an observation |

None is buildable before the plan reaches stage 13.

---

## 4. Housekeeping

- `pyproject.toml` declares `opencv-python` and `av` in the `vision` extra; both
  are installed locally. `ultralytics`, `torch`, `torchvision`, `paddleocr`, and
  `pillow` are declared and **not** installed.
- The CI workflow runs the full gate including the coverage threshold. It has
  never been exercised against a real Postgres service container — verify that
  when Docker becomes available.
- `docs/` now has nine files. `ARCHITECTURE.md` predates stages 08–10 and could
  use a pass to mention `pathing`, `timesync`, and `ingest`.
