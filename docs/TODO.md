# TODO

What is left. Three kinds of work: the remaining stages, the debts that are not
stages, and the requested features that are outside the plan.

---

## 1. Remaining stages

Each is specified in `coding-agent-prompts/`. **One per instruction, then stop.**

### Next: stage 12 — Plate detection and OCR

`coding-agent-prompts/12_plate_detection_ocr.json`. Find the plate within a
vehicle crop, rectify it, read it across several frames, and fuse the readings
into one string with a calibrated confidence.

What is already in place for it:

- `VehicleTrack.ranked` gives the highest-quality crops of each pass, best
  first, already padded and owned. **That ranking is what determines plate
  accuracy** — more than any threshold stage 12 will add. See
  [`DETECTION.md`](DETECTION.md).
- `track_to_sighting` produces a `Sighting` with `plate_text_raw`,
  `plate_text_normalized`, and `plate_confidence` all `None`, which is where
  stage 12 writes.
- Stage 06 already owns normalization, ambiguity folding, and the confusion map;
  stage 12 supplies readings to it and must not reimplement any of it.
- `paddleocr` is declared in the `vision` extra and **not installed**. The same
  decision stage 11 faced applies: keep OCR behind an interface with a fixture
  implementation so CI needs no model.

### Then

| stage | what it produces |
|---|---|
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

### Stage 11's thresholds were never swept

Every value in the stage 11 block of `thresholds.yaml` — minimum box area,
association IoU, track max age, the four best-frame weights — is a documented
starting point with a stated reason, **not a measurement**. The only detector
available here is synthetic, and sweeping against it would calibrate the
fixture rather than the system. The file says so; do not cite the numbers as
measured. Stage 13 or 20 must re-derive them against a real detector.

### The CI detection fixtures were not recorded from YOLO

`tests/fixtures/detections/sample_clips.json` was recorded with the reference
blob detector in `tests/fixtures/vision.py`, because no weights are installed.
The replay is exact and the recording declares its own provenance, but the
detection *patterns* downstream stages are tested against are a bright
rectangle's, not a vehicle's. Re-record with
`scripts/record_detection_fixtures.py --backend yolo` once weights exist.

Related: `tests/fixtures/images/` are synthetic rectangles, so the stage 11
prompt's case "detects vehicles in a committed sample image containing known
vehicles" is skipped rather than faked. It needs a photograph of a real road,
which carries licensing and privacy questions.

### No model weights are installed at all

`ultralytics`, `torch`, `torchvision`, `paddleocr`, and `pillow` are declared in
the `vision` extra and absent. Nine stage 11 integration tests skip on that,
including every real-inference case. **A release cannot ship on a machine that
has never run the model path.**

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
| operator picks a target from the live feed | 15 / 17 | 15, 17 — **stage 11 is now done**, so this needs only the live runner and the UI; still the cheapest and most reliable of the three modes |
| predict the next camera and arrival window | after 08, used by 15 and 17 | stage 08 (done) + real detections; must never be renderable as an observation |

None is buildable before the plan reaches stage 13, except that the picking
mode's detection half now exists.

---

## 4. Housekeeping

- `pyproject.toml` declares `opencv-python` and `av` in the `vision` extra; both
  are installed locally. `ultralytics`, `torch`, `torchvision`, `paddleocr`, and
  `pillow` are declared and **not** installed.
- The CI workflow runs the full gate including the coverage threshold. It has
  never been exercised against a real Postgres service container — verify that
  when Docker becomes available.
- `docs/` now has ten files. `ARCHITECTURE.md` predates stages 08–11 and could
  use a pass to mention `pathing`, `timesync`, `ingest`, and `vision`.
