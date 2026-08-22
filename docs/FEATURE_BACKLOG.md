# Feature Backlog

Requested capabilities that are **not** part of the 20-stage plan in
`coding-agent-prompts/`, recorded here so they are not lost and so each stage
that touches their foundations can build with them in mind.

Nothing here is implemented yet. Each entry names the stage it should attach to
and what has to exist first. Items are only built when the stage they attach to
is reached, or on explicit instruction.

---

## A. Target selection modes

**Requested 2026-08-22.** Today a search starts from a plate query or from
reference embeddings already attached to a `Target`. The system should instead
support three explicit modes, chosen by the user per search, because each has a
different accuracy profile and a different failure mode the operator needs to
understand before they trust the result.

The mode must be an explicit, recorded parameter — never inferred. A trajectory
assembled from an uploaded photograph and one assembled from an operator-picked
detection carry very different evidential weight, and the audit trail has to say
which it was.

### A1. Match from a provided image

The user uploads a photograph of the vehicle to track. The system extracts an
embedding from it and matches probabilistically against sightings.

* **Attaches to:** stage 13 (embedding extraction) for the extractor; stage 14
  (`search_target`) for the entry point; stage 16 for upload and stage 17 for
  the picker UI.
* **Needs first:** a real re-id model (stage 13). Stage 07's thresholds are
  calibrated on synthetic embeddings and will not transfer.
* **Open question — cross-domain similarity.** A user-supplied photo is not a
  camera frame: different resolution, angle, lighting, often a marketing or
  phone image. Similarity between a query photo and a CCTV crop is systematically
  lower than between two CCTV crops, so this mode needs **its own calibrated
  thresholds**, swept separately. Reusing the sighting-to-sighting numbers would
  silently destroy recall.
* **Confidence must be surfaced as a distribution, not a verdict.** The stage 07
  machinery already supports this: two-tier thresholds, the margin rule, and
  `pending_review` as the default outcome. An image-seeded search should
  auto-accept nothing without a human.

### A2. Track everything

Assemble trajectories for every distinct vehicle observed, rather than for one
nominated target.

* **Attaches to:** stage 14 (batch orchestration), with the clustering itself a
  new component.
* **Needs first:** stages 11-13, and stage 08's path reconstruction generalised
  from "one target's candidates" to "cluster then reconstruct per cluster".
* **Cost note.** This is a quadratic problem in the number of sightings unless
  it is blocked by camera and time window first — the same topology constraint
  that stage 07 uses for precision becomes a scaling requirement here. Expect to
  need approximate clustering (HNSW neighbourhoods plus connected components
  under topology constraints), not all-pairs comparison.
* **Retention interacts.** Tracking everyone by default is a far broader data
  footprint than tracking one nominated vehicle. The retention TTL and the
  access-control matrix (stage 19) should be revisited before this ships, not
  after.

### A3. Operator picks from the feed

The user selects a vehicle directly from a camera view, and that detection
becomes the reference.

* **Attaches to:** stage 17 (UI selection) and stage 15 (live streams), reading
  the detections stage 11 already produces.
* **Needs first:** stage 11 (detections with bounding boxes), stage 15 (live
  feed), stage 17 (the view to click in).
* **Cheapest of the three modes and the most reliable**, because the reference
  embedding comes from the same domain as every candidate — a camera crop
  matched against camera crops.

---

## B. Movement prediction

**Requested 2026-08-22.** Predict where a tracked vehicle goes next: the most
likely next camera, an arrival-time window, and a ranked set of candidate
onward routes.

* **Attaches to:** a new component after stage 08, consumed by stage 15
  (live alerting: "watch these cameras next") and stage 17 (predicted path
  drawn distinctly from observed path).
* **Needs first:** stage 08's trajectory and per-hop confidence; stage 04's
  travel-time windows, which already give the arrival-window arithmetic.
* **Approach.** The topology graph plus observed transition frequencies gives a
  first-order Markov model over cameras; direction of travel and speed from
  stage 8.9 narrow it further. Start there rather than with a learned model:
  the graph-based version is explainable, needs no training data, and is a
  baseline any learned model must beat.
* **Non-negotiable presentation rule.** A prediction must never be rendered in a
  way that lets it be mistaken for an observation. Predicted cameras, predicted
  arrival windows, and predicted routes are drawn and labelled distinctly, and
  are excluded from the trajectory's evidential record. A predicted sighting
  that leaks into the audit trail as a real one is the worst failure this system
  could have.
* **Evaluation before it ships.** Measured against the synthetic scenarios the
  same way everything else is: top-1 and top-3 next-camera accuracy, and
  arrival-window calibration (what fraction of actual arrivals fall inside the
  predicted window). A prediction feature with no measured accuracy is a
  guess with a user interface.

---

## Sequencing

None of this is buildable before the plan reaches stage 13 — every mode above
depends on real embeddings, and prediction depends on stage 08. The plan
continues in order; these attach at the stages named above.
