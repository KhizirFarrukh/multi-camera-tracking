# Decisions and findings

Decisions that were made by measurement, deviations from the plan, and bugs that
were caught and are worth not repeating. Recorded so a later session does not
re-litigate settled questions or re-introduce fixed defects.

Every number here came from a committed script and is guarded by a committed
test or baseline.

---

## 1. Calibrations that contradicted the prior

### Fuzzy plate distance — 0.9, not 1.5 (stage 06)

The sweep found a *structural* cliff at weighted distance 1.0: precision 1.000 at
0.9 and below, 0.625 at 1.0 and above. An in-group substitution (0 read as O)
costs 0.5 and an arbitrary one costs 1.0, so a cutoff just under 1.0 admits every
genuine OCR confusion and rejects every near-miss decoy.

### Appearance auto-accept — 0.95, not the contract's 0.92 (stage 07)

At 0.92, twenty-one hard-negative decoys are auto-accepted. At 0.95, **none**
are, and recall is unchanged at 1.000 with every plate failure still recovered.
The decoy band tops out at 0.9389, so 0.95 clears it entirely.

Consequence worth stating: at 0.95, only `clean` produces any candidate above
the auto-accept threshold at all. On every other scenario the appearance path is
effectively **review-only**. That is the intended posture, and it is easy to
misread the zero auto-accepted false positives as "the classifier is excellent"
when part of the explanation is "it rarely asserts anything".

### Path inclusion bonus — 0.0, against the contract (stage 08)

The contract expected a positive per-node bonus to prevent short confident
fragments. Measurement says it is unnecessary *and* harmful.

Unnecessary: hop weights are multiplicative, so a well-supported hop contributes
~0.8 and extension is already worth far more than stopping.

Harmful: with multiplicative weights, inserting a candidate of confidence *c*
between neighbours *a* and *b* changes the score by `c(a+b) − ab + bonus`, which
is exactly zero at `c = ab/(a+b)` — 0.45 for two 0.9 neighbours, which is
precisely stage 07's visual-only ceiling. **The bonus is the whole of the margin
by which a visual-only candidate gets chained in.** At 0.05 the engine strings
decoys through the adversarial route until no competing account survives and
reports it as unambiguous.

### Gap penalty — 0.05, from a narrow principled band (stage 08)

A gap hop between two confident matches is worth `0.9 × 0.9 × 0.1 = 0.081`
before the penalty; between two visual-only matches, `0.020`. The penalty must
sit between those two numbers for strong evidence to bridge unmonitored ground
while weak evidence does not. 0.05 is the middle of that band.

### Stage 11 calibrated nothing, and says so (stage 11)

Every earlier stage measured its thresholds against stage 05's ground truth.
Stage 11 could not, and the honest response was to record that rather than to
produce a sweep against a fixture and call it a measurement.

The only detector available here is `ReferenceBlobDetector`, which finds a bright
rectangle. Sweeping the minimum box area or the association IoU against it would
calibrate the shape of a drawn bar. So the stage 11 block of `thresholds.yaml`
opens with **NOT MEASURED**, each value carries the reasoning that produced it,
and stage 13 or 20 owns re-deriving them.

One value *is* structurally argued rather than guessed: the four best-frame
weights must sum to 1.0, because each criterion scores in `[0, 1]` and any other
sum puts the composite outside that range. That is enforced at startup.

### The best-frame edge penalty is multiplicative (stage 11)

Written first as a subtraction, which is the obvious reading of "penalty". At
0.5 it drove most edge-touching frames to the clamp at zero, where they all tied
and frame index decided the ranking -- precisely the wrong answer for a track
seen *only* at the frame edge, where one of those frames really is the best
available. A discount preserves their relative order and keeps the score inside
`[0, 1]` without a clamp.

### Drift detection needed two extra guards (stage 09)

The shipped script, run against real generated data, flagged **every camera in
the network** as drifting. An alert that fires everywhere is an alert nobody
reads. Two guards were added: a rate needs an hour of observation span, and a
bias must exceed its own scatter.

---

## 2. Bugs found, and what they teach

### The offset solver was wrong twice (stage 09)

**Jacobi relaxation oscillates on a chain of cameras.** A path graph is
bipartite, and Jacobi on a bipartite Laplacian does not converge — it returned a
stable-looking 36.7 s where 12 s was correct. Replaced with a direct
least-squares factorisation.

**Iterative reweighting failed on the case it existed for.** A three-hour
outlier skewed the seed fit so far that the *good* passes looked like outliers,
and the estimate came back thirty times too large. Robustness that depends on
the first guess being roughly right is not robustness. Replaced with per-link
medians, which one outlier cannot move however extreme.

### The rotation inverse was off by one (stage 10)

Exact at one corner of the frame and off by the full width at the other — the
shape of bug that reads as a flaky detector rather than as arithmetic.
`np.rot90(k=1)` maps source `(x, y)` to `(y, width − 1 − x)`, and the `−1` is
easy to lose. A known point now goes through every transform, alone and
composed, in both directions.

### Abandoning a frame generator leaked the decoder (stage 10)

Only the file source released on iteration end; the base class did nothing, so
breaking out of a frame loop held the handle until the collector ran — far too
late on a batch of ten thousand files. Closing on iteration end is now the
default, with the live source overriding it because a pause in reading is not
the end of a stream.

### FFmpeg identifies almost anything (stage 10)

A file of ASCII text opened as `bintext`, 640×64 at 25 fps, and yielded a junk
frame. **A decoder agreeing to open something is not evidence that it is a
video.** PyAV now identifies and OpenCV decodes.

Related: a clean end-of-file and a truncated one are *identical* through
OpenCV's API — `read()` simply returns `False`. Without checking the container's
frame count, every complete file was reported as truncated.

### The rotation inverse returns -1 at the frame boundary (stage 11)

Not the stage 10 off-by-one, which was a real bug and is fixed. This is the
residue of a convention: stage 10's mapping works in pixel *centres*, where the
inverse of a 90-degree rotation is `width - 1 - x`. Box coordinates are
half-open, so a box edge legitimately sits at `x = width`, and mapping that edge
back gives -1.

Found by pushing a real rotated clip through the whole chain, not by reading the
arithmetic -- which is the argument for the end-to-end test existing.
`Detection.in_source_coordinates` clamps that one pixel and documents why;
anything larger than one pixel is a genuine mapping fault and still fails
against the non-negativity check.

### Two fixtures were not what their names implied (stage 11)

**`sample_clean.mp4` contains two vehicle passes, not one.** Its bar is drawn at
`(index * 3) % 52`, so on frame 18 it wraps from the right edge back to the
left. The tracker called it a second vehicle and was right to: a real object
cannot cross a frame in one frame interval, and a tracker that stitched the
teleport into one track would be the one that merges two vehicles into a single
trajectory. The test now asserts two passes and explains the wrap.

**`SyntheticVideoSource(moving_rectangle=False)` is not a static scene.** It
shifts its whole background by two or three greyscale levels every frame by
design, which is above the motion gate's default sensitivity. The first version
of the prefilter test asserted a high skip ratio against it and failed at 0.095
-- which read as a broken gate and was a scene that was never still. Feeding
genuinely identical frames gives a skip ratio of 0.97.

Both are the same lesson: a fixture's name is a claim, and a test that trusts
the name rather than the behaviour measures the claim.

### The cloned-plate decoy tested nothing (found in stage 06)

Stage 05's clone departed at a random time, so conflict detection found nothing
and the scenario tested nothing it claimed to. Fixed: the clone now departs with
the target from the far end of its route.

### Ground truth was internally inconsistent (stage 05)

Injected duplicates appeared in `sighting_to_vehicle` but not in the vehicle's
trajectory, because truth was assembled before injections rather than after.

---

## 3. Deviations from the stage prompts

| stage | the prompt expected | what was built, and why |
|---|---|---|
| 07 | one false-positive-rate target | Stated per scenario class. On `hard_negatives`, 21 decoys sit *inside* the true-match similarity band, so no threshold separates them; the meaningful target there is that none is auto-accepted. |
| 07 | "combined never scores below plate-only" | Holds except on exactly two sightings, both the cloned-plate vehicle. That is the disagreement rule forcing review, which necessarily drops the score. Asserted in that qualified form. |
| 07 | constraint proves itself on `sparse_coverage` | It does not — that scenario has no lookalikes, so both configurations score 1.000. The constraint pays off on `hard_negatives` and `adversarial`; on the others it is asserted as a smaller candidate set at equal recall. |
| 08 | a positive inclusion bonus | Measured to zero. See above. |
| 08 | modules `{...evaluation}.py` | Added `reconstruction.py` (not in the list) to hold assembly, so incremental extension can reuse it without re-running the search. |
| 08 | `ARITHMETIC_MEAN` confidence strategy | Removed. With the weakest-link cap applied it cannot produce a different number — configuration with no reachable behaviour. |
| 09 | read container creation time | Deferred to stage 10, which owns the demuxer. Stage 09 owns the seam (`select_source`) and the reliability tier. |
| 10 | a genuinely unsupported codec fixture | Not authorable — with FFmpeg present essentially every codec is supported. The adjacent real failure (no decodable stream) exercises the same path. |

---

## 4. Standing design rules

These are settled and load-bearing. Changing any of them changes what the system
means, not just how it works.

1. **Appearance evidence can never outrank a plate match.** Enforced by an
   inequality checked at startup, not by tuning.
2. **Timestamps are never interpolated.** A sampled frame carries the capture
   time of the frame that was decoded.
3. **Coordinates are always in original source-frame space.** Preprocessing
   returns the mapping that undoes it.
4. **Correction happens once, at one boundary**, recomputed from
   `raw_timestamp`, which is what makes it idempotent.
5. **Unmonitored ground is reported, never interpolated across.**
6. **Ambiguity is surfaced with its alternatives**, never resolved silently.
7. **Nothing is dropped without being counted.** Silent loss in a system whose
   output is "here is everywhere the vehicle went" makes the answer look
   complete when it is not.
8. **A gate that fires always is a gate nobody reads.** Every threshold that
   raises an alert has been checked against the false-positive case.
9. **One vehicle pass is one sighting**, emitted only once the pass has ended --
   because which frame of a pass is worth keeping cannot be known until it is
   over.
10. **A sighting is dated from its track's midpoint frame.** Dating from the
    first frame places every vehicle systematically early, by an amount that
    varies with the camera's field of view, which corrupts travel-time
    plausibility across the whole topology while passing every local test.

---

## 5. Known measurement limits

Worth knowing before trusting a number:

- **All quality numbers are synthetic.** They measure the engine against stage
  05's noise model. The *shape* of each analysis carries to real data; the
  specific thresholds do not, and stage 13 must re-run the sweeps against real
  embeddings.
- **Drift detection cannot resolve offsets below about a minute** on this
  topology, because the travel windows are minutes wide. The graph estimator
  finds them exactly; the monitor does not. That division of labour is
  deliberate and documented.
- **Absolute clock-offset estimates carry the bias of the expected transit
  times.** Differences do not. Prefer the direct clock probe where a device
  exposes one.
- **No stage 11 threshold is measured.** The detection and tracking values are
  reasoned starting points. The only detector available here is synthetic, and
  the CI detection fixtures record a bright rectangle rather than a vehicle, so
  nothing in this stage has met a real detector.
- **Precision on `hard_negatives` and `adversarial` is genuinely poor** (0.16,
  0.27 for appearance; 0.44, 0.39 for pathing) and no tuning fixes it. Those
  decoys are indistinguishable by construction. What the system guarantees there
  is that it says so.
