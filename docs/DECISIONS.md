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
- **Precision on `hard_negatives` and `adversarial` is genuinely poor** (0.16,
  0.27 for appearance; 0.44, 0.39 for pathing) and no tuning fixes it. Those
  decoys are indistinguishable by construction. What the system guarantees there
  is that it says so.
