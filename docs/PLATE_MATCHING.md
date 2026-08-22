# Plate matching

The primary identification path. Given a target plate, find the sightings that
are the same vehicle.

**Precision matters more than recall here.** A false positive puts an innocent
vehicle on a stolen-car trajectory, and that error is expensive in a way a miss
is not: a missed sighting leaves a visible gap an operator can question, while a
wrong one looks exactly like a real hit. So thresholds are biased toward
precision, the uncertain middle goes to human review, and what plate matching
genuinely cannot find is left for re-id in stage 07.

Everything is pure string logic — no database, no video, no models — which is
what lets it be measured exactly against stage 05's ground truth.

---

## The pipeline

```
raw OCR string
  → normalize_plate()      uppercase, strip non-alphanumerics, reject homoglyphs
  → fold_ambiguous()       collapse confusion groups (COMPARISON ONLY)
  → classify_plate_match() exact / fuzzy / no_match
  → score_plate_match()    one number in [0, 1]
  → threshold              auto_accepted / pending_review / discarded
```

### Normalization

Uppercase, strip every non-alphanumeric character, optionally strip a configured
regional prefix or suffix (off by default).

Two rules that are easy to get wrong:

- **Unicode is transliterated where unambiguous and rejected otherwise.** An
  accented Latin character decomposes to one base letter — certain. A Cyrillic
  capital A renders identically to a Latin one but is a different character, and
  guessing would silently create two identities for one vehicle. It raises,
  naming the code point.
- **An empty result is `None`, not `""`.** The empty string compares equal to
  itself, so every plate that normalized to nothing would match every other one
  — a whole class of unreadable plates collapsing into a single false identity.

### Folding

Collapses each confusion group to one representative, turning fuzzy candidate
lookup into an indexed equality query.

| Confusable | Representative |
|---|---|
| `O`, `Q` | `0` |
| `I`, `L` | `1` |
| `S` | `5` |
| `B` | `8` |
| `Z` | `2` |
| `G` | `6` |

**Folding is for comparison only.** The folded form is never written to
`plate_text_normalized` and never shown as the plate. Displaying `A8C1234` for a
vehicle that reads `ABC1234` would put a wrong plate in front of a human
decision.

The map lives in `config/confusion_map.yaml` so it can be tuned per region — but
it is **also compiled into the `plate_folded` generated column by migration
0001**. Editing one without the other makes the database prefilter and
in-process matching disagree, and they disagree by *missing* candidates rather
than mis-scoring them, which leaves no trace in the results.
`assert_schema_fold_matches()` exists so that drift fails loudly, and a unit test
calls it.

### Distance

Damerau-Levenshtein (optimal string alignment), because OCR swaps adjacent
characters and a transposition costing 2 would push a one-error read outside a
distance-2 threshold meant to catch it.

The **weighted** variant is what makes the thresholds work:

| Operation | Cost |
|---|---|
| In-confusion-group substitution (`0` → `O`) | **0.5** |
| Arbitrary substitution (`0` → `W`) | 1.0 |
| Insertion / deletion | 1.0 |
| Adjacent transposition | 1.0 |

A cutoff supplied to either variant is an **early exit**: exceeding it returns a
value strictly above the cutoff rather than the true distance. The caller asked
whether the distance was within a threshold, not what it was.

---

## The threshold sweep

Thresholds were **measured, not chosen**. Reproduce with:

```bash
python scripts/evaluate_matching.py --sweep
```

Abridged output (full grid: 8 distances × 5 review floors):

| `max_weighted` | `review_min` | realistic P | realistic R | degraded P | degraded R |
|---|---|---|---|---|---|
| 0.40 | 0.50 | **1.000** | **1.000** | 1.000 | 0.200 |
| 0.50 | 0.50 | **1.000** | **1.000** | 1.000 | 0.200 |
| 0.60 | 0.50 | **1.000** | **1.000** | 1.000 | 0.200 |
| **0.90** | **0.50** | **1.000** | **1.000** | 1.000 | 0.200 |
| 1.00 | 0.50 | 0.625 | 1.000 | 1.000 | 0.200 |
| 1.10 | 0.50 | 0.625 | 1.000 | 1.000 | 0.200 |
| 1.50 | 0.50 | 0.625 | 1.000 | 1.000 | 0.200 |
| 2.00 | 0.50 | 0.500 | 1.000 | 1.000 | 0.200 |
| 0.90 | 0.60 | 1.000 | 0.600 | 1.000 | 0.000 |

Two things fall straight out of the table.

**There is a cliff at exactly 1.00, and it is structural.** An in-group
substitution costs 0.5; an arbitrary one costs 1.0. The near-miss decoys in the
`realistic` scenario differ from the target by *unrelated* characters, so a
cutoff at or above 1.0 admits them and precision falls to 0.625. A cutoff just
below 1.0 rejects every one of them while still admitting genuine OCR confusions
— and admits multi-character confusions too, because `folded_equal` short-
circuits the distance check entirely.

Intuition would have picked 1.5 or 2.0 here ("tolerate up to two errors"), which
measures at 0.625 precision. This is why the stage forbids picking by intuition.

**Raising the review floor to 0.60 buys nothing.** Recall drops from 1.000 to
0.600 while precision is already 1.000 — every candidate it removes was correct.

### Chosen values

```yaml
plate_fuzzy_max_weighted_distance: 0.9   # cliff is at 1.0; sit just below it
plate_review_min_confidence: 0.5         # 0.6 costs recall and buys no precision
plate_confusion_substitution_cost: 0.5   # what creates the 0.5/1.0 separation
plate_max_length_delta: 2
plate_auto_accept_min_confidence: 0.85
```

Re-run the sweep before changing any of them.

---

## Measured results

```bash
python scripts/evaluate_matching.py
```

| scenario | precision | recall | F1 | auto-accept P | misses |
|---|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 1.000 | — |
| realistic | **1.000** | 1.000 | 1.000 | 1.000 | — |
| degraded | **1.000** | 0.200 | 0.333 | 1.000 | dropout 1, read_failure 2, substitution 1 |
| hard_negatives | 1.000 | 0.800 | 0.889 | 1.000 | dropout 1 |
| sparse_coverage | 1.000 | 0.333 | 0.500 | 1.000 | substitution 2 |
| adversarial | 0.667 | 0.800 | 0.727 | 0.667 | substitution 1 |

Reading the two numbers that look like failures:

**`degraded` recall of 0.200 is correct, not a defect.** 40% of plates in that
scenario are completely unreadable and the rest are heavily corrupted. No string
logic recovers a plate that was never read. Precision holding at 1.000 is the
property that matters — the matcher does not compensate for missing data by
accepting worse candidates. Closing that recall gap is exactly what stage 07's
embedding fallback is for.

**`adversarial` precision of 0.667 is provably irreducible.** That scenario
contains a *cloned plate*: a second vehicle wearing the target's exact plate.
Both vehicles match, correctly, because by plate they are indistinguishable. All
false positives are attributed to `clone_00`, and a test asserts that. The system
does not pretend otherwise — it detects the physical impossibility instead.

Some misses are attributed to `substitution` even though folding should catch
those. That is deliberate: the stage 05 noise model draws from a **wider**
confusion table than the fold covers (it will read `B` as `6`, which folds to
`8` and `6` respectively). Real OCR also makes confusions outside any fold
table, and modelling only the ones we handle would be a self-fulfilling test.

A miss attributed to `clean` would mean the plate was read correctly and the
matcher still missed it — a matcher bug. A test asserts that count is zero.

---

## Plate conflict detection

When two sightings share a plate but sit on cameras closer together in time than
the fastest possible drive between them, one vehicle cannot account for both.

```python
conflicts = detect_plate_conflicts(plate, matched_sightings, topology)
```

Left undetected, path reconstruction would emit a route in which a vehicle
teleports — a confident, plausible-looking, wrong answer. So it is surfaced
instead:

> plate ABC1234 was seen on cam_01 and cam_05 0s apart, 135s faster than the
> 135s minimum possible transit; one vehicle cannot account for both

Details worth knowing:

- **Every pair is examined**, not just consecutive ones. A clone on a parallel
  route can leave each neighbouring pair looking fine while a wider pair is
  impossible.
- **Same-camera pairs are never conflicts.** A vehicle lingering in frame or
  circling back is ordinary; stage 04's `is_same_pass` handles that.
- **Feed it the matched set, not exact-plate equality.** The clone's readings are
  corrupted too, so an exact-plate filter misses half of them.
- **Unlinked pairs are ignored by default.** An undeclared route is far more
  often a surveying gap than a clone, and reporting every one would bury the real
  conflicts. Pass `include_unlinked=True` when investigating a specific plate.

---

## Candidate search

Two stages, and the split is the point.

**Stage one is indexed and database-side.** `find_by_plate_exact` and
`find_by_plate_folded` both hit partial B-tree indexes and return a handful of
rows. **Stage two scores those in Python.**

A full scan is never performed — query cost scales with the number of
*prefiltered* candidates, not with dataset size. That claim cannot be verified
from results (a full scan returns the same answers, just slower), so the test
wraps the repository in a recorder and asserts on the call log.

```python
candidates = find_plate_matches(
    target_plate="ABC1234",
    sighting_repo=repo,
    target_id=target.target_id,
    time_window=window,  # optional
    camera_ids=["cam_01"],  # optional
)
```

Returns `MatchCandidate` objects ranked by descending score, each with a review
status: **auto_accepted** at or above 0.85, **pending_review** down to 0.5, and
anything below discarded entirely rather than queued — a review queue full of
noise is a review queue nobody reads.

---

## Regression guard

`tests/integration/matching/baselines/plate_metrics_baseline.json` holds the
measured metrics for all six scenarios. A change that degrades matching fails CI
within a 0.02 tolerance.

Regenerate **deliberately**, once a change is understood and intended:

```bash
python scripts/evaluate_matching.py --write-baseline
```

Never regenerate it to make a failing test pass.
