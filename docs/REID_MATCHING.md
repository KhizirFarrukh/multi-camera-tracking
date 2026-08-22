# Visual Re-Identification Matching

How the system finds a vehicle when its plate is unreadable, why the thresholds
are the numbers they are, and what appearance evidence is *not* allowed to do.

Every number on this page was produced by
`python scripts/evaluate_matching.py --reid`, `--reid --sweep`, and
`--compare-constraint` against the committed scenarios in
`tests/fixtures/scenarios/`. Re-run them before changing a threshold.

---

## 1. The one number that justifies the stage

Plate matching cannot find a sighting whose plate was never legible. No amount
of string logic recovers a plate the camera did not capture. Stage 06 measured
that gap honestly and counted those sightings as misses.

Appearance closes it:

| scenario | plate-path failures | recovered by appearance | rate |
|---|---|---|---|
| `degraded` | 3 | 3 | 100% |
| `sparse_coverage` | 2 | 2 | 100% |
| `hard_negatives` | 1 | 1 | 100% |
| `adversarial` | 1 | 1 | 100% |

If this column were zeros, re-id would be complexity with no return. It is the
first thing to check after any change to the embedding path, and
`test_every_plate_failure__is_recovered_by_appearance` fails CI if it regresses.

Appearance earns its place a second way, in section 6: it is what catches a
cloned plate.

---

## 2. Results at the shipped configuration

Topology-constrained, `embedding_auto_accept_min_similarity: 0.95`,
`embedding_review_min_similarity: 0.75`, `embedding_margin_min: 0.04`:

| scenario | emb P | emb FP rate | auto-accepted FP | comb P | comb R | recovered | rank-1 | rank-5 |
|---|---|---|---|---|---|---|---|---|
| `clean` | 1.000 | 0.000 | 0 | 1.000 | 1.000 | 0/0 | 1.00 | 1.00 |
| `realistic` | 1.000 | 0.000 | 0 | 1.000 | 1.000 | 0/0 | 1.00 | 1.00 |
| `degraded` | 1.000 | 0.000 | 0 | 1.000 | 1.000 | 3/3 | 1.00 | 1.00 |
| `hard_negatives` | 0.160 | 0.840 | **0** | 0.160 | 1.000 | 1/1 | 1.00 | 1.00 |
| `sparse_coverage` | 1.000 | 0.000 | 0 | 1.000 | 1.000 | 2/2 | 1.00 | 1.00 |
| `adversarial` | 0.267 | 0.733 | **0** | 0.235 | 1.000 | 1/1 | 1.00 | 1.00 |

Recall is 1.000 everywhere and rank-1 accuracy is 1.000 everywhere: the true
match is always found, and always ranked first.

Precision on `hard_negatives` and `adversarial` is poor, and that is a property
of the data rather than a defect to tune away. Section 4 explains why, and why
the number that matters on those scenarios is the bolded zero.

### Definitions

A **positive** on the embedding path is a candidate returned at or above
`embedding_review_min_similarity` — everything put in front of a human or
accepted outright.

A **combined positive** is the union of what each path would return alone: a
plate score at or above stage 06's review floor (0.5), *or* a candidate the
embedding path retained. Requiring the plate floor of everything would discard
every visual-only match by construction — the visual score ceiling sits below
that floor deliberately (section 5) — and would amount to measuring plate
matching a second time under a new name.

**CMC** is reported over a single query per scenario (one target, one anchor),
so each rank is 0 or 1. It is a regression guard on ranking, not a population
statistic.

---

## 3. Similarity distributions: what the thresholds are actually cutting

The whole calibration follows from where true matches and decoys sit. Measured
against the target's first sighting as the sole reference:

| scenario | true matches | best decoy | decoys ≥ 0.75 |
|---|---|---|---|
| `clean` | 0.9889 – 0.9902 | — | 0 |
| `realistic` | 0.9383 – 0.9445 | 0.0625 | 0 |
| `degraded` | 0.8545 – 0.8707 | 0.0916 | 0 |
| `sparse_coverage` | 0.9405 – 0.9427 | 0.0955 | 0 |
| `adversarial` | 0.9376 – 0.9451 | 0.9346 | 11 |
| `hard_negatives` | 0.9379 – 0.9423 | **0.9389** | 21 |

On ordinary data the gap is enormous — true matches near 0.94, everything else
below 0.10 — so any review floor between 0.1 and 0.85 gives identical results.
The floor is set at 0.75 because that is comfortably inside the empty band while
still admitting the degraded true matches at 0.854.

On `hard_negatives` the bands **overlap**: the third-ranked candidate is a decoy
scoring above two genuine sightings of the target. No scalar threshold separates
them, which is the finding that drives everything in section 4.

---

## 4. Threshold calibration

### 4.1 Auto-accept similarity — 0.95, not the contract's 0.92

`python scripts/evaluate_matching.py --reid --sweep`:

| accept | margin | degraded R | recovered | hardneg P | hardneg FP rate | hardneg auto-FP | hardneg downgrades | clean downgrades |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.00 | 1.000 | 3/3 | 0.160 | 0.840 | 21 | 0 | 0 |
| 0.80 | 0.04 | 1.000 | 3/3 | 0.160 | 0.840 | 21 | 1 | 1 |
| 0.90 | 0.04 | 1.000 | 3/3 | 0.160 | 0.840 | 21 | 1 | 1 |
| 0.92 | 0.00 | 1.000 | 3/3 | 0.160 | 0.840 | **21** | 0 | 0 |
| 0.92 | 0.04 | 1.000 | 3/3 | 0.160 | 0.840 | **21** | 1 | 1 |
| **0.95** | **0.04** | **1.000** | **3/3** | 0.160 | 0.840 | **0** | 0 | 1 |
| 0.97 | 0.04 | 1.000 | 3/3 | 0.160 | 0.840 | 0 | 0 | 1 |

The contract proposed 0.92 as a prior. The measurement says otherwise: at 0.92,
**21 hard-negative decoys are auto-accepted** — asserted to an operator as fact
with no human in the loop. At 0.95 that count is zero, and nothing is paid for
it: recall stays 1.000 and every plate failure is still recovered.

The reason is section 3. The decoy band tops out at 0.9389, so 0.95 clears it
entirely. This is a real property of the data, not a coincidence of one seed:
the decoys are generated by perturbing the target's embedding, and 0.95 sits
above the perturbation radius.

### 4.2 What that conservatism costs

At 0.95, only `clean` produces any candidate above the auto-accept threshold at
all. On every other scenario the embedding path is effectively **review-only**:
it ranks candidates and queues them, and a human decides.

That is the intended posture for a system where a false accept means an operator
is told the wrong vehicle is the target. It is stated plainly here because it is
easy to misread the zero auto-accepted false positives as "the classifier is
excellent" when part of the explanation is "the classifier rarely asserts
anything at all".

### 4.3 Review floor — 0.75

Swept independently — the second table printed by
`scripts/evaluate_matching.py --reid --sweep`:

| review floor | degraded R | degraded recovered | hardneg P | hardneg R | hardneg auto-FP |
|---|---|---|---|---|---|
| 0.70 | 1.000 | 3/3 | 0.160 | 1.000 | 0 |
| **0.75** | **1.000** | **3/3** | 0.160 | 1.000 | 0 |
| 0.80 | 1.000 | 3/3 | 0.160 | 1.000 | 0 |
| 0.85 | 1.000 | 3/3 | 0.160 | 1.000 | 0 |
| 0.90 | 0.000 | 0/3 | 0.160 | 1.000 | 0 |
| 0.92 | 0.000 | 0/3 | 0.160 | 1.000 | 0 |
| 0.95 | 0.000 | 0/3 | 1.000 | **0.000** | 0 |

There is a cliff at 0.90: the `degraded` true matches sit at 0.854–0.871, so a
floor above them destroys the recovery this stage exists for. And note the last
row — a floor of 0.95 buys precision 1.000 on `hard_negatives` only by returning
nothing at all, including the true match. Precision without recall is not a
result.

No floor separates hard-negative decoys from true matches. That is the honest
conclusion: on adversarial appearance data, thresholding is the wrong tool, and
what protects the operator is the auto-accept ceiling plus human review.

### 4.4 Margin rule — 0.04

The margin rule demotes the top candidate to review when the runner-up is within
`embedding_margin_min` of it, regardless of absolute score. Two vehicles that
appearance cannot tell apart must never be resolved automatically; a confident
choice between two indistinguishable options is a coin flip wearing a number.

The sweep shows exactly where it bites, and it is worth being precise about:

* On `clean`, the four true sightings score 0.9889–0.9902 — gaps of ~0.001. The
  rule fires and demotes the leader. The runner-up is *another sighting of the
  same vehicle*, so this costs one extra human review for no safety gain.
* On `hard_negatives` at accept ≤ 0.92, the rule fires once (the leader's gap to
  the next candidate is 0.003) but the auto-accepted false-positive count does
  not move, because the rule as specified guards only the **top-1** decision and
  the 21 decoys ranked below it are unaffected.

So the margin rule is not what makes hard negatives safe — the 0.95 threshold
is. The rule is a guard on the top-1 assertion, and it is kept because that is
the assertion an operator acts on first.

Distinguishing "several sightings of one vehicle" from "several vehicles that
look alike" needs route consistency across sightings, which is stage 08. Until
then the rule errs toward review.

### 4.5 These numbers are calibrated on synthetic embeddings

The absolute similarity values above are artifacts of stage 05's embedding noise
model, not measurements of a real re-id network. The *shape* of the analysis —
sweep, find the cliff, take the safe side of it — carries over; the specific
0.95 does not. Stage 13 introduces real extraction, and this page must be
regenerated against real embeddings before those thresholds are trusted in
production.

---

## 5. Combining plate and appearance evidence

The governing rule: **an embedding-only match can never outrank a plate match.**
Many vehicles share a make, model, and colour; almost none share a plate.

This is enforced structurally, not by tuning. A visual-only score is capped at
`embedding_only_score_ceiling` (0.45), which sits below
`plate_review_min_confidence` (0.5) — the lowest score any retained plate match
can carry. No combination of inputs can invert the ordering, and a property test
asserts it across randomized inputs.

Four cases:

| plate | appearance | result |
|---|---|---|
| strong | strong | plate method, confidence boosted by 0.05, bounded at 1.0 |
| absent | strong | `embedding` method, score capped at 0.45, never auto-accepted |
| strong | weak (< 0.4) | **conflicting** → forced to review |
| weak | strong | **conflicting** → forced to review |

Disagreement is the interesting case. Two independent signals pointing different
ways is *more* alarming than one weak signal, because one of them is wrong and
nothing in the data says which. Averaging them would produce a middling score
that hides the contradiction; forcing review surfaces it.

### The one place combined evidence scores lower than plate evidence

The stage prompt states that combined evidence never scores below plate-only
evidence — "the fallback never hurts". Measured across all six scenarios, that
holds for every sighting **except two**, and both are the disagreement rule
working as designed: forcing a strong-plate match to human review necessarily
means dropping it below the auto-accept threshold.

Both of those sightings belong to the cloned-plate vehicle. The property is
asserted in that form —
`test_combined_evidence__never_scores_below_plate_only_unless_the_signals_conflict`
— rather than weakened or dropped.

---

## 6. Appearance catches a cloned plate

On `adversarial`, a second vehicle wears the target's plate perfectly. Plate
matching alone cannot separate them: both read identically. Stage 06 handled
this by detecting the physical impossibility of one vehicle being in two places.

Appearance handles it directly. The clone's embedding disagrees with the
target's references, so both of its sightings produce a conflicting verdict:

```
adversarial: conflicts=2   owner=clone_00 (both)   scored_below_plate_only=2
realistic:   conflicts=0
degraded:    conflicts=0
hard_negatives: conflicts=0
```

Zero conflicts on ordinary data is as important as two on the clone. A conflict
that fired routinely would train an operator to dismiss it.

---

## 7. Constrained retrieval beats raising the threshold

Given an anchor — a sighting already confirmed to be the target — the topology
says which cameras the vehicle could have reached and when. Searching only those
turns a global similarity contest into a local one. A decoy that looks identical
to the target is harmless if it was never anywhere the target could have been.

`python scripts/evaluate_matching.py --compare-constraint`, identical thresholds
in both columns:

| scenario | unconstrained P | constrained P | unconstrained R | constrained R |
|---|---|---|---|---|
| `clean` | 1.000 | 1.000 | 1.000 | 1.000 |
| `realistic` | 1.000 | 1.000 | 1.000 | 1.000 |
| `degraded` | 1.000 | 1.000 | 1.000 | 1.000 |
| `hard_negatives` | 0.100 | **0.160** | 1.000 | 1.000 |
| `sparse_coverage` | 1.000 | 1.000 | 1.000 | 1.000 |
| `adversarial` | 0.250 | **0.267** | 1.000 | 1.000 |

A 60% relative improvement in precision on `hard_negatives` at zero cost to
recall. Raising the similarity threshold instead trades recall for precision on
every sighting equally; constraining the space costs nothing on the true match
while removing most opportunities to be wrong.

Where there are no lookalikes the constraint cannot improve precision — it is
already 1.000. What it improves there is how much work the query does:

| scenario | unconstrained candidates | constrained | reduction |
|---|---|---|---|
| `clean` | 4 | 4 | 0% |
| `realistic` | 85 | 46 | 45.9% |
| `degraded` | 64 | 43 | 32.8% |
| `hard_negatives` | 64 | 36 | 43.8% |
| `sparse_coverage` | 26 | 19 | 26.9% |
| `adversarial` | 58 | 41 | 29.3% |

At production volume that is the difference between a query and an outage, and
it is asserted in
`test_constrained_retrieval__searches_a_smaller_space_at_equal_recall`.

**Note the deviation from the stage prompt.** It expected the constraint to show
its value on `sparse_coverage`. Measured, it does not: that scenario contains
nothing resembling the target, so both configurations score 1.000. The
constraint pays off where lookalikes exist — `hard_negatives` and `adversarial`
— and pays off in candidate-set size everywhere else. The tests assert what was
measured.

Without an anchor the search falls back to a global window and logs a warning at
`WARNING` level. It is a materially weaker query and a caller should know when
they are getting one.

---

## 8. Multi-reference targets

A target may accumulate several reference embeddings from confirmed sightings at
different angles. Three aggregation strategies are available; `max` is the
default.

| strategy | question it asks |
|---|---|
| `max` | does this look like the target from *any* confirmed angle? |
| `mean` | does this look like the average of the target's appearances? |
| `topk_mean` | does this look like the target's *k* most similar appearances? |

`max` is the default because averaging a front view and a rear view produces a
vector that resembles neither, and a candidate matching the rear view strongly
would score poorly against that average. All three are identical for a
single-reference target.

The trade-off `max` carries: it never falls as references accumulate, so a wrong
reference inflates scores permanently and produces more wrong confirmations. That
is why `add_reference_embedding` **rejects** a sighting whose match is only
`pending_review` — the system may not confirm its own guesses.

Pruning keeps a diverse subset by greedy max-min-distance selection rather than
the most recent, so the reference set spans viewpoints instead of clustering
around whichever angle was seen last. Default cap: 8.

---

## 9. Model-version safety

Two embeddings are comparable only if the same model produced them. Vectors from
different models occupy unrelated spaces, so their cosine similarity is a number
with no meaning — and, fatally, a *plausible* one. It will not be NaN or out of
range; it will be 0.31, and a ranking built on it looks entirely reasonable
while being noise.

Two layers prevent it:

1. The repository query filters by `embedding_model_version`. The rows never
   reach the comparison.
2. The scoring path calls `require_same_model_version` anyway, which raises
   `MatchingError` naming both versions.

Layer 2 exists because layer 1 is a query parameter a caller can forget.

### Re-embedding migration path

1. Deploy the new model, writing a new `embedding_model_version`.
2. Backfill historical sightings by re-running extraction over stored
   thumbnails, writing new rows alongside the old.
3. Switch searches to the new version once backfill covers the retention window.
4. Drop the old vectors after the retention TTL expires them.

Steps 1 and 2 overlap deliberately: during backfill both versions are present on
purpose. That is exactly why the filter belongs in the query — a search that
merely *rejected* mismatched rows after reading them would return a page of
unusable candidates, or raise, and be unusable itself.

---

## 10. Regenerating these numbers

```bash
python scripts/evaluate_matching.py --reid                  # the table in section 2
python scripts/evaluate_matching.py --reid --sweep          # section 4.1
python scripts/evaluate_matching.py --compare-constraint    # section 7
python scripts/evaluate_matching.py --reid --write-baseline # update the regression baseline
```

The baseline lives at
`tests/integration/matching/baselines/embedding_metrics_baseline.json` and is
asserted with a 0.02 tolerance by `test_metrics__match_the_committed_baseline`.
Rewriting it is a deliberate act: a diff there is a claim that matching quality
changed on purpose.
