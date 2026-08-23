# Path Reconstruction

How a cloud of scored matches becomes one ordered route, what the engine
maximises to choose it, and what it does when the evidence does not single one
out.

Every number here comes from `python scripts/evaluate_pathing.py` and its
`--sweep` mode against the committed scenarios. Re-run them before changing a
threshold.

---

## 1. Why not greedy

The obvious algorithm walks forward in time taking the best next sighting. It
fails on exactly the case that matters: **one false positive in the middle of a
timeline**. Greedy chaining has already committed to everything before it, so it
takes the bad hop, and every later decision is made from the wrong place. One
spurious match teleports the route across the city and the reconstruction is
worthless from there on.

Global scoring cannot be fooled that way. A false positive is accepted only if
the whole route *through* it beats the whole route *around* it — and reaching an
implausible location and coming back costs two bad hops, which the genuine route
does not pay. That is the property the stage exists for, and
`tests/unit/pathing/test_adversarial_paths.py` tests it at each of the three
positions that behave differently: mid-chain, first, and last.

Because every edge points forward in time, index order is a topological order,
so the exact optimum is one linear pass — O(V + E), no heuristic search, no
beam, no approximation.

---

## 2. The objective

```
score(route) = Σ hop weights  +  inclusion_bonus × (number of sightings)

hop weight   = origin.confidence × destination.confidence × topology_plausibility
               × implausible_penalty   (if the travel-time window was violated)
               − gap_penalty           (if the hop crosses unmonitored ground)
```

**Multiplicative, not additive.** A hop is only as good as its weakest
component: a confident match at each end means nothing if the transit was
impossible, and a perfect transit means nothing between two sightings that are
probably not the target. A sum would let strong plausibility paper over a weak
match, which is precisely how a false positive gets in.

### Edge kinds

| kind | meaning | price |
|---|---|---|
| `direct` | declared link, traversed inside its window | none |
| `indirect` | no direct link, but reachable via intermediate cameras inside the arrival window | none; reported as a coverage gap |
| `implausible` | a route exists and the observed time violates it | × `hop_implausible_penalty` (0.5) |
| `gap` | no route at all — the vehicle left monitored ground | − `path_gap_edge_penalty` (0.05) |

Splitting the last two mattered more than it looks. Both were once "gap", which
priced an 18-second violation of a travel-time window the same as a drive
through an unwatched district. They are different claims about the world: one
says *the timetable is wrong*, the other says *nobody was looking*.

An `indirect` hop scores as fully plausible rather than at the unlinked floor.
The reachability query bounded both the earliest and latest arrival and the
observation fell inside; what is uncertain about such a hop is that nothing was
seen in between, and that is reported as a gap rather than discounted twice.

### No edge is created for a physically impossible transition

When camera coordinates are available, a pair implying a speed above
`implausible_speed_kph` produces **no edge at all**. A hard constraint, not a
penalty: travel-time windows are derived from a speed model and can be
optimistic, and a 700 km/h hop that merely scores badly is still a hop the
search will take when everything else scores worse. It did, in an early version
of this engine, on `hard_negatives`.

---

## 3. Calibration

`python scripts/evaluate_pathing.py --sweep`. The two columns to read first are
the ambiguity flags: a setting that turns either to `False` has made the engine
*confident* about a case where two routes explain the evidence equally well, and
no amount of precision elsewhere compensates for that.

| bonus | gapPen | exact | meanP | meanR | decoys | teleport | ambigHN | ambigADV |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.02 | 3 | 0.778 | 0.933 | 18 | 0 | True | True |
| **0.00** | **0.05** | **3** | **0.805** | **0.933** | 13 | **0** | **True** | **True** |
| 0.00 | 0.08 | 3 | 0.810 | 0.933 | 12 | 0 | True | True |
| 0.00 | 0.50 | 3 | 0.810 | 0.933 | 12 | 0 | True | True |
| 0.02 | 0.02 | 3 | 0.771 | 0.933 | 20 | 0 | True | **False** |
| 0.05 | 0.02 | 3 | 0.778 | 0.967 | 20 | 0 | **False** | **False** |
| 0.05 | 0.05 | 3 | 0.778 | 0.967 | 20 | 0 | **False** | **False** |
| 0.05 | 0.15 | 3 | 0.807 | 0.967 | 14 | 0 | True | True |
| 0.25 | 0.35 | 3 | 0.802 | 0.967 | 15 | 0 | True | True |

### 3.1 Inclusion bonus — 0.0, against the prior

The contract expected a positive per-node bonus to stop the engine returning
short confident fragments. The measurement says it is unnecessary *and*
harmful.

**Unnecessary**, because hop weights are multiplicative: a well-supported hop
contributes ~0.8, so extending a route is already worth far more than stopping.
`clean`, `realistic`, and `sparse_coverage` all reconstruct their full routes
exactly at a bonus of zero.

**Harmful**, because a per-node bonus pays a fixed amount for including a
sighting whatever the evidence behind it. There is arithmetic behind this: with
multiplicative weights, inserting a candidate of confidence *c* between
neighbours *a* and *b* changes the score by `c(a+b) − ab + bonus`, which is
exactly *zero* at `c = ab/(a+b)` — 0.45 for two 0.9 neighbours, which is
precisely the visual-only score ceiling from stage 07. So the bonus is the whole
of the margin by which a visual-only candidate gets chained in. At 0.05 the
engine strings decoys through the adversarial route until no competing account
survives and reports it as unambiguous:
`test_a_positive_inclusion_bonus__makes_the_adversarial_answer_confident`.

It stays configurable rather than being removed: a deployment whose matcher
produces systematically weaker scores may need it, and should sweep it there.

### 3.2 Gap penalty — 0.05, a narrow but principled band

The scale is set by the topology's unlinked plausibility floor (0.1). A gap hop
between two confident matches (0.9) is worth `0.9 × 0.9 × 0.1 = 0.081` before
the penalty; between two visual-only matches (0.45) it is `0.020`. The penalty
has to sit **between** those two numbers, and 0.05 is the middle of that band:

* a strongly supported vehicle still bridges an unmonitored district — the
  requirement skip edges exist for;
* two weak guesses do not reach across it to find each other.

Both halves are asserted directly
(`test_a_confident_match_across_unmonitored_ground__is_still_reached` and
`test_excluded_candidates__are_recorded_alongside_the_route`).

Raising it to 0.50 buys about half a point of mean precision and disables skip
edges entirely — a strong gap hop becomes −0.42 and can never be selected. That
operating point is available to a deployment that would rather fragment a route
than bridge one, and the sweep records what it costs.

---

## 4. Results

`python scripts/evaluate_pathing.py`:

| scenario | exact | P | R | hop acc | teleports | gaps | ambiguous | confidence | decoys |
|---|---|---|---|---|---|---|---|---|---|
| `clean` | ✔ | 1.000 | 1.000 | 1.000 | 0 | 0 | no | 0.972 | 0 |
| `realistic` | ✔ | 1.000 | 1.000 | 1.000 | 0 | 0 | no | 0.557 | 0 |
| `degraded` | ✘ | 1.000 | 0.800 | 0.500 | 0 | 1 | no | 0.253 | 0 |
| `hard_negatives` | ✘ | 0.444 | 0.800 | 0.250 | **0** | 6 | **yes** | 0.060 | 5 |
| `sparse_coverage` | ✔ | 1.000 | 1.000 | 1.000 | 0 | 2 | no | 0.187 | 0 |
| `adversarial` | ✘ | 0.385 | 1.000 | 0.250 | **0** | 5 | **yes** | 0.059 | 8 |

Reconstruction runs on the **matchers' real output**, not on ground truth —
stage 06 plate matching plus stage 07 appearance, combined. That matters: on
`degraded`, plate matching alone returns one sighting of five, so any pathing
metric computed from it would be measuring stage 06. Appearance recovers the
rest.

### Reading the adversarial rows honestly

Precision of 0.385 is poor, and no objective setting fixes it. The decoys in
those scenarios are constructed to be indistinguishable — a cloned plate, or a
vehicle drawn deliberately close in embedding space, moving along plausible
routes. There is no evidence available to the engine that separates them.

What the engine does instead is refuse to pretend:

* the trajectory is **flagged ambiguous**, with the competing routes attached;
* overall confidence reads **0.06**, because the weakest-link aggregation lets
  one contested hop dominate rather than being averaged away;
* every included and excluded sighting carries a reason an operator can read.

A system that reported 0.385-precision routes at high confidence would be far
more dangerous than one that reports them at 0.06 and says so.

---

## 5. Confidence aggregation

**A chain is not stronger than its weakest link.**

```
overall = min( geometric_mean(hop confidences), min(hops) + weakest_link_tolerance )
```

An arithmetic mean of `[1.0, 1.0, 0.1]` reads 0.70; a geometric mean reads 0.46.
Both are too generous for a route whose middle link is barely evidence at all,
so the aggregate is capped at the weakest hop plus a small configured allowance
(0.05). The allowance is not a fudge: a 0.1 hop between two independently
confirmed sightings genuinely is better supported than a 0.1 hop standing alone.

The guarantee is asserted as a guarantee, over randomized inputs: the reported
confidence never exceeds the weakest hop by more than the tolerance, and never
leaves `[0, 1]`.

An arithmetic-mean strategy is deliberately *not* offered. With the cap applied
it cannot report anything the default does not, so it would be a configuration
option with no reachable behaviour.

---

## 6. Gaps

A gap is not a defect. It is the engine stating plainly that it does not know
what happened between two sightings, which is more honest than interpolating a
route through cameras that saw nothing.

| kind | meaning | what an operator does |
|---|---|---|
| `coverage` | no camera was watching the ground in between | nothing — this is the network's shape |
| `temporal` | a route existed and the time does not fit it | investigate: a stop, a detour, or a bad match |
| `outage` | cameras on the route recorded **nothing at all** | fix the camera; this is a system fault |

The outage case needs a cross-reference that the trajectory alone cannot
provide. "This camera saw nothing" and "this camera was not working" are
indistinguishable without the wider sighting feed, and reporting the second as
the first turns an absence of evidence into evidence of absence. When no
activity feed is supplied, **no outage is claimed** — silence about silence.

---

## 7. Alternatives and ambiguity

The engine returns the top *k* routes and the normalized margin between the best
two:

```
margin = (best − second) / max(|best|, ε)
```

Normalized because a raw difference is meaningless across routes of different
lengths: 0.4 between two ten-hop paths is noise, 0.4 between two two-hop paths
is decisive.

**Alternatives have to disagree about which sightings belong.** A route whose
sightings are a subset of a better one — or a superset of it — is not a
competing account of what happened; it is the same account with more or less of
it. Every route has dozens of such shadows, and letting them fill the list would
make the margin measure "how much did the final hop contribute" rather than "is
there another story here", flagging perfectly decisive trajectories as too close
to call. Two routes compete only when each contains a sighting the other does
not — which is exactly what an operator has to adjudicate.

---

## 8. Incremental extension

The live path (stage 15) reconstructs the same target every few seconds.
Appending to the tail is cheap and wrong: streams deliver out of order, and a
sighting that arrives after later ones must be inserted where it belongs, with
everything downstream of it recomputed.

The rule is **exactness, not speed at any cost**: the incremental result is
identical to a full recomputation over the same candidates, asserted as a
property over randomized arrival orders. What incremental extension saves is
work, never correctness.

The mechanism follows from the objective: the dynamic program is a forward pass,
so the score at node *i* depends only on nodes before it. Inserting a node at
position *p* invalidates exactly the suffix from *p* onward — the prefix table
stays valid and only the edges arriving at *p* or later are rebuilt.

Sightings older than `reorder_buffer_sec` (120 s) are **rejected and reported**,
not silently dropped and not silently accepted. A stream running minutes behind
is a fault an operator needs to see, and a trajectory that keeps being rewritten
hours after the fact is not one anyone can act on.

---

## 9. Explainability

Every reconstruction carries a serializable account of itself: why each sighting
is in the route, why the highest-confidence rejects are not, which constraint
governed each decision, and the objective in words.

The exclusions are the half that matters. A search that quietly drops a
candidate looks identical to one that never saw it, and the difference matters
most for the candidate someone was expecting. Each excluded record names its
reason — `no_plausible_connection`, `not_reachable_in_time`, or
`lower_scoring_path` — and the score the best route *through* it would have had,
so "including it scores 3.1 against 3.8" replaces "it was excluded".

The list of individually explained exclusions is capped at ten, ranked by
confidence, with the total still reported. An explanation nobody can read is not
an explanation.

Inspect any scenario with:

```bash
python scripts/evaluate_pathing.py --explain adversarial
```

---

## 10. Known limits

* **Repeat passes at one camera.** Two sightings at the same camera are joined
  by a `gap` edge, since whatever route ran between them was not observed. A
  vehicle circling a block therefore accumulates gap reports. Correct, but
  noisy on dense urban topologies.
* **Decoys on plausible routes are not separable here.** Stage 07 established
  that appearance cannot distinguish them; neither can pathing, and the
  ambiguity flag is the honest response. Route *consistency across multiple
  targets* — which vehicle can account for which sightings globally — is a
  larger problem than this stage takes on.
* **Clock drift is not yet corrected.** Stage 09 does that, and the adversarial
  scenario's drifted camera is expected to reorder correctly only once it lands.
* **The metrics are synthetic.** They measure the engine against stage 05's
  noise model. The shape of the analysis carries to real data; the specific
  numbers do not.

---

## 11. Regenerating these numbers

```bash
python scripts/evaluate_pathing.py                     # the table in section 4
python scripts/evaluate_pathing.py --sweep             # section 3
python scripts/evaluate_pathing.py --explain degraded  # section 9
python scripts/evaluate_pathing.py --write-baseline    # update the regression baseline
```

The baseline lives at
`tests/integration/pathing/baselines/pathing_metrics_baseline.json` and is
asserted with a 0.02 tolerance. Rewriting it is a deliberate act: a diff there
is a claim that reconstruction quality changed on purpose.
