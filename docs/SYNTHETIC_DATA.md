# Synthetic data and ground truth

This is the highest-leverage piece of the project.

On real footage, nobody ever truly knows the right answer. You can look at a
reconstructed route and think it seems plausible, but you cannot compute
precision or recall, you cannot tune a threshold empirically, and you cannot
tell a regression from a hard day's traffic.

With a ground-truth generator, all three become possible. The generator knows
which vehicle produced every sighting, so stages 06–08 are *measured* rather
than eyeballed — and when computer vision arrives in stages 11–13, any drop in
accuracy is attributable to the vision layer by elimination.

**The realism is the point.** A generator that only ever substituted one
character in a plate, or drew decoys uniformly at random, would make the matcher
look excellent and then collapse on real data.

---

## Quick start

```bash
# Generate and inspect
python scripts/generate_dataset.py tests/fixtures/scenarios/realistic.yaml

# Write JSON (dataset + <out>.truth.json)
python scripts/generate_dataset.py tests/fixtures/scenarios/degraded.yaml --out build/degraded.json

# Override the seed and load into the configured database
python scripts/generate_dataset.py tests/fixtures/scenarios/realistic.yaml --seed 42 --to-database
```

```python
from multicam_tracker.synth import generate_from_file

dataset = generate_from_file("tests/fixtures/scenarios/realistic.yaml", seed=42)
dataset.sightings  # list[Sighting]
dataset.ground_truth.vehicle_for(sid)  # which vehicle really produced it
dataset.sightings_of("target")  # the target's sightings, in time order
```

---

## Determinism

**The same scenario file plus the same seed produces byte-identical output.**
Non-negotiable: stages 06–08 tune thresholds against these datasets, and a
generator that drifted would turn every regression test above it into noise —
invisibly, because each run would look internally consistent.

Three things make it hold:

- Every source of randomness is a seeded `random.Random`, never the module-level
  functions.
- Identifiers come from `deterministic_uuid(rng)`, not `uuid.uuid4()`, which
  reads the OS entropy pool.
- Each component draws from **its own stream**, derived by `stream_seed(seed,
  name)` using CRC32 of the component name. So turning on background traffic
  does not shift the target's route timings, and Python's per-process string
  hash randomization cannot leak in.

The seed used is recorded in the ground truth. A dataset you cannot regenerate
is not reproducible, however deterministic the run was.

---

## Scenario format

```yaml
name: realistic
description: "Moderate OCR noise and moderate traffic."
topology_file: config/topology.yaml
seed: 2002
jitter_sec: 5.0          # uniform, clamped inside each link's window
ingest_delay_sec: 1.0    # gap between timestamp_utc and created_at

vehicles:
  - vehicle_id: target
    plate: ABC1234       # the TRUE plate, before any OCR noise
    route: [cam_01, cam_02, cam_03, cam_04, cam_05]
    departure_utc: '2026-08-10T14:00:00.000Z'
    object_class: car
    speed_profile: 0.45  # 0 = fastest plausible transit, 1 = slowest
    is_target: true

noise: {...}            # see below
embeddings: {...}
traffic: {...}
adversarial: {...}
```

Consecutive route entries **need not be directly linked**. A leg across
uncovered ground is timed against the summed window of the fastest connecting
path, min-with-min and max-with-max — which is how a missed detection is
simulated.

---

## Route simulation

Timestamps derive from the topology's own travel-time windows, so generated data
is **topology-consistent by construction**:

```
nominal = min + speed_profile × (max − min)
elapsed = clamp(nominal + U(−jitter, +jitter), min, max)
```

Clamping is what matters. Jitter can never push a hop outside the constraint the
matcher later checks it against — even jitter of 10,000 seconds. The correct
answer is not asserted afterwards; it is what the generator built. When
`test_simulate_route__every_hop__is_plausible_even_with_absurd_jitter` fails, the
*generator* is broken, and every measurement resting on it would have been wrong.

---

## OCR noise model

Real ALPR fails in four characteristic ways. All four are modelled, because a
generator reproducing only the first would flatter fuzzy matching badly.

| Mode | What it simulates | Effect on the plate |
|---|---|---|
| `substitution` | A character read as its lookalike | Same length, ≥1 character differs |
| `dropout` | Occlusion — tow bar, dirt, frame edge | **Strictly shorter** |
| `read_failure` | Nothing legible at all | `plate_text = None`, **embedding survives** |
| `spurious` | A sticker or frame border read as text | **Longer** |

Substitutions are weighted by which confusions actually happen: `0/O` far more
often than `6/G`. A uniform model would produce a distribution of errors no real
engine makes.

`read_failure` is the mode that matters most — it is what forces re-id to carry
the match, and the `degraded` scenario exists to make it common.

### Confidence tracks severity

```
confidence = U(clean_min, clean_max) − penalty_per_severity × edit_distance
```

This coupling is load-bearing. If the generator emitted high confidence on
mangled reads, stage 06 could reach excellent precision by *ignoring confidence
entirely* — and would then collapse on real OCR, which does report low
confidence on the reads it gets wrong. A test asserts the correlation
coefficient is below −0.5 over a large sample.

### Parameters

| Field | Default | Meaning |
|---|---|---|
| `corruption_probability` | 0.0 | Chance a reading is corrupted at all |
| `substitution_weight` | 6.0 | Relative weights, not probabilities |
| `dropout_weight` | 2.0 | |
| `read_failure_weight` | 1.5 | |
| `spurious_weight` | 1.0 | |
| `max_substitutions` / `max_dropped` / `max_spurious` | 2 | Bounds the edit distance |
| `clean_confidence_min` / `_max` | 0.88 / 0.99 | Confidence band for a correct read |
| `confidence_penalty_per_severity` | 0.22 | Confidence lost per edit operation |
| `min_confidence` | 0.05 | Floor |

An honest fallback: if the chosen mode has nothing to work with — a plate with no
confusable characters, or one character long — the reading is reported as
**clean**. Inventing a different corruption would make the requested
failure-mode mix a lie, and a test asserting on that mix would silently pass.

---

## Embedding simulation

Each vehicle gets a base identity vector; each sighting is that base plus
Gaussian noise, re-normalized.

**Sigmas are relative, not absolute.** They are scaled by `1/√dimension`, the
natural component magnitude of a unit vector. Treating them as absolute would
make similarity depend on the vector width — a sigma that barely perturbs a
64-dimensional vector would obliterate a 512-dimensional one, and every scenario
would need retuning whenever stage 13 picked a different model.

| Field | Default | Meaning |
|---|---|---|
| `dimension` | 512 | Must equal `vision.embedding_dim`; a mismatch is rejected up front |
| `intra_class_sigma` | 0.25 | How much the *same* vehicle varies across cameras |
| `hard_negative_fraction` | 0.0 | Share of decoys drawn near the target |
| `hard_negative_sigma` | 0.15 | Spread of those decoys. Smaller = harder |

### Hard negatives are the whole game

Decoys drawn uniformly at random sit near-orthogonal to the target in high
dimensions — cosine around zero — so *any* threshold separates them and re-id
looks flawless. Real traffic contains other silver hatchbacks.

The default `hard_negative_sigma` lands hard negatives around **cosine 0.93**,
deliberately just *above* the contract's 0.92 auto-accept threshold. Some are
therefore wrongly auto-accepted on appearance alone, and only topology can
reject them. A value that kept them safely below the threshold would make the
scenario decorative, which is exactly what the exit criterion forbids.

---

## Background traffic

| Field | Meaning |
|---|---|
| `decoy_vehicles` | Volume of background traffic |
| `decoy_route_length` | Cameras per decoy route |
| `near_miss_edit_distances` | One extra decoy per entry, at *exactly* that plate distance |

Near-miss plates are built by substitution only, so the Levenshtein distance is
exactly the requested one. That stresses fuzzy matching **at the threshold**
rather than wherever random plates happen to land.

Decoy routes are drawn by walking the topology, so they are as plausible as the
target's. A decoy on an impossible route would be rejected by the travel-time
constraint alone and would add no difficulty at all.

No decoy may accidentally share the target's plate — collisions are retried,
because an accidental clone would look like a matching bug rather than a
scenario. Deliberate cloning has its own switch.

---

## Adversarial injections

| Injection | What it does | What it breaks |
|---|---|---|
| `clock_drift` | Shifts one camera's reported time. **Uncorrected**: `clock_offset_applied_ms` stays 0 | Stage 09 must detect it; the true instants live in ground truth |
| `outages` | Removes a camera's sightings in a window | Path reconstruction must produce a gap, not invent a route |
| `duplicate_passes` | Records a pass twice, ~0.4s apart, distinct id | Dedup must collapse these without collapsing a genuine revisit |
| `clone_target_plate` | Gives a decoy the target's exact plate | Plate matching alone cannot separate them |
| `boundary_hops` | Pins the target's first hop to exactly `min` and second to exactly `max` | Exercises the inclusive-boundary convention with real data |
| `shuffle_emission_order` | Emits records out of chronological order | Anything assuming a sorted input stream |

Every injection is recorded in `ground_truth.injections` with enough context to
assert on it — including how many sightings an outage actually removed, since an
outage that removed nothing tested nothing.

---

## The ground-truth artifact

```python
truth.sighting_to_vehicle  # sighting_id -> vehicle_id, for every emitted sighting
truth.vehicles  # per-vehicle: true plate, flags, time-ordered sighting ids
truth.true_timestamps  # what really happened, before injected clock drift
truth.corruptions  # every OCR failure, with mode and severity
truth.injections  # every adversarial event
truth.seed  # regenerate exactly this dataset
truth.read_failure_rate  # asserted on by the 'degraded' scenario
```

The artifact describes **what was emitted**, not what was planned: outage-removed
sightings are absent, and injected duplicates are present and attributed to the
vehicle they duplicate. Tests assert there are no orphans (a sighting with no
owner) and no phantoms (an owner claiming a sighting that was never emitted) —
either would make every precision and recall number downstream quietly wrong.

---

## The six committed scenarios

| Scenario | Purpose | Asserted characteristic |
|---|---|---|
| `clean` | The floor. Anything failing here is broken, not imprecise | Zero corruptions, zero decoys |
| `realistic` | The everyday case thresholds are tuned against | 15–60% of reads corrupted, ≥20 decoys, <20% unreadable |
| `degraded` | Plate matching cannot carry it; re-id must | **>25% of plates unreadable**, all modes present |
| `hard_negatives` | Decoys that look like the target | At least one decoy above cosine 0.92 |
| `sparse_coverage` | Target crosses uncovered ground | Every target leg spans ≥2 links |
| `adversarial` | Everything at once, because they interact | All six injection kinds present |

Each assertion is quantitative on purpose. A `degraded` scenario whose
read-failure rate had drifted to 2% would still pass every generic
self-consistency check while measuring nothing.

---

## Loading into a repository

```python
from multicam_tracker.synth import load_dataset_into

load_dataset_into(dataset, sighting_repo, ignore_conflicts=True)
```

Works against the in-memory fakes and Postgres alike, since both satisfy the
same protocol. **Cameras must exist first** — a sighting carries a foreign key to
its camera, so run `sync_topology_to_db` before loading.
