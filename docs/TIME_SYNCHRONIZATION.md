# Time Synchronization

The whole system rests on one assumption: that a sighting at 14:02 on one camera
and 14:05 on another are three minutes apart in reality. If camera 3's clock is
five minutes fast, the route built from it is confidently wrong in a way nothing
downstream can detect — not the matcher, not the path search, not the confidence
score. The trajectory looks exactly like a correct one.

This page is what an operator needs to keep that assumption true, and what an
engineer needs to know about where it can still fail.

---

## 1. The measured demonstration

Same data, same matcher. The only difference is whether camera 3's clock
correction was applied.

| | route reconstructed | confidence | gaps reported |
|---|---|---|---|
| ground truth | `cam_01 → cam_02 → cam_03 → cam_04 → cam_05` | 0.97 | 0 |
| **5-minute drift, uncorrected** | `cam_01 → cam_02 → cam_04 → cam_03 → cam_05` | **0.10** | **2** |
| the same drift, corrected | `cam_01 → cam_02 → cam_03 → cam_04 → cam_05` | 0.97 | 0 |

The uncorrected route puts the vehicle at cam_04 before cam_03 — a road it never
travelled, in an order that never happened. Both directions are asserted in
`tests/integration/timesync/test_drift_correction_end_to_end.py`, because either
alone proves nothing.

**A 45-second drift on this network changes no route at all.** The topology's
travel-time windows are minutes wide and absorb it. That is a real finding, not
a gap in the test: it is why the integrity gate blocks on offsets exceeding the
*shortest transit between the queried cameras* rather than on any drift at all.
A gate that refused to answer whenever a clock was imperfect would refuse
almost always, and be switched off within a week.

---

## 2. Reliability tiers

A timestamp from a synchronised stream and one parsed out of a filename are both
`datetime` objects. Treating them alike is how a system presents a guess as a
measurement, so every source declares what it is worth, and the tier travels
into the integrity gate and onto the trajectory an operator reads.

| tier | sources | fails how |
|---|---|---|
| **high** | `StreamClockSource` — the device's own clock | only if the device's clock is wrong, which drift detection watches for |
| **medium** | `FileMetadataSource` — container creation time; `ManualOffsetSource` — an operator-stated start | a copy or remux rewrites container metadata without saying so; a human can be wrong but can be asked |
| **low** | `FilenameSource` — a naming convention; `OverlayOcrSource` — burned-in pixels | both fail *silently*: a pattern that stops matching yields a plausible wrong date, and OCR misreads a digit without any signal that it did |

`select_source` picks the highest tier available and **reports the fallback**
rather than performing it quietly. A file whose metadata was stripped still gets
a timestamp — from its name — and the route built from it carries a caveat
saying so.

When nothing at all is available, ingestion refuses. A capture time that is
unknown cannot be invented: inventing one puts a vehicle somewhere at a time
nobody observed.

---

## 3. Deriving a frame's instant

```
timestamp = source_start + frame_index / fps
```

is wrong in a way that hides for a long time, and both halves of it are wrong.

**The rate is rational.** 29.97 fps is really 30000/1001. The arithmetic here
uses `fractions.Fraction` throughout and rounds exactly once, at the end, to the
millisecond the Sighting contract stores. The tempting alternative — store each
frame's offset to the millisecond and add them up — loses 0.3667 ms per frame at
that rate, which is **over half a minute across 100,000 frames**. The
demonstration is in
`tests/unit/timesync/test_derivation.py::test_per_frame_millisecond_rounding__is_the_error_this_module_avoids`.

**Presentation timestamps win over the nominal rate.** Real recordings drop
frames and vary their rate; multiplying an index by a nominal fps assumes
neither ever happens and shifts every timestamp after the first drop. When the
container supplies PTS they are used and `fps` is ignored entirely. A frame
outside the recorded PTS raises rather than falling back — mixing two timing
models inside one video is worse than refusing.

---

## 4. Local time, DST, and leap seconds

Two hours a year are genuinely ambiguous in every zone that observes daylight
saving: when the clocks go back 01:30 happens twice, and when they go forward
02:30 does not happen at all. Libraries usually pick one silently and are wrong
half the time.

Both cases **raise**, naming the instant and the camera:

```
Local time 2026-10-25T01:30:00 occurs twice in Europe/London: the clocks went
back across it, so it could be 2026-10-25T01:30:00+01:00 or
2026-10-25T01:30:00+00:00. Supply an explicit utc_offset to say which
```

The operator answers by passing the offset that applied. An offset the zone does
not use at that instant is itself rejected — that would be a second guess
dressed as a decision.

**Leap seconds are not represented.** Python's `datetime` has no 23:59:60, the
POSIX clock does not tick it, and every device this system ingests from has
already smeared or repeated the second. Within one second the system's
conclusions do not change: the smallest travel-time window in a realistic
topology is tens of seconds. Leap seconds are therefore absorbed by whatever the
source did with them, and never corrected for here. The policy is explicit so it
is not mistaken for an oversight.

---

## 5. Measuring a camera's offset

Two instruments, and the division of labour between them matters.

### 5.1 The clock probe — preferred

Where a device exposes its time (ONVIF, an RTCP sender report, a vendor API),
comparing it against the server's clock is a direct measurement. It needs no
reference vehicle, no topology assumptions, and works on a camera nothing has
driven past today. Half the probe round-trip is subtracted as the transport
correction, which is why a slow probe is a weak sample rather than a precise one.

Samples are kept as a series, not a latest value: one comparison cannot tell a
constant offset from a drifting clock, and that difference decides whether an
operator corrects the camera once or replaces its time client.

### 5.2 Reference-event estimation — the fallback

For cameras that expose nothing. A vehicle with a known plate drives a known
route; each leg gives one equation:

```
offset[to] − offset[from] = observed_elapsed − expected_elapsed
```

solved as least squares over the camera graph with **one camera pinned to zero**
— the equations constrain only differences, so without a pin the solver returns
one of infinitely many equally good answers.

**Robustness is structural.** Many vehicles cross the same link, so each link is
summarised by the *median* of its crossings before anything is solved. A median
cannot be moved by one mismatched vehicle however extreme.

> An earlier version reweighted iteratively from a plain fit instead, and failed
> exactly where it mattered: a three-hour outlier skewed the first fit so far
> that the *good* passes looked like the outliers, and the estimate came back
> thirty times too large. Robustness that depends on the first guess being
> roughly right is not robustness.
>
> An earlier version also solved by Jacobi relaxation. A chain of cameras is a
> bipartite graph, and Jacobi on a bipartite Laplacian oscillates rather than
> converging — it produced a stable-looking answer that was simply wrong. It is
> a direct least-squares factorisation now.

**Refusing to answer is a feature.** A camera with one reference pass is not
estimated, it is guessed at, and a guessed offset applied to real data is worse
than none because it looks like a correction. Below `min_reference_passes` the
estimator names the camera and stops.

### 5.3 The accuracy limit, measured

The estimator is only as good as the expected transit times it is given. Feeding
the midpoint of a 60–300 s window as "expected" makes every camera absorb the
difference between that midpoint and how traffic actually moves, so **absolute**
offsets come back biased even on data with no drift at all.

**Differences do not carry that bias.** On the adversarial scenario — a clone,
decoys, an outage, and an injected 45-second drift — comparing the estimate
before and after the injection attributes exactly −45,000 ms to `cam_03` and
0 ms to every other camera. That is what the graph buys: a camera late by X
makes its incoming legs long by X *and* its outgoing legs short by X, and no
other explanation fits both.

---

## 6. Drift detection, and its noise floor

Fast traffic and a fast clock look identical in one observation. What separates
them is consistency: fast driving is a property of one journey, a fast clock
shows up on every link, in both directions, all day.

Three guards keep the alerts worth reading:

* **Both directions count.** An arrival that is late and a departure that makes
  the next leg short are the same evidence with opposite signs.
* **A rate needs a span.** A slope measured across twenty minutes and quoted in
  ms/hour multiplies that window's noise by three. Below one hour no rate is
  fitted.
* **The signal must exceed its own scatter.** A mean offset smaller than the
  spread it was averaged from is traffic variance, not a clock.

Those guards were added because the shipped `estimate_camera_offsets.py` script,
run against the adversarial scenario, flagged **every camera in the network** as
drifting. An alert that fires everywhere is an alert nobody reads.

**The measured sensitivity limit**: with this topology's travel windows, a
45-second offset is *below* the detector's noise floor and it correctly declines
to call it — while the estimator finds the same offset exactly. The monitor
watches for gross faults; the estimator measures.

---

## 7. The integrity gate

Called before path reconstruction. Four findings, and they differ in how badly
they undermine a route:

| code | severity | meaning |
|---|---|---|
| `stale_verification` | warning | the camera's clock has not been checked recently, or ever |
| `drift_alert` | warning (blocking by config) | drift detection has an open alert |
| `low_reliability_source` | warning (low) / info (medium) | timestamps came from a filename or an overlay |
| `offset_spread` | **blocking** | known offsets span more than the shortest transit between the queried cameras |

The blocking case is the one worth dwelling on. Past that point two sightings
can be reordered by the clock error alone — and a trajectory *is* an ordering,
so there is no useful route to return and returning one anyway is the failure
this stage exists to prevent.

Everything else is a **caveat attached to the trajectory**, not a refusal. An
operator who knows a camera is 45 seconds fast can still read a route usefully,
as long as the route says so. A caveat in a log file is a caveat nobody acts on.

`Trajectory.temporal_integrity` is nullable, and the three states are distinct:

* `None` — assembled without checking. **Not** the same as clean.
* `verified=True` — checked, nothing to report.
* `verified=False` — checked, and here is what is wrong with it.

---

## 8. Correcting a clock

A correction rewrites history. Every timestamp that camera ever recorded moves,
and so does every conclusion drawn from them.

```python
change = apply_offset_change(
    sighting_repo,
    trajectory_repo,
    "cam_03",
    -45_000,
    clock,
    previous_offset_ms=0,
    reason="ntp sweep found the camera 45s fast",
)
```

Three things happen inside the caller's transaction, all or none:

1. the camera's sightings are recomputed **from `raw_timestamp`**, which is what
   makes running it twice harmless;
2. every trajectory built from those sightings is flagged
   `requires_recomputation`;
3. the change is recorded with **both** offsets — "the offset is now −45000" is
   unanswerable after the fact without knowing what it was.

The transaction belongs to the caller. Committing inside would take a decision
that is not this function's to make, and would stop several cameras being
corrected atomically.

**Why the flag exists.** A stored trajectory names its sightings and keeps its
own start and end bounds. After a correction those bounds no longer match the
sightings they name, so the conclusion no longer follows from the evidence. The
flag is how an operator finds the routes that need rebuilding; without it they
would have to work that out by hand.

---

## 9. Live ingestion: the watermark

Streams deliver out of order. Something has to decide when 14:02 is settled, and
that decision is a trade: wait longer and results are later, wait less and more
records arrive too late to use.

* **The watermark never moves backward.** It is a promise that everything before
  it is accounted for, and a promise that can be withdrawn is not one.
* **Nothing is dropped silently.** Every late record is counted, and under the
  default policy still delivered with its lateness attached. Silent loss in a
  system whose output is "here is everywhere the vehicle went" makes the answer
  look complete when it is not.

| policy | what happens past the allowance |
|---|---|
| `accept_and_recompute` (default) | take it, flag whatever was built from the old data |
| `queue` | hold it for a separate late-arrivals path |
| `drop` | discard it — counted and logged, never silent |

---

## 10. Operator procedures

**Verify a camera's clock**

```bash
python scripts/estimate_camera_offsets.py --scenario adversarial
python scripts/estimate_camera_offsets.py --scenario adversarial --drift
```

The script reports; it changes nothing. Correcting a clock is a decision made
after reading the numbers, not a side effect of running a diagnostic.

**Interpreting the output**

* a wide 95% interval means the estimate is not worth acting on yet — drive the
  reference vehicle again;
* `constant_offset` → correct it once;
* `linear_drift` → correcting will not hold; the camera needs a working time
  client;
* cameras listed as "not estimated" are still unverified, and their routes will
  carry a `stale_verification` caveat.

**After applying a correction**, rebuild the trajectories flagged
`requires_recomputation`. Until then they are stale conclusions resting on
evidence that has moved.

---

## 11. Configuration

| setting | default | what it controls |
|---|---|---|
| `timesync.verification_staleness_hours` | 168 | how long a clock check stays good |
| `timesync.drift_alert_ms` | 2000 | offset magnitude that raises an alert |
| `timesync.drift_rate_alert_ms_per_hour` | 500 | fitted drift rate that raises an alert |
| `timesync.min_reference_passes` | 3 | fewest passes before a camera is estimated at all |
| `timesync.watermark_lateness_sec` | 120 | how far behind a live record may arrive |
| `timesync.block_on_drift_alert` | false | whether an alert refuses reconstruction or annotates it |

---

## 12. Known limits

* **Detector sensitivity is bounded by the travel windows.** On this topology it
  cannot resolve offsets below roughly a minute. Narrower windows — from a
  better speed model or measured transit statistics — would tighten it.
* **Absolute offset estimates carry the bias of the expected transit times.**
  Differences do not. Prefer the clock probe where a device exposes one.
* **Container creation time is read by stage 10.** Stage 09 owns the seam and
  the tier; the demuxer is not a stage 09 dependency. See
  `tests/fixtures/media/README.md`.
* **Reading a stale trajectory raises** rather than serving a conclusion whose
  evidence moved. That is deliberate, and the recomputation flag is how those
  routes are found — but it does mean a correction must be followed by a rebuild
  before those routes are readable again.
