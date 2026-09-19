# Vehicle Detection and Single-Camera Tracking

How frames become sightings, why one vehicle pass produces exactly one row, and
what every tunable parameter actually does.

---

## 1. The chain

```
Frame  ->  Preprocessor          stage 10: mask, rotate, resize
       ->  Detector              boxes, in the PROCESSED frame
       ->  in_source_coordinates boxes, in the ORIGINAL frame
       ->  DetectionFilters      specks, whole-frame boxes, outside the ROI
       ->  SingleCameraTracker   detections bound into tracks over time
       ->  VehicleTrack          emitted when the vehicle leaves
       ->  track_to_sighting     one row, one thumbnail
```

The third step is the one that gets skipped and the one that breaks everything
downstream. A detector run on a rotated, resized frame reports boxes in *that*
frame. Stored as-is, the box is perfectly plausible and in the wrong part of the
picture — the thumbnail crops the wrong region, review shows the wrong car, and
every symptom reads as a detector problem rather than as arithmetic.

---

## 2. One pass, one sighting

A car crossing one camera at 5 fps generates about forty detections. Without
something binding them together, that is forty sightings of forty vehicles as
far as everything downstream can tell: forty rows, forty thumbnails, forty
matching candidates, and a trajectory that appears to visit one camera forty
times in eight seconds.

A `VehicleTrack` is the answer, and **nothing is emitted until the track ends** —
because which frame of a pass is worth keeping cannot be known until the pass is
over.

---

## 3. Tracker parameters

| threshold | value | what it decides |
|---|---|---|
| `track_association_min_iou` | 0.3 | overlap below which a detection cannot continue a track |
| `track_max_age_frames` | 5 | consecutive updates a track survives unmatched — the occlusion allowance |
| `track_min_hits_to_confirm` | 3 | detections before a track may emit a sighting |
| `max_retained_frames` | 5 | cropped images held per track (memory bound) |

**Age counts tracker updates, not source frame indices.** The tracker is fed
sampled frames, so consecutive updates can be six source frames apart. The gap
that matters is the gap in what the tracker *saw*; counting source indices would
make the occlusion allowance depend on the sampling rate.

The allowance is inclusive: a track survives up to and including
`max_age_frames` consecutive unmatched updates, and the one after that ends it.
Both sides of that boundary are asserted.

**Too generous an allowance stitches two different vehicles into one track**,
which is the confident-wrong-answer failure in miniature. Too tight fragments
one pass into several sightings, which looks like a vehicle circling.

### Why a motion model

Matching on raw box overlap works until two vehicles pass each other, at which
point their boxes overlap each other as much as they overlap themselves, and the
identities swap. A Kalman filter carrying each track's velocity keeps predicting
them *through* the crossing, so the overlap that matters is with where each
vehicle should be rather than where it was.

State is `[cx, cy, scale, ratio, dcx, dcy, dscale]` — SORT's. Aspect ratio is
modelled as constant: a vehicle does not change shape; what changes is how much
of the frame it fills and where it is.

### Why optimal assignment, not greedy

Given two tracks and two detections, greedy matching takes the single best pair
first and leaves the other track whatever remains. That can be the wrong
detection even when a different pairing is better overall — which is an identity
swap, happening exactly when two vehicles are close together, which is exactly
when a swap matters.

So matching is a minimum-cost assignment problem, solved with Jonker-Volgenant.
The unit tests include the matrix where greedy is wrong:

```
cost = [[1, 2],
        [2, 100]]
```

Greedy takes `(0,0)` at 1.0 and is then forced into `(1,1)` at 100, for 101. The
optimum pairs across for 4.

Written out rather than taken from SciPy, which is not otherwise a dependency of
this project and would be pulled in for this one function.

---

## 4. Best-frame ranking

A vehicle is seen forty times and read once. **Which frame the plate is read
from determines plate accuracy more than any threshold in the OCR stage**, so
the ranking is explicit and configurable rather than a `max(key=...)`.

| criterion | weight | why |
|---|---|---|
| detection confidence | 0.35 | the detector's own opinion; a frame it was unsure about is usually occluded or blurred |
| sharpness | 0.30 | Laplacian variance of the crop — separates two frames the detector is equally happy with |
| area | 0.20 | a closer vehicle carries more plate detail |
| centrality | 0.15 | a central vehicle is fully visible and less distorted |
| edge discount | ×0.5 | a vehicle the frame cut in half is missing what it would be selected for |

The four weights sum to 1.0, enforced at startup. Each criterion scores in
`[0, 1]`, so the composite does too — any other sum puts the score outside that
range and makes it unreadable in a review UI.

**Area and sharpness are normalized within the track**, not against the frame.
Every vehicle is a small fraction of a frame, so an absolute area ratio would
make that criterion do nothing. The consequence: a ranking score is not
comparable between tracks.

**The edge penalty is multiplicative, not subtractive.** Subtracting a fixed 0.5
drives most edge frames to the clamp at zero, where they all tie and frame index
decides — which is exactly the wrong answer for a track that is *entirely* at
the frame edge, where one of those frames really is the best available.

---

## 5. Track to sighting

**Timestamp: the frame nearest the track's temporal midpoint.** Not the first
frame, and the reason is not aesthetics. A track begins with the vehicle
entering frame — clipped by the edge, partly occluded, at the periphery of the
lens — and ends the same way leaving. The midpoint is where the vehicle is most
fully in view, and it is the instant least sensitive to exactly when the
detector acquired and lost the track.

Dating from the first frame would place every vehicle systematically early, by
an amount that varies with the camera's field of view. That is a bias that
survives every local test and quietly corrupts travel-time plausibility across
the whole topology.

**Box: the same frame.** A timestamp and a box describing different instants
describe nothing.

**Confidence: the median across the track.** Not the mean — a track's first and
last frames are legitimately uncertain and drag a mean down without saying
anything about whether a vehicle passed. Not the maximum — one lucky frame would
promote a track of marginal detections to near-certainty.

**Class: a confidence-weighted vote.** A detector calling something a truck in
six uncertain frames and a car in five confident ones is telling you something a
plain majority discards.

**Thumbnail: the highest-ranked frame — which may not be the midpoint frame.**
The sighting is dated and located from the midpoint; the image is cropped from
the clearest view, because that image exists to be *read*, by OCR in stage 12
and by a human in stage 18. The two are usually the same frame and deliberately
need not be. A reviewer comparing `frame_index` against the thumbnail should
expect them to differ.

### Long tracks

A vehicle parked in view for ten minutes is one track, and collapsing it to one
instantaneous sighting misrepresents it in both directions: it claims the
vehicle was there at one moment rather than throughout, and it gives the path
reconstructor a single point where it should see a stationary period.

`split_long_tracks_sec` emits one sighting per interval. **Off by default**,
because for a vehicle that merely drives past — almost all of them — splitting
would manufacture several sightings of one transit and make a single pass look
like a vehicle circling.

---

## 6. Memory

A track holds a `TrackObservation` per frame (a handful of numbers) and a
cropped image for only its highest-ranked few. That split is what lets a vehicle
sit in view for ten minutes without accumulating six hundred crops.

Retention is a **sliding top-K**: each new crop is ranked against the retained
set and the lowest-scoring one is released. A frame evicted early cannot come
back even if every later frame turns out worse. That is the price of a fixed
memory bound, and the alternative — holding every crop to be sure — is what the
bound exists to prevent.

Crops are **owned copies**. A decoder hands out a view into a buffer it reuses
for the next frame, so a crop kept as a view would change underneath the track,
producing a thumbnail of a vehicle from a later frame — a symptom that reads as
a cropping bug rather than an ownership one.

---

## 7. Filtering, and counting what went

| filter | threshold | rejects |
|---|---|---|
| minimum area | 400 px | specks: distant pedestrians, birds, compression artifacts |
| maximum area | 0.5 of frame | a box covering the frame is a failed detection |
| aspect ratio | 0.25–5.0 | boxes no vehicle shape produces |
| region of interest | centroid or 0.5 overlap | the neighbouring property |
| edge policy | flag / drop / keep | a vehicle the frame cut in half |

**Bounds are inclusive at both ends**, stated once and tested at the exact
values, because "at the threshold" is where a filter gets argued about.

**A detection is counted under the first criterion it fails**, not all of them,
so `rejected_total` never exceeds `seen`.

**Every rejection is counted by reason**, and that matters more than the
filtering. A pipeline quietly discarding three quarters of its detections and
reporting healthy is the failure this project exists to avoid: the operator sees
a sparse trajectory and concludes the vehicle was not there, when in fact the
aspect-ratio window was misconfigured.

The edge policy defaults to **flag**, not drop. A clipped box usually has an
unreadable plate, but dropping it loses the only evidence that something passed
— and every vehicle is edge-clipped at the start and end of its track. Flagging
preserves the record and lets the ranking push those frames down, which is where
the decision belongs.

---

## 8. Swapping the detector

One setting, no caller anywhere naming a class:

```bash
MCT_DETECTION__DETECTOR_BACKEND=fixture
MCT_DETECTION__FIXTURE_DETECTIONS_PATH=tests/fixtures/detections/sample_clips.json
```

| backend | what it is | when |
|---|---|---|
| `yolo` | Ultralytics, lazily loaded | production |
| `fixture` | replays a committed recording | **CI**: no weights, no GPU, no network |
| `fake` | scripted per frame index | unit tests, where the right answer must be known exactly |

**Weights load on first use, never at import.** A unit test about bounding-box
arithmetic must not require a 50 MB download, and a pipeline must be
constructible on a machine that has not been provisioned yet. A missing weights
file raises `VisionError` naming the path and the environment variable, because
the operator's next action is to put a file somewhere.

**CUDA is requested, not assumed.** A machine configured for `cuda` without CUDA
runs on the CPU with a warning. Slow is a problem somebody can see and plan
around; a service that will not start at three in the morning is a different
kind of problem.

**Provenance travels with every detection.** `model_id` and `model_version` —
the Ultralytics release plus a digest of the weights file. Six months later,
"which model put this box here" has to be answerable from the record alone, and
after a model upgrade it is the only way to tell which stored sightings predate
the change.

---

## 9. What running it found

Four things, none of which came from reading the code.

**The rotation inverse can return −1.** Stage 10's mapping works in pixel
*centres*, where the inverse of a 90° rotation is `width − 1 − x`. A box edge
legitimately sits at `x = width`, because box coordinates are half-open, and
mapping that edge back gives −1. It happens only at the frame boundary, only by
one pixel, and only for a rotated camera. `in_source_coordinates` clamps that
single pixel; anything larger is a genuine mapping fault and is left to fail
against the non-negativity check.

**`sample_clean.mp4` contains two passes, not one.** The bar is drawn at
`(index * 3) % 52`, so on frame 18 it wraps from the right edge back to the
left. The tracker calls it a second vehicle, which is right: a real object
cannot cross the frame in one frame interval, and a tracker that accepted the
teleport would be the one that merges two vehicles into one trajectory.

**`SyntheticVideoSource(moving_rectangle=False)` is not a static scene.** It
shifts its whole background by two or three greyscale levels every frame by
design — above the motion gate's default sensitivity. A gate watching it
correctly reports motion and skips a tenth of the frames. The first version of
the prefilter test asserted against that source and failed at a skip ratio of
0.095, which looked like a broken gate and was a scene that was never still. The
test now feeds identical frames, and the gate skips 97%.

**A sighting's timestamp is truncated to milliseconds; a decoder reports
microseconds.** The domain model truncates on validation so a round trip through
storage is lossless (stage 02). Comparisons between a frame's timestamp and its
sighting's must be made at millisecond resolution — asserting equality at
microsecond resolution asserts something the schema does not promise.

---

## 10. Deviations from the stage prompt

| the prompt asked for | what was built, and why |
|---|---|
| fixtures recorded from a real model run | Recorded with `ReferenceBlobDetector` (in `tests/fixtures/vision.py`). No weights are installed here, and committing a 50 MB checkpoint is not this stage's decision. The recording names its own provenance and says `NOT with YOLO` in its `note` field. **Re-record with `--backend yolo` in stage 13 or 20.** |
| "detects vehicles in a committed sample image containing known vehicles" | Not assertable. The committed images are synthetic rectangles; a real model will not call them vehicles, and a photograph of a real road carries licensing and privacy questions a test fixture does not get to decide. The structural cases around it are tested. |
| "batched inference outperforms per-frame at batch size 8" | Skipped with a stated reason. That is a property of a GPU kernel, not of this code; with the reference detector batching is a Python loop and would measure nothing. The structural obligation — batched results *equal* sequential results — is asserted. |
| modules `{…thumbnails}.py` | Added `sightings.py`, `instrumentation.py`, and `factory.py`. The listed modules have no home for track-to-sighting conversion (it needs both ranking and thumbnails, and putting it in `track.py` would make the imports circular), for the performance counters, or for the backend switch that makes the swappability criterion true. |

---

## 11. What these numbers are not

**No threshold in this stage has been calibrated.** Unlike the matching and
pathing values, which were swept against ground truth, every number in the
stage 11 block of `thresholds.yaml` is a documented starting point with a stated
reason. The only detector available here is a synthetic one, and sweeping
against it would calibrate the fixture rather than the system.

Stage 13 or 20 must re-derive them against a real detector on real footage.
Until then, do not cite them as measured.

Measured with:

```bash
python scripts/benchmark_detection.py            # throughput, per stage
python scripts/record_detection_fixtures.py      # re-record the CI fixtures
```
