# Video Ingestion

How frames get from a file, a camera, or a generator into the rest of the
system, and what each sampling strategy costs.

Everything downstream consumes `Frame` objects and nothing else. That is the
point of this stage: stages 11–13 are tested against generated frames in CI with
no media files, no network, and no GPU.

---

## 1. The four standing rules

Each is enforced somewhere in the package, and each has tests whose only job is
to keep it true.

**Coordinates are always original source-frame coordinates.** A detector run on
a resized, rotated frame reports a box in *that* frame's space. Stored as-is it
puts the vehicle in the wrong part of the picture, and nothing downstream can
tell — the box is perfectly plausible, just wrong. Thumbnails crop the wrong
region, review shows the wrong car, and every symptom looks like a detector
problem rather than an arithmetic one.

**Timestamps are never interpolated.** A sampled frame carries the capture time
of the frame that was decoded. A sampler that smoothed the cadence to a regular
5 fps would move a vehicle to a moment nobody observed, and the trajectory built
from those times would be confidently wrong.

**Resources are released on every exit path.** Normal exit, exception, and
abandoning the generator half way. The third is the one that bites: breaking out
of a frame loop leaves a suspended generator, and without the `try/finally`
around the yield the decoder is held until the collector happens to run — far
too late on a batch of ten thousand files.

**Latency beats completeness on live sources.** The bounded buffer drops the
*oldest* frames. An operator watching for a vehicle now has no use for a tidy
record of an increasingly stale past.

---

## 2. Sampling strategies

| strategy | cadence | cost | when it is right |
|---|---|---|---|
| `EveryNthFrame` | source-dependent | lowest | fixed-rate estate, predictable counts |
| **`TargetFpsSampler`** | **as configured, in real time** | low | **the default** |
| `KeyframeOnlySampler` | whatever the encoder chose | lowest of all | archive scanning where completeness is secondary |
| `AdaptiveSampler` | idle rate, raised on detection | variable | live watching, where events are rare and detail matters when they happen |

`TargetFpsSampler` is the default because `EveryNthFrame(6)` is 5 fps on a 30 fps
camera and 4 fps on a 25 fps one — a single configuration that means different
things per camera. Working from the frames' own timestamps gives the same
cadence across a mixed estate, and keeps giving it when a camera's rate varies.

`KeyframeOnlySampler` is the cheapest option available: a keyframe decodes
without reference to its neighbours. The catch is the cadence — a typical
two-second GOP is 0.5 fps, and fast traffic crosses a frame between two of them.
It is also **only exact on the PyAV path**: OpenCV does not expose keyframe
flags, so the file source approximates with a fixed interval there.

`AdaptiveSampler` is driven by the consumer: it cannot know what the detector
found, so `note_detection()` is called by whatever does. It refuses a
configuration whose active rate is below its idle rate — that would make a
detection *slow sampling down*.

---

## 3. Motion prefilter

Typically the largest compute saving in the pipeline. A street camera at night
sees the same empty road for hours, and a detector run on each of those frames
costs the same as one run on a busy frame and finds nothing.

Frame differencing at 64 pixels on the longest edge: motion large enough to
matter survives aggressive downsampling, and the reduction also suppresses the
sensor noise that a full-resolution difference mistakes for movement.

**The forced-sample interval is what keeps it honest.** A vehicle parked in view
is stationary, and pure motion detection stops reporting it — so a frame goes
through at least every `motion_force_interval_sec` however still the scene is.
Without that, a car that parks vanishes from the record, which reads exactly
like a car that drove away.

The gate reports `skip_ratio` because the saving is a claim, and a claim worth
making is worth measuring.

---

## 4. Preprocessing and coordinate recovery

Applied in a fixed order — **mask, rotate, crop, resize** — and every step
returns the inverse that undoes it:

```python
processed, mapping = preprocessor.apply_to_frame(frame)
boxes = detector.detect(processed.image)  # processed-frame coordinates
source_boxes = [mapping.bbox_to_source(b) for b in boxes]  # what gets stored
```

Rotation is limited to right angles. An arbitrary angle needs an interpolating
warp, and worse, its inverse is not exact: a box mapped back through 37 degrees
is an axis-aligned box larger than the object it bounds. A camera mounted at 37
degrees is a mounting problem.

> The rotation inverse was wrong when first written — exact at one corner of the
> frame and off by the full width at the other, which is precisely the shape of
> bug that reads as a flaky detector. `np.rot90(k=1)` maps source `(x, y)` to
> `(y, width − 1 − x)`, and the `−1` is easy to lose. The tests push a known
> point through every transform, individually and composed, in both directions.

Under a downsample the mapping cannot recover what the resize discarded, so the
tests assert exactness for the lossless transforms and one-processed-pixel
accuracy under resize. That is the best any inverse can do, and saying so is
better than a tolerance nobody can justify.

---

## 5. Sources

### `SyntheticVideoSource`

The one stages 11–13 depend on. Deterministic from a seed, seeded **per frame
index** rather than carried forward — so frame 400 is identical whether it was
reached by iterating or by seeking, which is what makes a seek test mean
anything.

Faults are injected by index: a corrupt frame at N, a stall of N seconds, a
disconnect that is permanent or recoverable. Code that has never seen those does
not handle them; it has merely never been asked to.

### `FileVideoSource`

OpenCV first, PyAV as the fallback. Neither is enough alone: OpenCV is fast and
ubiquitous but reports frame rates as floats and has no useful notion of a
presentation timestamp; PyAV exposes exact rational rates, per-frame PTS, and a
codec name for an error message.

**PyAV identifies; OpenCV decodes.** FFmpeg's raw-format guesser will happily
"open" an arbitrary byte stream — a file of ASCII text came back as 640×64 at 25
fps and yielded a junk frame — so OpenCV's willingness to open something is not
evidence that it is a video. The header is probed first.

Damage is handled in three ways:

| damage | behaviour |
|---|---|
| a corrupt frame mid-file | counted and skipped; an hour of footage with one bad frame is still an hour |
| a truncated file | yields what is readable, then stops; this is what a recording interrupted by a power cut looks like |
| no decodable stream | raises, naming what was identified — the operator's next step needs it |

A clean end-of-file and a damaged one look **identical** through OpenCV's API:
`read()` simply returns `False`. The container's frame count is what separates
them, and without that check every complete file would be reported as truncated
— a warning that fires always is a warning nobody reads.

### `LiveStreamSource`

Assumes the network is unreliable, because it is.

* **Bounded buffer, oldest dropped, every drop counted.** A system quietly
  discarding half its input while reporting healthy is worse than one visibly
  falling behind.
* **The decoder never blocks.** It runs on its own thread and pushes into a
  buffer that never waits. A blocked decoder fills the socket's receive buffer,
  then the camera's transmit buffer, and the far end eventually resets the
  connection — turning a slow consumer into a disconnection.
* **Backoff is exponential with jitter.** The exponent stops a dead camera being
  hammered; the jitter stops forty cameras behind one failed switch from
  reconnecting in lockstep and knocking it over again.
* **Retries are finite by default.** A camera that is genuinely gone reaches a
  terminal `FAILED` state rather than being reported as permanently
  "connecting".

---

## 6. Multi-source reading

**Threads, not asyncio.** OpenCV and PyAV are blocking C extensions with no
async interface; driving them from an event loop means a thread pool anyway,
with the event loop's complexity and none of its benefit. They release the GIL
while decoding, so threads genuinely run in parallel here.

The requirement that matters is not throughput — it is that **one failing source
never stalls the others**. A deployment has forty cameras and at any moment one
is rebooting, one has a damaged file, and one is behind a switch being replaced.
Every per-source failure is caught, recorded in the health report, and left to
that camera.

Frames merge into one bounded queue. Ordering *within* a source is preserved;
ordering *between* sources is not, and must not be relied on — two cameras have
no shared clock at this layer, which is what stage 09's corrected timestamps are
for.

---

## 7. Frame memory ownership

`frame.image` is **read-only to consumers**. A decoder hands out a view into a
buffer it will reuse, so a consumer that writes into it corrupts the next frame
as well as its own, and one that keeps a reference holds a buffer that changes
underneath it. Both produce images that are subtly wrong rather than obviously
broken.

The array is marked non-writeable so the mistake raises, and
`frame.owned_copy()` is the one supported way to get an array a consumer may
keep or modify.

---

## 8. Test fixtures

Five clips, a few kilobytes each, regenerated by
`python scripts/make_sample_clips.py`:

| fixture | what it exercises |
|---|---|
| `sample_clean.mp4` | the ordinary case: 20 frames, 10 fps, 64×48 |
| `sample_truncated.ts` | a recording cut off mid-write |
| `sample_corrupt.ts` | a hole punched through the payload |
| `sample_vfr.mp4` | variable frame rate whose nominal rate is a lie |
| `sample_undecodable.mp4` | bytes that are not a video |

Transport streams for the damaged cases and MP4 for the rest, for a reason: an
MP4 keeps its index at the end, so a truncated one cannot be opened *at all* —
a different failure. A TS decodes from any point, which is what a camera writing
when the power failed actually leaves behind.

Live-source tests run against `tests/fixtures/local_stream_server.py`, which
serves a clip over loopback and can be told to drop connections or stall.
Depending on an external endpoint would make the suite flaky and the failures
someone else's to fix.

---

## 9. Known limits

* **Keyframe flags are exact only on the PyAV path.** OpenCV does not expose
  them, so the file source approximates with a fixed interval there, and
  `KeyframeOnlySampler` is correspondingly approximate.
* **Live frames are timestamped on arrival.** There is no container to ask, so
  network latency is inside the timestamp. Where a camera exposes its own clock,
  stage 09's probe is the better source.
* **Resizing is nearest-neighbour.** It works without OpenCV and the inverse is
  a pure scale either way. A detector that wants better resampling can have
  OpenCV do it; the coordinate arithmetic does not change.
* **The undecodable fixture is not a genuinely unsupported codec.** With FFmpeg
  present essentially every codec is supported, and an exotic encoder is not a
  reasonable test dependency. The adjacent real failure — a file with no
  decodable stream — exercises the same path.

---

## 10. Configuration

| setting | default | what it controls |
|---|---|---|
| `ingest.target_fps` | 5.0 | detection cadence |
| `ingest.motion_sensitivity` | 2.0 | greyscale change that counts as motion |
| `ingest.motion_force_interval_sec` | 5.0 | forced sample in a static scene |
| `ingest.live_buffer_frames` | 8 | frames between a live decoder and its consumer |
| `ingest.live_read_timeout_sec` | 10.0 | silence before a stream is treated as dead |
| `ingest.max_queued_frames` | 64 | merged-queue bound across all cameras |
