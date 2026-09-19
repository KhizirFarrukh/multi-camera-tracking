"""Throughput and memory, measured rather than asserted from intuition.

Two rules keep these honest.

**No wall-clock thresholds on a shared machine.** "Under 50 ms" fails on a
loaded build agent and teaches nobody anything. Where a floor is asserted it is
deliberately loose, and where a comparison is asserted it is between two runs on
the same machine in the same process, which is the only fair one.

**The claims that need a real model say so and skip.** Batched inference
outperforming per-frame inference is a property of a GPU kernel, not of this
code. With the reference detector, batching is a Python loop and would measure
nothing. Better a skip that names what is missing than a green tick against a
number that means nothing.
"""

from __future__ import annotations

import importlib.util
import time
import tracemalloc
from pathlib import Path

import pytest

from multicam_tracker.ingest import FileVideoSource, MotionGate, SyntheticVideoSource
from multicam_tracker.vision import (
    DetectionFilters,
    DetectionMetrics,
    FakeDetector,
    SingleCameraTracker,
    TrackerConfig,
)
from tests.fixtures.vision import ReferenceBlobDetector, frame_sequence, linear_script

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MEDIA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "media"
CLIP = MEDIA_DIR / "sample_clean.mp4"

MINIMUM_FPS = 5.0
"""A floor, not a target.

The sample clip is 64x48 and the reference detector's flood fill is the slowest
thing in the loop, so this measures pipeline overhead rather than inference. Set
an order of magnitude below what the machine actually does, because the failure
worth catching is an accidental quadratic -- something that makes throughput
collapse -- and not a build agent having a slow minute."""


def _has_opencv() -> bool:
    """Return whether OpenCV is importable.

    Returns:
        ``True`` when it imports.
    """
    return importlib.util.find_spec("cv2") is not None


requires_opencv = pytest.mark.skipif(not _has_opencv(), reason="OpenCV is not installed")
requires_models = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="batched-vs-sequential throughput is a property of the model kernel, not of this code",
)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


@requires_opencv
def test_throughput_on_the_sample_clip__clears_the_documented_floor() -> None:
    """Catches an accidental quadratic, which is what a throughput floor is for."""
    detector = ReferenceBlobDetector()
    metrics = DetectionMetrics(source_id=CLIP.name)
    tracker = SingleCameraTracker("cam_01", CLIP.name, TrackerConfig(min_hits=2, min_iou=0.2))
    filters = DetectionFilters(min_area_px=1)

    with FileVideoSource(path=CLIP, camera_id="cam_01") as source:
        for frame in source.frames():
            metrics.record_frame(ran_detector=True)
            with metrics.timer.measure("detect"):
                detections = detector.detect(frame)
            with metrics.timer.measure("track"):
                tracker.update(
                    frame,
                    filters.apply(detections, frame_width=frame.width, frame_height=frame.height),
                )
    tracker.flush()

    assert metrics.frames_read > 0
    assert metrics.frames_per_second >= MINIMUM_FPS


@requires_models
def test_batched_inference__outperforms_per_frame_inference_at_batch_size_eight() -> None:
    """Needs a real model; see the module docstring for why this cannot be faked."""
    pytest.skip("requires weights; recorded here so the gap is visible rather than forgotten")


# ---------------------------------------------------------------------------
# The stage 10 motion gate, measured against the detector it exists to save
# ---------------------------------------------------------------------------


def test_the_motion_prefilter__measurably_reduces_detector_invocations() -> None:
    """The largest compute saving in the pipeline, asserted as a number.

    A claim worth making is worth measuring: the gate is only worth its
    complexity if the detector really is called fewer times.

    The frames are built here rather than taken from ``SyntheticVideoSource``
    with ``moving_rectangle=False``, and the reason is a finding worth keeping.
    That source shifts its whole background every frame by design -- two or
    three greyscale levels, which is *above* the configured sensitivity -- so a
    gate watching it correctly reports motion and skips only a tenth of the
    frames. The first version of this test asserted against that source and
    failed at 0.095, which looked like a broken gate and was a scene that was
    never static.

    "A street camera at night sees the same empty road for hours" means
    identical frames, so that is what this feeds it.
    """
    detector = FakeDetector(linear_script(frames=200))
    gate = MotionGate(sensitivity=2.0, force_interval_sec=5.0)
    frames = frame_sequence(200, width=160, height=120, camera_id="cam_static")

    for frame in frames:
        if gate.should_pass(frame):
            detector.detect(frame)

    assert gate.stats.seen == 200
    assert detector.call_count == gate.stats.passed
    assert gate.stats.skip_ratio > 0.9

    # The forced-sample interval is what keeps the saving honest: a vehicle
    # parked in view stops producing motion, and a frame has to go through
    # anyway or a car that parks vanishes from the record. At 10 fps over 20
    # seconds with a 5-second interval, that is a handful of frames.
    assert gate.stats.forced >= 3


def test_the_motion_prefilter__still_lets_a_moving_object_through() -> None:
    """A saving that also loses the vehicles is not a saving.

    Asserted directly against the static case above, because the gate is only
    correct if the two differ.
    """
    static_gate = MotionGate(sensitivity=2.0, force_interval_sec=60.0)
    moving_gate = MotionGate(sensitivity=2.0, force_interval_sec=60.0)

    for gate, moving in ((static_gate, False), (moving_gate, True)):
        source = SyntheticVideoSource(
            camera_id="cam_01",
            frame_count=60,
            width=160,
            height=120,
            moving_rectangle=moving,
            fps=10.0,
        )
        with source:
            for frame in source.frames():
                gate.should_pass(frame)

    assert moving_gate.stats.passed > static_gate.stats.passed


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory__across_a_long_clip__does_not_grow_with_frame_count() -> None:
    """A tracker that held every crop would exhaust a machine on an hour of footage.

    Measured as growth between two equal halves of the same run rather than as
    an absolute figure, because an absolute figure depends on the interpreter,
    the allocator, and whatever else the process is doing.
    """
    config = TrackerConfig(retain_crops=True, max_retained_frames=3, min_hits=2, min_iou=0.2)
    tracker = SingleCameraTracker("cam_01", "cam_01_long", config)
    filters = DetectionFilters(min_area_px=1)
    detector = ReferenceBlobDetector()

    def run_frames(count: int) -> None:
        """Feed the tracker a run of generated frames.

        Args:
            count: How many frames to feed.
        """
        source = SyntheticVideoSource(
            camera_id="cam_01",
            source_id="cam_01_long",
            frame_count=count,
            width=160,
            height=120,
            fps=10.0,
        )
        with source:
            for frame in source.frames():
                detections = detector.detect(frame)
                tracker.update(
                    frame,
                    filters.apply(detections, frame_width=frame.width, frame_height=frame.height),
                )

    tracemalloc.start()
    try:
        run_frames(150)
        first_half = tracemalloc.get_traced_memory()[0]
        run_frames(150)
        second_half = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()

    growth = second_half - first_half

    # Generous: the point is that it is bounded, not that it is zero. A tracker
    # keeping every crop of a 160x120 clip would add roughly 20 MB over 150
    # frames, which this catches by two orders of magnitude.
    assert growth < 2_000_000, f"memory grew by {growth} bytes over the second half"
    assert tracker.retained_image_count <= config.max_retained_frames * max(
        1, len(tracker.active_track_ids)
    )


def test_retained_images__never_exceed_the_cap_at_any_point_in_a_run() -> None:
    """The invariant itself, checked on every single update rather than at the end."""
    config = TrackerConfig(retain_crops=True, max_retained_frames=2, min_hits=2, min_iou=0.2)
    tracker = SingleCameraTracker("cam_01", "cam_01_long", config)
    filters = DetectionFilters(min_area_px=1)
    detector = ReferenceBlobDetector()

    source = SyntheticVideoSource(
        camera_id="cam_01", source_id="cam_01_long", frame_count=120, width=160, height=120
    )
    with source:
        for frame in source.frames():
            tracker.update(
                frame,
                filters.apply(
                    detector.detect(frame), frame_width=frame.width, frame_height=frame.height
                ),
            )
            live = max(1, len(tracker.active_track_ids))
            assert tracker.retained_image_count <= config.max_retained_frames * live


def test_processing_a_long_clip__completes_without_unbounded_slowdown() -> None:
    """Per-frame cost must not rise with how many frames have already been seen.

    Compared within one process, second half against first, which is the only
    comparison a shared machine makes fairly.
    """
    config = TrackerConfig(retain_crops=False, min_hits=2, min_iou=0.2)
    tracker = SingleCameraTracker("cam_01", "cam_01_long", config)
    detector = FakeDetector(linear_script(frames=400, start=(4, 40), step=(1, 0)))

    source = SyntheticVideoSource(
        camera_id="cam_01", source_id="cam_01_long", frame_count=400, width=160, height=120
    )
    with source:
        frames = list(source.frames())

    started = time.perf_counter()
    for frame in frames[:200]:
        tracker.update(frame, detector.detect(frame))
    first_half = time.perf_counter() - started

    started = time.perf_counter()
    for frame in frames[200:]:
        tracker.update(frame, detector.detect(frame))
    second_half = time.perf_counter() - started

    # Ten times, not two: the assertion is about complexity, not about timing
    # noise, and on a run this short the noise floor is large relative to the
    # work. A quadratic would show up as hundreds of times, not ten.
    assert second_half < max(first_half, 0.001) * 10
