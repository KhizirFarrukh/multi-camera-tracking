"""Throughput and yield counters.

Structure only, never durations. A test asserting that a stage took under N
milliseconds is a test that fails on a loaded build machine and teaches nobody
anything.
"""

from __future__ import annotations

import pytest

from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import DetectionMetrics, StageTimer

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------


def test_measure__accumulates_time_and_calls_per_stage() -> None:
    """A pipeline spending its time decoding needs different hardware from one in the model."""
    timer = StageTimer()

    for _ in range(3):
        with timer.measure("detect"):
            pass
    with timer.measure("decode"):
        pass

    assert timer.calls["detect"] == 3
    assert timer.calls["decode"] == 1
    assert timer.total_sec("detect") >= 0.0


def test_measure__a_stage_that_raises__still_records_the_time_it_burned() -> None:
    """Exactly the stage you want timings for is the one that failed."""
    timer = StageTimer()

    with pytest.raises(RuntimeError), timer.measure("detect"):
        raise RuntimeError("model fell over")

    assert timer.calls["detect"] == 1


def test_total_and_mean__for_a_stage_that_never_ran__are_zero() -> None:
    """Boundary: a stage with no calls has no mean, and zero is the honest answer."""
    timer = StageTimer()

    assert timer.total_sec("crop") == pytest.approx(0.0)
    assert timer.mean_ms("crop") == pytest.approx(0.0)


def test_as_dict__reports_only_stages_that_ran() -> None:
    """A new key means something new actually happened."""
    timer = StageTimer()
    with timer.measure("track"):
        pass

    flat = timer.as_dict()

    assert "track_total_sec" in flat
    assert "track_mean_ms" in flat
    assert "detect_total_sec" not in flat


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def test_record_frame__separates_frames_read_from_frames_the_detector_saw() -> None:
    """The gap between them is the saving the stage 10 motion gate claims."""
    metrics = DetectionMetrics(source_id="cam_01_test")

    for index in range(10):
        metrics.record_frame(ran_detector=index % 2 == 0)

    assert metrics.frames_read == 10
    assert metrics.frames_detected_on == 5
    assert metrics.prefilter_skip_ratio == pytest.approx(0.5)


def test_prefilter_skip_ratio__with_no_frames__is_zero_rather_than_dividing() -> None:
    """Boundary: a run that has read nothing has skipped nothing."""
    assert DetectionMetrics().prefilter_skip_ratio == pytest.approx(0.0)


def test_record_detection__builds_a_histogram_by_class() -> None:
    """A model that has quietly stopped finding trucks shows up here first."""
    metrics = DetectionMetrics()

    for _ in range(3):
        metrics.record_detection(ObjectClass.CAR)
    metrics.record_detection(ObjectClass.TRUCK)

    assert metrics.detections_total == 4
    assert metrics.detections_by_class["car"] == 3


def test_frames_per_second__with_nothing_timed__is_zero() -> None:
    """Boundary: dividing by an elapsed time of zero is not a throughput figure."""
    metrics = DetectionMetrics()
    metrics.record_frame(ran_detector=True)

    assert metrics.frames_per_second == pytest.approx(0.0)


def test_frames_per_second__after_timing__is_positive() -> None:
    """Structure, not a duration: the number has to exist and be finite."""
    metrics = DetectionMetrics()
    for _ in range(4):
        with metrics.timer.measure("detect"):
            pass
        metrics.record_frame(ran_detector=True)

    assert metrics.frames_per_second > 0.0


def test_as_dict__flattens_every_counter_for_an_exporter() -> None:
    """Stage 19 consumes this shape; a nested structure would need a translator."""
    metrics = DetectionMetrics(source_id="cam_01_test")
    metrics.record_frame(ran_detector=True)
    metrics.record_detection(ObjectClass.BUS)
    metrics.thumbnails_written = 1
    with metrics.timer.measure("crop"):
        pass

    flat = metrics.as_dict()

    assert flat["frames_read"] == 1
    assert flat["detections_total"] == 1
    assert flat["detections_bus"] == 1
    assert flat["thumbnails_written"] == 1
    assert "crop_total_sec" in flat
    assert all(isinstance(value, int | float) for value in flat.values())


def test_as_dict__omits_classes_that_were_never_detected() -> None:
    """Zero rows for every class in the enum would bury the ones that moved."""
    metrics = DetectionMetrics()
    metrics.record_detection(ObjectClass.CAR)

    assert "detections_truck" not in metrics.as_dict()
