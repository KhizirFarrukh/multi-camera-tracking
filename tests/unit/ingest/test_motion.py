"""Unit tests for the motion prefilter.

The forced-sample interval is the case worth reading. Pure motion detection
stops reporting a vehicle the moment it parks, and a car that vanishes from the
record looks exactly like a car that drove away.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import Frame, MotionGate, downsample_grey

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
WIDTH, HEIGHT = 96, 72


def _frame(index: int, image: np.ndarray, seconds: float | None = None) -> Frame:
    """Build a frame carrying a given image.

    Args:
        index: Frame index.
        image: The image.
        seconds: Offset from the base instant. Defaults to one second per index.

    Returns:
        The frame.
    """
    offset = index if seconds is None else seconds
    moment = START + timedelta(seconds=offset)
    return Frame(
        image=image,
        frame_index=index,
        raw_timestamp=moment,
        timestamp_utc=moment,
        source_id="src",
        camera_id="cam_01",
    )


def _flat(value: int = 100) -> np.ndarray:
    """Return a uniform image.

    Args:
        value: The grey level.

    Returns:
        A BGR image.
    """
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def _with_object(value: int = 100) -> np.ndarray:
    """Return an image with a bright block in it.

    Args:
        value: The background level.

    Returns:
        A BGR image.
    """
    image = _flat(value)
    image[20:50, 20:60] = 240
    return image


def test_identical_consecutive_frames__are_skipped() -> None:
    """The whole saving: a static scene costs nothing beyond the comparison."""
    gate = MotionGate(sensitivity=2.0, force_interval_sec=1000.0)
    still = _flat()

    passed = [gate.should_pass(_frame(index, still)) for index in range(5)]

    assert passed == [True, False, False, False, False]
    assert gate.stats.skipped == 4


def test_a_frame_with_significant_change__passes() -> None:
    """The control: something moved, and the detector needs to see it."""
    gate = MotionGate(sensitivity=2.0, force_interval_sec=1000.0)
    gate.should_pass(_frame(0, _flat()))

    assert gate.should_pass(_frame(1, _with_object()))


def test_the_first_frame__always_passes() -> None:
    """There is nothing to compare it against, and it establishes the reference."""
    assert MotionGate().should_pass(_frame(0, _flat()))


def test_the_forced_interval__guarantees_periodic_sampling_in_a_static_scene() -> None:
    """A parked vehicle stays in the record.

    Without this a car that stops being detected is indistinguishable from a car
    that left.
    """
    gate = MotionGate(sensitivity=2.0, force_interval_sec=3.0)
    still = _flat()

    passed = [gate.should_pass(_frame(index, still, seconds=float(index))) for index in range(8)]

    assert passed[0] is True
    assert passed[3] is True
    assert gate.stats.forced >= 2


def test_the_skip_ratio__is_accurate_against_a_known_input() -> None:
    """The metric the compute saving is claimed from, so it has to be right."""
    gate = MotionGate(sensitivity=2.0, force_interval_sec=1000.0)
    frames = [_frame(index, _flat()) for index in range(10)]

    list(gate.filter(frames))

    assert gate.stats.seen == 10
    assert gate.stats.passed == 1
    assert gate.stats.skipped == 9
    assert gate.stats.skip_ratio == pytest.approx(0.9)


def test_the_skip_ratio__of_an_unused_gate__is_zero() -> None:
    """A gate that processed nothing has saved nothing, and must not claim to."""
    assert MotionGate().stats.skip_ratio == 0.0


def test_sensitivity__measurably_changes_the_skip_rate() -> None:
    """The knob has to do something, in the direction the name implies."""
    frames = []
    for index in range(10):
        image = _flat()
        # A small, steady change: below a coarse threshold, above a fine one.
        image[10:20, 10:20] = 100 + index * 3
        frames.append(_frame(index, image, seconds=float(index)))

    sensitive = MotionGate(sensitivity=0.05, force_interval_sec=1000.0)
    insensitive = MotionGate(sensitivity=50.0, force_interval_sec=1000.0)

    list(sensitive.filter(frames))
    list(insensitive.filter(frames))

    assert sensitive.stats.passed > insensitive.stats.passed


def test_a_negative_sensitivity__is_rejected() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="cannot be negative"):
        MotionGate(sensitivity=-1.0)


def test_reset__forgets_the_reference_frame() -> None:
    """A reconnected source may be looking at a different scene entirely.

    Comparing against a stale reference would report motion that is really just
    the gap.
    """
    gate = MotionGate(sensitivity=2.0, force_interval_sec=1000.0)
    still = _flat()
    gate.should_pass(_frame(0, still))
    assert not gate.should_pass(_frame(1, still))

    gate.reset()

    assert gate.should_pass(_frame(2, still))


def test_the_filter__yields_only_the_frames_that_passed() -> None:
    """The gate as a pipeline stage rather than a predicate."""
    gate = MotionGate(sensitivity=2.0, force_interval_sec=1000.0)
    frames = [_frame(0, _flat()), _frame(1, _flat()), _frame(2, _with_object())]

    kept = list(gate.filter(frames))

    assert [frame.frame_index for frame in kept] == [0, 2]


def test_downsampling__shrinks_a_large_frame_and_keeps_a_small_one() -> None:
    """The comparison runs small: motion that matters survives, noise does not."""
    large = downsample_grey(np.zeros((1080, 1920, 3), dtype=np.uint8))
    small = downsample_grey(np.zeros((48, 64, 3), dtype=np.uint8))

    assert max(large.shape) <= 64
    assert small.shape == (48, 64)


def test_seconds_until_forced__counts_down_toward_the_next_forced_sample() -> None:
    """Lets a caller reason about when the gate will next let something through."""
    gate = MotionGate(sensitivity=2.0, force_interval_sec=10.0)
    gate.should_pass(_frame(0, _flat(), seconds=0.0))

    assert gate.seconds_until_forced(START + timedelta(seconds=4)) == pytest.approx(6.0)
    assert gate.seconds_until_forced(START + timedelta(seconds=30)) == 0.0
