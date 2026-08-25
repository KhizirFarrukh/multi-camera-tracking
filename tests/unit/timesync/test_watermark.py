"""Unit tests for the watermark and late-arrival policy.

Two properties carry this module: the watermark never moves backward, because a
promise that can be withdrawn is not one, and nothing is dropped without being
counted, because silent data loss in a system whose output is "here is
everywhere the vehicle went" makes the answer look complete when it is not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.timesync import LatePolicy, WatermarkTracker, WatermarkVerdict

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
ALLOWANCE_SEC = 120.0


def _tracker(policy: LatePolicy = LatePolicy.ACCEPT_AND_RECOMPUTE) -> WatermarkTracker:
    """Return a tracker with a fixed allowance.

    Args:
        policy: What to do with records past the allowance.

    Returns:
        The tracker.
    """
    return WatermarkTracker(lateness_allowance_sec=ALLOWANCE_SEC, policy=policy)


def test_records_arriving_in_order__are_on_time() -> None:
    """The ordinary case."""
    tracker = _tracker()

    verdicts = [tracker.observe(START + timedelta(seconds=offset))[0] for offset in (0, 30, 60)]

    assert verdicts == [WatermarkVerdict.ON_TIME] * 3
    assert tracker.metrics.accepted == 3


def test_a_record_behind_the_watermark_but_inside_the_allowance__is_buffered() -> None:
    """Reordering within the allowance is what the allowance is for."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=300))

    verdict, lateness = tracker.observe(START + timedelta(seconds=120))

    assert verdict is WatermarkVerdict.BUFFERED
    assert lateness == pytest.approx(60.0)
    assert tracker.metrics.buffered == 1


def test_a_record_past_the_allowance__is_reported_late() -> None:
    """Beyond here the stream has a problem an operator should see."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=600))

    verdict, lateness = tracker.observe(START)

    assert verdict is WatermarkVerdict.LATE
    assert lateness == pytest.approx(480.0)
    assert tracker.metrics.late == 1


def test_the_watermark__never_moves_backward() -> None:
    """A promise that everything before it is settled cannot be withdrawn."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=600))
    high_watermark = tracker.watermark

    tracker.observe(START + timedelta(seconds=10))
    tracker.observe(START + timedelta(seconds=20))

    assert tracker.watermark == high_watermark


def test_the_watermark__advances_with_the_highest_event_time_seen() -> None:
    """It trails the furthest point by exactly the allowance."""
    tracker = _tracker()

    tracker.observe(START + timedelta(seconds=600))

    assert tracker.watermark == START + timedelta(seconds=600 - ALLOWANCE_SEC)


def test_the_first_record__establishes_the_watermark() -> None:
    """Boundary: before anything arrives there is nothing to be late for."""
    tracker = _tracker()

    assert tracker.watermark is None
    assert tracker.observe(START)[0] is WatermarkVerdict.ON_TIME
    assert tracker.watermark is not None


@pytest.mark.parametrize(
    ("policy", "accepted"),
    [
        (LatePolicy.ACCEPT_AND_RECOMPUTE, True),
        (LatePolicy.QUEUE, False),
        (LatePolicy.DROP, False),
    ],
)
def test_the_configured_policy__decides_what_reaches_the_pipeline(
    policy: LatePolicy, accepted: bool
) -> None:
    """All three policies are legitimate; the choice must be explicit."""
    tracker = _tracker(policy)
    tracker.observe(START + timedelta(seconds=600))

    verdict, _lateness = tracker.observe(START)

    assert tracker.accepts(verdict) is accepted


def test_dropped_and_queued_records__are_counted_separately() -> None:
    """Silent loss is the failure; a counter is the whole defence against it."""
    dropping = _tracker(LatePolicy.DROP)
    dropping.observe(START + timedelta(seconds=600))
    dropping.observe(START)

    queueing = _tracker(LatePolicy.QUEUE)
    queueing.observe(START + timedelta(seconds=600))
    queueing.observe(START)

    assert dropping.metrics.dropped == 1
    assert dropping.metrics.queued == 0
    assert queueing.metrics.queued == 1
    assert queueing.metrics.dropped == 0


def test_the_maximum_lateness__is_tracked_for_the_metrics_exporter() -> None:
    """How far behind the worst straggler was is what sizes the allowance."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=900))
    tracker.observe(START + timedelta(seconds=60))
    tracker.observe(START + timedelta(seconds=300))

    assert tracker.metrics.max_lateness_sec == pytest.approx(720.0)
    assert tracker.metrics.as_dict()["max_lateness_sec"] == pytest.approx(720.0)


def test_a_record_exactly_at_the_watermark__is_on_time() -> None:
    """Boundary, stated: the watermark is inclusive of its own instant."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=600))

    verdict, lateness = tracker.observe(tracker.watermark or START)

    assert verdict is WatermarkVerdict.ON_TIME
    assert lateness == 0.0


def test_a_record_exactly_at_the_allowance_boundary__is_buffered_not_late() -> None:
    """The other boundary, equally explicit."""
    tracker = _tracker()
    tracker.observe(START + timedelta(seconds=600))
    watermark = tracker.watermark
    assert watermark is not None

    verdict, _lateness = tracker.observe(watermark - timedelta(seconds=ALLOWANCE_SEC))

    assert verdict is WatermarkVerdict.BUFFERED
