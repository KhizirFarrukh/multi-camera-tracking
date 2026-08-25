"""Unit tests for drift detection.

The false-positive case is the important one. Fast traffic and a fast clock look
identical in a single observation, and a detector that cannot tell them apart
would cry wolf on every busy afternoon until nobody read its alerts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.models import TimeWindow
from multicam_tracker.timesync import ReferencePass, detect_drift

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 6, 0, 0, tzinfo=UTC)
EXPECTED_SEC = 120.0
ALERT_MS = 2000.0
RATE_ALERT_MS_PER_HOUR = 500.0


def _passes_into(
    camera_id: str,
    offsets_sec: list[float],
    *,
    neighbours: tuple[str, ...] = ("cam_01", "cam_04"),
    spacing_hours: float = 1.0,
) -> list[ReferencePass]:
    """Build arrivals at one camera, each shifted by a known amount.

    Args:
        camera_id: The camera under analysis.
        offsets_sec: One shift per pass, in the order they occur.
        neighbours: Cameras the vehicles arrive from, cycled through so a bias
            is seen from more than one direction.
        spacing_hours: Hours between successive passes.

    Returns:
        The reference passes.
    """
    passes = []
    for index, offset in enumerate(offsets_sec):
        departure = START + timedelta(hours=spacing_hours * index)
        passes.append(
            ReferencePass(
                from_camera_id=neighbours[index % len(neighbours)],
                to_camera_id=camera_id,
                departure_utc=departure,
                arrival_utc=departure + timedelta(seconds=EXPECTED_SEC + offset),
                expected_sec=EXPECTED_SEC,
                vehicle_id=f"ref_{index}",
            )
        )
    return passes


def test_a_constant_offset__is_detected_with_the_right_sign_and_magnitude() -> None:
    """A camera reading 45 seconds late, consistently, from every neighbour."""
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [45.0] * 6),
        alert_ms=ALERT_MS,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert analysis.has_alert
    assert analysis.alert is not None
    assert analysis.alert.kind == "constant_offset"
    assert analysis.alert.magnitude_ms == pytest.approx(45_000, abs=100)


def test_a_camera_reading_early__is_detected_with_a_negative_magnitude() -> None:
    """Sign matters: the correction goes the other way."""
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [-30.0] * 6),
        alert_ms=ALERT_MS,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert analysis.alert is not None
    assert analysis.alert.magnitude_ms == pytest.approx(-30_000, abs=100)


def test_linear_drift__is_detected_and_its_rate_is_fitted() -> None:
    """A clock gaining steadily needs a time client, not another correction.

    Two seconds per hour over six hours: the fitted rate is what tells an
    operator that correcting the current offset will not hold.
    """
    gain_per_hour = 2.0
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [gain_per_hour * hour for hour in range(6)]),
        alert_ms=1e9,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert analysis.alert is not None
    assert analysis.alert.kind == "linear_drift"
    assert analysis.rate_ms_per_hour == pytest.approx(gain_per_hour * 1000, rel=0.05)


def test_a_healthy_clock__produces_no_alert() -> None:
    """The control. Small scatter around zero is ordinary traffic variance."""
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [0.4, -0.3, 0.2, -0.5, 0.1, 0.3]),
        alert_ms=ALERT_MS,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert not analysis.has_alert


def test_genuinely_fast_traffic__does_not_raise_a_drift_alert() -> None:
    """The false positive that would destroy trust in the detector.

    Every vehicle crosses quickly, but the bias is well under the alert
    threshold -- which is what separates "the traffic is moving" from "the clock
    is wrong".
    """
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [-1.5, -1.2, -1.8, -1.0, -1.4, -1.6]),
        alert_ms=ALERT_MS,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert not analysis.has_alert


@pytest.mark.parametrize(
    ("offset_sec", "expected_alert"),
    [(1.999, False), (2.0, True)],
    ids=["just below the threshold", "exactly at the threshold"],
)
def test_the_alert_threshold__is_tested_at_its_exact_boundary(
    offset_sec: float, expected_alert: bool
) -> None:
    """Inclusive at the threshold, stated rather than left to be discovered."""
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [offset_sec] * 6),
        alert_ms=ALERT_MS,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    )

    assert analysis.has_alert is expected_alert


def test_departures_contribute_with_the_opposite_sign() -> None:
    """A camera whose clock reads late makes the next leg look short.

    Using only arrivals would halve the evidence and miss a camera at the end of
    every route.
    """
    departures = [
        ReferencePass(
            from_camera_id="cam_03",
            to_camera_id="cam_04",
            departure_utc=START + timedelta(hours=index),
            # The reported departure is 45 s late, so the leg appears 45 s short.
            arrival_utc=START + timedelta(hours=index, seconds=EXPECTED_SEC - 45.0),
            expected_sec=EXPECTED_SEC,
        )
        for index in range(6)
    ]

    analysis = detect_drift(
        "cam_03", departures, alert_ms=ALERT_MS, rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR
    )

    assert analysis.alert is not None
    assert analysis.alert.magnitude_ms == pytest.approx(45_000, abs=100)


def test_the_analysis__records_which_neighbours_contributed() -> None:
    """A bias seen from one neighbour is more likely a bad link than a bad clock."""
    analysis = detect_drift(
        "cam_03",
        _passes_into("cam_03", [45.0] * 6, neighbours=("cam_01", "cam_04")),
        alert_ms=ALERT_MS,
    )

    assert analysis.directions_seen == {"cam_01", "cam_04"}


def test_passes_outside_the_window__are_ignored() -> None:
    """Drift is analysed over a period, not over everything ever recorded."""
    passes = _passes_into("cam_03", [45.0] * 6)
    window = TimeWindow(start_utc=START, end_utc=START + timedelta(hours=2))

    analysis = detect_drift("cam_03", passes, window=window, alert_ms=ALERT_MS)

    assert analysis.sample_count == 2


def test_a_camera_with_no_passes__produces_no_analysis_and_no_alert() -> None:
    """Silence is not evidence of a healthy clock, and must not be reported as one."""
    analysis = detect_drift("cam_09", _passes_into("cam_03", [45.0] * 4))

    assert analysis.sample_count == 0
    assert not analysis.has_alert


def test_two_samples__do_not_produce_a_fitted_rate() -> None:
    """Two points always fit a line perfectly, which is a confident claim from
    no evidence."""
    analysis = detect_drift(
        "cam_03", _passes_into("cam_03", [10.0, 20.0]), alert_ms=1e9, rate_alert_ms_per_hour=1.0
    )

    assert analysis.rate_ms_per_hour == 0.0
    assert not analysis.has_alert


def test_the_alert_text__explains_what_to_do_about_it() -> None:
    """The distinction an operator acts on: correct it once, or fix the clock."""
    constant = detect_drift("cam_03", _passes_into("cam_03", [45.0] * 6), alert_ms=ALERT_MS).alert
    drifting = detect_drift(
        "cam_03",
        _passes_into("cam_03", [2.0 * hour for hour in range(6)]),
        alert_ms=1e9,
        rate_alert_ms_per_hour=RATE_ALERT_MS_PER_HOUR,
    ).alert

    assert constant is not None
    assert drifting is not None
    assert "not traffic" in constant.describe()
    assert "time client" in drifting.describe()
