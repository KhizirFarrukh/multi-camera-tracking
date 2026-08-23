"""Unit tests for direction and speed inference.

The bearings are checked against compass directions that can be verified by
inspection -- due east is 90 degrees, due north is 0 -- rather than against
whatever the implementation happens to return.
"""

from __future__ import annotations

import pytest

from multicam_tracker.models import GeoPoint
from multicam_tracker.pathing import bearing_degrees, compass_point, summarize_movement
from tests.fixtures.factories import make_camera
from tests.fixtures.pathing import prepared

pytestmark = pytest.mark.unit

ORIGIN = GeoPoint(lat=0.0, lon=0.0)


def _cameras(positions: dict[str, tuple[float, float]]) -> dict[str, object]:
    """Build cameras at explicit coordinates.

    Args:
        positions: ``camera_id -> (lat, lon)``.

    Returns:
        Cameras by id.
    """
    return {
        camera_id: make_camera(camera_id=camera_id, lat=lat, lon=lon)
        for camera_id, (lat, lon) in positions.items()
    }


# ---------------------------------------------------------------------------
# Bearings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon", "expected", "label"),
    [
        (0.0, 1.0, 90.0, "due east"),
        (1.0, 0.0, 0.0, "due north"),
        (0.0, -1.0, 270.0, "due west"),
        (-1.0, 0.0, 180.0, "due south"),
    ],
)
def test_bearing__matches_the_expected_compass_direction(
    lat: float, lon: float, expected: float, label: str
) -> None:
    """Verifiable by inspection, which is the point of choosing these four."""
    assert bearing_degrees(ORIGIN, GeoPoint(lat=lat, lon=lon)) == pytest.approx(expected, abs=0.5)


def test_bearing__between_two_identical_points__is_zero_rather_than_undefined() -> None:
    """A repeat detection has no heading, and raising would push that on callers."""
    assert bearing_degrees(ORIGIN, ORIGIN) == 0.0


@pytest.mark.parametrize(
    ("bearing", "expected"),
    [(0.0, "N"), (45.0, "NE"), (90.0, "E"), (180.0, "S"), (359.0, "N")],
)
def test_compass_point__labels_the_eight_sectors(bearing: float, expected: str) -> None:
    """Including the wrap at 360, where an off-by-one is easy."""
    assert compass_point(bearing) == expected


# ---------------------------------------------------------------------------
# Speeds
# ---------------------------------------------------------------------------


def test_speed__is_distance_over_elapsed_time() -> None:
    """One degree of longitude at the equator is about 111.3 km."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 1.0)})
    path = [prepared("cam_a", 0.0, 0.9), prepared("cam_b", 3600.0, 0.9)]

    summary = summarize_movement(path, cameras, implausible_speed_kph=1000.0)

    assert summary.hops[0].speed_kph == pytest.approx(111.3, rel=0.01)


def test_an_impossible_speed__is_flagged_and_the_hop_is_kept() -> None:
    """The cause might be the match, the clock, or the topology.

    Deleting the hop would hide all three, so it is reported instead.
    """
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 1.0)})
    path = [prepared("cam_a", 0.0, 0.9), prepared("cam_b", 60.0, 0.9)]

    summary = summarize_movement(path, cameras, implausible_speed_kph=200.0)

    assert summary.has_implausible_speed
    assert summary.implausible_hops == [0]
    assert len(summary.hops) == 1


def test_a_plausible_speed__is_not_flagged() -> None:
    """The control."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.01)})
    path = [prepared("cam_a", 0.0, 0.9), prepared("cam_b", 120.0, 0.9)]

    summary = summarize_movement(path, cameras, implausible_speed_kph=200.0)

    assert not summary.has_implausible_speed


def test_zero_elapsed_time__does_not_divide_by_zero() -> None:
    """Boundary. The graph never emits such a hop, but this function is public."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.01)})
    path = [prepared("cam_a", 0.0, 0.9), prepared("cam_b", 0.0, 0.9)]

    summary = summarize_movement(path, cameras)

    assert summary.hops[0].speed_kph == 0.0


# ---------------------------------------------------------------------------
# Trajectory-level statistics
# ---------------------------------------------------------------------------


def test_a_doubling_back_route__is_detected_as_a_reversal() -> None:
    """East then west is the vehicle turning around, and worth surfacing."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.02), "cam_c": (0.0, 0.0)})
    path = [
        prepared("cam_a", 0.0, 0.9),
        prepared("cam_b", 120.0, 0.9),
        prepared("cam_c", 240.0, 0.9),
    ]

    summary = summarize_movement(path, cameras, implausible_speed_kph=1000.0)

    assert summary.reversals == [1]


def test_a_straight_route__reports_no_reversals() -> None:
    """The control."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.01), "cam_c": (0.0, 0.02)})
    path = [
        prepared("cam_a", 0.0, 0.9),
        prepared("cam_b", 120.0, 0.9),
        prepared("cam_c", 240.0, 0.9),
    ]

    summary = summarize_movement(path, cameras, implausible_speed_kph=1000.0)

    assert summary.reversals == []


def test_the_dominant_direction__is_the_net_displacement_not_an_average_heading() -> None:
    """A vehicle that goes east then back west ends up where it started.

    Averaging the headings would report a direction it never travelled; the net
    displacement says where it actually ended up.
    """
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.02), "cam_c": (0.0, 0.01)})
    path = [
        prepared("cam_a", 0.0, 0.9),
        prepared("cam_b", 120.0, 0.9),
        prepared("cam_c", 240.0, 0.9),
    ]

    summary = summarize_movement(path, cameras, implausible_speed_kph=1000.0)

    assert summary.dominant_direction == "E"


def test_totals__are_summed_across_the_route() -> None:
    """What the UI puts under the map."""
    cameras = _cameras({"cam_a": (0.0, 0.0), "cam_b": (0.0, 0.01), "cam_c": (0.0, 0.02)})
    path = [
        prepared("cam_a", 0.0, 0.9),
        prepared("cam_b", 120.0, 0.9),
        prepared("cam_c", 240.0, 0.9),
    ]

    summary = summarize_movement(path, cameras, implausible_speed_kph=1000.0)

    assert summary.total_elapsed_sec == pytest.approx(240.0)
    assert summary.total_distance_meters == pytest.approx(
        summary.hops[0].distance_meters + summary.hops[1].distance_meters
    )
    assert summary.average_speed_kph > 0.0


def test_a_single_sighting__implies_no_movement_at_all() -> None:
    """Boundary."""
    summary = summarize_movement([prepared("cam_a", 0.0, 0.9)], _cameras({"cam_a": (0.0, 0.0)}))

    assert summary.hops == []
    assert summary.total_distance_meters == 0.0


def test_a_camera_missing_from_the_map__is_skipped_rather_than_fatal() -> None:
    """A camera decommissioned after a sighting must not make it unreadable."""
    cameras = _cameras({"cam_a": (0.0, 0.0)})
    path = [prepared("cam_a", 0.0, 0.9), prepared("cam_gone", 120.0, 0.9)]

    summary = summarize_movement(path, cameras)

    assert summary.hops == []
