"""Unit tests for gap detection and annotation.

The outage case is the one that needs the cross-reference. "This camera saw
nothing" and "this camera was not working" are indistinguishable from the
trajectory alone, and reporting the second as the first turns an absence of
evidence into evidence of absence.
"""

from __future__ import annotations

import pytest

from multicam_tracker.models import TimeWindow
from multicam_tracker.pathing import GapKind, build_graph, detect_gaps, find_camera_outages
from multicam_tracker.pathing.optimal_path import best_path
from tests.fixtures.factories import BASE_INSTANT, make_sighting
from tests.fixtures.pathing import (
    CHAIN_MAX_SEC,
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    prepared,
)

pytestmark = pytest.mark.unit


def _reports(entries: list[tuple[str, float, float]], **kwargs: object) -> list[object]:
    """Reconstruct a route and return its gap reports.

    Args:
        entries: ``(camera, offset_sec, score)`` per candidate.
        **kwargs: Passed through to ``detect_gaps``.

    Returns:
        The gap reports.
    """
    topology = chain_topology_with_outlier()
    nodes = [prepared(camera, offset, score) for camera, offset, score in entries]
    graph = build_graph(nodes, topology, cameras=chain_cameras())
    path = best_path(graph)
    return detect_gaps(graph, path.node_indices, path.edges, topology, **kwargs)  # type: ignore[arg-type]


def test_a_fully_covered_trajectory__reports_no_gaps() -> None:
    """Every hop on a declared link inside its window is fully accounted for."""
    assert _reports([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)]) == []


def test_a_hop_with_no_topology_link__is_reported_as_a_coverage_gap() -> None:
    """Unmonitored ground, named as such."""
    reports = _reports([("cam_01", 0.0, 0.95), (FAR_CAMERA, 4000.0, 0.95)])

    assert [report.kind for report in reports] == [GapKind.COVERAGE]  # type: ignore[attr-defined]
    assert "no known route" in reports[0].describe() or "no direct link" in reports[0].describe()  # type: ignore[attr-defined]


def test_a_transit_far_beyond_the_plausible_maximum__is_reported_as_temporal() -> None:
    """The vehicle stopped, parked, or left the network and came back."""
    reports = _reports([("cam_01", 0.0, 0.9), ("cam_02", CHAIN_MAX_SEC * 4, 0.9)])

    assert [report.kind for report in reports] == [GapKind.TEMPORAL]  # type: ignore[attr-defined]


def test_an_unobserved_but_plausible_multi_hop_transit__is_reported_as_coverage() -> None:
    """Passing a camera that did not see the vehicle is worth saying out loud."""
    reports = _reports([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)])

    assert [report.kind for report in reports] == [GapKind.COVERAGE]  # type: ignore[attr-defined]
    assert "unobserved" in reports[0].describe()  # type: ignore[attr-defined]


def test_gap_reasons__are_distinct_per_gap_type() -> None:
    """An operator reads the sentence, not the enum."""
    coverage = _reports([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)])[0]
    temporal = _reports([("cam_01", 0.0, 0.9), ("cam_02", CHAIN_MAX_SEC * 4, 0.9)])[0]

    assert coverage.describe() != temporal.describe()  # type: ignore[attr-defined]
    assert coverage.describe() and temporal.describe()  # type: ignore[attr-defined]


def test_every_gap__carries_the_domain_model_the_trajectory_stores() -> None:
    """The report is for the operator; the model is what gets persisted."""
    report = _reports([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)])[0]

    assert report.gap.elapsed_sec == pytest.approx(240.0)  # type: ignore[attr-defined]
    assert report.gap.reason  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Outages
# ---------------------------------------------------------------------------


def test_a_silent_camera_on_the_route__is_reported_as_an_outage() -> None:
    """A camera that logged no traffic at all was probably not working.

    Reporting that as "the vehicle was not seen here" would present a system
    fault as evidence about the vehicle.
    """
    reports = _reports(
        [("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)],
        activity=[make_sighting("cam_01", offset_sec=0.0)],
    )

    assert [report.kind for report in reports] == [GapKind.OUTAGE]  # type: ignore[attr-defined]
    assert reports[0].silent_cameras == ["cam_02"]  # type: ignore[attr-defined]
    assert "outage" in reports[0].describe()  # type: ignore[attr-defined]


def test_a_working_camera_that_saw_other_traffic__is_not_an_outage() -> None:
    """The control. It was watching and the vehicle was genuinely not there."""
    reports = _reports(
        [("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)],
        activity=[make_sighting("cam_02", offset_sec=100.0)],
    )

    assert [report.kind for report in reports] == [GapKind.COVERAGE]  # type: ignore[attr-defined]
    assert reports[0].silent_cameras == []  # type: ignore[attr-defined]


def test_without_an_activity_feed__no_outage_is_claimed() -> None:
    """Silence about silence. Without the cross-reference there is no evidence."""
    reports = _reports([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)])

    assert all(report.kind is not GapKind.OUTAGE for report in reports)  # type: ignore[attr-defined]


def test_find_camera_outages__returns_only_cameras_with_no_activity() -> None:
    """The primitive, tested directly."""
    window = TimeWindow(
        start_utc=BASE_INSTANT, end_utc=BASE_INSTANT.replace(minute=BASE_INSTANT.minute + 5)
    )
    activity = [make_sighting("cam_02", offset_sec=30.0)]

    silent = find_camera_outages(activity, window, ["cam_02", "cam_03", "cam_04"])

    assert silent == ["cam_03", "cam_04"]


def test_find_camera_outages__ignores_activity_outside_the_window() -> None:
    """A camera working an hour later was not working during the gap."""
    window = TimeWindow(
        start_utc=BASE_INSTANT, end_utc=BASE_INSTANT.replace(minute=BASE_INSTANT.minute + 1)
    )
    activity = [make_sighting("cam_02", offset_sec=3600.0)]

    assert find_camera_outages(activity, window, ["cam_02"]) == ["cam_02"]
