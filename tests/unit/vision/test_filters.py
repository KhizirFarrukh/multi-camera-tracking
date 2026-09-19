"""Post-detection filtering, and the counters that keep the loss visible.

Every boundary here is asserted at its exact value rather than near it. "At the
threshold" is where a filter is argued about, and a convention that only exists
in a docstring is one that changes by accident.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from multicam_tracker.config import Settings
from multicam_tracker.ingest.preprocess import RegionOfInterest
from multicam_tracker.vision import (
    DetectionFilters,
    EdgePolicy,
    FilterStats,
    RejectionReason,
    RoiMode,
)
from tests.fixtures.vision import make_detection

pytestmark = pytest.mark.unit

FRAME = {"frame_width": 160, "frame_height": 120}
FRAME_AREA = 160 * 120


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


def test_apply__box_below_the_minimum_area__is_rejected_and_counted() -> None:
    """A speck is a distant pedestrian or a compression artifact, not a vehicle.

    The counter matters as much as the rejection: a pipeline silently dropping
    most of its detections looks identical to an empty road.
    """
    filters = DetectionFilters(min_area_px=400)
    stats = FilterStats()

    kept = filters.apply([make_detection((0, 0, 10, 10))], stats=stats, **FRAME)

    assert kept == []
    assert stats.rejected[RejectionReason.TOO_SMALL.value] == 1
    assert stats.rejected_total == 1
    assert stats.seen == 1


def test_apply__box_exactly_at_the_minimum_area__is_kept() -> None:
    """Boundary: the minimum is inclusive, as the module docstring states."""
    filters = DetectionFilters(min_area_px=400)

    kept = filters.apply([make_detection((0, 0, 20, 20))], **FRAME)

    assert len(kept) == 1


def test_apply__box_above_the_maximum_area__is_rejected() -> None:
    """A box covering most of the frame is a failed detection, not a large vehicle."""
    filters = DetectionFilters(max_area_fraction=0.5)
    stats = FilterStats()

    kept = filters.apply([make_detection((0, 0, 160, 120))], stats=stats, **FRAME)

    assert kept == []
    assert stats.rejected[RejectionReason.TOO_LARGE.value] == 1


def test_apply__box_exactly_at_the_maximum_area__is_kept() -> None:
    """Boundary: the maximum is inclusive too, so the window is closed at both ends."""
    filters = DetectionFilters(max_area_fraction=0.5)
    half_area_box = make_detection((0, 0, 160, 60))

    assert half_area_box.area == FRAME_AREA // 2
    assert len(filters.apply([half_area_box], **FRAME)) == 1


# ---------------------------------------------------------------------------
# Aspect ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bbox", "label"),
    [((0, 0, 100, 10), "too wide"), ((0, 0, 10, 100), "too tall")],
)
def test_apply__aspect_ratio_outside_the_window__is_rejected(
    bbox: tuple[int, int, int, int], label: str
) -> None:
    """Both ends of the window reject, and both are counted under one reason."""
    filters = DetectionFilters(min_aspect_ratio=0.25, max_aspect_ratio=5.0)
    stats = FilterStats()

    kept = filters.apply([make_detection(bbox)], stats=stats, **FRAME)

    assert kept == [], label
    assert stats.rejected[RejectionReason.ASPECT_RATIO.value] == 1


@pytest.mark.parametrize("bbox", [(0, 0, 40, 160), (0, 0, 200, 40)])
def test_apply__aspect_ratio_exactly_at_a_bound__is_kept(
    bbox: tuple[int, int, int, int],
) -> None:
    """Boundary: 0.25 and 5.0 are both admitted."""
    filters = DetectionFilters(min_aspect_ratio=0.25, max_aspect_ratio=5.0)

    assert len(filters.apply([make_detection(bbox)], **FRAME)) == 1


# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------


def _left_half_roi() -> RegionOfInterest:
    """Return a region covering the left half of a 160x120 frame.

    Returns:
        The region.
    """
    return RegionOfInterest(polygon=[(0, 0), (80, 0), (80, 120), (0, 120)])


def test_apply__centroid_outside_the_region__is_rejected() -> None:
    """A vehicle on the neighbouring property is not this camera's business."""
    filters = DetectionFilters(roi=_left_half_roi(), roi_mode=RoiMode.CENTROID)
    stats = FilterStats()

    kept = filters.apply([make_detection((100, 40, 140, 70))], stats=stats, **FRAME)

    assert kept == []
    assert stats.rejected[RejectionReason.OUTSIDE_ROI.value] == 1


def test_apply__centroid_just_inside_the_region__is_kept() -> None:
    """Boundary: membership is decided at the centroid, one pixel inside counts."""
    filters = DetectionFilters(roi=_left_half_roi(), roi_mode=RoiMode.CENTROID)
    # Centroid x = 79, one pixel inside the region edge at x = 80.
    detection = make_detection((59, 40, 99, 70))

    assert detection.centroid[0] == pytest.approx(79.0)
    assert len(filters.apply([detection], **FRAME)) == 1


def test_apply__overlap_mode__keeps_a_box_mostly_inside_and_drops_one_mostly_outside() -> None:
    """Overlap mode asks how much of the vehicle is inside, not merely where its centre is."""
    filters = DetectionFilters(roi=_left_half_roi(), roi_mode=RoiMode.OVERLAP, roi_min_overlap=0.5)

    mostly_inside = make_detection((40, 40, 100, 70))
    mostly_outside = make_detection((70, 40, 130, 70))

    assert len(filters.apply([mostly_inside], **FRAME)) == 1
    assert filters.apply([mostly_outside], **FRAME) == []


def test_apply__overlap_mode__a_degenerate_box_is_outside_rather_than_crashing() -> None:
    """A zero-area box has no fraction inside; that is a rejection, not a division."""
    filters = DetectionFilters(roi=_left_half_roi(), roi_mode=RoiMode.OVERLAP)

    assert filters.apply([make_detection((10, 10, 10, 10))], **FRAME) == []


def test_apply__centroid_outside_the_frame_entirely__is_rejected() -> None:
    """A centroid off the image cannot be inside any region drawn on it."""
    filters = DetectionFilters(roi=_left_half_roi(), roi_mode=RoiMode.CENTROID)

    assert filters.apply([make_detection((300, 300, 340, 330))], **FRAME) == []


# ---------------------------------------------------------------------------
# Edge policy -- both modes, because the trade has no universally right answer
# ---------------------------------------------------------------------------


def test_apply__edge_policy_drop__discards_an_edge_touching_box() -> None:
    """Drop is for deployments where an unreadable plate is worse than a missing row."""
    filters = DetectionFilters(edge_policy=EdgePolicy.DROP, edge_margin_px=2)
    stats = FilterStats()

    kept = filters.apply([make_detection((0, 40, 40, 70))], stats=stats, **FRAME)

    assert kept == []
    assert stats.rejected[RejectionReason.EDGE_TOUCHING.value] == 1


def test_apply__edge_policy_flag__keeps_it_and_marks_it() -> None:
    """Flag preserves the record and lets the best-frame ranking push it down."""
    filters = DetectionFilters(edge_policy=EdgePolicy.FLAG, edge_margin_px=2)
    stats = FilterStats()

    kept = filters.apply([make_detection((0, 40, 40, 70))], stats=stats, **FRAME)

    assert len(kept) == 1
    assert kept[0].touches_edge is True
    assert stats.flagged_edge == 1
    assert stats.rejected_total == 0


def test_apply__edge_policy_keep__neither_drops_nor_marks() -> None:
    """Keep is the do-nothing policy, and must genuinely do nothing."""
    filters = DetectionFilters(edge_policy=EdgePolicy.KEEP, edge_margin_px=2)

    kept = filters.apply([make_detection((0, 40, 40, 70))], **FRAME)

    assert kept[0].touches_edge is False


def test_apply__edge_policy_flag__leaves_a_central_box_unmarked() -> None:
    """The flag must mean something; a central box that carried it would be noise."""
    filters = DetectionFilters(edge_policy=EdgePolicy.FLAG, edge_margin_px=2)

    kept = filters.apply([make_detection((60, 40, 100, 70))], **FRAME)

    assert kept[0].touches_edge is False


def test_apply__edge_margin_boundary__is_inclusive_of_the_margin() -> None:
    """Boundary: a box starting exactly at the margin counts as touching.

    Stated once and asserted here, because a detector rarely puts a box exactly
    on the edge even when the vehicle is cut off by it -- which is the whole
    reason the margin exists.
    """
    at_margin = DetectionFilters(edge_policy=EdgePolicy.FLAG, edge_margin_px=2)
    inside_margin = DetectionFilters(edge_policy=EdgePolicy.FLAG, edge_margin_px=2)

    assert at_margin.apply([make_detection((2, 40, 42, 70))], **FRAME)[0].touches_edge is True
    assert inside_margin.apply([make_detection((3, 40, 43, 70))], **FRAME)[0].touches_edge is False


# ---------------------------------------------------------------------------
# Composition, pass-through, and the counters
# ---------------------------------------------------------------------------


def test_apply__all_filters_disabled__passes_everything_through_unchanged() -> None:
    """The default is a pass-through, which is what a test of something else wants."""
    filters = DetectionFilters()
    detections = [
        make_detection((0, 0, 2, 2)),
        make_detection((0, 0, 160, 120)),
        make_detection((0, 0, 200, 4)),
    ]

    assert filters.is_pass_through is True
    assert filters.apply(detections, **FRAME) == detections


def test_apply__a_detection_failing_several_criteria__is_counted_once() -> None:
    """Counting a box under three reasons would make the rejection ratio exceed one."""
    filters = DetectionFilters(min_area_px=400, min_aspect_ratio=0.5, max_aspect_ratio=2.0)
    stats = FilterStats()

    filters.apply([make_detection((0, 0, 2, 20))], stats=stats, **FRAME)

    assert stats.rejected_total == 1
    assert stats.rejection_ratio == pytest.approx(1.0)


def test_apply__no_stats_supplied__still_filters() -> None:
    """Counters are for observability; their absence must not change behaviour."""
    filters = DetectionFilters(min_area_px=400)

    assert filters.apply([make_detection((0, 0, 10, 10))], **FRAME) == []


def test_apply__empty_input__reports_nothing_and_counts_nothing() -> None:
    """Boundary: a frame with no detections is the common case at night."""
    stats = FilterStats()

    assert DetectionFilters(min_area_px=400).apply([], stats=stats, **FRAME) == []
    assert stats.seen == 0
    assert stats.rejection_ratio == pytest.approx(0.0)


def test_filter_stats__as_dict__omits_reasons_that_never_fired() -> None:
    """A new key appearing in the metrics means something new actually happened."""
    filters = DetectionFilters(min_area_px=400)
    stats = FilterStats()

    filters.apply(
        [make_detection((0, 0, 10, 10)), make_detection((0, 0, 40, 30))],
        stats=stats,
        **FRAME,
    )
    flat = stats.as_dict()

    assert flat["seen"] == 2
    assert flat["accepted"] == 1
    assert flat[f"rejected_{RejectionReason.TOO_SMALL.value}"] == 1
    assert f"rejected_{RejectionReason.TOO_LARGE.value}" not in flat


def test_filter_stats__accumulates_across_frames() -> None:
    """Ratios are meaningless per frame; the same instance spans the run."""
    filters = DetectionFilters(min_area_px=400)
    stats = FilterStats()

    for _ in range(4):
        filters.apply([make_detection((0, 0, 10, 10))], stats=stats, **FRAME)

    assert stats.seen == 4
    assert stats.rejected_total == 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_from_settings__builds_every_configured_filter(
    build_settings: Callable[..., Settings],
) -> None:
    """Production gets all of them from config; nothing is a magic number in logic."""
    settings = build_settings()
    filters = DetectionFilters.from_settings(settings, roi=_left_half_roi())

    assert filters.min_area_px == settings.thresholds.detection_min_bbox_area_px
    assert filters.max_area_fraction == pytest.approx(
        settings.thresholds.detection_max_bbox_area_fraction
    )
    assert filters.min_aspect_ratio == pytest.approx(settings.thresholds.detection_min_aspect_ratio)
    assert filters.edge_policy is EdgePolicy(settings.detection.edge_policy)
    assert filters.roi is not None
    assert filters.is_pass_through is False
