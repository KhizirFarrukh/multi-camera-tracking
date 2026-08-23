"""Unit tests for candidate preparation.

The collapsing rule is the one worth reading. A camera sampling at 3 fps sees a
vehicle in a dozen consecutive frames, and left in, those become a dozen nodes
joined by same-camera hops the topology cannot explain.
"""

from __future__ import annotations

import pytest

from multicam_tracker.models import MatchMethod, ReviewStatus
from multicam_tracker.pathing import prepare_candidates
from tests.fixtures.factories import make_sighting
from tests.fixtures.pathing import scored_candidate

pytestmark = pytest.mark.unit

GAP_SEC = 60.0
"""The configured same-pass gap, restated so the boundary cases read clearly."""


def _prepare(entries: list[tuple[str, float, float, ReviewStatus]], **kwargs: object) -> list[str]:
    """Prepare candidates and return the surviving camera sequence.

    Args:
        entries: ``(camera, offset_sec, score, review_status)`` per candidate.
        **kwargs: Passed through to ``prepare_candidates``.

    Returns:
        Camera ids of the survivors, in order.
    """
    sightings = [make_sighting(camera, offset_sec=offset) for camera, offset, _, _ in entries]
    candidates = [
        scored_candidate(sighting, score, review_status=status)
        for sighting, (_, _, score, status) in zip(sightings, entries, strict=True)
    ]
    prepared = prepare_candidates(
        candidates,
        {sighting.sighting_id: sighting for sighting in sightings},
        **kwargs,  # type: ignore[arg-type]
    )
    return [entry.camera_id for entry in prepared]


def test_rejected_candidates__are_always_excluded() -> None:
    """A human said no. No mode admits it."""
    for confirmed_only in (True, False):
        cameras = _prepare(
            [
                ("cam_01", 0.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
                ("cam_02", 120.0, 0.9, ReviewStatus.REJECTED),
            ],
            confirmed_only=confirmed_only,
        )
        assert cameras == ["cam_01"]


def test_pending_review__is_included_by_default_and_excluded_when_asked() -> None:
    """Excluding a pending candidate silently answers the review question as no."""
    entries = [
        ("cam_01", 0.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
        ("cam_02", 120.0, 0.6, ReviewStatus.PENDING_REVIEW),
    ]

    assert _prepare(entries) == ["cam_01", "cam_02"]
    assert _prepare(entries, confirmed_only=True) == ["cam_01"]


def test_confirmed_candidates__are_admitted_in_both_modes() -> None:
    """A human said yes; nothing may drop it."""
    entries = [("cam_01", 0.0, 0.9, ReviewStatus.CONFIRMED)]

    assert _prepare(entries) == ["cam_01"]
    assert _prepare(entries, confirmed_only=True) == ["cam_01"]


def test_near_simultaneous_same_camera_sightings__collapse_to_one() -> None:
    """One pass through a camera is one node, however many frames caught it."""
    cameras = _prepare(
        [
            ("cam_01", 0.0, 0.80, ReviewStatus.AUTO_ACCEPTED),
            ("cam_01", 5.0, 0.95, ReviewStatus.AUTO_ACCEPTED),
            ("cam_01", 10.0, 0.85, ReviewStatus.AUTO_ACCEPTED),
        ]
    )

    assert cameras == ["cam_01"]


def test_collapsing__retains_the_highest_confidence_representative() -> None:
    """The clearest view of the vehicle survives, not the first frame of it."""
    sightings = [make_sighting("cam_01", offset_sec=offset) for offset in (0.0, 5.0, 10.0)]
    candidates = [
        scored_candidate(sighting, score)
        for sighting, score in zip(sightings, (0.80, 0.95, 0.85), strict=True)
    ]

    prepared = prepare_candidates(
        candidates, {sighting.sighting_id: sighting for sighting in sightings}
    )

    assert len(prepared) == 1
    assert prepared[0].confidence == pytest.approx(0.95)
    assert prepared[0].sighting_id == sightings[1].sighting_id


def test_same_camera_beyond_the_redetection_gap__are_both_kept() -> None:
    """A vehicle that circles the block and returns has made two transits."""
    cameras = _prepare(
        [
            ("cam_01", 0.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
            ("cam_01", GAP_SEC + 1.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
        ]
    )

    assert cameras == ["cam_01", "cam_01"]


def test_same_camera_at_exactly_the_redetection_gap__counts_as_a_separate_pass() -> None:
    """Boundary, stated explicitly: the comparison is strict, as stage 04 documents."""
    cameras = _prepare(
        [
            ("cam_01", 0.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
            ("cam_01", GAP_SEC, 0.9, ReviewStatus.AUTO_ACCEPTED),
        ]
    )

    assert cameras == ["cam_01", "cam_01"]


def test_different_cameras__are_never_collapsed_however_close_in_time() -> None:
    """Two cameras seeing a vehicle seconds apart is a transition, not a repeat."""
    cameras = _prepare(
        [
            ("cam_01", 0.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
            ("cam_02", 2.0, 0.9, ReviewStatus.AUTO_ACCEPTED),
        ]
    )

    assert cameras == ["cam_01", "cam_02"]


def test_output__is_strictly_time_ordered_whatever_the_input_order() -> None:
    """Everything downstream assumes index order is time order."""
    sightings = [make_sighting("cam_01", offset_sec=offset) for offset in (300.0, 0.0, 150.0)]
    candidates = [scored_candidate(sighting, 0.9) for sighting in sightings]

    prepared = prepare_candidates(
        candidates, {sighting.sighting_id: sighting for sighting in sightings}
    )

    timestamps = [entry.sighting.timestamp_utc for entry in prepared]
    assert timestamps == sorted(timestamps)


def test_empty_input__returns_an_empty_list_without_raising() -> None:
    """A target with no candidates is an ordinary case, not an error."""
    assert prepare_candidates([], {}) == []


def test_a_candidate_whose_sighting_is_missing__is_skipped_not_fatal() -> None:
    """Retention can purge the row a candidate points at.

    One missing sighting must not abort a reconstruction that is otherwise
    perfectly answerable.
    """
    present = make_sighting("cam_01", offset_sec=0.0)
    absent = make_sighting("cam_02", offset_sec=120.0)

    prepared = prepare_candidates(
        [scored_candidate(present, 0.9), scored_candidate(absent, 0.9)],
        {present.sighting_id: present},
    )

    assert [entry.camera_id for entry in prepared] == ["cam_01"]


def test_embedding_candidates__are_admitted_like_any_other_evidence() -> None:
    """Stage 07's visual-only matches are what recover an unreadable plate."""
    sighting = make_sighting("cam_02", offset_sec=120.0)

    prepared = prepare_candidates(
        [
            scored_candidate(
                sighting,
                0.45,
                method=MatchMethod.EMBEDDING,
                review_status=ReviewStatus.PENDING_REVIEW,
            )
        ],
        {sighting.sighting_id: sighting},
    )

    assert len(prepared) == 1
    assert prepared[0].candidate.match_method is MatchMethod.EMBEDDING
