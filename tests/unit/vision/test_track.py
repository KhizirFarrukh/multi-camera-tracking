"""The track object: its invariants and the aggregates built on it.

Two of these matter more than the rest. Ascending frame order is what makes the
midpoint rule select a defensible frame rather than an arbitrary one, and the
confidence aggregate is the number every downstream weighting multiplies by.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from multicam_tracker.exceptions import VisionError
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import TrackObservation, TrackStatus, VehicleTrack
from tests.fixtures.vision import FIXTURE_START, make_detection, make_observation, make_track

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_vehicle_track__with_no_observations__is_rejected() -> None:
    """An empty track is not a vehicle that was seen zero times; it is a bug."""
    with pytest.raises(VisionError, match="at least one observation"):
        VehicleTrack(
            track_id="cam_01-00000",
            camera_id="cam_01",
            source_id="cam_01_test",
            observations=(),
        )


def test_vehicle_track__observations_out_of_order__are_rejected() -> None:
    """Out-of-order history makes the midpoint rule select an arbitrary frame.

    That is the kind of defect that produces a plausible timestamp nobody can
    trace back to a cause.
    """
    with pytest.raises(VisionError, match="ascend by frame index"):
        VehicleTrack(
            track_id="cam_01-00000",
            camera_id="cam_01",
            source_id="cam_01_test",
            observations=(make_observation(5), make_observation(2)),
        )


def test_vehicle_track__a_single_observation__is_valid() -> None:
    """Boundary: one frame is a short track, not an invalid one."""
    track = make_track([make_observation(7)])

    assert track.first_frame_index == track.last_frame_index == 7
    assert track.duration_sec == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bounds and derived values
# ---------------------------------------------------------------------------


def test_vehicle_track__bounds__come_from_the_first_and_last_observations() -> None:
    """These are what stage 12 seeks back to and what the sighting is dated from."""
    track = make_track([make_observation(index) for index in (4, 5, 9)])

    assert track.first_frame_index == 4
    assert track.last_frame_index == 9
    assert track.hit_count == 3
    assert track.duration_sec == pytest.approx(0.5)
    assert track.first_timestamp_utc == FIXTURE_START + timedelta(seconds=0.4)


def test_aggregate_confidence__is_the_median_not_the_mean_or_the_maximum() -> None:
    """A track begins and ends with the vehicle half out of frame.

    Those frames drag a mean down and say nothing about how sure the system is
    that a vehicle passed; a maximum has the opposite problem, promoting a track
    of marginal detections on one lucky frame.
    """
    confidences = [0.10, 0.80, 0.85, 0.90, 0.99]
    track = make_track(
        [make_observation(index, confidence=value) for index, value in enumerate(confidences)]
    )

    assert track.aggregate_confidence == pytest.approx(0.85)
    assert track.aggregate_confidence != pytest.approx(sum(confidences) / len(confidences))
    assert track.aggregate_confidence != pytest.approx(max(confidences))


def test_object_class__is_decided_by_confidence_weighted_vote() -> None:
    """A plain majority discards what a confident minority is telling you.

    Six uncertain "truck" readings against five confident "car" ones is a car.
    """
    observations = [make_observation(index, confidence=0.30) for index in range(6)] + [
        make_observation(6 + index, confidence=0.95) for index in range(5)
    ]
    observations = [
        TrackObservation(
            frame_index=entry.frame_index,
            raw_timestamp=entry.raw_timestamp,
            timestamp_utc=entry.timestamp_utc,
            detection=make_detection(
                entry.detection.bbox,
                confidence=entry.detection.confidence,
                object_class=ObjectClass.TRUCK if index < 6 else ObjectClass.CAR,
            ),
        )
        for index, entry in enumerate(observations)
    ]

    assert make_track(observations).object_class is ObjectClass.CAR


def test_object_class__a_unanimous_track__reports_that_class() -> None:
    """The ordinary case has to work before the tie-breaking does."""
    assert make_track().object_class is ObjectClass.CAR


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def test_midpoint_observation__picks_the_frame_nearest_the_temporal_middle() -> None:
    """The documented selection rule, asserted exactly."""
    track = make_track([make_observation(index) for index in range(11)])

    assert track.midpoint_observation.frame_index == 5


def test_midpoint_observation__with_an_even_number_of_frames__breaks_toward_the_earlier() -> None:
    """Ties resolve on something stated, not on iteration order."""
    track = make_track([make_observation(index) for index in range(4)])

    assert track.midpoint_observation.frame_index == 1


def test_observation_nearest__an_instant_outside_the_track__clamps_to_an_endpoint() -> None:
    """Asking for a moment before the track began yields its first frame."""
    track = make_track([make_observation(index) for index in range(5)])

    assert track.observation_nearest(FIXTURE_START - timedelta(hours=1)).frame_index == 0
    assert track.observation_nearest(FIXTURE_START + timedelta(hours=1)).frame_index == 4


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------


def test_best__with_no_retained_imagery__is_none_rather_than_a_fallback() -> None:
    """An honest absence. A caller wanting imagery has to handle it either way."""
    assert make_track().best is None


def test_best__with_retained_imagery__is_the_highest_ranked_frame() -> None:
    """What the thumbnail and stage 12's OCR are taken from."""
    crop = np.full((20, 20, 3), 120, dtype=np.uint8)
    track = make_track(
        [
            make_observation(0, confidence=0.20, crop=crop, sharpness=5.0),
            make_observation(1, confidence=0.99, crop=crop, sharpness=900.0),
        ]
    )

    assert track.best is not None
    assert track.best.observation.frame_index == 1


def test_retained_image_count__counts_only_observations_holding_an_image() -> None:
    """The invariant the tracker asserts against its configured cap."""
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    track = make_track(
        [make_observation(0, crop=crop), make_observation(1), make_observation(2, crop=crop)]
    )

    assert track.retained_image_count() == 2


def test_without_image__releases_the_pixels_and_keeps_the_numbers() -> None:
    """A track's history must stay complete when an image falls out of the cap."""
    observation = make_observation(3, crop=np.zeros((8, 8, 3), dtype=np.uint8), sharpness=42.0)

    released = observation.without_image()

    assert released.has_image is False
    assert released.sharpness == pytest.approx(42.0)
    assert released.frame_index == 3
    assert released.detection == observation.detection


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_repr__does_not_print_any_imagery() -> None:
    """A track holds arrays; a repr that rendered them would be unusable in a log."""
    text = repr(make_track([make_observation(0, crop=np.zeros((4, 4, 3), dtype=np.uint8))]))

    assert "cam_01-00000" in text
    assert "array" not in text


def test_track_status__has_a_tentative_state_distinct_from_confirmed() -> None:
    """Tentative is what stops one false positive becoming a sighting."""
    assert TrackStatus.TENTATIVE.value == "tentative"
    assert TrackStatus.TENTATIVE is not TrackStatus.CONFIRMED
