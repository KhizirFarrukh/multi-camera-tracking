"""Track to sighting: the seam where a claim becomes a record.

Everything asserted here will be read back as fact months later, so each rule is
tested exactly rather than approximately -- particularly the timestamp, which
biases the whole topology if it is systematically early.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.exceptions import VisionError
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.models.sighting import Sighting
from multicam_tracker.vision import (
    ThumbnailWriter,
    VehicleTrack,
    track_to_sighting,
    track_to_sightings,
)
from tests.fixtures.vision import FIXTURE_START, make_observation, make_track

pytestmark = pytest.mark.unit

CREATED_AT = datetime(2026, 8, 10, 15, 0, 0, tzinfo=UTC)
CLOCK = FixedClock(CREATED_AT)


def _has_opencv() -> bool:
    """Return whether OpenCV is importable.

    Returns:
        ``True`` when it imports.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


requires_opencv = pytest.mark.skipif(not _has_opencv(), reason="OpenCV is not installed")


def sequential_ids() -> Callable[[], str]:
    """Return an id factory producing predictable identifiers.

    Deterministic ids make the thumbnail path assertable, which a UUID would
    not be.

    Returns:
        A callable yielding ``00000000-...-0000`` style ids.
    """
    counter: Iterator[int] = itertools.count()
    return lambda: f"{next(counter):08d}-0000-0000-0000-000000000000"


# ---------------------------------------------------------------------------
# One track, one sighting
# ---------------------------------------------------------------------------


def test_track_to_sighting__produces_exactly_one_sighting() -> None:
    """The stage exit criterion: one vehicle pass is one row, not forty."""
    track = make_track([make_observation(index) for index in range(40)])

    sighting = track_to_sighting(track, clock=CLOCK)

    assert isinstance(sighting, Sighting)
    assert sighting.camera_id == "cam_01"
    assert sighting.source_id == "cam_01_test"


def test_track_to_sighting__timestamp_follows_the_midpoint_rule_exactly() -> None:
    """Dating from the first frame would place every vehicle systematically early.

    By an amount that varies with the camera's field of view, which corrupts
    travel-time plausibility across the whole topology while passing every
    local test.
    """
    track = make_track([make_observation(index) for index in range(11)])

    sighting = track_to_sighting(track, clock=CLOCK)

    assert sighting.frame_index == 5
    assert sighting.timestamp_utc == FIXTURE_START + timedelta(seconds=0.5)


def test_track_to_sighting__box_comes_from_the_same_frame_as_the_timestamp() -> None:
    """A timestamp and a box that describe different instants describe nothing."""
    observations = [
        make_observation(index, bbox=(10 + index * 5, 40, 50 + index * 5, 70)) for index in range(5)
    ]
    track = make_track(observations)

    sighting = track_to_sighting(track, clock=CLOCK)

    assert sighting.frame_index == 2
    assert sighting.bbox == list(observations[2].detection.bbox)


def test_track_to_sighting__confidence_is_the_track_median() -> None:
    """The documented aggregation, asserted against a hand-computed value."""
    track = make_track(
        [
            make_observation(index, confidence=value)
            for index, value in enumerate([0.10, 0.80, 0.85, 0.90, 0.99])
        ]
    )

    assert track_to_sighting(track, clock=CLOCK).detection_confidence == pytest.approx(0.85)


def test_track_to_sighting__carries_the_clock_offset_the_frames_were_corrected_by() -> None:
    """Stage 09 recomputes corrections from the raw value, so both must be stored."""
    track = make_track(
        [make_observation(index, clock_offset_ms=1500) for index in range(5)],
    )

    sighting = track_to_sighting(track, clock=CLOCK)

    assert sighting.clock_offset_applied_ms == 1500
    assert sighting.timestamp_utc == sighting.raw_timestamp + timedelta(milliseconds=1500)


def test_track_to_sighting__created_at_comes_from_the_injected_clock() -> None:
    """Nothing in this project reads the wall clock on its own."""
    assert track_to_sighting(make_track(), clock=CLOCK).created_at == CREATED_AT


def test_track_to_sighting__passes_the_stage_two_model_validators() -> None:
    """The model is the contract; producing something it rejects is a stage 11 bug."""
    sighting = track_to_sighting(make_track(), clock=CLOCK)

    assert sighting.object_class is ObjectClass.CAR
    assert len(sighting.bbox) == 4
    assert sighting.bbox[2] > sighting.bbox[0]
    assert sighting.has_plate is False
    assert sighting.has_embedding is False


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_track_to_sighting__a_box_past_the_frame_edge__is_clamped_into_bounds() -> None:
    """Every emitted bbox lies within the original frame bounds after clamping."""
    track = make_track(
        [make_observation(index, bbox=(140, 100, 220, 180)) for index in range(3)],
        frame_size=(160, 120),
    )

    assert track_to_sighting(track, clock=CLOCK).bbox == [140, 100, 160, 120]


def test_track_to_sighting__a_box_entirely_outside_its_frame__raises() -> None:
    """A box outside its own frame is a coordinate-mapping fault, not a small box.

    Inventing a one-pixel box to keep going would hide exactly the failure
    stage 10 exists to prevent.
    """
    track = make_track(
        [make_observation(index, bbox=(400, 400, 440, 430)) for index in range(3)],
        frame_size=(160, 120),
    )

    with pytest.raises(VisionError, match="coordinate-mapping fault"):
        track_to_sighting(track, clock=CLOCK)


def test_track_to_sighting__an_unknown_frame_size__skips_clamping_rather_than_guessing() -> None:
    """A track assembled without frames has no bounds to clamp against."""
    track = VehicleTrack(
        track_id="cam_01-00000",
        camera_id="cam_01",
        source_id="cam_01_test",
        observations=(make_observation(0, bbox=(300, 300, 340, 330)),),
        frame_size=(0, 0),
    )

    assert track_to_sighting(track, clock=CLOCK).bbox == [300, 300, 340, 330]


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def test_track_to_sighting__without_a_writer__emits_no_thumbnail_path() -> None:
    """Storage is optional; claiming a path for a file nobody wrote would not be."""
    assert track_to_sighting(make_track(), clock=CLOCK).thumbnail_path is None


@requires_opencv
def test_track_to_sighting__with_a_writer__writes_the_best_frame_and_records_it(
    tmp_path: Path,
) -> None:
    """The image comes from the clearest frame; the timestamp from the midpoint one.

    Those are usually the same frame and deliberately need not be, which is why
    the sighting records the midpoint frame's index rather than the thumbnail's.
    """
    crop = np.full((24, 24, 3), 90, dtype=np.uint8)
    track = make_track(
        [
            make_observation(0, confidence=0.20, crop=crop, sharpness=1.0),
            make_observation(1, confidence=0.30, crop=crop, sharpness=2.0),
            make_observation(2, confidence=0.99, crop=crop, sharpness=900.0),
        ]
    )
    writer = ThumbnailWriter(root=tmp_path, width=32, height=32)

    sighting = track_to_sighting(
        track, clock=CLOCK, thumbnail_writer=writer, id_factory=sequential_ids()
    )

    assert track.best is not None
    assert track.best.observation.frame_index == 2
    assert sighting.frame_index == 1
    assert sighting.thumbnail_path is not None
    assert (tmp_path / sighting.thumbnail_path).is_file()


@requires_opencv
def test_track_to_sighting__a_track_with_no_retained_image__emits_no_path(
    tmp_path: Path,
) -> None:
    """Arithmetic-only tracking has no imagery, and must say so rather than invent one."""
    sighting = track_to_sighting(
        make_track(), clock=CLOCK, thumbnail_writer=ThumbnailWriter(root=tmp_path)
    )

    assert sighting.thumbnail_path is None


# ---------------------------------------------------------------------------
# Long tracks
# ---------------------------------------------------------------------------


def test_track_to_sightings__a_short_track__still_produces_exactly_one() -> None:
    """Splitting must not manufacture several sightings of one ordinary transit."""
    track = make_track([make_observation(index) for index in range(10)])

    assert len(track_to_sightings(track, clock=CLOCK, split_interval_sec=60.0)) == 1


def test_track_to_sightings__splitting_disabled__produces_one_however_long() -> None:
    """Off by default: a vehicle merely driving past is one sighting."""
    track = make_track([make_observation(index) for index in range(600)])

    assert len(track_to_sightings(track, clock=CLOCK)) == 1


def test_track_to_sightings__a_long_track__is_split_at_the_configured_interval() -> None:
    """A vehicle parked in view for a minute is not one instantaneous sighting.

    At 10 fps, 300 frames is 29.9 seconds, so a ten-second interval yields three
    windows.
    """
    track = make_track([make_observation(index) for index in range(300)])

    sightings = track_to_sightings(track, clock=CLOCK, split_interval_sec=10.0)

    assert len(sightings) == 3
    assert [entry.frame_index for entry in sightings] == sorted(
        entry.frame_index for entry in sightings
    )


def test_track_to_sightings__each_split_sighting__is_dated_within_its_own_window() -> None:
    """A split that reported the same instant three times would be worse than not splitting."""
    track = make_track([make_observation(index) for index in range(300)])

    sightings = track_to_sightings(track, clock=CLOCK, split_interval_sec=10.0)
    stamps = [entry.timestamp_utc for entry in sightings]

    assert len(set(stamps)) == 3
    assert all(
        (later - earlier).total_seconds() >= 5.0 for earlier, later in itertools.pairwise(stamps)
    )


def test_track_to_sightings__split_sightings__have_distinct_identifiers() -> None:
    """Rows that cannot be told apart cannot be audited."""
    track = make_track([make_observation(index) for index in range(300)])

    sightings = track_to_sightings(
        track, clock=CLOCK, split_interval_sec=10.0, id_factory=sequential_ids()
    )

    assert len({entry.sighting_id for entry in sightings}) == len(sightings)
