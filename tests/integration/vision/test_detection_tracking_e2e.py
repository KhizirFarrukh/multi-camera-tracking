"""The whole chain, against real decoded media: clip in, sightings out.

The unit tests prove each piece. This proves they fit together over a file that
a real decoder actually opened -- which is the part a fake frame cannot vouch
for, because the seam where a box leaves processed-frame space is exactly where
the unit tests stop.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path

import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.ingest import FileVideoSource, Preprocessor, RegionOfInterest
from multicam_tracker.models.sighting import Sighting
from multicam_tracker.vision import (
    DetectionFilters,
    Detector,
    FilterStats,
    FixtureDetector,
    SingleCameraTracker,
    ThumbnailWriter,
    TrackerConfig,
    VehicleTrack,
    track_to_sighting,
)
from tests.fixtures.vision import ReferenceBlobDetector

pytestmark = pytest.mark.integration

MEDIA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "media"
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "detections" / "sample_clips.json"
CLIP = MEDIA_DIR / "sample_clean.mp4"

CLOCK = FixedClock(datetime(2026, 8, 10, 15, 0, 0, tzinfo=UTC))

EXPECTED_PASSES = 2
"""How many passes the clip actually contains, which is not the obvious answer.

The bar is drawn at ``(index * 3) % 52``, so on frame 18 it wraps from the right
edge back to the left. That is a teleport, and the tracker is right to call it a
second vehicle rather than stitching it into the first: a real object cannot
cross the frame in one frame interval, and a tracker that accepted it would be
the one that merges two different vehicles into one trajectory.

Discovered by running the clip rather than by reading the generator, which is
the argument for this test existing at all."""


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


def process_clip(
    detector: Detector,
    *,
    preprocessor: Preprocessor | None = None,
    config: TrackerConfig | None = None,
) -> tuple[list[VehicleTrack], FilterStats]:
    """Run a clip through the full detection and tracking chain.

    The chain, in the order that keeps coordinates honest: decode, preprocess,
    detect on the processed image, map every box back to source coordinates,
    filter, track.

    Args:
        detector: The detector to run.
        preprocessor: Frame preparation, if any. The mapping it returns is what
            puts boxes back into source-frame space.
        config: Tracker parameters.

    Returns:
        ``(tracks, filter_stats)``.
    """
    tracker_config = config or TrackerConfig(min_hits=2, min_iou=0.2, max_age_frames=3)
    stats = FilterStats()
    filters = DetectionFilters(min_area_px=1)

    with FileVideoSource(path=CLIP, camera_id="cam_01") as source:
        tracker = SingleCameraTracker("cam_01", source.source_id, tracker_config)
        tracks: list[VehicleTrack] = []

        for frame in source.frames():
            if preprocessor is None:
                detections = detector.detect(frame)
            else:
                processed, mapping = preprocessor.apply_to_frame(frame)
                detections = [
                    detection.in_source_coordinates(mapping)
                    for detection in detector.detect(processed)
                ]

            kept = filters.apply(
                detections,
                frame_width=frame.width,
                frame_height=frame.height,
                stats=stats,
            )
            tracks.extend(tracker.update(frame, kept))

        tracks.extend(tracker.flush())

    return tracks, stats


def summarise(tracks: list[VehicleTrack]) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """Return a comparable summary of a set of tracks.

    Compares what a change would actually break -- bounds, length, geometry --
    rather than object identity, which no two runs share.

    Args:
        tracks: The tracks to summarise.

    Returns:
        One tuple per track: first frame, last frame, hit count, midpoint box.
    """
    return [
        (
            track.first_frame_index,
            track.last_frame_index,
            track.hit_count,
            track.midpoint_observation.detection.bbox,
        )
        for track in sorted(tracks, key=lambda entry: entry.first_frame_index)
    ]


# ---------------------------------------------------------------------------
# One pass, one track
# ---------------------------------------------------------------------------


@requires_opencv
def test_a_clip_with_known_passes__produces_one_track_per_pass() -> None:
    """The stage exit criterion, over a file a real decoder opened.

    Eighteen frames of one bar crossing the frame produce **one** track, not
    eighteen sightings of eighteen vehicles. The wrap on frame 18 produces a
    second, which is the right answer: see EXPECTED_PASSES.
    """
    tracks, _ = process_clip(ReferenceBlobDetector())
    ordered = sorted(tracks, key=lambda track: track.first_frame_index)

    assert len(ordered) == EXPECTED_PASSES
    assert ordered[0].first_frame_index == 0
    assert ordered[0].hit_count >= 15
    assert ordered[1].first_frame_index > ordered[0].last_frame_index


@requires_opencv
def test_the_same_clip_processed_twice__produces_identical_results() -> None:
    """Determinism is what makes every other assertion in this file meaningful."""
    first, _ = process_clip(ReferenceBlobDetector())
    second, _ = process_clip(ReferenceBlobDetector())

    assert summarise(first) == summarise(second)
    assert [track.track_id for track in first] == [track.track_id for track in second]


# ---------------------------------------------------------------------------
# Fixture replay
# ---------------------------------------------------------------------------


@requires_opencv
def test_fixture_replay__reproduces_the_recorded_run_exactly() -> None:
    """What makes the committed recording worth trusting.

    If the replay drifted from the run it was recorded from, every downstream
    stage tested against it would be tested against fiction.
    """
    live, _ = process_clip(ReferenceBlobDetector())
    replayed, _ = process_clip(FixtureDetector.from_file(FIXTURE_PATH))

    assert summarise(live) == summarise(replayed)


def test_the_committed_fixture__declares_how_it_was_recorded() -> None:
    """A recording that does not say what made it is one nobody can date.

    The committed file was recorded with the reference blob detector rather than
    YOLO, because no weights are available here. That is a deviation, and it has
    to be legible from the artefact itself rather than only from a document.
    """
    detector = FixtureDetector.from_file(FIXTURE_PATH)

    assert detector.model_id == "reference-blob"
    assert "NOT with YOLO" in detector.note
    assert detector.recorded_at != ""


def test_the_committed_fixture__covers_the_clips_it_claims_to() -> None:
    """A source missing from the recording fails loudly; this is the positive case."""
    assert FixtureDetector.from_file(FIXTURE_PATH).covered_sources == (
        "sample_clean.mp4",
        "sample_vfr.mp4",
    )


# ---------------------------------------------------------------------------
# Coordinates through the full preprocessing chain
# ---------------------------------------------------------------------------


@requires_opencv
def test_boxes_survive_preprocessing__and_land_in_source_frame_coordinates() -> None:
    """The exit criterion that stage 10 exists to make possible.

    The detector sees a rotated, resized image and reports boxes in that space.
    Every box stored has to be in the coordinates of the frame as captured, or
    it is a plausible box in the wrong part of the picture and nothing
    downstream can tell.
    """
    preprocessor = Preprocessor(rotation_degrees=90, target_width=48, target_height=64)

    tracks, _ = process_clip(
        ReferenceBlobDetector(),
        preprocessor=preprocessor,
        config=TrackerConfig(min_hits=2, min_iou=0.1, max_age_frames=4),
    )

    assert tracks, "the chain produced no tracks at all"
    for track in tracks:
        for observation in track.observations:
            x1, y1, x2, y2 = observation.detection.bbox
            assert 0 <= x1 < x2 <= 64, "box outside the source frame width"
            assert 0 <= y1 < y2 <= 48, "box outside the source frame height"


@requires_opencv
def test_a_region_of_interest__excludes_what_falls_outside_it() -> None:
    """A camera overlooking a neighbouring property must be able to ignore it."""
    right_half = RegionOfInterest(polygon=[(32, 0), (64, 0), (64, 48), (32, 48)])
    stats = FilterStats()
    filters = DetectionFilters(roi=right_half, min_area_px=1)
    detector = ReferenceBlobDetector()

    with FileVideoSource(path=CLIP, camera_id="cam_01") as source:
        for frame in itertools.islice(source.frames(), 4):
            filters.apply(
                detector.detect(frame),
                frame_width=frame.width,
                frame_height=frame.height,
                stats=stats,
            )

    # The bar starts at the left edge, so the early frames are outside the
    # right-hand region and must be rejected rather than silently kept.
    assert stats.rejected_total > 0


# ---------------------------------------------------------------------------
# Sightings and thumbnails
# ---------------------------------------------------------------------------


@requires_opencv
def test_every_emitted_track__produces_one_readable_thumbnail(tmp_path: Path) -> None:
    """A sighting that points at a file nobody can open is a broken record."""
    import cv2

    tracks, _ = process_clip(ReferenceBlobDetector())
    writer = ThumbnailWriter(root=tmp_path, width=32, height=32)

    sightings = [track_to_sighting(track, clock=CLOCK, thumbnail_writer=writer) for track in tracks]

    assert len(sightings) == len(tracks)
    for sighting in sightings:
        assert isinstance(sighting, Sighting)
        assert sighting.thumbnail_path is not None
        written = tmp_path / sighting.thumbnail_path
        assert written.is_file()
        assert cv2.imread(str(written)) is not None


@requires_opencv
def test_emitted_sightings__carry_the_clips_own_timestamps(tmp_path: Path) -> None:
    """The sighting must be dated from the frame, not from when it was processed."""
    tracks, _ = process_clip(ReferenceBlobDetector())
    sighting = track_to_sighting(tracks[0], clock=CLOCK)

    midpoint = tracks[0].midpoint_observation

    # Compared at millisecond resolution because that is what a sighting stores.
    # The domain model truncates on validation so a round trip through storage
    # is lossless (stage 02); a decoder reports microseconds. The two agree to
    # the precision the record actually keeps, and asserting equality at
    # microsecond resolution would be asserting something the schema does not
    # promise.
    assert sighting.timestamp_utc == midpoint.timestamp_utc.replace(
        microsecond=(midpoint.timestamp_utc.microsecond // 1000) * 1000
    )
    assert sighting.frame_index == midpoint.frame_index
    assert sighting.created_at == CLOCK.now_utc()
    assert sighting.timestamp_utc != sighting.created_at
