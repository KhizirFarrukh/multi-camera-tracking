"""Association, lifecycle, and the assignment solver underneath them.

Driven entirely by the scripted fake detector, because that is the only way the
right answer is known exactly. Tested against a real model, these would be
measuring the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from multicam_tracker.config import Settings
from multicam_tracker.vision import (
    Detection,
    FakeDetector,
    KalmanBoxTracker,
    SingleCameraTracker,
    TrackerConfig,
    VehicleTrack,
    solve_min_cost_assignment,
)
from tests.fixtures.vision import blank_frame, linear_script, make_detection

pytestmark = pytest.mark.unit

ARITHMETIC_ONLY = TrackerConfig(retain_crops=False, min_hits=3, max_age_frames=5, min_iou=0.3)
"""Crops cost time and say nothing about association, which is what these test."""


def run_script(
    script: dict[int, list[Detection]],
    *,
    frames: int,
    config: TrackerConfig = ARITHMETIC_ONLY,
    camera_id: str = "cam_01",
) -> tuple[list[VehicleTrack], SingleCameraTracker]:
    """Drive a tracker over a scripted sequence and collect everything it emits.

    Args:
        script: Detections per frame index.
        frames: How many frames to feed, including ones the script omits.
        config: Tracker parameters.
        camera_id: Which camera the frames claim to be from.

    Returns:
        ``(emitted_tracks, tracker)``. The tracker is returned so a test can
        assert on its counters.
    """
    detector = FakeDetector(script)
    tracker = SingleCameraTracker(camera_id, f"{camera_id}_test", config)

    emitted: list[VehicleTrack] = []
    for index in range(frames):
        frame = blank_frame(index, camera_id=camera_id, source_id=f"{camera_id}_test")
        emitted.extend(tracker.update(frame, detector.detect(frame)))
    emitted.extend(tracker.flush())
    return emitted, tracker


def centroids_x(track: VehicleTrack) -> list[float]:
    """Return the horizontal centre of every detection in a track.

    Args:
        track: The track to read.

    Returns:
        One centre per observation, in frame order.
    """
    return [observation.detection.centroid[0] for observation in track.observations]


# ---------------------------------------------------------------------------
# The assignment solver
# ---------------------------------------------------------------------------


def test_solve_min_cost_assignment__square_matrix__finds_the_optimum() -> None:
    """A worked three-by-three, so an off-by-one in the potentials cannot pass."""
    cost = [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]

    pairs = solve_min_cost_assignment(cost)

    assert sorted(pairs) == [(0, 1), (1, 0), (2, 2)]
    assert sum(cost[row][column] for row, column in pairs) == pytest.approx(5.0)


def test_solve_min_cost_assignment__where_greedy_is_wrong__is_still_optimal() -> None:
    """The reason this is an assignment problem and not a sorted loop.

    Greedy takes the single cheapest pair, ``(0, 0)`` at 1.0, and is then forced
    into ``(1, 1)`` at 100.0 for a total of 101. The optimum pairs across for a
    total of 4. In the tracker that difference is an identity swap between two
    vehicles that were briefly close together.
    """
    cost = [
        [1.0, 2.0],
        [2.0, 100.0],
    ]

    pairs = solve_min_cost_assignment(cost)

    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert sum(cost[row][column] for row, column in pairs) == pytest.approx(4.0)


def test_solve_min_cost_assignment__more_columns_than_rows__assigns_every_row() -> None:
    """More detections than tracks: every track gets one, the rest start tracks."""
    pairs = solve_min_cost_assignment([[5.0, 1.0, 9.0]])

    assert pairs == [(0, 1)]


def test_solve_min_cost_assignment__more_rows_than_columns__assigns_every_column() -> None:
    """More tracks than detections: the transposed path must give the same answer."""
    pairs = solve_min_cost_assignment([[5.0], [1.0], [9.0]])

    assert pairs == [(1, 0)]


@pytest.mark.parametrize("matrix", [[], [[]]])
def test_solve_min_cost_assignment__empty_input__returns_nothing(
    matrix: Sequence[Sequence[float]],
) -> None:
    """Boundary: no tracks, or no detections, is the ordinary state of a quiet road."""
    assert solve_min_cost_assignment(matrix) == []


# ---------------------------------------------------------------------------
# The motion model
# ---------------------------------------------------------------------------


def test_kalman__repeated_updates_at_one_place__predicts_that_place() -> None:
    """A stationary vehicle must not be predicted to drift off on its own."""
    kalman = KalmanBoxTracker((50, 50, 90, 80))
    for _ in range(6):
        kalman.predict()
        kalman.update((50, 50, 90, 80))

    predicted = kalman.predict()

    assert predicted[0] == pytest.approx(50, abs=2)
    assert predicted[2] == pytest.approx(90, abs=2)


def test_kalman__constant_velocity__predicts_ahead_of_the_last_observation() -> None:
    """Carrying velocity is what keeps two crossing vehicles apart in the matching."""
    kalman = KalmanBoxTracker((10, 50, 50, 80))
    for step in range(1, 7):
        kalman.predict()
        kalman.update((10 + 10 * step, 50, 50 + 10 * step, 80))

    predicted = kalman.predict()

    assert predicted[0] > 70


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_update__one_object_moving_linearly__produces_exactly_one_track() -> None:
    """The whole point of the stage: one pass is one track, not twenty."""
    emitted, tracker = run_script(linear_script(frames=20), frames=20)

    assert len(emitted) == 1
    assert emitted[0].hit_count == 20
    assert tracker.stats.tracks_started == 1


def test_update__two_well_separated_objects__produce_two_stable_distinct_ids() -> None:
    """Two vehicles on opposite sides of the road are never one vehicle."""
    script = linear_script(frames=15, start=(10, 10), step=(4, 0))
    other = linear_script(frames=15, start=(10, 80), step=(4, 0))
    for index, detections in other.items():
        script[index] = script[index] + detections

    emitted, _ = run_script(script, frames=15)

    assert len(emitted) == 2
    assert len({track.track_id for track in emitted}) == 2
    assert all(track.hit_count == 15 for track in emitted)


def test_update__two_objects_crossing_paths__do_not_swap_track_ids() -> None:
    """The case the Kalman model and the optimal assignment both exist for.

    One vehicle moves left to right, the other right to left, along the same
    line, so their boxes overlap heavily as they pass. If identities swapped,
    one track would reverse direction halfway through -- which is what the
    monotonicity assertion catches, and which no assertion on ids alone could.
    """
    rightward = linear_script(frames=16, start=(4, 40), size=(20, 20), step=(8, 0))
    leftward = linear_script(frames=16, start=(124, 40), size=(20, 20), step=(-8, 0))
    script = {index: rightward[index] + leftward[index] for index in rightward}

    emitted, _ = run_script(script, frames=16)

    assert len(emitted) == 2
    paths = sorted((centroids_x(track) for track in emitted), key=lambda path: path[0])

    assert paths[0] == sorted(paths[0]), "the rightward vehicle must never turn back"
    assert paths[1] == sorted(paths[1], reverse=True), "the leftward vehicle must never turn back"


def test_update__gap_shorter_than_max_age__keeps_the_same_track_id() -> None:
    """A vehicle passing behind a pole is the same vehicle when it comes out."""
    script = linear_script(frames=20, start=(10, 40), step=(5, 0))
    for missing in range(8, 13):
        script[missing] = []

    emitted, tracker = run_script(script, frames=20)

    assert len(emitted) == 1
    assert emitted[0].first_frame_index == 0
    assert emitted[0].last_frame_index == 19
    assert tracker.stats.tracks_started == 1


def test_update__gap_longer_than_max_age__starts_a_new_track() -> None:
    """Boundary: one frame more than the allowance ends the track.

    Deliberately asserted directly against the previous test. Too generous an
    allowance stitches two different vehicles into one track, which is the
    confident-wrong-answer failure in miniature.
    """
    script = linear_script(frames=20, start=(10, 40), step=(5, 0))
    for missing in range(8, 14):
        script[missing] = []

    emitted, tracker = run_script(script, frames=20)

    assert len(emitted) == 2
    assert len({track.track_id for track in emitted}) == 2
    assert tracker.stats.tracks_started == 2


def test_update__fewer_hits_than_the_minimum__emits_nothing_and_is_counted() -> None:
    """A detector firing twice in the same place is not a vehicle passing."""
    emitted, tracker = run_script(linear_script(frames=2), frames=10)

    assert emitted == []
    assert tracker.stats.tracks_discarded_unconfirmed == 1
    assert tracker.stats.tracks_emitted == 0


def test_update__exactly_the_minimum_hits__is_confirmed_and_emitted() -> None:
    """Boundary: the confirmation threshold is inclusive."""
    emitted, _ = run_script(linear_script(frames=3), frames=12)

    assert len(emitted) == 1
    assert emitted[0].hit_count == 3


def test_update__no_detections_at_all__produces_no_tracks() -> None:
    """Boundary: an empty road is the normal state of most cameras most nights."""
    emitted, tracker = run_script({}, frames=12)

    assert emitted == []
    assert tracker.active_track_ids == ()
    assert tracker.stats.tracks_started == 0


def test_update__a_stationary_object__produces_one_continuous_track() -> None:
    """A parked vehicle is one track, not one track per frame it fails to move."""
    box = (50, 40, 90, 70)
    script = {index: [make_detection(box)] for index in range(18)}

    emitted, _ = run_script(script, frames=18)

    assert len(emitted) == 1
    assert emitted[0].hit_count == 18


def test_update__object_entering_and_leaving__records_the_true_frame_bounds() -> None:
    """The frame indices are what stage 12 seeks back to; an off-by-one loses the plate."""
    script = linear_script(frames=10, start=(10, 40), step=(6, 0), first_frame=3)

    emitted, _ = run_script(script, frames=20)

    assert len(emitted) == 1
    assert emitted[0].first_frame_index == 3
    assert emitted[0].last_frame_index == 12


def test_update__a_detection_too_far_from_every_prediction__starts_its_own_track() -> None:
    """Below the IoU floor is not a continuation, however few candidates there are."""
    script = {
        0: [make_detection((10, 10, 40, 40))],
        1: [make_detection((10, 10, 40, 40))],
        2: [make_detection((10, 10, 40, 40))],
        3: [make_detection((120, 90, 150, 115))],
    }

    _, tracker = run_script(script, frames=4)

    assert tracker.stats.tracks_started == 2


def test_update__tracks_still_live_at_the_end__are_emitted_by_flush() -> None:
    """Without the flush, every vehicle still in view when a clip ends is lost."""
    detector = FakeDetector(linear_script(frames=6))
    tracker = SingleCameraTracker("cam_01", "cam_01_test", ARITHMETIC_ONLY)

    during: list[VehicleTrack] = []
    for index in range(6):
        frame = blank_frame(index)
        during.extend(tracker.update(frame, detector.detect(frame)))

    assert during == []
    assert len(tracker.flush()) == 1
    assert tracker.active_track_ids == ()


# ---------------------------------------------------------------------------
# Retention and memory
# ---------------------------------------------------------------------------


def test_update__with_crops_retained__holds_no_more_than_the_configured_cap() -> None:
    """The invariant that keeps a long clip from growing without bound.

    A track of any length holds at most ``max_retained_frames`` images; the rest
    of its history is a few numbers per frame.
    """
    config = TrackerConfig(retain_crops=True, max_retained_frames=3, min_hits=2, min_iou=0.3)
    detector = FakeDetector(linear_script(frames=40, start=(4, 40), step=(3, 0)))
    tracker = SingleCameraTracker("cam_01", "cam_01_test", config)

    for index in range(40):
        frame = blank_frame(index)
        tracker.update(frame, detector.detect(frame))
        assert tracker.retained_image_count <= 3

    emitted = tracker.flush()

    assert emitted[0].hit_count == 40
    assert emitted[0].retained_image_count() == 3
    assert len(emitted[0].ranked) == 3


def test_update__terminated_tracks__are_released_from_the_tracker() -> None:
    """Emitting a track must also stop holding it, or the run is one long leak."""
    config = TrackerConfig(retain_crops=True, max_retained_frames=2, min_hits=2, min_iou=0.3)
    script = linear_script(frames=4, start=(10, 40), step=(4, 0))

    detector = FakeDetector(script)
    tracker = SingleCameraTracker("cam_01", "cam_01_test", config)
    for index in range(20):
        frame = blank_frame(index)
        tracker.update(frame, detector.detect(frame))

    assert tracker.active_track_ids == ()
    assert tracker.retained_image_count == 0


def test_update__without_crop_retention__emits_tracks_with_no_imagery() -> None:
    """Arithmetic-only mode is honest about having no best frame, rather than guessing."""
    emitted, _ = run_script(linear_script(frames=6), frames=6)

    assert emitted[0].ranked == ()
    assert emitted[0].best is None


# ---------------------------------------------------------------------------
# Counters and configuration
# ---------------------------------------------------------------------------


def test_stats__after_a_run__describe_what_happened() -> None:
    """These are what stage 19 exports; a fragmenting tracker shows up here first."""
    _, tracker = run_script(linear_script(frames=10), frames=10)
    flat = tracker.stats.as_dict()

    assert flat["updates"] == 10
    assert flat["detections_seen"] == 10
    assert flat["associations"] == 9
    assert flat["tracks_emitted"] == 1
    assert flat["association_rate"] == pytest.approx(0.9)


def test_stats__nothing_seen__reports_a_zero_rate_rather_than_dividing() -> None:
    """Boundary: a tracker that has processed nothing has associated nothing."""
    _, tracker = run_script({}, frames=3)

    assert tracker.stats.association_rate == pytest.approx(0.0)


def test_tracker_config__from_settings__reads_every_value_from_configuration(
    build_settings: Callable[..., Settings],
) -> None:
    """No magic numbers in logic: the tracker's behaviour is configured, not coded."""
    settings = build_settings()
    config = TrackerConfig.from_settings(settings)

    assert config.min_iou == pytest.approx(settings.thresholds.track_association_min_iou)
    assert config.max_age_frames == settings.thresholds.track_max_age_frames
    assert config.min_hits == settings.thresholds.track_min_hits_to_confirm
    assert config.max_retained_frames == settings.detection.max_retained_frames
    assert config.weights.confidence == pytest.approx(
        settings.thresholds.best_frame_confidence_weight
    )


def test_track_ids__are_deterministic_across_identical_runs() -> None:
    """A tracker whose ids change between runs makes every comparison flaky."""
    script = linear_script(frames=8)

    first, _ = run_script(script, frames=8)
    second, _ = run_script(script, frames=8)

    assert [track.track_id for track in first] == [track.track_id for track in second]
    assert first[0].track_id.startswith("cam_01-")
