"""Properties that must hold for detection sequences nobody thought to script.

The example-based tests check the cases a person imagined. These check the
invariants that have to survive inputs nobody imagined -- which is where a
tracker actually breaks, because real traffic is stranger than any script.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.vision import (
    Detection,
    FakeDetector,
    SingleCameraTracker,
    TrackerConfig,
    crop_with_padding,
    solve_min_cost_assignment,
)
from tests.fixtures.vision import blank_frame, make_detection

pytestmark = pytest.mark.unit

FRAME_WIDTH = 160
FRAME_HEIGHT = 120

PROPERTY_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

boxes = st.builds(
    lambda x, y, width, height: (x, y, x + width, y + height),
    x=st.integers(min_value=0, max_value=FRAME_WIDTH - 1),
    y=st.integers(min_value=0, max_value=FRAME_HEIGHT - 1),
    width=st.integers(min_value=1, max_value=60),
    height=st.integers(min_value=1, max_value=50),
)

detection_sequences = st.lists(
    st.lists(st.builds(make_detection, boxes), min_size=0, max_size=3),
    min_size=1,
    max_size=25,
)


def run(per_frame: list[list[Detection]]) -> list:
    """Drive a tracker over a generated sequence and return everything it emitted.

    Args:
        per_frame: Detections for each frame, in order.

    Returns:
        The emitted tracks.
    """
    script = dict(enumerate(per_frame))
    detector = FakeDetector(script)
    tracker = SingleCameraTracker(
        "cam_01", "cam_01_test", TrackerConfig(retain_crops=False, min_hits=1)
    )

    emitted = []
    for index in range(len(per_frame)):
        frame = blank_frame(index, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        emitted.extend(tracker.update(frame, detector.detect(frame)))
    emitted.extend(tracker.flush())
    return emitted


@PROPERTY_SETTINGS
@given(per_frame=detection_sequences)
def test_every_emitted_track__ends_no_earlier_than_it_began(
    per_frame: list[list[Detection]],
) -> None:
    """A track running backwards would make every downstream time calculation wrong."""
    for track in run(per_frame):
        assert track.last_frame_index >= track.first_frame_index
        assert track.hit_count >= 1


@PROPERTY_SETTINGS
@given(per_frame=detection_sequences)
def test_no_two_simultaneously_active_tracks__share_an_identifier(
    per_frame: list[list[Detection]],
) -> None:
    """Two live tracks under one id are two vehicles the system cannot tell apart."""
    script = dict(enumerate(per_frame))
    detector = FakeDetector(script)
    tracker = SingleCameraTracker(
        "cam_01", "cam_01_test", TrackerConfig(retain_crops=False, min_hits=1)
    )

    for index in range(len(per_frame)):
        frame = blank_frame(index, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        tracker.update(frame, detector.detect(frame))
        active = tracker.active_track_ids
        assert len(set(active)) == len(active)


@PROPERTY_SETTINGS
@given(per_frame=detection_sequences)
def test_every_track__has_observations_in_ascending_frame_order(
    per_frame: list[list[Detection]],
) -> None:
    """The invariant the midpoint rule depends on, asserted against generated input."""
    for track in run(per_frame):
        indices = [observation.frame_index for observation in track.observations]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)


@PROPERTY_SETTINGS
@given(bbox=boxes)
def test_clamping__always_lands_inside_the_frame(bbox: tuple[int, int, int, int]) -> None:
    """Every emitted bbox lies within the original frame bounds after clamping."""
    x1, y1, x2, y2 = make_detection(bbox).clamped_to(FRAME_WIDTH, FRAME_HEIGHT).bbox

    assert 0 <= x1 <= x2 <= FRAME_WIDTH
    assert 0 <= y1 <= y2 <= FRAME_HEIGHT


@PROPERTY_SETTINGS
@given(
    bbox=st.builds(
        lambda x, y, width, height: (x, y, x + width, y + height),
        x=st.integers(min_value=-50, max_value=FRAME_WIDTH + 50),
        y=st.integers(min_value=-50, max_value=FRAME_HEIGHT + 50),
        width=st.integers(min_value=0, max_value=300),
        height=st.integers(min_value=0, max_value=300),
    ),
    padding=st.floats(min_value=0.0, max_value=1.0),
)
def test_cropping__any_box_at_all__yields_a_non_empty_image(
    bbox: tuple[int, int, int, int], padding: float
) -> None:
    """Including boxes with negative corners, which clamping is what saves.

    The thumbnail writer must never raise: the reviewer needs to see that there
    is nothing there, and an exception three layers down does not say that.
    """
    import numpy as np

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    crop = crop_with_padding(frame, bbox, padding_fraction=padding)

    assert crop.size > 0
    assert crop.shape[0] >= 1
    assert crop.shape[1] >= 1


@PROPERTY_SETTINGS
@given(
    matrix=st.lists(
        st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=5),
        min_size=1,
        max_size=5,
    ).filter(lambda rows: len({len(row) for row in rows}) == 1)
)
def test_assignment__pairs_each_row_and_column_at_most_once(
    matrix: list[list[float]],
) -> None:
    """A solver that double-assigned would silently merge two vehicles into one track."""
    pairs = solve_min_cost_assignment(matrix)

    rows = [row for row, _ in pairs]
    columns = [column for _, column in pairs]

    assert len(set(rows)) == len(rows)
    assert len(set(columns)) == len(columns)
    assert len(pairs) == min(len(matrix), len(matrix[0]))
