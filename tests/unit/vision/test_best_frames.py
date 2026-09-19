"""Which frame of a pass gets read, and why.

This ranking determines plate accuracy more than any threshold in the OCR stage,
so each criterion is tested in isolation: frames identical in every way but one,
so a failure names the criterion that broke.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import pytest

from multicam_tracker.config import Settings
from multicam_tracker.vision import (
    BestFrameWeights,
    TrackObservation,
    centrality,
    laplacian_variance,
    rank_observations,
)
from tests.fixtures.vision import FIXTURE_START, make_detection

pytestmark = pytest.mark.unit

FRAME_SIZE = (160, 120)
WEIGHTS = BestFrameWeights()


def observation(
    frame_index: int,
    bbox: tuple[int, int, int, int] = (60, 45, 100, 75),
    *,
    confidence: float = 0.8,
    sharpness: float = 100.0,
    touches_edge: bool = False,
) -> TrackObservation:
    """Return an observation with everything but the varied criterion held fixed.

    Args:
        frame_index: Which frame.
        bbox: The detection box.
        confidence: Detector confidence.
        sharpness: Precomputed Laplacian variance.
        touches_edge: Whether a filter flagged it as cut off.

    Returns:
        The observation.
    """
    stamp = FIXTURE_START + timedelta(seconds=frame_index / 10)
    return TrackObservation(
        frame_index=frame_index,
        raw_timestamp=stamp,
        timestamp_utc=stamp,
        detection=make_detection(bbox, confidence=confidence, touches_edge=touches_edge),
        sharpness=sharpness,
    )


def sharp_image(size: int = 24) -> npt.NDArray[np.uint8]:
    """Return a high-frequency image, as a focused crop looks to a Laplacian.

    Args:
        size: Side length.

    Returns:
        A BGR checkerboard.
    """
    board = np.indices((size, size)).sum(axis=0) % 2
    return np.repeat((board * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def blurred_image(size: int = 24) -> npt.NDArray[np.uint8]:
    """Return a low-frequency image, as a motion-blurred crop looks.

    A smooth horizontal ramp: the same mean brightness as the checkerboard and
    almost none of its high-frequency content, which is the difference the
    Laplacian measures.

    Args:
        size: Side length.

    Returns:
        A BGR gradient.
    """
    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (size, 1))[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


def test_laplacian_variance__a_blurred_image__scores_far_below_a_sharp_one() -> None:
    """The criterion that separates two frames a detector is equally happy with."""
    assert laplacian_variance(sharp_image()) > 10 * laplacian_variance(blurred_image())


def test_laplacian_variance__a_flat_image__is_zero() -> None:
    """Boundary: a featureless crop has no high-frequency content at all."""
    assert laplacian_variance(np.full((20, 20, 3), 120, dtype=np.uint8)) == pytest.approx(0.0)


@pytest.mark.parametrize("shape", [(0, 0, 3), (2, 2, 3), (1, 30, 3)])
def test_laplacian_variance__an_image_too_small_to_convolve__is_zero(
    shape: tuple[int, int, int],
) -> None:
    """A 2x2 crop carries no information about focus, and saying zero is honest."""
    assert laplacian_variance(np.zeros(shape, dtype=np.uint8)) == pytest.approx(0.0)


def test_laplacian_variance__a_greyscale_crop__is_handled_like_a_colour_one() -> None:
    """Two axes rather than three is a shape difference, not an error."""
    grey = (np.indices((20, 20)).sum(axis=0) % 2 * 255).astype(np.uint8)

    assert laplacian_variance(grey) > 0.0


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------


def test_centrality__at_the_exact_centre__is_one() -> None:
    """The anchor value the criterion is scaled against."""
    assert centrality((80.0, 60.0), FRAME_SIZE) == pytest.approx(1.0)


def test_centrality__at_a_corner__is_zero() -> None:
    """Boundary: the corner is a half-diagonal away, which is the full scale."""
    assert centrality((0.0, 0.0), FRAME_SIZE) == pytest.approx(0.0)


def test_centrality__with_an_unknown_frame_size__is_zero_not_one() -> None:
    """Unknown is not central. Defaulting to 1.0 would hand every frame full marks."""
    assert centrality((80.0, 60.0), (0, 0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Each criterion in isolation
# ---------------------------------------------------------------------------


def test_rank__frames_differing_only_in_confidence__put_the_most_confident_first() -> None:
    """The detector's own opinion, which correlates with occlusion and blur."""
    ranked = rank_observations(
        [observation(0, confidence=0.4), observation(1, confidence=0.95)],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1


def test_rank__frames_differing_only_in_sharpness__put_the_sharpest_first() -> None:
    """Measured from real images rather than an asserted number, so the metric is tested too."""
    sharp = laplacian_variance(sharp_image())
    blurred = laplacian_variance(blurred_image())

    ranked = rank_observations(
        [observation(0, sharpness=blurred), observation(1, sharpness=sharp)],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1


def test_rank__frames_differing_only_in_area__put_the_larger_box_first() -> None:
    """A closer vehicle covers more pixels and carries more plate detail."""
    ranked = rank_observations(
        [observation(0, bbox=(70, 50, 90, 70)), observation(1, bbox=(50, 30, 110, 90))],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1


def test_rank__an_edge_touching_frame__ranks_below_an_otherwise_equal_centred_one() -> None:
    """A vehicle the frame has cut in half is missing what it would be selected for."""
    ranked = rank_observations(
        [observation(0, touches_edge=True), observation(1)],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1
    assert "edge_discount" in ranked[1].components


def test_rank__a_box_literally_on_the_border__is_discounted_without_a_filter_flag() -> None:
    """Ranking must not lose a criterion when filtering is turned off.

    The filter sets the flag under its configured margin; the geometric test
    backs it up, because literally touching the border is unambiguous whatever
    the policy.
    """
    ranked = rank_observations(
        [observation(0, bbox=(0, 45, 40, 75)), observation(1)],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1


def test_rank__a_track_entirely_at_the_edge__still_orders_its_frames() -> None:
    """Why the edge penalty is a discount rather than a subtraction.

    Subtracting a fixed 0.5 drives every edge frame to the clamp at zero, where
    they tie and frame index decides -- which is the wrong answer for a track
    where one of those frames really is the best available.
    """
    ranked = rank_observations(
        [
            observation(0, bbox=(0, 45, 40, 75), confidence=0.3),
            observation(1, bbox=(0, 45, 40, 75), confidence=0.95),
        ],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert ranked[0].observation.frame_index == 1
    assert ranked[0].score > ranked[1].score


# ---------------------------------------------------------------------------
# Determinism, weighting, and boundaries
# ---------------------------------------------------------------------------


def test_rank__identical_inputs__produce_an_identical_order() -> None:
    """A ranking that reorders between runs makes every downstream comparison flaky."""
    observations = [observation(index, confidence=0.5 + index / 100) for index in range(6)]

    first = [
        entry.observation.frame_index
        for entry in rank_observations(observations, WEIGHTS, FRAME_SIZE)
    ]
    second = [
        entry.observation.frame_index
        for entry in rank_observations(observations, WEIGHTS, FRAME_SIZE)
    ]

    assert first == second


def test_rank__tied_scores__break_toward_the_earlier_frame() -> None:
    """Ties must resolve on something stated, not on dictionary ordering."""
    observations = [observation(5), observation(2), observation(9)]

    ranked = rank_observations(observations, WEIGHTS, FRAME_SIZE)

    assert [entry.observation.frame_index for entry in ranked] == [2, 5, 9]


def test_rank__changing_the_weights__measurably_changes_the_order() -> None:
    """The weights have to actually do something, or they are decoration."""
    frames = [
        observation(0, confidence=0.95, sharpness=10.0),
        observation(1, confidence=0.40, sharpness=500.0),
    ]

    confidence_led = BestFrameWeights(confidence=0.9, sharpness=0.1, area=0.0, centrality=0.0)
    sharpness_led = BestFrameWeights(confidence=0.1, sharpness=0.9, area=0.0, centrality=0.0)

    assert rank_observations(frames, confidence_led, FRAME_SIZE)[0].observation.frame_index == 0
    assert rank_observations(frames, sharpness_led, FRAME_SIZE)[0].observation.frame_index == 1


def test_rank__no_observations__returns_nothing() -> None:
    """Boundary: a track with no retained imagery has no best frame."""
    assert rank_observations([], WEIGHTS, FRAME_SIZE) == []


def test_rank__a_single_observation__is_its_own_best() -> None:
    """Boundary: within-set normalization must not divide by zero on one element."""
    ranked = rank_observations([observation(0)], WEIGHTS, FRAME_SIZE)

    assert len(ranked) == 1
    assert 0.0 <= ranked[0].score <= 1.0


def test_rank__all_sharpness_zero__does_not_divide_by_zero() -> None:
    """Boundary: arithmetic-only mode computes no sharpness at all."""
    ranked = rank_observations(
        [observation(0, sharpness=0.0), observation(1, sharpness=0.0)],
        WEIGHTS,
        FRAME_SIZE,
    )

    assert all(entry.components["sharpness"] == pytest.approx(0.0) for entry in ranked)


def test_rank__scores_stay_within_the_unit_interval() -> None:
    """The weights sum to one so the composite score means the same thing everywhere."""
    observations = [
        observation(0, confidence=1.0, sharpness=999.0, bbox=(70, 50, 90, 70)),
        observation(1, confidence=0.0, sharpness=0.0, touches_edge=True),
    ]

    assert all(
        0.0 <= entry.score <= 1.0 for entry in rank_observations(observations, WEIGHTS, FRAME_SIZE)
    )


def test_rank__components_are_reported_alongside_the_score() -> None:
    """A ranking nobody can explain is a ranking nobody can tune."""
    ranked = rank_observations([observation(0)], WEIGHTS, FRAME_SIZE)

    assert set(ranked[0].components) >= {"confidence", "sharpness", "area", "centrality"}


def test_weights__from_settings__come_from_the_thresholds_file(
    build_settings: Callable[..., Settings],
) -> None:
    """The ranking is configured rather than coded, like every other threshold."""
    settings = build_settings()
    weights = BestFrameWeights.from_settings(settings)

    assert weights.confidence == pytest.approx(settings.thresholds.best_frame_confidence_weight)
    assert weights.sharpness == pytest.approx(settings.thresholds.best_frame_sharpness_weight)
    assert weights.area == pytest.approx(settings.thresholds.best_frame_area_weight)
    assert weights.centrality == pytest.approx(settings.thresholds.best_frame_centrality_weight)
    assert weights.edge_penalty == pytest.approx(settings.thresholds.best_frame_edge_penalty)
