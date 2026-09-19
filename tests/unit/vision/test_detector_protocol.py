"""The detection type and the interface every detector answers to.

The coordinate mapping is the important part here. A box that reaches storage in
processed-frame space is the stage 10 failure mode: a plausible box in the wrong
place, which nothing downstream can detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_tracker.exceptions import VisionError
from multicam_tracker.ingest.preprocess import Preprocessor, RegionOfInterest
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import (
    BaseDetector,
    Detection,
    Detector,
    FakeDetector,
    iou,
)
from tests.fixtures.vision import blank_frame, make_detection

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Detection geometry and validation
# ---------------------------------------------------------------------------


def test_detection__ordinary_box__reports_its_geometry() -> None:
    """The derived properties are what every filter and the tracker read."""
    detection = make_detection((10, 20, 50, 40))

    assert detection.width == 40
    assert detection.height == 20
    assert detection.area == 800
    assert detection.aspect_ratio == pytest.approx(2.0)
    assert detection.centroid == (30.0, 30.0)


def test_detection__zero_area_box__is_accepted_rather_than_rejected() -> None:
    """A collapsed box is what a model reports when it has latched onto nothing.

    Rejecting it here would move the failure out of the filter that exists to
    count it and into an exception halfway through a batch.
    """
    detection = make_detection((10, 10, 10, 10))

    assert detection.area == 0
    assert detection.aspect_ratio == pytest.approx(0.0)


def test_detection__inverted_box__is_rejected() -> None:
    """An inverted box has negative width, which is not a degenerate box but a bug."""
    with pytest.raises(VisionError, match="inverted"):
        make_detection((50, 10, 10, 40))


def test_detection__negative_coordinate__is_rejected() -> None:
    """Source-frame coordinates start at zero; a negative one means a bad mapping."""
    with pytest.raises(VisionError, match="non-negative"):
        Detection(
            bbox=(-5, 10, 50, 40),
            object_class=ObjectClass.CAR,
            confidence=0.9,
            model_id="m",
            model_version="1",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection__confidence_outside_the_unit_interval__is_rejected(confidence: float) -> None:
    """Confidence is a probability everywhere else in the system."""
    with pytest.raises(VisionError, match=r"\[0, 1\]"):
        make_detection((0, 0, 10, 10), confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_detection__confidence_at_each_boundary__is_accepted(confidence: float) -> None:
    """Boundary: the interval is closed at both ends."""
    assert make_detection((0, 0, 10, 10), confidence=confidence).confidence == confidence


def test_detection__wrong_length_box__is_rejected() -> None:
    """A three-element box would silently unpack wrongly somewhere later."""
    with pytest.raises(VisionError, match=r"\(x1, y1, x2, y2\)"):
        Detection(
            bbox=(1, 2, 3),  # type: ignore[arg-type]
            object_class=ObjectClass.CAR,
            confidence=0.5,
            model_id="m",
            model_version="1",
        )


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_clamped_to__box_running_past_the_edge__is_clipped_to_the_frame() -> None:
    """A vehicle entering frame has a box past the edge; that is not an error."""
    clamped = make_detection((140, 100, 200, 160)).clamped_to(160, 120)

    assert clamped.bbox == (140, 100, 160, 120)


def test_clamped_to__box_entirely_outside__collapses_rather_than_raising() -> None:
    """The filters count this; an exception here would say nothing about how often."""
    clamped = make_detection((300, 300, 400, 400)).clamped_to(160, 120)

    assert clamped.area == 0


def test_clamped_to__box_already_inside__is_unchanged() -> None:
    """Idempotence: clamping a box that needs no clamping must not move it."""
    detection = make_detection((10, 20, 50, 40))

    assert detection.clamped_to(160, 120).bbox == detection.bbox


# ---------------------------------------------------------------------------
# Coordinate recovery through the stage 10 preprocessing chain
# ---------------------------------------------------------------------------


def test_in_source_coordinates__after_a_resize__recovers_the_original_box() -> None:
    """The exit criterion: every emitted box is in original source coordinates."""
    frame = blank_frame(0, width=160, height=120)
    preprocessor = Preprocessor(target_width=80, target_height=60)
    _, mapping = preprocessor.apply_to_frame(frame)

    # A box the detector would report in the half-size processed frame.
    recovered = make_detection((10, 15, 30, 25)).in_source_coordinates(mapping)

    assert recovered.bbox == (20, 30, 60, 50)


def test_in_source_coordinates__after_rotation_and_resize__recovers_the_original_box() -> None:
    """The composed chain is where the off-by-one in the rotation inverse hid.

    Pushed through the real preprocessor rather than a hand-built mapping, so
    this breaks if either side of the seam changes.
    """
    frame = blank_frame(0, width=160, height=120)
    preprocessor = Preprocessor(rotation_degrees=90, target_width=60, target_height=80)
    processed, mapping = preprocessor.apply_to_frame(frame)

    assert processed.shape == (60, 80)

    recovered = make_detection((5, 10, 25, 30)).in_source_coordinates(mapping)
    x1, y1, x2, y2 = recovered.bbox

    assert 0 <= x1 < x2 <= frame.width
    assert 0 <= y1 < y2 <= frame.height


def test_in_source_coordinates__through_a_cropping_roi__lands_inside_the_region() -> None:
    """A cropped region shifts the origin, and the mapping has to undo that too."""
    frame = blank_frame(0, width=160, height=120)
    roi = RegionOfInterest(polygon=[(40, 30), (120, 30), (120, 90), (40, 90)], crop_to_bounds=True)
    preprocessor = Preprocessor(roi=roi)
    processed, mapping = preprocessor.apply_to_frame(frame)

    assert processed.shape == (80, 60)

    recovered = make_detection((0, 0, 20, 20)).in_source_coordinates(mapping)

    assert recovered.bbox == (40, 30, 60, 50)


def test_in_source_coordinates__identity_preprocessing__changes_nothing() -> None:
    """Boundary: a preprocessor that does nothing must map boxes to themselves."""
    frame = blank_frame(0)
    _, mapping = Preprocessor().apply_to_frame(frame)
    detection = make_detection((10, 20, 50, 40))

    assert detection.in_source_coordinates(mapping).bbox == detection.bbox


def test_in_source_coordinates__preserves_everything_but_the_box() -> None:
    """Provenance and class must survive the mapping; only the geometry moves."""
    _, mapping = Preprocessor(target_width=80, target_height=60).apply_to_frame(
        blank_frame(0, width=160, height=120)
    )
    detection = make_detection((10, 10, 20, 20), object_class=ObjectClass.TRUCK, confidence=0.77)

    recovered = detection.in_source_coordinates(mapping)

    assert recovered.object_class is ObjectClass.TRUCK
    assert recovered.confidence == pytest.approx(0.77)
    assert recovered.model_id == detection.model_id


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def test_iou__identical_boxes__is_one() -> None:
    """The association metric's anchor value."""
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou__disjoint_boxes__is_zero() -> None:
    """No overlap is no association."""
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou__boxes_touching_at_an_edge__is_zero() -> None:
    """Boundary: sharing a border is not sharing area."""
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == pytest.approx(0.0)


def test_iou__half_overlap__is_the_computed_ratio() -> None:
    """A worked case, so a sign error in the union cannot pass."""
    # Intersection 5x10 = 50; union 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou__degenerate_box__is_zero_rather_than_undefined() -> None:
    """A collapsed box must not associate with everything through a zero union."""
    assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The interface itself
# ---------------------------------------------------------------------------


def test_fake_detector__satisfies_the_detector_protocol() -> None:
    """Swappability is only real if the implementations actually conform."""
    assert isinstance(FakeDetector(), Detector)


def test_base_detector__detect_is_not_implemented() -> None:
    """The base supplies batching, not detection; a subclass that forgets must fail."""
    with pytest.raises(NotImplementedError, match="detect"):
        BaseDetector("m", "1").detect(blank_frame(0))


def test_base_detector__batch_default__matches_sequential_detection() -> None:
    """Batching is a throughput optimisation and must not change the answer."""
    detector = FakeDetector({0: [make_detection((0, 0, 10, 10))], 1: []})
    frames = [blank_frame(0), blank_frame(1)]

    batched = detector.detect_batch(frames)
    sequential = [detector.detect(frame) for frame in frames]

    assert batched == sequential


def test_base_detector__empty_batch__returns_nothing_without_raising() -> None:
    """Boundary: an empty sample is the normal end of a stream."""
    assert FakeDetector().detect_batch([]) == []


def test_detection__mask_is_optional_and_carried_through() -> None:
    """A segmentation detector has one; a box detector does not. Both are valid."""
    mask = np.ones((10, 10), dtype=np.bool_)
    detection = Detection(
        bbox=(0, 0, 10, 10),
        object_class=ObjectClass.CAR,
        confidence=0.5,
        model_id="m",
        model_version="1",
        mask=mask,
    )

    assert detection.mask is not None
    assert make_detection((0, 0, 10, 10)).mask is None
