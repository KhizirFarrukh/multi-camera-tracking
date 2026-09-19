"""The scripted detector the tracking tests are built on."""

from __future__ import annotations

import pytest

from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import FakeDetector
from tests.fixtures.vision import blank_frame, make_detection

pytestmark = pytest.mark.unit


def test_detect__a_scripted_frame__returns_what_was_scripted() -> None:
    """The point of the fake: the right answer is known exactly."""
    detector = FakeDetector({3: [make_detection((10, 10, 50, 40))]})

    assert len(detector.detect(blank_frame(3))) == 1


def test_detect__an_unscripted_frame__returns_nothing() -> None:
    """The honest reading of "the script does not mention this frame"."""
    detector = FakeDetector({3: [make_detection((10, 10, 50, 40))]})

    assert detector.detect(blank_frame(4)) == []


def test_detect__with_a_default__returns_it_for_unscripted_frames() -> None:
    """A background detection on every frame is occasionally what a test needs."""
    detector = FakeDetector({}, default=[make_detection((0, 0, 20, 20))])

    assert len(detector.detect(blank_frame(99))) == 1


def test_detect__restamps_provenance_with_its_own_identity() -> None:
    """Provenance that is sometimes a lie is provenance nobody can use.

    A detection copied in from elsewhere must not keep claiming a lineage it no
    longer has.
    """
    detector = FakeDetector(
        {0: [make_detection((0, 0, 20, 20), model_id="somebody-else", model_version="9")]},
        model_id="fake-detector",
        model_version="1.0",
    )

    detection = detector.detect(blank_frame(0))[0]

    assert detection.model_id == "fake-detector"
    assert detection.model_version == "1.0"


def test_detect__preserves_everything_but_the_provenance() -> None:
    """Restamping must not quietly change the geometry or the class."""
    original = make_detection((5, 6, 25, 26), confidence=0.42, object_class=ObjectClass.BUS)
    detector = FakeDetector({0: [original]})

    detection = detector.detect(blank_frame(0))[0]

    assert detection.bbox == original.bbox
    assert detection.confidence == pytest.approx(0.42)
    assert detection.object_class is ObjectClass.BUS


def test_call_count__counts_every_invocation() -> None:
    """The motion-prefilter claim is only real if the detector is invoked less."""
    detector = FakeDetector()

    for index in range(5):
        detector.detect(blank_frame(index))

    assert detector.call_count == 5


def test_detect_batch__matches_sequential_detection() -> None:
    """Batching must not change the answer, in the fake as in the real model."""
    detector = FakeDetector({0: [make_detection((0, 0, 20, 20))], 2: []})
    frames = [blank_frame(index) for index in range(3)]

    assert detector.detect_batch(frames) == [detector.detect(frame) for frame in frames]


def test_scripted_frame_indices__reports_the_script_in_order() -> None:
    """A test asserting it scripted what it meant to needs somewhere to look."""
    detector = FakeDetector({9: [], 1: [], 4: []})

    assert detector.scripted_frame_indices() == (1, 4, 9)


def test_empty_script__detects_nothing_anywhere() -> None:
    """Boundary: the default construction is a detector that finds nothing."""
    assert FakeDetector().detect(blank_frame(0)) == []
