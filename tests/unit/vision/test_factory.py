"""Swapping the detector through configuration alone.

The stage exit criterion lives or dies here: if a caller anywhere has to name a
concrete class, the detector is not swappable and CI cannot run without weights.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from multicam_tracker.config import Settings
from multicam_tracker.exceptions import ConfigurationError, VisionError
from multicam_tracker.vision import (
    FakeDetector,
    FixtureDetector,
    YoloDetector,
    build_fixture_payload,
    create_detector,
    write_fixture,
)
from tests.fixtures.vision import FIXTURE_START, blank_frame, linear_script

pytestmark = pytest.mark.unit


def test_create_detector__yolo_backend__builds_without_touching_the_weights(
    build_settings: Callable[..., Settings],
) -> None:
    """Lazy loading is what lets a machine with no weights still construct the pipeline."""
    settings = build_settings(detection={"detector_backend": "yolo"})

    detector = create_detector(settings)

    assert isinstance(detector, YoloDetector)
    assert detector.is_loaded is False


def test_create_detector__fake_backend__builds_a_detector_that_finds_nothing(
    build_settings: Callable[..., Settings],
) -> None:
    """A fake that invented detections from configuration would be placeholder logic."""
    settings = build_settings(detection={"detector_backend": "fake"})

    detector = create_detector(settings)

    assert isinstance(detector, FakeDetector)
    assert detector.detect(blank_frame(0)) == []


def test_create_detector__fixture_backend__replays_the_configured_recording(
    build_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """How CI runs every downstream stage with no weights, no GPU, and no network."""
    path = tmp_path / "detections.json"
    frames = [blank_frame(index) for index in range(4)]
    write_fixture(
        path,
        build_fixture_payload(
            FakeDetector(linear_script(frames=4)),
            {"cam_01_test": frames},
            recorded_at=FIXTURE_START,
        ),
    )
    settings = build_settings(
        detection={"detector_backend": "fixture", "fixture_detections_path": path}
    )

    detector = create_detector(settings)

    assert isinstance(detector, FixtureDetector)
    assert len(detector.detect(frames[0])) == 1


def test_create_detector__fixture_backend_with_no_recording__is_a_configuration_error(
    build_settings: Callable[..., Settings],
) -> None:
    """Falling back to an empty fixture would make every clip look like an empty road."""
    settings = build_settings(detection={"detector_backend": "fixture"})

    with pytest.raises(ConfigurationError, match="needs a recording to replay"):
        create_detector(settings)


def test_create_detector__fixture_backend_with_a_missing_file__names_the_path(
    build_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """An error that does not say where it looked costs an hour."""
    settings = build_settings(
        detection={
            "detector_backend": "fixture",
            "fixture_detections_path": tmp_path / "absent.json",
        }
    )

    with pytest.raises(VisionError, match=r"absent\.json"):
        create_detector(settings)


def test_create_detector__the_environment_alone__selects_the_backend(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[..., Settings]
) -> None:
    """The exit criterion, stated as a test: configuration, not code.

    No caller names a class. An operator changes one environment variable and
    the whole pipeline runs against a different detector.
    """
    monkeypatch.setenv("MCT_DETECTION__DETECTOR_BACKEND", "fake")

    assert isinstance(create_detector(build_settings()), FakeDetector)


def test_yolo_detector__from_settings__reads_its_parameters_from_configuration(
    build_settings: Callable[..., Settings],
) -> None:
    """Model paths and inference parameters are deployment decisions, not constants."""
    settings = build_settings()
    detector = YoloDetector.from_settings(settings)

    assert detector.weights_path == settings.vision.vehicle_detector_path
    assert detector.min_confidence == pytest.approx(settings.vision.detection_min_confidence)
    assert detector.nms_iou == pytest.approx(settings.vision.detection_nms_iou)
    assert detector.requested_device == settings.vision.device


def test_yolo_detector__model_id__is_available_before_the_weights_load() -> None:
    """The factory logs it, so asking for it must not pull in a 50 MB file."""
    detector = YoloDetector(Path("weights/vehicle_detector.pt"))

    assert detector.model_id == "vehicle_detector"
    assert detector.is_loaded is False


def test_yolo_detector__class_map__excludes_anything_that_is_not_a_vehicle() -> None:
    """Bicycle has no class in the domain model, and mapping it to car is worse than dropping it."""
    detector = YoloDetector(Path("weights/vehicle_detector.pt"))

    assert set(detector.class_map) == {2, 3, 5, 7}
    assert 1 not in detector.class_map
    assert 0 not in detector.class_map
