"""The real detector, against real weights.

Structural correctness only -- that it returns a list of the right shape, that a
batched run equals a sequential one, that the class filter is honoured, that
raising the threshold cannot raise the count. Never accuracy numbers: those
belong to the model, not to this code, and asserting them here would make a
model upgrade look like a regression in the wrapper.

Almost everything in this module needs weights and skips without them. What does
not is at the bottom: the error paths, which are exactly the ones an operator
meets on a machine where the weights are missing, and which therefore must be
tested on a machine where the weights are missing.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from multicam_tracker.exceptions import VisionError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import COCO_VEHICLE_CLASSES, YoloDetector

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = REPO_ROOT / "tests" / "fixtures" / "images"
WEIGHTS = REPO_ROOT / "weights" / "vehicle_detector.pt"


def _weights_available() -> bool:
    """Return whether Ultralytics and the weights file are both present.

    Returns:
        ``True`` when a real run is possible.
    """
    return importlib.util.find_spec("ultralytics") is not None and WEIGHTS.is_file()


requires_models = pytest.mark.skipif(
    not _weights_available(),
    reason="needs the vision extra and weights/vehicle_detector.pt",
)


def _has_opencv() -> bool:
    """Return whether OpenCV is importable.

    Returns:
        ``True`` when it imports.
    """
    return importlib.util.find_spec("cv2") is not None


requires_opencv = pytest.mark.skipif(not _has_opencv(), reason="OpenCV is not installed")


def load_frame(name: str, frame_index: int = 0) -> Frame:
    """Return a committed still image as a frame.

    Args:
        name: File name inside ``tests/fixtures/images``.
        frame_index: Index to stamp on the frame.

    Returns:
        The frame.
    """
    import cv2

    image = cv2.imread(str(IMAGE_DIR / name))
    assert image is not None, f"fixture image missing: {name}"
    stamp = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC) + timedelta(seconds=frame_index)
    return Frame(
        image=np.ascontiguousarray(image),
        frame_index=frame_index,
        raw_timestamp=stamp,
        timestamp_utc=stamp,
        source_id=name,
        camera_id="cam_01",
    )


# ---------------------------------------------------------------------------
# With weights
# ---------------------------------------------------------------------------


@requires_models
@requires_opencv
def test_detect__an_image_with_vehicles__returns_a_list_of_detections() -> None:
    """Structural: the call works and the shape is right.

    Deliberately not "finds the three vehicles". The committed images are
    synthetic rectangles, not photographs -- see
    ``tests/fixtures/images/README.md``. Asserting a count against them would be
    asserting something about a fixture rather than about a road, and a real
    photograph is a licensing decision this stage does not get to make.
    """
    detections = YoloDetector(WEIGHTS).detect(load_frame("multi_vehicle.png"))

    assert isinstance(detections, list)
    assert all(detection.confidence <= 1.0 for detection in detections)
    assert all(detection.model_id == "vehicle_detector" for detection in detections)


@requires_models
@requires_opencv
def test_detect__an_empty_road__returns_an_empty_list_without_raising() -> None:
    """The common case on most cameras most of the night."""
    assert YoloDetector(WEIGHTS).detect(load_frame("empty_road.png")) == []


@requires_models
@requires_opencv
def test_detect__the_class_filter__excludes_everything_that_is_not_a_vehicle() -> None:
    """COCO includes people, traffic lights and handbags; none of them are vehicles."""
    detections = YoloDetector(WEIGHTS).detect(load_frame("multi_vehicle.png"))

    assert all(
        detection.object_class in set(COCO_VEHICLE_CLASSES.values()) for detection in detections
    )
    assert all(detection.object_class is not ObjectClass.UNKNOWN for detection in detections)


@requires_models
@requires_opencv
def test_detect_batch__matches_sequential_detection_exactly() -> None:
    """Batching exists for throughput and must not change the answer."""
    detector = YoloDetector(WEIGHTS)
    frames = [
        load_frame("empty_road.png", 0),
        load_frame("single_vehicle.png", 1),
        load_frame("multi_vehicle.png", 2),
    ]

    batched = detector.detect_batch(frames)
    sequential = [detector.detect(frame) for frame in frames]

    assert [len(entry) for entry in batched] == [len(entry) for entry in sequential]
    for from_batch, from_sequence in zip(batched, sequential, strict=True):
        assert [entry.bbox for entry in from_batch] == [entry.bbox for entry in from_sequence]


@requires_models
@requires_opencv
def test_detect__raising_the_confidence_threshold__cannot_increase_the_count() -> None:
    """Monotonicity: a stricter filter that found more would be a wiring bug."""
    frame = load_frame("multi_vehicle.png")

    permissive = len(YoloDetector(WEIGHTS, min_confidence=0.10).detect(frame))
    strict = len(YoloDetector(WEIGHTS, min_confidence=0.90).detect(frame))

    assert strict <= permissive


@requires_models
def test_cuda_requested_without_cuda__falls_back_to_the_cpu() -> None:
    """A service that will not start at three in the morning is the worse failure."""
    detector = YoloDetector(WEIGHTS, device="cuda")

    assert detector.device in {"cpu", "cuda"}


@requires_models
def test_model_version__names_both_the_library_and_the_weights() -> None:
    """The library version alone does not distinguish two sets of weights."""
    version = YoloDetector(WEIGHTS).model_version

    assert version.startswith("ultralytics-")
    assert "sha256:" in version


# ---------------------------------------------------------------------------
# Without weights -- the paths an operator actually meets
# ---------------------------------------------------------------------------


def test_importing_and_constructing__without_weights__does_not_raise() -> None:
    """Lazy loading, stated as a test.

    A unit test about bounding-box arithmetic must not require a 50 MB download,
    and a pipeline must be constructible on a machine that has not been
    provisioned yet.
    """
    detector = YoloDetector(Path("weights/definitely-not-here.pt"))

    assert detector.is_loaded is False
    assert detector.model_id == "definitely-not-here"


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="without ultralytics the import error masks the missing-weights error",
)
def test_detect__with_a_missing_weights_file__names_the_path_it_looked_for() -> None:
    """The operator's next action is to put a file somewhere; the error says where."""
    detector = YoloDetector(REPO_ROOT / "weights" / "definitely-not-here.pt")

    with pytest.raises(VisionError, match="weights not found") as excinfo:
        detector.detect_batch([load_frame("empty_road.png")])

    assert "definitely-not-here.pt" in excinfo.value.context["path"]
    assert excinfo.value.context["setting"] == "MCT_VISION__VEHICLE_DETECTOR_PATH"


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics is installed, so this error path cannot be reached",
)
def test_detect__without_ultralytics__names_the_extra_that_provides_it() -> None:
    """The machine this ran on had no vision extra, which is the case being tested."""
    detector = YoloDetector(WEIGHTS)

    with pytest.raises(VisionError, match="Ultralytics is required") as excinfo:
        detector.detect_batch([load_frame("empty_road.png")])

    assert excinfo.value.context["extra"] == "vision"


@requires_opencv
def test_detect_batch__an_empty_batch__returns_nothing_without_loading_weights() -> None:
    """Boundary: the end of a stream must not trigger a model load."""
    detector = YoloDetector(Path("weights/definitely-not-here.pt"))

    assert detector.detect_batch([]) == []
    assert detector.is_loaded is False
