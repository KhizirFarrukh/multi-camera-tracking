"""Builders and a weights-free reference detector for the stage 11 tests.

Two things live here.

**Builders.** Frames and detections with sensible defaults, so a test about
track association is not three quarters frame construction. Shared rather than
redefined per module, per the contract's fixture rule.

**A reference detector.** ``ReferenceBlobDetector`` finds the bright rectangle
that both ``SyntheticVideoSource`` and the committed sample clips draw. It is
deterministic, needs no weights, and is *not* a general vehicle detector -- it
would find nothing at all in real footage. Its job is to give the recorded
detection fixtures something real to be recorded from on a machine with no model
weights, and to let the end-to-end test compare a live run against that
recording.

That substitution is a deviation from the stage prompt, which asks for fixtures
recorded from a real model run. It is recorded in ``docs/DETECTION.md`` and
``docs/DECISIONS.md``, and the fixture file itself says so in its ``note``
field, so nobody can mistake the recording for a YOLO one.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt

from multicam_tracker.ingest.frame import Frame
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision.best_frames import BestFrameWeights, rank_observations
from multicam_tracker.vision.detector_protocol import BaseDetector, Detection
from multicam_tracker.vision.track import TrackObservation, VehicleTrack

__all__ = [
    "REFERENCE_MODEL_ID",
    "ReferenceBlobDetector",
    "blank_frame",
    "frame_sequence",
    "linear_script",
    "make_detection",
    "make_observation",
    "make_track",
]

REFERENCE_MODEL_ID = "reference-blob"
"""Identifier the reference detector stamps on everything it produces."""

FIXTURE_START = datetime(2026, 8, 10, 14, 22, 11, tzinfo=UTC)
"""Capture time of frame zero in the built sequences."""

_BRIGHT_THRESHOLD = 170
_MIN_BLOB_PIXELS = 12


def make_detection(
    bbox: tuple[int, int, int, int],
    *,
    confidence: float = 0.9,
    object_class: ObjectClass = ObjectClass.CAR,
    model_id: str = "test-detector",
    model_version: str = "1.0",
    touches_edge: bool = False,
) -> Detection:
    """Return a detection with test-friendly defaults.

    Args:
        bbox: ``(x1, y1, x2, y2)``.
        confidence: Detector confidence.
        object_class: What it is.
        model_id: Provenance identifier.
        model_version: Provenance version.
        touches_edge: Whether a filter flagged it as cut off.

    Returns:
        The detection.
    """
    return Detection(
        bbox=bbox,
        object_class=object_class,
        confidence=confidence,
        model_id=model_id,
        model_version=model_version,
        touches_edge=touches_edge,
    )


def blank_frame(
    frame_index: int,
    *,
    width: int = 160,
    height: int = 120,
    camera_id: str = "cam_01",
    source_id: str = "cam_01_test",
    fps: float = 10.0,
    fill: int = 40,
    clock_offset_ms: int = 0,
) -> Frame:
    """Return a featureless frame at a computed capture time.

    Args:
        frame_index: Position in the source.
        width: Frame width.
        height: Frame height.
        camera_id: Which camera.
        source_id: Which source.
        fps: Rate used to derive the timestamp from the index.
        fill: Grey level of the image.
        clock_offset_ms: Offset between raw and corrected timestamps.

    Returns:
        The frame.
    """
    raw = FIXTURE_START + timedelta(seconds=frame_index / fps)
    image = np.full((height, width, 3), fill, dtype=np.uint8)
    return Frame(
        image=image,
        frame_index=frame_index,
        raw_timestamp=raw,
        timestamp_utc=raw + timedelta(milliseconds=clock_offset_ms),
        source_id=source_id,
        camera_id=camera_id,
    )


def frame_sequence(count: int, **kwargs: object) -> list[Frame]:
    """Return consecutive blank frames.

    Args:
        count: How many.
        **kwargs: Passed to :func:`blank_frame`.

    Returns:
        Frames indexed from zero.
    """
    return [blank_frame(index, **kwargs) for index in range(count)]  # type: ignore[arg-type]


def linear_script(
    *,
    frames: int,
    start: tuple[int, int] = (10, 40),
    size: tuple[int, int] = (30, 20),
    step: tuple[int, int] = (6, 0),
    confidence: float = 0.9,
    first_frame: int = 0,
) -> dict[int, list[Detection]]:
    """Return a script of one object moving in a straight line.

    The canonical tracker input: a vehicle crossing the frame at a constant
    rate, which must produce exactly one track.

    Args:
        frames: How many frames the object appears in.
        start: ``(x, y)`` of the box at the first frame.
        size: ``(width, height)`` of the box.
        step: Per-frame movement.
        confidence: Detector confidence on every frame.
        first_frame: Frame index the object first appears in.

    Returns:
        A mapping suitable for :class:`~multicam_tracker.vision.fake_detector.FakeDetector`.
    """
    script: dict[int, list[Detection]] = {}
    for offset in range(frames):
        x = start[0] + step[0] * offset
        y = start[1] + step[1] * offset
        script[first_frame + offset] = [
            make_detection((x, y, x + size[0], y + size[1]), confidence=confidence)
        ]
    return script


def _blobs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int, int, int, int]]:
    """Return connected bright regions as boxes plus their pixel counts.

    Four-connected flood fill. Written out rather than taken from OpenCV so the
    fixture works in the core install, and because the images are at most a few
    hundred pixels on a side.

    Args:
        mask: Which pixels are bright.

    Returns:
        ``(x1, y1, x2, y2, pixel_count)`` per region, in raster order of their
        first pixel, which makes the result deterministic.
    """
    height, width = mask.shape
    seen = np.zeros_like(mask)
    regions: list[tuple[int, int, int, int, int]] = []

    for row in range(height):
        for column in range(width):
            if not mask[row, column] or seen[row, column]:
                continue

            queue = deque([(row, column)])
            seen[row, column] = True
            min_row = max_row = row
            min_col = max_col = column
            count = 0

            while queue:
                current_row, current_col = queue.popleft()
                count += 1
                min_row, max_row = min(min_row, current_row), max(max_row, current_row)
                min_col, max_col = min(min_col, current_col), max(max_col, current_col)

                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row, next_col = current_row + delta_row, current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        queue.append((next_row, next_col))

            regions.append((min_col, min_row, max_col + 1, max_row + 1, count))

    return regions


class ReferenceBlobDetector(BaseDetector):
    """Finds the bright rectangle in a generated frame.

    Deterministic and weights-free. Not a vehicle detector: it detects
    brightness, and the only reason that is useful is that the synthetic source
    and the committed sample clips both draw a bright rectangle where a vehicle
    would be.

    Args:
        threshold: Grey level above which a pixel counts as bright.
        min_pixels: Smallest region reported, which suppresses the speckle that
            video compression leaves around a hard edge.
    """

    def __init__(self, *, threshold: int = _BRIGHT_THRESHOLD, min_pixels: int = _MIN_BLOB_PIXELS):
        super().__init__(REFERENCE_MODEL_ID, "1.0")
        self.threshold = threshold
        self.min_pixels = min_pixels

    def detect(self, frame: Frame) -> list[Detection]:
        """Return one detection per bright region.

        Confidence is the region's fill ratio -- how much of its own bounding
        box is actually bright. A clean rectangle scores near 1.0 and a
        compression-smeared one scores lower, which gives the recorded fixtures
        the ragged confidences a real detector produces rather than a column of
        identical numbers.

        Args:
            frame: The frame to search.

        Returns:
            Detections in the frame's own coordinates.
        """
        grey = frame.image.astype(np.int16).mean(axis=2)
        mask: npt.NDArray[np.bool_] = grey > self.threshold

        detections: list[Detection] = []
        for x1, y1, x2, y2, count in _blobs(mask):
            if count < self.min_pixels:
                continue
            box_area = max(1, (x2 - x1) * (y2 - y1))
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    object_class=ObjectClass.CAR,
                    confidence=round(min(1.0, count / box_area), 6),
                    model_id=self.model_id,
                    model_version=self.model_version,
                )
            )
        return detections

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return detections for several frames.

        Args:
            frames: The frames to search.

        Returns:
            One list per frame, in order.
        """
        return [self.detect(frame) for frame in frames]


def make_observation(
    frame_index: int,
    bbox: tuple[int, int, int, int] = (60, 45, 100, 75),
    *,
    confidence: float = 0.8,
    sharpness: float = 100.0,
    touches_edge: bool = False,
    crop: npt.NDArray[np.uint8] | None = None,
    fps: float = 10.0,
    clock_offset_ms: int = 0,
) -> TrackObservation:
    """Return a track observation with everything but the varied field held fixed.

    Args:
        frame_index: Which frame.
        bbox: The detection box, in source-frame coordinates.
        confidence: Detector confidence.
        sharpness: Precomputed Laplacian variance.
        touches_edge: Whether a filter flagged it as cut off.
        crop: The retained image, if this observation kept one.
        fps: Rate used to derive the timestamp from the index.
        clock_offset_ms: Offset between raw and corrected timestamps.

    Returns:
        The observation.
    """
    raw = FIXTURE_START + timedelta(seconds=frame_index / fps)
    return TrackObservation(
        frame_index=frame_index,
        raw_timestamp=raw,
        timestamp_utc=raw + timedelta(milliseconds=clock_offset_ms),
        detection=make_detection(bbox, confidence=confidence, touches_edge=touches_edge),
        crop=crop,
        sharpness=sharpness,
    )


def make_track(
    observations: Sequence[TrackObservation] | None = None,
    *,
    track_id: str = "cam_01-00000",
    camera_id: str = "cam_01",
    source_id: str = "cam_01_test",
    frame_size: tuple[int, int] = (160, 120),
    weights: BestFrameWeights | None = None,
) -> VehicleTrack:
    """Return a completed track, with its retained frames ranked.

    Args:
        observations: The track's history. Defaults to three ordinary frames.
        track_id: Its identifier.
        camera_id: Which camera saw it.
        source_id: Which source the frames came from.
        frame_size: ``(width, height)`` of the source frames.
        weights: Ranking weights, defaulting to the dataclass defaults.

    Returns:
        The track.
    """
    entries = (
        list(observations)
        if observations is not None
        else [make_observation(index) for index in range(3)]
    )
    ranked = rank_observations(
        [entry for entry in entries if entry.has_image],
        weights or BestFrameWeights(),
        frame_size,
    )
    return VehicleTrack(
        track_id=track_id,
        camera_id=camera_id,
        source_id=source_id,
        observations=tuple(entries),
        ranked=tuple(ranked),
        frame_size=frame_size,
    )


def scene_image(
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    width: int = 160,
    height: int = 120,
) -> npt.NDArray[np.uint8]:
    """Return a road-like scene with bright rectangles where vehicles would be.

    Deliberately noiseless. Random texture would be the honest way to imitate a
    sensor, and it also makes a PNG incompressible -- these images went from a
    few hundred bytes to 126 KB each with noise on. Nothing that uses them
    exercises a noise model, and a fixture directory that costs half a megabyte
    is one nobody wants in a repository.

    These are **synthetic**, not photographs. The reference blob detector finds
    the rectangles; a real model will not call them vehicles. See
    ``tests/fixtures/images/README.md``.

    Args:
        boxes: Where to draw rectangles, as ``(x1, y1, x2, y2)``.
        width: Image width.
        height: Image height.

    Returns:
        A BGR image.
    """
    image = np.full((height, width, 3), 55, dtype=np.uint8)

    # A lighter band across the middle, so the scene has some structure rather
    # than being one flat colour.
    image[height // 3 : 2 * height // 3] = 85

    for x1, y1, x2, y2 in boxes:
        image[y1:y2, x1:x2] = np.array([225, 228, 235], dtype=np.uint8)

    return image
