"""What a detector is, and what it hands back.

One interface over every way of finding vehicles in a frame: a real model, a
scripted fake, a recorded replay. Nothing downstream knows which it is holding,
which is what lets stages 12-20 be tested in CI with no weights and no GPU, and
what lets a deployment swap YOLO for whatever replaces it without touching a
line of calling code.

**A detection's box is always in original source-frame coordinates.** This is
the rule the whole vision package inherits from stage 10, and it is worth
restating because a detector is exactly where it gets broken. Preprocessing
hands the model a masked, rotated, resized image; the model reports boxes in
*that* image. A box stored in processed-frame space is a perfectly plausible box
in the wrong part of the picture, and nothing downstream can tell -- the
thumbnail crops the wrong region, the reviewer sees the wrong car, and every
symptom reads as a detector problem rather than as arithmetic.

So a detector run on preprocessed frames must map its output back through the
:class:`~multicam_tracker.ingest.preprocess.CoordinateMapping` the preprocessor
returned, and :meth:`Detection.in_source_coordinates` is the one supported way
to do it.

**Provenance travels with the detection.** Every detection records which model
produced it and at what version. Six months later, when a trajectory is
questioned, "which model put this box here" is a question the record has to be
able to answer on its own -- and when a model is upgraded, it is the only way to
tell which stored sightings predate the change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import VisionError
from multicam_tracker.models.enums import ObjectClass

if TYPE_CHECKING:
    from multicam_tracker.ingest.frame import Frame
    from multicam_tracker.ingest.preprocess import CoordinateMapping

__all__ = [
    "BaseDetector",
    "Detection",
    "Detector",
    "iou",
]

_BBOX_LENGTH = 4


@dataclass(frozen=True)
class Detection:
    """One object found in one frame.

    Args:
        bbox: ``(x1, y1, x2, y2)`` in original source-frame pixel coordinates.
        object_class: What the detector thinks it is.
        confidence: How sure it is, in ``[0, 1]``.
        model_id: Which model produced this, for provenance.
        model_version: That model at which version.
        mask: Optional per-pixel segmentation, same size as the box. Kept
            optional because a box detector has none and most of the pipeline
            does not need one.
        touches_edge: Whether a filter judged this box to be cut off by the
            frame edge. Set by the keep-but-flag policy in
            :mod:`~multicam_tracker.vision.filters`; see that module for why the
            distinction matters.
    """

    bbox: tuple[int, int, int, int]
    object_class: ObjectClass
    confidence: float
    model_id: str
    model_version: str
    mask: npt.NDArray[np.bool_] | None = None
    touches_edge: bool = False

    def __post_init__(self) -> None:
        """Validate the box and the confidence.

        A zero-area box is permitted on purpose. It is what a model reports when
        it has latched onto nothing, and rejecting it here would move the
        failure from the filter that exists to count it (task 11.4) into an
        exception halfway through a batch.

        Raises:
            VisionError: If the box is inverted, has a negative coordinate, or
                the confidence is outside ``[0, 1]``.
        """
        if len(self.bbox) != _BBOX_LENGTH:
            raise VisionError("A detection box must be (x1, y1, x2, y2)", {"bbox": self.bbox})

        x1, y1, x2, y2 = self.bbox
        if any(coordinate < 0 for coordinate in self.bbox):
            raise VisionError(
                "Detection box coordinates must be non-negative",
                {"bbox": list(self.bbox), "model_id": self.model_id},
            )
        if x2 < x1 or y2 < y1:
            raise VisionError(
                "Detection box is inverted; x2 >= x1 and y2 >= y1 are required",
                {"bbox": list(self.bbox), "model_id": self.model_id},
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise VisionError(
                "Detection confidence must be in [0, 1]",
                {"confidence": self.confidence, "model_id": self.model_id},
            )

    @property
    def width(self) -> int:
        """Return the box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Return the box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Return the box area in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Return width divided by height.

        Returns:
            The ratio, or ``0.0`` for a box of zero height -- which is
            degenerate rather than infinitely wide, and reporting infinity would
            put it outside every configured window for the wrong reason.
        """
        return 0.0 if self.height == 0 else self.width / self.height

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the box centre as ``(x, y)``."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def in_source_coordinates(self, mapping: CoordinateMapping) -> Detection:
        """Return this detection with its box mapped back to source coordinates.

        Call this on every detection produced from a preprocessed frame, before
        anything else touches it. See the module docstring for what goes wrong
        otherwise.

        Args:
            mapping: The mapping the preprocessor returned alongside the
                processed image.

        Returns:
            A detection identical but for the box, with any coordinate the
            mapping put at -1 pulled back to 0.

            That one-pixel clamp is not papering over an error. Stage 10's
            mapping works in pixel *centres*, where the rotation inverse is
            ``width - 1 - x``; a box edge legitimately sits at ``x = width``,
            because box coordinates are half-open. Mapping that edge back gives
            -1. It happens only at the frame boundary, only ever by one pixel,
            and only for a rotated camera -- and it was found by running a real
            rotated clip through the chain, not by reading the arithmetic.
            Anything larger than one pixel is a genuine mapping fault and is
            left to fail against the non-negativity check.
        """
        x1, y1, x2, y2 = mapping.bbox_to_source(list(self.bbox))
        return replace(self, bbox=(max(0, x1), max(0, y1), max(0, x2), max(0, y2)))

    def clamped_to(self, width: int, height: int) -> Detection:
        """Return this detection with its box clipped to a frame.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            A detection whose box lies inside ``[0, width] x [0, height]``. A
            box entirely outside collapses to zero area rather than raising: the
            filters count it, which is more useful than an exception that says
            nothing about how often it happens.
        """
        x1, y1, x2, y2 = self.bbox
        clipped = (
            min(max(x1, 0), width),
            min(max(y1, 0), height),
            min(max(min(x2, width), 0), width),
            min(max(min(y2, height), 0), height),
        )
        ordered = (
            clipped[0],
            clipped[1],
            max(clipped[0], clipped[2]),
            max(clipped[1], clipped[3]),
        )
        return replace(self, bbox=ordered)


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Return the intersection-over-union of two boxes.

    The association metric the tracker runs on, and the only geometric
    similarity in this package. Defined here rather than in the tracker because
    the filters and the tests need it too, and two implementations of it would
    eventually disagree.

    Args:
        first: ``(x1, y1, x2, y2)``.
        second: ``(x1, y1, x2, y2)``.

    Returns:
        A value in ``[0, 1]``. Zero when either box is degenerate, which keeps a
        collapsed box from associating with everything through a zero-over-zero.
    """
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    overlap_width = min(ax2, bx2) - max(ax1, bx1)
    overlap_height = min(ay2, by2) - max(ay1, by1)
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0

    intersection = float(overlap_width * overlap_height)
    union = float((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)) - intersection
    return 0.0 if union <= 0.0 else intersection / union


@runtime_checkable
class Detector(Protocol):
    """Finds objects in frames.

    Implementations are interchangeable through configuration (see
    :func:`~multicam_tracker.vision.factory.create_detector`). Nothing that
    consumes detections may depend on which one it holds.
    """

    @property
    def model_id(self) -> str:
        """Return the identifier recorded on every detection this produces."""
        ...

    @property
    def model_version(self) -> str:
        """Return the version recorded on every detection this produces."""
        ...

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the objects found in one frame.

        Args:
            frame: The frame to search.

        Returns:
            Detections in the coordinate space of the supplied frame.
        """
        ...

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return the objects found in several frames.

        Args:
            frames: The frames to search.

        Returns:
            One list per input frame, in the same order.
        """
        ...


class BaseDetector:
    """Shared machinery for detector implementations.

    Supplies :meth:`detect_batch` by looping, so an implementation only has to
    write :meth:`detect`. A model that genuinely batches -- YOLO does -- overrides
    it, and the two are required by test to agree: a batched run and a
    sequential run over the same frames must produce identical detections, or
    batching has quietly changed the answer.

    Args:
        model_id: Identifier recorded on every detection.
        model_version: Version recorded on every detection.
    """

    def __init__(self, model_id: str, model_version: str) -> None:
        self._model_id = model_id
        self._model_version = model_version

    @property
    def model_id(self) -> str:
        """Return the identifier recorded on every detection this produces."""
        return self._model_id

    @property
    def model_version(self) -> str:
        """Return the version recorded on every detection this produces."""
        return self._model_version

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the objects found in one frame.

        Args:
            frame: The frame to search.

        Returns:
            Detections in the coordinate space of the supplied frame.

        Raises:
            NotImplementedError: Always; subclasses implement this.
        """
        raise NotImplementedError("Detector subclasses must implement detect()")

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return the objects found in several frames, one frame at a time.

        Args:
            frames: The frames to search.

        Returns:
            One list per input frame, in the same order.
        """
        return [self.detect(frame) for frame in frames]
