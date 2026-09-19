"""The real detector: Ultralytics YOLO behind the project's interface.

Everything specific to Ultralytics is confined to this file. That is the point
of the interface -- when the model is replaced, and it will be, the change is
this module and the configuration, not the tracker, not the pipeline, not a
single test of either.

**Weights load on first use, never at import.** A test environment with no
weights must still be able to import this module, construct the detector, and
reason about configuration; only actually asking it to detect something should
fail. Loading in ``__init__`` would make the CI install of this package require
a 50 MB download to run a unit test about bounding-box arithmetic.

**A missing weights file names the path it looked for.** The operator's next
action is to put a file somewhere, and an error that does not say where is an
error that costs an hour.

**CUDA is requested, not assumed.** A machine configured for ``cuda`` that has
no CUDA runs on the CPU with a warning rather than crashing. Slow is a problem
somebody can see and plan around; a service that will not start at three in the
morning is a different kind of problem.

**Class filtering happens here, not downstream.** The COCO classes YOLO reports
include people, traffic lights, and handbags. Mapping the vehicle classes onto
:class:`~multicam_tracker.models.enums.ObjectClass` and discarding the rest at
the boundary means nothing downstream has to know what COCO is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multicam_tracker.config import Settings
from multicam_tracker.exceptions import VisionError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision.detector_protocol import BaseDetector, Detection

__all__ = ["COCO_VEHICLE_CLASSES", "YoloDetector"]

logger = get_logger(__name__)

COCO_VEHICLE_CLASSES: dict[int, ObjectClass] = {
    2: ObjectClass.CAR,
    3: ObjectClass.MOTORCYCLE,
    5: ObjectClass.BUS,
    7: ObjectClass.TRUCK,
}
"""COCO class indices this project treats as vehicles.

Bicycle (1) is deliberately absent: the domain model has no class for it, and
mapping it to ``car`` would put cyclists into vehicle trajectories. A detector
trained on a different taxonomy supplies its own map."""

_DIGEST_CHUNK = 1 << 20


class YoloDetector(BaseDetector):
    """Ultralytics YOLO, wrapped.

    Args:
        weights_path: The ``.pt`` file to load.
        device: ``cpu`` or ``cuda``. A ``cuda`` request on a machine without it
            falls back to the CPU.
        min_confidence: Detections below this are not reported at all.
        nms_iou: Overlap above which the model merges two boxes.
        class_map: COCO index to project class. Anything not in the map is
            discarded.
    """

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str = "cpu",
        min_confidence: float = 0.25,
        nms_iou: float = 0.45,
        class_map: dict[int, ObjectClass] | None = None,
    ) -> None:
        super().__init__(weights_path.stem, "unloaded")
        self.weights_path = weights_path
        self.requested_device = device
        self.min_confidence = min_confidence
        self.nms_iou = nms_iou
        self.class_map = dict(class_map or COCO_VEHICLE_CLASSES)
        self._model: Any | None = None
        self._device = device

    @classmethod
    def from_settings(cls, settings: Settings) -> YoloDetector:
        """Build the configured detector.

        Args:
            settings: Loaded settings.

        Returns:
            A detector pointed at the configured weights. Nothing is loaded
            yet.
        """
        vision = settings.vision
        return cls(
            vision.vehicle_detector_path,
            device=vision.device,
            min_confidence=vision.detection_min_confidence,
            nms_iou=vision.detection_nms_iou,
        )

    @property
    def is_loaded(self) -> bool:
        """Return whether the weights have been read yet."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Return the device inference will actually run on.

        Resolving this requires knowing whether CUDA exists, which requires
        torch, so it loads the model.

        Returns:
            ``cpu`` or ``cuda``.
        """
        self._ensure_loaded()
        return self._device

    @property
    def model_version(self) -> str:
        """Return the loaded model's version string.

        Asking a detector what version it is is a question only the loaded model
        can answer, so this loads the weights. The string combines the
        Ultralytics release with a digest of the weights file, because the
        library version alone does not distinguish two sets of weights and the
        digest alone does not explain how they are interpreted.

        Returns:
            Something like ``ultralytics-8.3.0/sha256:1f4c9a2b``.
        """
        self._ensure_loaded()
        return self._model_version

    def _ensure_loaded(self) -> None:
        """Load the weights if they are not loaded already.

        Raises:
            VisionError: If Ultralytics is not installed, the weights file is
                missing, or the model cannot be constructed.
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionError(
                "Ultralytics is required for YoloDetector; install the 'vision' extra",
                {"package": "ultralytics", "extra": "vision"},
            ) from exc

        if not self.weights_path.is_file():
            raise VisionError(
                "Vehicle detector weights not found",
                {
                    "path": str(self.weights_path),
                    "setting": "MCT_VISION__VEHICLE_DETECTOR_PATH",
                },
            )

        self._device = self._resolve_device(self.requested_device)

        try:
            model = YOLO(str(self.weights_path))
            model.to(self._device)
        except Exception as exc:  # the library raises bare exceptions
            raise VisionError(
                "Vehicle detector weights could not be loaded",
                {"path": str(self.weights_path), "reason": str(exc)},
            ) from exc

        self._model = model
        self._model_version = f"{self._ultralytics_version()}/sha256:{self._weights_digest()}"
        logger.info(
            "vehicle_detector_loaded",
            model_id=self.model_id,
            model_version=self._model_version,
            device=self._device,
            weights=str(self.weights_path),
        )

    def _resolve_device(self, requested: str) -> str:
        """Return the device to actually use.

        Args:
            requested: What configuration asked for.

        Returns:
            ``cuda`` when it was asked for and is available, ``cpu`` otherwise.
        """
        if requested != "cuda":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:  # pragma: no cover - torch ships with ultralytics
            pass

        logger.warning(
            "cuda_unavailable_falling_back_to_cpu",
            model_id=self.model_id,
            detail=(
                "inference will be slower; a service that starts slowly beats "
                "one that does not start"
            ),
        )
        return "cpu"

    @staticmethod
    def _ultralytics_version() -> str:
        """Return the installed Ultralytics version, for the provenance string.

        Returns:
            ``ultralytics-<version>``, or ``ultralytics-unknown`` when the
            package does not report one.
        """
        try:
            import ultralytics

            return f"ultralytics-{ultralytics.__version__}"
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            return "ultralytics-unknown"

    def _weights_digest(self) -> str:
        """Return a short digest of the weights file.

        Returns:
            The first sixteen hex characters of its SHA-256.
        """
        digest = hashlib.sha256()
        with self.weights_path.open("rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the vehicles found in one frame.

        Args:
            frame: The frame to search.

        Returns:
            Detections in the coordinate space of the supplied frame.
        """
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return the vehicles found in several frames, in one inference call.

        Batching exists for throughput and must not change the answer: a batched
        run and a sequential run over the same frames are required by test to
        produce identical detections.

        Args:
            frames: The frames to search.

        Returns:
            One list per input frame, in the same order.

        Raises:
            VisionError: If the weights cannot be loaded or inference fails.
        """
        if not frames:
            return []

        self._ensure_loaded()
        assert self._model is not None

        try:
            results = self._model.predict(
                [frame.image for frame in frames],
                conf=self.min_confidence,
                iou=self.nms_iou,
                classes=sorted(self.class_map),
                device=self._device,
                verbose=False,
            )
        except Exception as exc:  # the library raises bare exceptions
            raise VisionError(
                "Vehicle detection failed",
                {
                    "model_id": self.model_id,
                    "frames": len(frames),
                    "reason": str(exc),
                },
            ) from exc

        return [self._parse(result) for result in results]

    def _parse(self, result: Any) -> list[Detection]:
        """Convert one Ultralytics result into project detections.

        Args:
            result: One element of the library's result list.

        Returns:
            The vehicle detections it contains, with anything outside the class
            map discarded.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        detections: list[Detection] = []
        for corners, confidence, class_index in zip(coordinates, confidences, classes, strict=True):
            object_class = self.class_map.get(int(class_index))
            if object_class is None:
                continue
            x1, y1, x2, y2 = (max(0, round(value)) for value in corners)
            detections.append(
                Detection(
                    bbox=(x1, y1, max(x1, x2), max(y1, y2)),
                    object_class=object_class,
                    confidence=min(1.0, max(0.0, float(confidence))),
                    model_id=self.model_id,
                    model_version=self._model_version,
                )
            )
        return detections
