"""Replaying detections that were recorded once from a real run.

Between the scripted fake and the real model there is a gap: the fake proves the
tracker handles the cases a test imagined, and the model proves nothing
reproducible because its output changes with its weights. The fixture detector
closes that gap. It replays exactly what a real detector produced on the
committed sample clips, so downstream stages are exercised against real
detection patterns -- the ragged confidences, the frames where the vehicle is
missed, the spurious box on frame 12 -- with none of the cost or the variance.

This is what makes the stage 10 exit criterion survive contact with stage 11:
**every downstream stage can run in CI with no weights, no GPU, and no
network.**

**The recording names its own provenance.** A fixture file records which
detector produced it, at what version, and when. A replay therefore cannot
silently claim to be something it is not, and when the model is upgraded the
stale fixtures are identifiable rather than merely suspected.

**A source the fixture does not cover is an error, not an empty result.**
Replaying against a clip that was never recorded would return no detections for
every frame, which is indistinguishable from a road with no traffic on it -- and
a test asserting "no false positives" would pass triumphantly against nothing at
all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from multicam_tracker.exceptions import VisionError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision.detector_protocol import BaseDetector, Detection, Detector

__all__ = [
    "FIXTURE_FORMAT_VERSION",
    "FixtureDetector",
    "build_fixture_payload",
    "write_fixture",
]

logger = get_logger(__name__)

FIXTURE_FORMAT_VERSION = 1
"""Bumped when the on-disk shape changes, so an old file fails loudly."""

_BBOX_LENGTH = 4


def _detection_to_json(detection: Detection) -> dict[str, Any]:
    """Return one detection as a JSON-serializable mapping.

    The mask is deliberately not recorded. A per-pixel mask would dominate the
    file size of a fixture that exists to be small and committed, and nothing
    before stage 13 reads one.

    Args:
        detection: The detection to serialize.

    Returns:
        A plain mapping.
    """
    return {
        "bbox": list(detection.bbox),
        "object_class": detection.object_class.value,
        "confidence": round(detection.confidence, 6),
    }


def build_fixture_payload(
    detector: Detector,
    frames_by_source: Mapping[str, Sequence[Frame]],
    *,
    recorded_at: datetime,
    note: str = "",
) -> dict[str, Any]:
    """Run a detector over frames and return the recording.

    Args:
        detector: The detector to record. Its identifier and version are written
            into the file.
        frames_by_source: Frames to run, grouped by the source they came from.
        recorded_at: When the recording was made, from an injected clock.
        note: Free text explaining how the recording was produced. Written into
            the file because a fixture recorded from something other than the
            production model must say so where the next reader will see it.

    Returns:
        The payload, ready for :func:`write_fixture`.
    """
    sources: dict[str, dict[str, Any]] = {}

    for source_id, frames in frames_by_source.items():
        per_frame: dict[str, list[dict[str, Any]]] = {}
        for frame, detections in zip(frames, detector.detect_batch(list(frames)), strict=True):
            if detections:
                per_frame[str(frame.frame_index)] = [
                    _detection_to_json(detection) for detection in detections
                ]
        sources[source_id] = {
            "frame_count": len(frames),
            "frames": per_frame,
        }

    return {
        "format_version": FIXTURE_FORMAT_VERSION,
        "model_id": detector.model_id,
        "model_version": detector.model_version,
        "recorded_at": recorded_at.isoformat(),
        "note": note,
        "sources": sources,
    }


def write_fixture(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a fixture payload to disk.

    Sorted keys and a fixed indent, so re-recording an unchanged run produces a
    byte-identical file and a diff means the detections really changed.

    Args:
        path: Where to write it.
        payload: The payload from :func:`build_fixture_payload`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class FixtureDetector(BaseDetector):
    """Replays recorded detections.

    Args:
        payload: A recording, as produced by :func:`build_fixture_payload`.
        path: Where it was read from, for error messages.

    Raises:
        VisionError: If the payload is not a recording this version understands.
    """

    def __init__(self, payload: Mapping[str, Any], *, path: Path | None = None) -> None:
        version = payload.get("format_version")
        if version != FIXTURE_FORMAT_VERSION:
            raise VisionError(
                "Detection fixture format is not supported by this version",
                {
                    "found": version,
                    "expected": FIXTURE_FORMAT_VERSION,
                    "path": str(path) if path else None,
                },
            )

        super().__init__(
            str(payload.get("model_id", "unknown")),
            str(payload.get("model_version", "unknown")),
        )
        self.path = path
        self.note = str(payload.get("note", ""))
        self.recorded_at = str(payload.get("recorded_at", ""))
        self._sources: dict[str, dict[int, tuple[Detection, ...]]] = {}

        raw_sources = payload.get("sources", {})
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise VisionError(
                "Detection fixture contains no sources",
                {"path": str(path) if path else None},
            )

        for source_id, entry in raw_sources.items():
            frames = entry.get("frames", {}) if isinstance(entry, dict) else {}
            self._sources[str(source_id)] = {
                int(index): tuple(self._detection_from_json(item) for item in items)
                for index, items in frames.items()
            }

    @classmethod
    def from_file(cls, path: Path) -> FixtureDetector:
        """Load a recording from disk.

        Args:
            path: The fixture file.

        Returns:
            A detector replaying it.

        Raises:
            VisionError: If the file is missing or is not valid JSON.
        """
        if not path.is_file():
            raise VisionError(
                "Detection fixture not found; record one with scripts/record_detection_fixtures.py",
                {"path": str(path)},
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisionError(
                "Detection fixture could not be read",
                {"path": str(path), "reason": str(exc)},
            ) from exc
        return cls(payload, path=path)

    def _detection_from_json(self, item: Mapping[str, Any]) -> Detection:
        """Return one detection from its recorded form.

        Args:
            item: The recorded mapping.

        Returns:
            The detection, stamped with the recording's provenance.

        Raises:
            VisionError: If the record is malformed.
        """
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != _BBOX_LENGTH:
            raise VisionError(
                "Recorded detection has no usable bbox",
                {"item": dict(item), "path": str(self.path) if self.path else None},
            )
        return Detection(
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            object_class=ObjectClass(item.get("object_class", ObjectClass.UNKNOWN.value)),
            confidence=float(item.get("confidence", 0.0)),
            model_id=self.model_id,
            model_version=self.model_version,
        )

    @property
    def covered_sources(self) -> tuple[str, ...]:
        """Return the source identifiers this recording covers, sorted."""
        return tuple(sorted(self._sources))

    def detect(self, frame: Frame) -> list[Detection]:
        """Return what the recorded detector found in this frame.

        Args:
            frame: The frame to look up, by source and frame index.

        Returns:
            The recorded detections, or an empty list for a frame the recording
            covers but found nothing in.

        Raises:
            VisionError: If the recording does not cover this frame's source.
                Silently returning nothing would be indistinguishable from an
                empty road.
        """
        source = self._sources.get(frame.source_id)
        if source is None:
            raise VisionError(
                "This detection fixture does not cover that source; replaying "
                "against an unrecorded source would look like an empty road",
                {
                    "source_id": frame.source_id,
                    "covered": list(self.covered_sources),
                    "path": str(self.path) if self.path else None,
                },
            )
        return list(source.get(frame.frame_index, ()))
