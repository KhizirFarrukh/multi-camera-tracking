"""A detector that returns whatever you told it to.

Tracking logic is where the subtle bugs live -- identity swaps, tracks that
fragment, tracks that merge two vehicles into one -- and none of them can be
tested against a real model, because with a real model you cannot say what the
right answer is. Testing a tracker against YOLO measures YOLO.

So the tracking tests run against a script: "frame 7 contains a car at exactly
these coordinates". The expected track is then known exactly, and a failure
means the tracker is wrong rather than that the detector had a bad day.

**Provenance is restamped.** A detection handed in with some other model id is
re-labelled with this detector's, because the record of which model produced a
box has to be true. A fixture built from copied detections would otherwise claim
a lineage it does not have, and provenance that is sometimes a lie is provenance
nobody can use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from multicam_tracker.ingest.frame import Frame
from multicam_tracker.vision.detector_protocol import BaseDetector, Detection

__all__ = ["FakeDetector"]


class FakeDetector(BaseDetector):
    """Returns scripted detections for given frame indices.

    Args:
        script: Detections per frame index. A frame index absent from the
            script yields ``default``.
        default: What to return for an unscripted frame. Empty by default,
            which is the honest reading of "the script does not mention this
            frame".
        model_id: Recorded on every detection.
        model_version: Recorded on every detection.
    """

    def __init__(
        self,
        script: Mapping[int, Sequence[Detection]] | None = None,
        *,
        default: Sequence[Detection] = (),
        model_id: str = "fake-detector",
        model_version: str = "1.0",
    ) -> None:
        super().__init__(model_id, model_version)
        self._script: dict[int, tuple[Detection, ...]] = {
            int(index): tuple(detections) for index, detections in (script or {}).items()
        }
        self._default = tuple(default)
        self.call_count = 0
        """How many times :meth:`detect` has been called.

        The motion-prefilter test asserts on this: the saving stage 10 claims
        only exists if the detector really is invoked fewer times."""

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the detections scripted for this frame.

        Args:
            frame: The frame, of which only :attr:`Frame.frame_index` is used.

        Returns:
            The scripted detections, restamped with this detector's provenance.
        """
        self.call_count += 1
        scripted = self._script.get(frame.frame_index, self._default)
        return [
            replace(detection, model_id=self.model_id, model_version=self.model_version)
            for detection in scripted
        ]

    def scripted_frame_indices(self) -> tuple[int, ...]:
        """Return the frame indices the script covers, ascending.

        Returns:
            The indices, which a test uses to assert it scripted what it meant
            to.
        """
        return tuple(sorted(self._script))
