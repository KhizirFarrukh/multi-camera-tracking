"""Skipping frames where nothing changed.

This is typically the single largest compute saving in the whole pipeline. A
street camera at night sees the same empty road for hours; running a detector on
each of those frames costs the same as running it on a busy one and finds
nothing. Frame differencing costs microseconds and answers the only question
that matters first: *did anything move?*

**A forced-sample interval keeps the gate honest.** A vehicle parked in view is
stationary, and pure motion detection would stop reporting it -- so a frame is
let through at least every ``force_interval_sec`` however still the scene is.
Without that, a car that parks vanishes from the record, which reads exactly
like a car that drove away.

Differencing rather than MOG2 background subtraction, because this runs before
OpenCV is necessarily available and the marginal accuracy of a learned
background model is not worth a hard dependency at the gate. A camera that needs
one can have it later; the interface here does not change.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.logging_config import get_logger

__all__ = ["MotionGate", "MotionStats", "downsample_grey"]

logger = get_logger(__name__)

_GREY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)
"""BGR luminance weights. Frames are BGR, so blue comes first."""

_DOWNSAMPLE_TARGET = 64
"""Longest edge the comparison runs at.

Motion large enough to matter survives aggressive downsampling, and comparing
64-pixel images costs a thousandth of comparing 1080p ones. Downsampling also
suppresses sensor noise, which is what a naive full-resolution difference
mistakes for movement."""


@dataclass
class MotionStats:
    """What the gate did, for the metrics the saving is claimed from."""

    seen: int = 0
    passed: int = 0
    skipped: int = 0
    forced: int = 0
    """Frames passed only because the forced-sample interval elapsed."""

    @property
    def skip_ratio(self) -> float:
        """Return the share of frames the gate skipped.

        Returns:
            ``0.0`` when nothing has been seen, which is honest: a gate that has
            processed nothing has saved nothing.
        """
        return 0.0 if self.seen == 0 else self.skipped / self.seen

    def as_dict(self) -> dict[str, float]:
        """Return the counters for a metrics exporter.

        Returns:
            A flat mapping.
        """
        return {
            "seen": self.seen,
            "passed": self.passed,
            "skipped": self.skipped,
            "forced": self.forced,
            "skip_ratio": round(self.skip_ratio, 4),
        }


def downsample_grey(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Return a small greyscale version of an image.

    Strided slicing rather than an interpolating resize: this runs on every
    frame before anything else does, and the comparison does not need the
    quality that averaging would buy.

    Args:
        image: A BGR image.

    Returns:
        A greyscale float array whose longest edge is at most the downsample
        target.
    """
    height, width = image.shape[:2]
    step = max(1, max(height, width) // _DOWNSAMPLE_TARGET)
    reduced = image[::step, ::step]
    return reduced.astype(np.float32) @ _GREY_WEIGHTS


@dataclass
class MotionGate:
    """Passes frames that changed, and skips the rest.

    Args:
        sensitivity: Mean absolute difference, in greyscale levels, above which
            a frame counts as changed. Lower is more sensitive. The default is
            tuned above typical sensor noise and below a vehicle entering frame.
        force_interval_sec: Longest a frame may go unpassed however static the
            scene. This is what keeps a parked vehicle in the record.

    Raises:
        IngestError: If the sensitivity is negative.
    """

    sensitivity: float = 2.0
    force_interval_sec: float = 5.0
    stats: MotionStats = field(default_factory=MotionStats)
    _reference: npt.NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _last_passed_at: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the sensitivity.

        Raises:
            IngestError: If it is negative.
        """
        if self.sensitivity < 0:
            raise IngestError(
                "Motion sensitivity cannot be negative", {"sensitivity": self.sensitivity}
            )

    def difference(self, image: npt.NDArray[np.uint8]) -> float:
        """Return how much an image differs from the current reference.

        Args:
            image: The frame to compare.

        Returns:
            Mean absolute difference in greyscale levels. ``inf`` when there is
            no reference yet -- the first frame of a stream is always new.
        """
        reduced = downsample_grey(image)
        if self._reference is None or self._reference.shape != reduced.shape:
            return float("inf")
        return float(np.abs(reduced - self._reference).mean())

    def should_pass(self, frame: Frame) -> bool:
        """Decide whether one frame reaches the detector, and update state.

        Args:
            frame: The frame to judge.

        Returns:
            ``True`` when the frame changed enough, or when the forced-sample
            interval has elapsed.
        """
        self.stats.seen += 1
        change = self.difference(frame.image)
        moved = change >= self.sensitivity

        forced = False
        if not moved and self._last_passed_at is not None:
            elapsed = (frame.timestamp_utc - self._last_passed_at).total_seconds()
            forced = elapsed >= self.force_interval_sec

        if moved or forced or self._last_passed_at is None:
            self._reference = downsample_grey(frame.image)
            self._last_passed_at = frame.timestamp_utc
            self.stats.passed += 1
            if forced and not moved:
                self.stats.forced += 1
            return True

        self.stats.skipped += 1
        return False

    def filter(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Yield only the frames worth processing.

        Args:
            frames: Every decoded frame.

        Yields:
            The frames that changed, plus the periodic forced samples.
        """
        for frame in frames:
            if self.should_pass(frame):
                yield frame

        if self.stats.seen:
            logger.info(
                "motion_gate_summary",
                seen=self.stats.seen,
                passed=self.stats.passed,
                skipped=self.stats.skipped,
                forced=self.stats.forced,
                skip_ratio=round(self.stats.skip_ratio, 3),
            )

    def reset(self) -> None:
        """Forget the reference frame and the forced-sample timer.

        Called when a source reconnects: the scene may have changed entirely
        while it was away, and comparing against a stale reference would report
        motion that is really just a gap.
        """
        self._reference = None
        self._last_passed_at = None

    def seconds_until_forced(self, now_utc: datetime) -> float:
        """Return how long until a frame would be passed regardless of motion.

        Args:
            now_utc: The current frame's capture time.

        Returns:
            Seconds remaining, or ``0.0`` when a forced sample is already due.
        """
        if self._last_passed_at is None:
            return 0.0
        due = self._last_passed_at + timedelta(seconds=self.force_interval_sec)
        return max(0.0, (due - now_utc).total_seconds())
