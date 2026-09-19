"""Measuring what the detection pipeline actually costs and actually finds.

Three questions this answers, all of which someone eventually has to:

**How many cameras will this machine run?** That is a throughput question, and
the answer is not a single number -- it is a breakdown. A pipeline spending 80%
of its time decoding needs different hardware from one spending 80% in the
model, and "12 fps" alone does not distinguish them. So timing is recorded per
stage: decode, detect, track, crop.

**Is the detector still working?** A model that has quietly stopped finding
trucks shows up as a class histogram that changed, long before anyone notices a
missing trajectory.

**Where is the data going?** Filter rejections by reason, from
:class:`~multicam_tracker.vision.filters.FilterStats`, plus tracks discarded
before confirmation. A pipeline discarding most of its detections may be
correctly rejecting noise or may be misconfigured, and those look identical from
the outside without these counters.

Everything here is a plain counter with an :meth:`as_dict`, ready for the
metrics system in stage 19. Deliberately no dependency on a metrics library: the
measurement should not have to wait for the exporter.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from multicam_tracker.models.enums import ObjectClass

__all__ = ["DetectionMetrics", "StageTimer"]


@dataclass
class StageTimer:
    """Accumulated wall time per named stage.

    Wall time rather than CPU time on purpose: the question is how many frames
    per second a machine delivers, and a decoder waiting on disk is time the
    pipeline really spent.
    """

    totals: dict[str, float] = field(default_factory=dict)
    calls: Counter[str] = field(default_factory=Counter)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time a block and add it to one stage.

        Args:
            stage: The stage name, such as ``decode`` or ``detect``.

        Yields:
            ``None``; the measurement is the effect. The timer is stopped in a
            ``finally``, so a stage that raises still records the time it burned
            -- which is exactly the stage you want timings for.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.totals[stage] = self.totals.get(stage, 0.0) + (time.perf_counter() - started)
            self.calls[stage] += 1

    def total_sec(self, stage: str) -> float:
        """Return the accumulated seconds for one stage.

        Args:
            stage: The stage name.

        Returns:
            Seconds, or ``0.0`` for a stage that never ran.
        """
        return float(self.totals.get(stage, 0.0))

    def mean_ms(self, stage: str) -> float:
        """Return the mean milliseconds per call for one stage.

        Args:
            stage: The stage name.

        Returns:
            Milliseconds, or ``0.0`` for a stage that never ran.
        """
        calls = self.calls.get(stage, 0)
        return 0.0 if calls == 0 else self.total_sec(stage) * 1000.0 / calls

    def as_dict(self) -> dict[str, float]:
        """Return per-stage totals and means.

        Returns:
            A flat mapping with ``<stage>_total_sec`` and ``<stage>_mean_ms``
            for every stage that ran.
        """
        flat: dict[str, float] = {}
        for stage in sorted(self.totals):
            flat[f"{stage}_total_sec"] = round(self.total_sec(stage), 6)
            flat[f"{stage}_mean_ms"] = round(self.mean_ms(stage), 4)
        return flat


@dataclass
class DetectionMetrics:
    """Throughput and yield for one detection run.

    Args:
        source_id: Which source this describes.
        timer: Per-stage timings.
    """

    source_id: str = ""
    timer: StageTimer = field(default_factory=StageTimer)
    frames_read: int = 0
    frames_detected_on: int = 0
    detections_by_class: Counter[str] = field(default_factory=Counter)
    thumbnails_written: int = 0

    def record_frame(self, *, ran_detector: bool) -> None:
        """Count one frame off the source.

        Args:
            ran_detector: Whether the detector was actually invoked, or the
                frame was skipped by the stage 10 motion gate. The gap between
                the two numbers is the saving that gate claims, which is why it
                is counted rather than asserted.
        """
        self.frames_read += 1
        if ran_detector:
            self.frames_detected_on += 1

    def record_detection(self, object_class: ObjectClass) -> None:
        """Count one detection under its class.

        Args:
            object_class: What the detector called it.
        """
        self.detections_by_class[object_class.value] += 1

    @property
    def detections_total(self) -> int:
        """Return how many detections were reported, across all classes."""
        return int(sum(self.detections_by_class.values()))

    @property
    def prefilter_skip_ratio(self) -> float:
        """Return the share of frames the detector never saw.

        Returns:
            ``0.0`` when no frames have been read.
        """
        if self.frames_read == 0:
            return 0.0
        return (self.frames_read - self.frames_detected_on) / self.frames_read

    @property
    def frames_per_second(self) -> float:
        """Return end-to-end throughput, in frames read per wall second.

        Measured against the total of every timed stage rather than against a
        clock read at the start and end, so a caller that times only part of its
        loop gets a number describing the part it timed rather than one silently
        including everything else it was doing.

        Returns:
            Frames per second, or ``0.0`` when nothing has been timed.
        """
        elapsed = float(sum(self.timer.totals.values()))
        return 0.0 if elapsed <= 0.0 else self.frames_read / elapsed

    def as_dict(self) -> dict[str, float]:
        """Return every counter flattened for a metrics exporter.

        Returns:
            A flat mapping, with one ``detections_<class>`` key per class that
            was actually seen.
        """
        flat: dict[str, float] = {
            "frames_read": self.frames_read,
            "frames_detected_on": self.frames_detected_on,
            "prefilter_skip_ratio": round(self.prefilter_skip_ratio, 4),
            "detections_total": self.detections_total,
            "thumbnails_written": self.thumbnails_written,
            "frames_per_second": round(self.frames_per_second, 3),
        }
        for object_class, count in sorted(self.detections_by_class.items()):
            flat[f"detections_{object_class}"] = count
        flat.update(self.timer.as_dict())
        return flat
