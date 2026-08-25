"""Asking a camera what time it thinks it is.

Where a device exposes its own clock -- ONVIF, an RTCP sender report, a vendor
API -- comparing it against the server's clock is a direct measurement, and a
direct measurement beats the inference in
:mod:`~multicam_tracker.timesync.estimation` on every count: it needs no
reference vehicle, no topology assumptions, and it works on a camera nothing has
driven past today.

So this is the preferred mechanism, and reference-event estimation is the
fallback for cameras that expose nothing.

The samples are kept as a series rather than a latest value, because one
comparison cannot tell a constant offset from a drifting clock -- and the
difference decides whether an operator corrects the camera once or replaces its
time client.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from multicam_tracker.clock import Clock, ensure_utc
from multicam_tracker.logging_config import get_logger

__all__ = ["ClockProbe", "ClockSample", "ClockSampleSeries", "probe_cameras"]

logger = get_logger(__name__)

_MS_PER_SECOND = 1000.0
_SECONDS_PER_HOUR = 3600.0
_MIN_SAMPLES_FOR_RATE = 3


@dataclass(frozen=True)
class ClockSample:
    """One comparison of a camera's reported time against the server's."""

    camera_id: str
    observed_at_utc: datetime
    """Server time when the probe ran."""

    camera_time_utc: datetime
    """What the camera said the time was."""

    round_trip_ms: float = 0.0
    """Probe round-trip. Half of it is the best available correction for
    transport delay, and its size bounds how precise the sample can be."""

    @property
    def offset_ms(self) -> float:
        """Return how far the camera's clock is ahead of the server's.

        Half the round trip is subtracted, on the usual assumption that the
        request and the response took equally long. That assumption is why a
        probe with a large round trip is a weak sample rather than a precise
        one.
        """
        raw = (self.camera_time_utc - self.observed_at_utc).total_seconds() * _MS_PER_SECOND
        return raw - self.round_trip_ms / 2.0


ClockProbe = Callable[[str], tuple[datetime, float]]
"""Asks one camera for its time.

Returns ``(camera_time_utc, round_trip_ms)``. Supplied by the caller because the
protocol differs per device, and stage 09 introduces no network dependency of
its own.
"""


@dataclass
class ClockSampleSeries:
    """A camera's clock comparisons over time."""

    camera_id: str
    samples: list[ClockSample] = field(default_factory=list)

    def record(self, sample: ClockSample) -> None:
        """Append one sample, keeping the series in observation order.

        Args:
            sample: The comparison to record.
        """
        self.samples.append(sample)
        self.samples.sort(key=lambda entry: entry.observed_at_utc)

    @property
    def latest_offset_ms(self) -> float | None:
        """Return the most recent measured offset, or ``None`` when empty."""
        return self.samples[-1].offset_ms if self.samples else None

    @property
    def verified_at_utc(self) -> datetime | None:
        """Return when this camera's clock was last measured."""
        return self.samples[-1].observed_at_utc if self.samples else None

    def drift_rate_ms_per_hour(self) -> float:
        """Return the fitted rate at which the offset is changing.

        Returns:
            Milliseconds per hour, positive when the camera is gaining. ``0.0``
            with fewer than three samples, where a line through the points says
            nothing.
        """
        if len(self.samples) < _MIN_SAMPLES_FOR_RATE:
            return 0.0

        origin = self.samples[0].observed_at_utc
        times = [
            (sample.observed_at_utc - origin).total_seconds() / _SECONDS_PER_HOUR
            for sample in self.samples
        ]
        values = [sample.offset_ms for sample in self.samples]

        mean_time = sum(times) / len(times)
        mean_value = sum(values) / len(values)
        variance = sum((time - mean_time) ** 2 for time in times)
        if variance <= 0.0:
            return 0.0

        covariance = sum(
            (time - mean_time) * (value - mean_value)
            for time, value in zip(times, values, strict=True)
        )
        return covariance / variance

    def suggested_offset_ms(self) -> int | None:
        """Return the correction to apply, rounded to whole milliseconds.

        Returns:
            The negation of the measured offset -- a camera running 800 ms fast
            needs 800 ms subtracted -- or ``None`` when nothing was measured.
        """
        latest = self.latest_offset_ms
        return None if latest is None else -round(latest)


def probe_cameras(
    camera_ids: Iterable[str],
    probe: ClockProbe,
    clock: Clock,
    *,
    series: dict[str, ClockSampleSeries] | None = None,
) -> dict[str, ClockSampleSeries]:
    """Ask each camera for its time and record the comparison.

    Args:
        camera_ids: Cameras to probe.
        probe: Callable that queries one camera.
        clock: Supplies server time at the moment of each probe.
        series: Existing series to append to. A fresh mapping is built when
            omitted.

    Returns:
        The series per camera, including those that failed -- a camera that
        could not be probed keeps whatever history it had rather than being
        dropped from the record.
    """
    collected = series if series is not None else {}

    for camera_id in camera_ids:
        entry = collected.setdefault(camera_id, ClockSampleSeries(camera_id=camera_id))
        observed_at = clock.now_utc()
        try:
            camera_time, round_trip_ms = probe(camera_id)
        except Exception as exc:
            # One unreachable camera must not abort a sweep of forty. The gap in
            # its series is what the staleness check later reads.
            logger.warning(
                "camera_clock_probe_failed",
                camera_id=camera_id,
                error=str(exc),
                detail="the camera keeps its previous samples and will read as stale",
            )
            continue

        entry.record(
            ClockSample(
                camera_id=camera_id,
                observed_at_utc=observed_at,
                camera_time_utc=ensure_utc(camera_time, field_name="camera_time_utc"),
                round_trip_ms=round_trip_ms,
            )
        )

    return collected
