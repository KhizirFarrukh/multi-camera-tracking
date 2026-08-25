"""Deciding how long to wait for a straggler.

A live stream does not deliver in order. Cameras buffer, networks retry, and a
sighting from 14:02 can arrive after one from 14:05. Something has to decide
when the system stops waiting and treats 14:02 as settled -- that decision is
the watermark, and it is a trade: wait longer and results are later; wait less
and more records arrive too late to use.

Two rules make it honest:

**The watermark never moves backward.** It is a promise that everything before
it has been accounted for, and a promise that can be withdrawn is not one.
A record arriving before the watermark does not drag it back; it is late, and
late is a state the system reports rather than hides.

**Nothing is dropped silently.** Every late record is counted and, under the
default policy, still delivered -- with the fact that it was late attached, so
the caller can decide whether to recompute. Silent data loss in a system whose
whole output is "here is everywhere the vehicle went" is the worst kind: the
answer looks complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from multicam_tracker.logging_config import get_logger

__all__ = ["LatePolicy", "WatermarkMetrics", "WatermarkTracker", "WatermarkVerdict"]

logger = get_logger(__name__)


class LatePolicy(StrEnum):
    """What to do with a record that arrives past the lateness allowance."""

    ACCEPT_AND_RECOMPUTE = "accept_and_recompute"
    """Take it and flag that whatever was built from the old data is stale.

    The default. A late sighting is still evidence, and discarding evidence
    because it was slow is a choice that should be made deliberately."""

    QUEUE = "queue"
    """Hold it for a separate late-arrivals path, out of the live stream."""

    DROP = "drop"
    """Discard it. Counted, logged, and never silent."""


class WatermarkVerdict(StrEnum):
    """What the tracker decided about one record."""

    ON_TIME = "on_time"
    """At or ahead of the watermark; ordinary."""

    BUFFERED = "buffered"
    """Behind the watermark but inside the allowance; reordered normally."""

    LATE = "late"
    """Past the allowance; the configured policy applies."""


@dataclass
class WatermarkMetrics:
    """Counters that make late and lost records visible."""

    accepted: int = 0
    buffered: int = 0
    late: int = 0
    dropped: int = 0
    queued: int = 0
    max_lateness_sec: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return the counters for a metrics exporter.

        Returns:
            A flat mapping of counter names to values.
        """
        return {
            "accepted": self.accepted,
            "buffered": self.buffered,
            "late": self.late,
            "dropped": self.dropped,
            "queued": self.queued,
            "max_lateness_sec": self.max_lateness_sec,
        }


@dataclass
class WatermarkTracker:
    """Tracks how far a stream has settled and judges each arrival.

    Args:
        lateness_allowance_sec: How far behind the watermark a record may arrive
            and still be reordered normally. Defaults to config.
        policy: What to do with records past the allowance.
    """

    lateness_allowance_sec: float | None = None
    policy: LatePolicy = LatePolicy.ACCEPT_AND_RECOMPUTE
    metrics: WatermarkMetrics = field(default_factory=WatermarkMetrics)

    _watermark: datetime | None = field(default=None, init=False)
    _highest_seen: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve the lateness allowance from config when not supplied."""
        if self.lateness_allowance_sec is None:
            from multicam_tracker.config import get_settings

            self.lateness_allowance_sec = get_settings().timesync.watermark_lateness_sec

    @property
    def watermark(self) -> datetime | None:
        """Return the current watermark, or ``None`` before the first record."""
        return self._watermark

    def observe(self, event_time_utc: datetime) -> tuple[WatermarkVerdict, float]:
        """Judge one arriving record and advance the watermark.

        Args:
            event_time_utc: When the record says it happened -- not when it
                arrived. The watermark tracks event time, because that is what
                downstream ordering is built on.

        Returns:
            ``(verdict, lateness_sec)``. Lateness is zero for on-time records
            and positive for anything behind the watermark.
        """
        allowance = float(self.lateness_allowance_sec or 0.0)

        if self._highest_seen is None or event_time_utc > self._highest_seen:
            self._highest_seen = event_time_utc

        candidate = self._highest_seen - timedelta(seconds=allowance)
        # Monotonic by construction: the watermark is the furthest it has ever
        # been, never the furthest it is right now.
        self._watermark = candidate if self._watermark is None else max(self._watermark, candidate)

        lateness = max(0.0, (self._watermark - event_time_utc).total_seconds())

        if lateness <= 0.0:
            self.metrics.accepted += 1
            return (WatermarkVerdict.ON_TIME, 0.0)

        self.metrics.max_lateness_sec = max(self.metrics.max_lateness_sec, lateness)

        if lateness <= allowance:
            self.metrics.buffered += 1
            return (WatermarkVerdict.BUFFERED, lateness)

        self.metrics.late += 1
        if self.policy is LatePolicy.DROP:
            self.metrics.dropped += 1
        elif self.policy is LatePolicy.QUEUE:
            self.metrics.queued += 1

        logger.warning(
            "stream_record_late",
            event_time_utc=event_time_utc.isoformat(),
            watermark_utc=self._watermark.isoformat(),
            lateness_sec=round(lateness, 3),
            policy=self.policy.value,
        )
        return (WatermarkVerdict.LATE, lateness)

    def accepts(self, verdict: WatermarkVerdict) -> bool:
        """Return whether a record with this verdict is passed downstream.

        Args:
            verdict: What :meth:`observe` decided.

        Returns:
            ``True`` unless the policy is to drop or queue a late record.
        """
        if verdict is not WatermarkVerdict.LATE:
            return True
        return self.policy is LatePolicy.ACCEPT_AND_RECOMPUTE
