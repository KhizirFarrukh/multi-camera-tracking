"""Half-open time intervals.

Used by stage 04 for travel-time plausibility windows and by stage 08 when
bounding the search for the next sighting in a trajectory.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import model_validator

from multicam_tracker.models.base import MCTBaseModel, UtcDatetime

__all__ = ["TimeWindow"]


class TimeWindow(MCTBaseModel):
    """A half-open interval ``[start_utc, end_utc)``.

    The half-open convention is deliberate: it makes consecutive windows tile
    the timeline without overlapping, so a sighting at the exact boundary
    between two windows belongs to exactly one of them. A closed interval would
    place it in both and double-count it.
    """

    start_utc: UtcDatetime
    end_utc: UtcDatetime

    @model_validator(mode="after")
    def _end_follows_start(self) -> TimeWindow:
        """Reject empty and inverted windows.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``end_utc`` is not strictly after ``start_utc``. A
                zero-length half-open window contains nothing, so it is almost
                always a caller bug rather than an intent.
        """
        if self.end_utc <= self.start_utc:
            msg = (
                f"end_utc ({self.end_utc.isoformat()}) must be strictly after "
                f"start_utc ({self.start_utc.isoformat()})"
            )
            raise ValueError(msg)
        return self

    @property
    def duration_sec(self) -> float:
        """Return the window length in seconds."""
        return (self.end_utc - self.start_utc).total_seconds()

    def contains(self, moment: datetime) -> bool:
        """Return whether ``moment`` falls inside the window.

        Inclusive at the start, exclusive at the end.

        Args:
            moment: An aware datetime. Compared in UTC.

        Returns:
            ``True`` when ``start_utc <= moment < end_utc``.

        Raises:
            ValueError: If ``moment`` is naive.
        """
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            msg = f"contains() received a naive datetime ({moment.isoformat()})"
            raise ValueError(msg)
        return self.start_utc <= moment < self.end_utc

    def overlaps(self, other: TimeWindow) -> bool:
        """Return whether this window shares any instant with ``other``.

        Consistent with :meth:`contains`: two adjacent windows where one ends
        exactly as the other begins do **not** overlap.

        Args:
            other: The window to compare against.

        Returns:
            ``True`` when the intersection is non-empty.
        """
        return self.start_utc < other.end_utc and other.start_utc < self.end_utc
