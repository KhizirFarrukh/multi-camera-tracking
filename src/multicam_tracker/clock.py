"""Time abstraction.

Nothing in :mod:`multicam_tracker` calls :func:`datetime.datetime.now` directly.
Code that needs the current time accepts a :class:`Clock` and calls
:meth:`Clock.now_utc`. That indirection is what makes time-dependent behaviour
testable: production wires in :class:`SystemClock`, tests wire in
:class:`FixedClock` and get a deterministic instant with no patching, no
sleeping, and no flakiness.

This matters more here than in most projects. The entire system reasons about
elapsed time between camera sightings, so a test that cannot pin "now" cannot
assert anything meaningful about travel-time plausibility.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from multicam_tracker.exceptions import ValidationError

__all__ = ["Clock", "FixedClock", "SystemClock", "ensure_utc"]


def ensure_utc(value: datetime, *, field_name: str = "timestamp") -> datetime:
    """Return ``value`` converted to UTC, rejecting naive datetimes.

    All internal datetimes are timezone-aware UTC (global contract,
    ``cross_cutting_requirements.timezones``). A naive datetime is ambiguous --
    it could be local time, camera time, or UTC -- and silently assuming one
    would corrupt every downstream elapsed-time calculation. So it is rejected
    rather than guessed at.

    Args:
        value: The datetime to normalize.
        field_name: Name reported in the error context when ``value`` is naive.

    Returns:
        An aware datetime whose ``tzinfo`` is UTC.

    Raises:
        ValidationError: If ``value`` has no timezone information.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValidationError(
            "Naive datetime rejected; timezone-aware UTC is required",
            {"field": field_name, "value": value.isoformat()},
        )
    return value.astimezone(UTC)


@runtime_checkable
class Clock(Protocol):
    """Source of the current time.

    Implementations must return timezone-aware UTC datetimes.
    """

    def now_utc(self) -> datetime:
        """Return the current instant as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """Clock backed by the operating system clock. Use this in production."""

    def now_utc(self) -> datetime:
        """Return the current instant as a timezone-aware UTC datetime."""
        return datetime.now(UTC)

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return "SystemClock()"


class FixedClock:
    """Clock pinned to a configured instant. Use this in tests.

    The instant does not move on its own; :meth:`advance` and :meth:`set_to` move
    it explicitly, which lets a test simulate elapsed time without sleeping.

    Args:
        instant: The instant to report. Converted to UTC on construction.

    Raises:
        ValidationError: If ``instant`` is naive.
    """

    def __init__(self, instant: datetime) -> None:
        self._instant = ensure_utc(instant, field_name="instant")

    def now_utc(self) -> datetime:
        """Return the configured instant, unchanged on every call."""
        return self._instant

    def advance(self, delta: timedelta | float) -> datetime:
        """Move the clock forward (or backward, for a negative delta).

        Args:
            delta: A :class:`~datetime.timedelta`, or a number of seconds.

        Returns:
            The new instant, for convenient chaining in tests.
        """
        step = delta if isinstance(delta, timedelta) else timedelta(seconds=delta)
        self._instant = self._instant + step
        return self._instant

    def set_to(self, instant: datetime) -> datetime:
        """Jump the clock to ``instant``.

        Args:
            instant: The new instant. Converted to UTC.

        Returns:
            The new instant.

        Raises:
            ValidationError: If ``instant`` is naive.
        """
        self._instant = ensure_utc(instant, field_name="instant")
        return self._instant

    def __repr__(self) -> str:
        """Return an unambiguous representation including the pinned instant."""
        return f"FixedClock(instant={self._instant.isoformat()})"
