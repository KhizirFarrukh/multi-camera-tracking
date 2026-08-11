"""Unit tests for :mod:`multicam_tracker.clock`.

Every elapsed-time calculation in the system -- travel-time plausibility, clock
drift correction, retention TTLs -- reads the current time through this module.
If ``FixedClock`` drifted, or if a naive datetime slipped through, the failure
would surface far downstream as an inexplicably implausible hop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from multicam_tracker.clock import Clock, FixedClock, SystemClock, ensure_utc
from multicam_tracker.exceptions import ValidationError

pytestmark = pytest.mark.unit

INSTANT = datetime(2026, 8, 10, 14, 22, 11, 500000, tzinfo=UTC)


# ---------------------------------------------------------------------------
# SystemClock
# ---------------------------------------------------------------------------


def test_system_clock__now_utc__returns_timezone_aware_utc() -> None:
    """Production time is always aware and always UTC."""
    now = SystemClock().now_utc()

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_system_clock__consecutive_calls__do_not_go_backwards() -> None:
    """A monotonic-enough guarantee for ordering sightings within a run."""
    clock = SystemClock()

    first = clock.now_utc()
    second = clock.now_utc()

    assert second >= first


def test_system_clock__satisfies_the_clock_protocol() -> None:
    """Dependency injection sites can type against the protocol."""
    assert isinstance(SystemClock(), Clock)


# ---------------------------------------------------------------------------
# FixedClock
# ---------------------------------------------------------------------------


def test_fixed_clock__repeated_calls__return_the_exact_configured_instant() -> None:
    """The whole point: time does not move unless a test moves it."""
    clock = FixedClock(INSTANT)

    assert clock.now_utc() == INSTANT
    assert clock.now_utc() == INSTANT
    assert clock.now_utc() == INSTANT


def test_fixed_clock__advance_timedelta__reflects_the_new_instant() -> None:
    """Simulating elapsed time needs no sleeping."""
    clock = FixedClock(INSTANT)

    returned = clock.advance(timedelta(minutes=5))

    assert returned == INSTANT + timedelta(minutes=5)
    assert clock.now_utc() == INSTANT + timedelta(minutes=5)


def test_fixed_clock__advance_seconds__accepts_a_plain_number() -> None:
    """Travel times are expressed in seconds throughout the contract."""
    clock = FixedClock(INSTANT)

    clock.advance(90)

    assert clock.now_utc() == INSTANT + timedelta(seconds=90)


def test_fixed_clock__advance_negative__moves_backwards() -> None:
    """Negative drift correction must be expressible."""
    clock = FixedClock(INSTANT)

    clock.advance(timedelta(seconds=-30))

    assert clock.now_utc() == INSTANT - timedelta(seconds=30)


def test_fixed_clock__advance_zero__leaves_the_instant_unchanged() -> None:
    """Boundary: a zero step is a no-op, not an error."""
    clock = FixedClock(INSTANT)

    clock.advance(0)

    assert clock.now_utc() == INSTANT


def test_fixed_clock__set_to__jumps_to_the_new_instant() -> None:
    """Absolute repositioning, converted to UTC on the way in."""
    clock = FixedClock(INSTANT)
    target = datetime(2026, 8, 11, 9, 0, tzinfo=timezone(timedelta(hours=5)))

    clock.set_to(target)

    assert clock.now_utc() == target.astimezone(UTC)
    assert clock.now_utc().tzinfo is UTC


def test_fixed_clock__non_utc_instant__is_converted_not_rejected() -> None:
    """An aware non-UTC instant is unambiguous, so it is normalized."""
    tashkent = timezone(timedelta(hours=5))
    clock = FixedClock(datetime(2026, 8, 10, 19, 22, 11, 500000, tzinfo=tashkent))

    assert clock.now_utc() == INSTANT


def test_fixed_clock__naive_instant__is_rejected() -> None:
    """A naive datetime could be local, camera, or UTC time. Guessing corrupts data."""
    with pytest.raises(ValidationError) as excinfo:
        FixedClock(datetime(2026, 8, 10, 14, 22, 11))

    assert excinfo.value.context["field"] == "instant"


def test_fixed_clock__set_to_naive__is_rejected() -> None:
    """The rejection applies on every entry point, not just construction."""
    clock = FixedClock(INSTANT)

    with pytest.raises(ValidationError):
        clock.set_to(datetime(2026, 8, 10, 14, 22, 11))


def test_fixed_clock__satisfies_the_clock_protocol() -> None:
    """A FixedClock is substitutable wherever a Clock is required."""
    assert isinstance(FixedClock(INSTANT), Clock)


def test_fixed_clock__repr__shows_the_pinned_instant() -> None:
    """Failure output should say what time the test thought it was."""
    assert repr(FixedClock(INSTANT)) == "FixedClock(instant=2026-08-10T14:22:11.500000+00:00)"


def test_system_clock__repr__is_stable() -> None:
    """A stable repr keeps failure output readable."""
    assert repr(SystemClock()) == "SystemClock()"


# ---------------------------------------------------------------------------
# ensure_utc
# ---------------------------------------------------------------------------


def test_ensure_utc__aware_non_utc__converts_to_utc() -> None:
    """Offsets are normalized rather than stripped."""
    tashkent = timezone(timedelta(hours=5))
    value = datetime(2026, 8, 10, 19, 22, 11, 500000, tzinfo=tashkent)

    assert ensure_utc(value) == INSTANT


def test_ensure_utc__already_utc__returns_an_equal_value() -> None:
    """Idempotency: normalizing twice changes nothing."""
    assert ensure_utc(ensure_utc(INSTANT)) == INSTANT


def test_ensure_utc__naive__raises_with_the_field_name() -> None:
    """The error names the field so the caller knows which input to fix."""
    with pytest.raises(ValidationError) as excinfo:
        ensure_utc(datetime(2026, 8, 10, 14, 22, 11), field_name="timestamp_utc")

    assert excinfo.value.context["field"] == "timestamp_utc"
    assert "Naive datetime" in str(excinfo.value)
