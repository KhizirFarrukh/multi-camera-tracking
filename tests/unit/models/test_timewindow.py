"""Unit tests for :mod:`multicam_tracker.models.timewindow`.

The half-open ``[start, end)`` convention is the thing under test. It is what
lets consecutive windows tile the timeline without double-counting a sighting
that lands exactly on a boundary.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from multicam_tracker.models import TimeWindow
from tests.fixtures.factories import BASE_INSTANT

pytestmark = pytest.mark.unit

START = BASE_INSTANT
END = BASE_INSTANT + timedelta(minutes=10)


def _window(start_offset_min: float = 0.0, end_offset_min: float = 10.0) -> TimeWindow:
    """Build a window at minute offsets from the base instant.

    Args:
        start_offset_min: Minutes after the base instant for the start.
        end_offset_min: Minutes after the base instant for the end.

    Returns:
        A validated window.
    """
    return TimeWindow(
        start_utc=BASE_INSTANT + timedelta(minutes=start_offset_min),
        end_utc=BASE_INSTANT + timedelta(minutes=end_offset_min),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_time_window__end_after_start__constructs() -> None:
    """The happy path."""
    window = TimeWindow(start_utc=START, end_utc=END)

    assert window.duration_sec == pytest.approx(600.0)


@pytest.mark.parametrize("end_offset", [0, -1], ids=["equal", "inverted"])
def test_time_window__end_not_after_start__is_rejected(end_offset: int) -> None:
    """A zero-length half-open window contains nothing, so it is a caller bug."""
    with pytest.raises(ValidationError, match="strictly after"):
        TimeWindow(start_utc=START, end_utc=START + timedelta(minutes=end_offset))


def test_time_window__naive_bound__is_rejected() -> None:
    """No model accepts a naive datetime anywhere."""
    with pytest.raises(ValidationError, match="naive datetime"):
        TimeWindow(start_utc=START.replace(tzinfo=None), end_utc=END)


def test_time_window__non_utc_bounds__are_normalized() -> None:
    """An aware offset datetime is unambiguous, so it is converted, not rejected."""
    from datetime import timezone

    tashkent = timezone(timedelta(hours=5))
    window = TimeWindow(start_utc=START.astimezone(tashkent), end_utc=END.astimezone(tashkent))

    assert window.start_utc == START
    assert window.end_utc == END


# ---------------------------------------------------------------------------
# contains: inclusive start, exclusive end
# ---------------------------------------------------------------------------


def test_contains__instant_inside__is_true() -> None:
    """The ordinary case."""
    assert _window().contains(START + timedelta(minutes=5)) is True


def test_contains__exactly_the_start__is_true() -> None:
    """Boundary: the start is inclusive."""
    assert _window().contains(START) is True


def test_contains__exactly_the_end__is_false() -> None:
    """Boundary: the end is exclusive, which is what makes windows tile cleanly."""
    assert _window().contains(END) is False


@pytest.mark.parametrize("offset_min", [-1, 11], ids=["before", "after"])
def test_contains__instant_outside__is_false(offset_min: int) -> None:
    """Outside on either side."""
    assert _window().contains(START + timedelta(minutes=offset_min)) is False


def test_contains__naive_instant__is_rejected() -> None:
    """Comparing a naive instant against a UTC window would silently assume a zone."""
    with pytest.raises(ValueError, match="naive datetime"):
        _window().contains(START.replace(tzinfo=None))


def test_contains__aware_non_utc_instant__is_compared_correctly() -> None:
    """Comparison is by instant, not by wall-clock reading."""
    from datetime import timezone

    tashkent = timezone(timedelta(hours=5))

    assert _window().contains((START + timedelta(minutes=5)).astimezone(tashkent)) is True


# ---------------------------------------------------------------------------
# overlaps
# ---------------------------------------------------------------------------


def test_overlaps__partial_overlap__is_true() -> None:
    """The windows share a stretch in the middle."""
    assert _window(0, 10).overlaps(_window(5, 15)) is True


def test_overlaps__containment__is_true() -> None:
    """One window entirely inside the other, in both directions."""
    outer, inner = _window(0, 20), _window(5, 10)

    assert outer.overlaps(inner) is True
    assert inner.overlaps(outer) is True


def test_overlaps__identical_windows__is_true() -> None:
    """A window overlaps itself."""
    assert _window().overlaps(_window()) is True


def test_overlaps__adjacent_windows__is_false() -> None:
    """Boundary: one ends exactly as the other begins. Consistent with contains()."""
    assert _window(0, 10).overlaps(_window(10, 20)) is False
    assert _window(10, 20).overlaps(_window(0, 10)) is False


def test_overlaps__disjoint_windows__is_false() -> None:
    """Separated in time."""
    assert _window(0, 5).overlaps(_window(10, 20)) is False


def test_overlaps__single_instant_of_intersection__is_true() -> None:
    """The smallest true overlap: the second window starts one millisecond early."""
    first = _window(0, 10)
    second = TimeWindow(
        start_utc=END - timedelta(milliseconds=1), end_utc=END + timedelta(minutes=5)
    )

    assert first.overlaps(second) is True


def test_overlaps__is_symmetric() -> None:
    """Overlap is a symmetric relation."""
    a, b = _window(0, 10), _window(5, 15)

    assert a.overlaps(b) == b.overlaps(a)


def test_time_window__round_trip__preserves_the_bounds() -> None:
    """Windows travel in API payloads alongside trajectories."""
    window = _window()

    assert TimeWindow.from_json_dict(window.to_json_dict()) == window
