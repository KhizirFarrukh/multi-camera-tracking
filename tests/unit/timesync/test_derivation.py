"""Unit tests for frame-index to timestamp derivation.

The 29.97 fps case is the one that matters. Float accumulation over 100,000
frames drifts by seconds, and seconds are exactly the scale at which travel-time
plausibility is decided -- so the error does not blur a timestamp, it moves a
sighting into or out of a hop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multicam_tracker.exceptions import IngestError, ValidationError
from multicam_tracker.timesync import FrameTiming, derive_timestamp, fps_as_fraction

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
"""A fixed instant. No test here reads the real clock."""

_HYPOTHESIS = settings(max_examples=200, deadline=None)


def test_frame_zero__is_the_start_instant_exactly() -> None:
    """The anchor everything else is measured from."""
    assert derive_timestamp(START, 0, 30.0) == START


def test_constant_thirty_fps__advances_by_exactly_one_thirtieth_per_frame() -> None:
    """Hand-computable: frame 90 at 30 fps is three seconds in."""
    assert derive_timestamp(START, 90, 30.0) == START + timedelta(seconds=3)


def test_twenty_nine_ninety_seven__is_treated_as_30000_over_1001() -> None:
    """The rate broadcast equipment actually uses.

    Taking 29.97 literally accumulates about 3 ms per 100 frames, which is a
    minute and a half over a day of footage.
    """
    exact = START + timedelta(seconds=float(Fraction(100_000) / Fraction(30000, 1001)))

    derived = derive_timestamp(START, 100_000, 29.97)

    assert abs((derived - exact).total_seconds()) < 0.001


def test_rational_fps_over_a_hundred_thousand_frames__stays_within_a_millisecond() -> None:
    """The exit criterion, asserted against exact rational arithmetic.

    A float accumulation -- ``start + sum(1/fps for _ in range(n))`` -- fails
    this by orders of magnitude, which is why the module uses fractions.
    """
    rate = Fraction(30000, 1001)
    expected_seconds = Fraction(100_000) / rate

    derived = derive_timestamp(START, 100_000, rate)
    error_seconds = abs((derived - START).total_seconds() - float(expected_seconds))

    assert error_seconds < 0.001


def test_per_frame_millisecond_rounding__is_the_error_this_module_avoids() -> None:
    """Documents the failure mode rather than only asserting the fix works.

    The tempting implementation stores each frame's offset to the millisecond
    the contract keeps and adds them up. At 29.97 fps one frame is 33.3667 ms,
    rounds to 33, and loses 0.3667 ms *per frame* -- over 100,000 frames that is
    more than half a minute, which moves a sighting clean out of its hop.

    Rounding once at the end, as the module does, keeps the same output
    precision and none of the drift.
    """
    rate = Fraction(30000, 1001)
    exact_seconds = float(Fraction(100_000) / rate)

    per_frame_ms = round(float(1 / rate) * 1000)
    accumulated_seconds = per_frame_ms * 100_000 / 1000

    assert abs(accumulated_seconds - exact_seconds) > 30.0
    assert (
        abs((derive_timestamp(START, 100_000, rate) - START).total_seconds() - exact_seconds)
        < 0.001
    )


@pytest.mark.parametrize("fps", [Fraction(30000, 1001), Fraction(24, 1), Fraction(25, 1)])
def test_common_rational_rates__round_trip_through_the_converter(fps: Fraction) -> None:
    """A caller passing an exact fraction gets it back unchanged."""
    assert fps_as_fraction(fps) == fps


def test_presentation_timestamps__are_used_in_preference_to_the_nominal_rate() -> None:
    """A real recording varies its rate, and the nominal rate assumes it does not."""
    timing = FrameTiming([0.0, 0.5, 2.0, 2.25])

    # The nominal rate would put frame 2 at 2/30 s; the PTS says 2 seconds.
    assert derive_timestamp(START, 2, 30.0, timing=timing) == START + timedelta(seconds=2)


def test_dropped_frames__do_not_shift_later_timestamps_when_pts_is_available() -> None:
    """The failure that makes frame-index arithmetic dangerous on real files.

    Frames 2 and 3 were dropped by the encoder. Frame 2 in the decoded order is
    really the sixth frame in time, and the PTS says so; multiplying an index by
    a nominal rate would put every later sighting two frames early, forever.
    """
    timing = FrameTiming([0.0, 1 / 30, 5 / 30, 6 / 30])

    assert derive_timestamp(START, 2, 30.0, timing=timing) == START + timedelta(seconds=5 / 30)


def test_a_frame_beyond_the_recorded_pts__raises_rather_than_falling_back() -> None:
    """Mixing two timing models within one video is worse than refusing."""
    timing = FrameTiming([0.0, 0.5])

    with pytest.raises(IngestError, match="outside the recorded presentation"):
        derive_timestamp(START, 5, 30.0, timing=timing)


def test_presentation_timestamps_running_backwards__are_rejected() -> None:
    """A stream demuxed wrongly would produce a route that goes back in time."""
    with pytest.raises(IngestError, match="non-decreasing"):
        FrameTiming([0.0, 1.0, 0.5])


def test_empty_presentation_timestamps__are_rejected() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="at least one"):
        FrameTiming([])


@pytest.mark.parametrize("fps", [0.0, -30.0])
def test_a_zero_or_negative_frame_rate__raises(fps: float) -> None:
    """One divides by zero, the other runs time backwards."""
    with pytest.raises(IngestError, match="must be positive"):
        derive_timestamp(START, 10, fps)


def test_a_non_finite_frame_rate__raises() -> None:
    """Boundary: NaN would propagate into every derived instant."""
    with pytest.raises(IngestError, match="finite"):
        derive_timestamp(START, 10, float("nan"))


def test_a_negative_frame_index__raises() -> None:
    """There is no frame before the first one."""
    with pytest.raises(IngestError, match="non-negative"):
        derive_timestamp(START, -1, 30.0)


def test_neither_a_rate_nor_timings__raises() -> None:
    """Nothing to derive from, and guessing a rate would invent a timestamp."""
    with pytest.raises(IngestError, match="either a frame rate or presentation"):
        derive_timestamp(START, 10)


def test_a_naive_start_instant__is_rejected() -> None:
    """A naive datetime could be local, camera, or UTC time."""
    with pytest.raises(ValidationError, match="Naive datetime"):
        derive_timestamp(datetime(2026, 8, 10, 14, 0, 0), 10, 30.0)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@_HYPOTHESIS
@given(
    frame=st.integers(min_value=0, max_value=500_000),
    step=st.integers(min_value=1, max_value=1000),
)
def test_derivation__is_strictly_increasing_in_frame_index(frame: int, step: int) -> None:
    """Property: a later frame is never an earlier instant."""
    rate = Fraction(30000, 1001)

    assert derive_timestamp(START, frame, rate) < derive_timestamp(START, frame + step, rate)


@_HYPOTHESIS
@given(
    first=st.integers(min_value=0, max_value=200_000),
    delta=st.integers(min_value=0, max_value=200_000),
)
def test_derivation__composes_exactly(first: int, delta: int) -> None:
    """Property: deriving frame N equals deriving frame M then advancing N-M.

    This is what fails under float accumulation, and it is the reason the
    project can re-derive a timestamp anywhere without the answer moving.
    """
    rate = Fraction(30000, 1001)
    direct = derive_timestamp(START, first + delta, rate)

    intermediate = derive_timestamp(START, first, rate)
    stepped = derive_timestamp(intermediate, delta, rate)

    assert abs((direct - stepped).total_seconds()) <= 0.001
