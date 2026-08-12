"""Unit tests for :mod:`multicam_tracker.topology.plausibility`.

Every boundary is asserted at its exact value. A window whose ends are off by
one second produces routes that look entirely reasonable and are wrong, which is
the failure mode this stage exists to prevent.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.topology import (
    PlausibilityReason,
    is_same_pass,
    is_transition_plausible,
    plausibility_score,
)
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

MIN_TRAVEL = 60.0
MAX_TRAVEL = 300.0

TOPOLOGY = make_topology(
    ["cam_a", "cam_b", "cam_lonely"],
    [("cam_a", "cam_b", MIN_TRAVEL, MAX_TRAVEL)],
)

_HYPOTHESIS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# is_transition_plausible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elapsed", [MIN_TRAVEL, 180.0, MAX_TRAVEL], ids=["at-min", "middle", "at-max"]
)
def test_is_plausible__inside_the_window__is_plausible(elapsed: float) -> None:
    """Both ends are inclusive, so a hop is never rejected on float equality."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", elapsed)

    assert verdict.plausible is True
    assert verdict.reason is PlausibilityReason.WITHIN_WINDOW
    assert verdict.margin_sec == 0.0


def test_is_plausible__one_second_below_the_minimum__is_too_fast_with_margin_one() -> None:
    """Boundary: one second under, and the margin says exactly how far under."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", MIN_TRAVEL - 1)

    assert verdict.plausible is False
    assert verdict.reason is PlausibilityReason.TOO_FAST
    assert verdict.margin_sec == pytest.approx(1.0)


def test_is_plausible__one_second_above_the_maximum__is_too_slow_with_margin_one() -> None:
    """Boundary on the other end."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", MAX_TRAVEL + 1)

    assert verdict.plausible is False
    assert verdict.reason is PlausibilityReason.TOO_SLOW
    assert verdict.margin_sec == pytest.approx(1.0)


def test_is_plausible__far_above_the_maximum__reports_the_full_margin() -> None:
    """The margin is how far outside, not merely that it is outside."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", MAX_TRAVEL + 240)

    assert verdict.margin_sec == pytest.approx(240.0)


def test_is_plausible__unlinked_pair__is_no_link() -> None:
    """No declared route means the topology cannot explain the hop at all."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_lonely", 120.0)

    assert verdict.reason is PlausibilityReason.NO_LINK
    assert verdict.margin_sec == 0.0


def test_is_plausible__reverse_of_a_one_way_link__is_no_link() -> None:
    """Direction is load-bearing: the reverse of a one-way link does not exist."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_b", "cam_a", 120.0)

    assert verdict.reason is PlausibilityReason.NO_LINK


def test_is_plausible__same_camera__is_reported_distinctly() -> None:
    """A repeat detection is not a transition, and must not be scored as one."""
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_a", 30.0)

    assert verdict.reason is PlausibilityReason.SAME_CAMERA
    assert verdict.plausible is False


def test_is_plausible__negative_elapsed__is_implausible_rather_than_raising() -> None:
    """Documented choice.

    Out-of-order timestamps mean bad data, but path reconstruction evaluates
    thousands of candidate pairs and one nonsense pair must not abort the
    search. It is reported as too_fast -- it "arrived before it left" -- with
    the margin measuring how far backwards it ran.
    """
    verdict = is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", -30.0)

    assert verdict.plausible is False
    assert verdict.reason is PlausibilityReason.TOO_FAST
    assert verdict.margin_sec == pytest.approx(MIN_TRAVEL + 30.0)


def test_is_plausible__zero_elapsed__is_too_fast() -> None:
    """Boundary: simultaneous sightings on two cameras are not a transit."""
    assert is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", 0.0).plausible is False


def test_plausibility_result__str__is_human_readable() -> None:
    """The verdict appears in operator-facing explanations."""
    rendered = str(is_transition_plausible(TOPOLOGY, "cam_a", "cam_b", 120.0))

    assert "plausible" in rendered
    assert "within_window" in rendered


# ---------------------------------------------------------------------------
# plausibility_score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elapsed", [MIN_TRAVEL, 100.0, 200.0, MAX_TRAVEL], ids=["min", "low", "high", "max"]
)
def test_score__inside_the_window__is_exactly_one(elapsed: float) -> None:
    """Anywhere inside the window is equally plausible; the window is the model."""
    assert plausibility_score(TOPOLOGY, "cam_a", "cam_b", elapsed) == 1.0


def test_score__decays_monotonically_as_it_gets_slower() -> None:
    """Further outside is less plausible, without a cliff."""
    scores = [
        plausibility_score(TOPOLOGY, "cam_a", "cam_b", MAX_TRAVEL + margin)
        for margin in (0, 30, 60, 120, 600)
    ]

    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
    assert scores[-1] < scores[1]


def test_score__decays_monotonically_as_it_gets_faster() -> None:
    """The decay is symmetric: arriving impossibly early is equally suspicious."""
    scores = [
        plausibility_score(TOPOLOGY, "cam_a", "cam_b", MIN_TRAVEL - margin)
        for margin in (0, 30, 60, 120)
    ]

    assert scores == sorted(scores, reverse=True)


def test_score__at_one_half_life_outside__is_one_half() -> None:
    """The half-life is the documented shape of the decay, not a vague 'falls off'."""
    score = plausibility_score(TOPOLOGY, "cam_a", "cam_b", MAX_TRAVEL + 120.0, half_life_sec=120.0)

    assert score == pytest.approx(0.5)


def test_score__never_reaches_zero_for_a_linked_pair() -> None:
    """A hop far outside its window is very unlikely, not impossible.

    The contract requires an implausible hop to be penalised rather than
    discarded, and a hard zero would discard it.
    """
    score = plausibility_score(TOPOLOGY, "cam_a", "cam_b", MAX_TRAVEL + 600, half_life_sec=120.0)

    assert 0.0 < score < 0.05


def test_score__unlinked_pair__returns_the_configured_floor_rather_than_raising() -> None:
    """An undeclared route is unexplained, not proven impossible."""
    score = plausibility_score(TOPOLOGY, "cam_a", "cam_lonely", 120.0, unlinked_score=0.1)

    assert score == pytest.approx(0.1)


def test_score__same_camera__returns_the_floor() -> None:
    """Same-camera repeats are handled by is_same_pass, not by hop scoring."""
    assert plausibility_score(TOPOLOGY, "cam_a", "cam_a", 5.0, unlinked_score=0.1) == pytest.approx(
        0.1
    )


@pytest.mark.parametrize(
    "elapsed",
    [-1e12, -1.0, 0.0, 1e12, float("inf")],
    ids=["huge-negative", "negative", "zero", "huge", "infinite"],
)
def test_score__extreme_elapsed_values__stay_within_the_unit_interval(elapsed: float) -> None:
    """Boundary: the score feeds a confidence product and must remain a probability."""
    score = plausibility_score(TOPOLOGY, "cam_a", "cam_b", elapsed)

    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


@_HYPOTHESIS
@given(elapsed=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_score__is_always_within_the_unit_interval(elapsed: float) -> None:
    """Property: no input produces a value outside [0, 1]."""
    assert 0.0 <= plausibility_score(TOPOLOGY, "cam_a", "cam_b", elapsed) <= 1.0


# ---------------------------------------------------------------------------
# Same-camera semantics
# ---------------------------------------------------------------------------


def test_is_same_pass__within_the_gap__is_one_pass() -> None:
    """A camera seeing a vehicle in consecutive frames observed one transit."""
    assert is_same_pass(5.0, min_redetection_gap_sec=60.0) is True


def test_is_same_pass__beyond_the_gap__is_two_transits() -> None:
    """A vehicle that circles the block and returns has made two."""
    assert is_same_pass(90.0, min_redetection_gap_sec=60.0) is False


def test_is_same_pass__exactly_at_the_gap__is_a_separate_transit() -> None:
    """Boundary, matching the half-open convention used for time windows."""
    assert is_same_pass(60.0, min_redetection_gap_sec=60.0) is False


def test_is_same_pass__negative_gap__is_judged_by_magnitude() -> None:
    """Whether two observations are one pass does not depend on their order."""
    assert is_same_pass(-5.0, min_redetection_gap_sec=60.0) is True


def test_is_same_pass__uses_the_configured_default_when_unspecified() -> None:
    """Every later stage reads the gap from here rather than picking its own."""
    assert is_same_pass(1.0) is True
    assert is_same_pass(10_000.0) is False
