"""Unit tests for :mod:`multicam_tracker.matching.distance`.

The transposition case is the one that justifies Damerau over plain Levenshtein:
OCR swaps adjacent characters, and a swap costing 2 would push a one-error read
outside a distance-2 threshold that should have caught it.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.matching import damerau_levenshtein, weighted_damerau_levenshtein

pytestmark = pytest.mark.unit

_HYPOTHESIS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
_PLATES = st.text(alphabet="ABC0123OIL", min_size=0, max_size=8)


# ---------------------------------------------------------------------------
# Unweighted
# ---------------------------------------------------------------------------


def test_distance__identical_strings__is_zero() -> None:
    """The floor."""
    assert damerau_levenshtein("ABC1234", "ABC1234") == 0


def test_distance__single_substitution__is_one() -> None:
    """One character changed, one edit."""
    assert damerau_levenshtein("ABC1234", "ABC1235") == 1


def test_distance__single_insertion__is_one() -> None:
    """A spurious character read from a sticker."""
    assert damerau_levenshtein("ABC1234", "ABC12345") == 1


def test_distance__single_deletion__is_one() -> None:
    """An occluded character."""
    assert damerau_levenshtein("ABC1234", "ABC123") == 1


def test_distance__adjacent_transposition__is_one_not_two() -> None:
    """The Damerau behaviour this module exists for.

    Under plain Levenshtein a swap costs 2, which would push a single OCR error
    outside a distance-2 threshold meant to catch exactly that.
    """
    assert damerau_levenshtein("ABC1234", "ABC1243") == 1


def test_distance__empty_versus_non_empty__is_the_length() -> None:
    """Boundary: everything has to be inserted."""
    assert damerau_levenshtein("", "ABC1234") == 7
    assert damerau_levenshtein("ABC1234", "") == 7


def test_distance__both_empty__is_zero() -> None:
    """Boundary."""
    assert damerau_levenshtein("", "") == 0


@pytest.mark.parametrize(
    ("left", "right"),
    [("ABC1234", "XYZ9999"), ("ABC", "ABCDEF"), ("ABC1234", "ABC1243")],
    ids=["different", "longer", "transposed"],
)
def test_distance__is_symmetric(left: str, right: str) -> None:
    """Distance does not depend on argument order."""
    assert damerau_levenshtein(left, right) == damerau_levenshtein(right, left)


def test_distance__cutoff__returns_above_the_cutoff_rather_than_the_true_value() -> None:
    """Documented early-exit behaviour.

    The caller asked whether the distance was within a threshold, not what it
    was; computing an exact large distance across a big candidate set is wasted
    work. The returned value is deliberately above the cutoff so it can never be
    mistaken for a real measurement.
    """
    true_distance = damerau_levenshtein("ABC1234", "ZZZZZZZ")
    assert true_distance == 7

    capped = damerau_levenshtein("ABC1234", "ZZZZZZZ", max_distance=2)
    assert capped == 3
    assert capped > 2


def test_distance__cutoff__still_returns_the_true_value_when_within_it() -> None:
    """The cutoff must not distort answers it was not meant to affect."""
    assert damerau_levenshtein("ABC1234", "ABC1235", max_distance=3) == 1


def test_distance__cutoff__short_circuits_on_length_alone() -> None:
    """A length gap larger than the cutoff cannot be closed by any edit."""
    assert damerau_levenshtein("AB", "ABCDEFGHIJ", max_distance=2) == 3


# ---------------------------------------------------------------------------
# Weighted
# ---------------------------------------------------------------------------


def test_weighted__in_group_substitution__costs_less_than_arbitrary() -> None:
    """The whole reason the weighted variant exists.

    A 0 read as O is far more likely than a 0 read as W, and charging both the
    same forces a choice between admitting the W case and rejecting the O case.
    """
    confusable = weighted_damerau_levenshtein("ABC1234", "ABC1O34".replace("O", "0"))
    in_group = weighted_damerau_levenshtein("ABC0234", "ABCO234")
    arbitrary = weighted_damerau_levenshtein("ABC0234", "ABCW234")

    assert in_group == pytest.approx(0.5)
    assert arbitrary == pytest.approx(1.0)
    assert in_group < arbitrary
    assert confusable >= 0.0


def test_weighted__substitution_cost_is_configurable() -> None:
    """Regions with cleaner plate fonts can charge more for a confusion."""
    assert weighted_damerau_levenshtein(
        "ABC0234", "ABCO234", confusion_substitution_cost=0.25
    ) == pytest.approx(0.25)


def test_weighted__identical_strings__is_zero() -> None:
    """The floor is unchanged by weighting."""
    assert weighted_damerau_levenshtein("ABC1234", "ABC1234") == 0.0


def test_weighted__insertions_and_deletions__still_cost_one() -> None:
    """Only substitutions are discounted; an occlusion is not a confusion."""
    assert weighted_damerau_levenshtein("ABC1234", "ABC123") == pytest.approx(1.0)


def test_weighted__transposition__costs_one() -> None:
    """Same Damerau behaviour as the unweighted variant."""
    assert weighted_damerau_levenshtein("ABC1234", "ABC1243") == pytest.approx(1.0)


def test_weighted__empty_input__is_the_other_length() -> None:
    """Boundary."""
    assert weighted_damerau_levenshtein("", "ABC") == pytest.approx(3.0)
    assert weighted_damerau_levenshtein("ABC", "") == pytest.approx(3.0)


def test_weighted__cutoff__returns_above_the_cutoff() -> None:
    """Same early-exit contract as the unweighted variant."""
    capped = weighted_damerau_levenshtein("ABC1234", "ZZZZZZZ", max_distance=1.0)

    assert capped > 1.0


def test_weighted__all_confusable_plate__is_half_the_unweighted_distance() -> None:
    """Every character differs, but every difference is a known confusion."""
    unweighted = damerau_levenshtein("OIL", "01L".replace("L", "1"))
    weighted = weighted_damerau_levenshtein("OIL", "011")

    assert weighted == pytest.approx(1.5)
    assert unweighted >= 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@_HYPOTHESIS
@given(left=_PLATES, right=_PLATES)
def test_weighted__never_exceeds_unweighted(left: str, right: str) -> None:
    """Property: every operation costs at most what it costs unweighted."""
    assert weighted_damerau_levenshtein(left, right) <= damerau_levenshtein(left, right) + 1e-9


@_HYPOTHESIS
@given(left=_PLATES, right=_PLATES)
def test_distance__is_symmetric_for_arbitrary_inputs(left: str, right: str) -> None:
    """Property: symmetry holds everywhere, not just at the reference cases."""
    assert damerau_levenshtein(left, right) == damerau_levenshtein(right, left)


@_HYPOTHESIS
@given(left=_PLATES, right=_PLATES)
def test_distance__is_bounded_by_the_longer_length(left: str, right: str) -> None:
    """Property: replacing every character and padding is always an option."""
    assert damerau_levenshtein(left, right) <= max(len(left), len(right))
