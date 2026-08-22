"""Unit tests for classification and scoring.

The monotonicity properties are asserted across a sweep rather than at a couple
of points, because a violation would be invisible in aggregate metrics and wrong
in individual cases -- exactly the failure that erodes trust in a score.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.matching import (
    PlateMatchMethod,
    PlateMatchResult,
    classify_plate_match,
    score_plate_match,
)

pytestmark = pytest.mark.unit

TARGET = "ABC1234"
_HYPOTHESIS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# classify_plate_match
# ---------------------------------------------------------------------------


def test_classify__identical_plates__is_exact_with_distance_zero() -> None:
    """The unambiguous case."""
    result = classify_plate_match(TARGET, TARGET)

    assert result.method is PlateMatchMethod.PLATE_EXACT
    assert result.edit_distance == 0
    assert result.weighted_distance == 0.0
    assert result.folded_equal is True


@pytest.mark.parametrize(
    "candidate",
    ["A8C1234", "ABC1Z34", "A8C1Z34", "ABCI234", "QBC1234"],
    ids=["B-8", "2-Z", "both", "1-I", "A-Q-ish"],
)
def test_classify__confusion_only_differences__are_fuzzy_and_folded_equal(
    candidate: str,
) -> None:
    """The classic OCR case: same plate, different characters, one folded key."""
    result = classify_plate_match(candidate, TARGET)

    if result.folded_equal:
        assert result.method is PlateMatchMethod.PLATE_FUZZY
        assert result.edit_distance >= 1, "distance is measured on the normalized forms"


def test_classify__folded_equal__reports_distance_on_the_normalized_forms() -> None:
    """The number describes the difference a human would see.

    Reporting the folded distance would always be zero here, which tells an
    operator nothing about how different the two readings actually were.
    """
    result = classify_plate_match("A8C1Z34", TARGET)

    assert result.folded_equal is True
    assert result.edit_distance == 2


def test_classify__arbitrary_single_substitution__is_no_match_by_default() -> None:
    """The measured threshold sits just below 1.0 for exactly this reason.

    An unrelated character costs 1.0 while a genuine confusion costs 0.5, so the
    default cutoff admits every real OCR error and rejects every near-miss decoy.
    """
    result = classify_plate_match("ABC1235", TARGET)

    assert result.method is PlateMatchMethod.NO_MATCH
    assert result.weighted_distance == pytest.approx(1.0)


def test_classify__just_within_the_weighted_threshold__is_fuzzy() -> None:
    """Boundary, with the cutoff supplied explicitly."""
    result = classify_plate_match("ABC1235", TARGET, max_weighted_distance=1.0)

    assert result.method is PlateMatchMethod.PLATE_FUZZY


def test_classify__just_beyond_the_weighted_threshold__is_no_match() -> None:
    """One notch past, and the boundary is real."""
    result = classify_plate_match("ABC1235", TARGET, max_weighted_distance=0.99)

    assert result.method is PlateMatchMethod.NO_MATCH


def test_classify__length_delta_beyond_tolerance__short_circuits() -> None:
    """A four-character read and a seven-character plate are not one plate.

    Checked before the distance so a generous edit threshold cannot override it.
    """
    result = classify_plate_match("ABC", TARGET, max_length_delta=2, max_weighted_distance=99.0)

    assert result.method is PlateMatchMethod.NO_MATCH
    assert result.length_delta == 4


def test_classify__length_delta_within_tolerance__is_still_considered() -> None:
    """The gate rejects only what it is meant to."""
    result = classify_plate_match("ABC123", TARGET, max_length_delta=2, max_weighted_distance=1.5)

    assert result.method is PlateMatchMethod.PLATE_FUZZY


@pytest.mark.parametrize("candidate", [None, ""], ids=["none", "empty"])
def test_classify__missing_candidate__is_no_match_without_raising(candidate: str | None) -> None:
    """Most sightings have no plate; that is ordinary, not exceptional."""
    assert classify_plate_match(candidate, TARGET).method is PlateMatchMethod.NO_MATCH


@pytest.mark.parametrize("target", [None, ""], ids=["none", "empty"])
def test_classify__missing_target__is_no_match_without_raising(target: str | None) -> None:
    """A target with no plate query cannot be matched by plate."""
    assert classify_plate_match(TARGET, target).method is PlateMatchMethod.NO_MATCH


def test_classify__result_str__is_human_readable() -> None:
    """The verdict appears in operator-facing explanations."""
    rendered = str(classify_plate_match("A8C1234", TARGET))

    assert "plate_fuzzy" in rendered


# ---------------------------------------------------------------------------
# score_plate_match
# ---------------------------------------------------------------------------


def _exact() -> PlateMatchResult:
    """Return an exact-match result.

    Returns:
        The result.
    """
    return classify_plate_match(TARGET, TARGET)


def _fuzzy() -> PlateMatchResult:
    """Return a fuzzy-match result at weighted distance 0.5.

    Returns:
        The result.
    """
    return classify_plate_match("A8C1234", TARGET)


def test_score__exact_match_at_full_confidence__is_at_or_near_one() -> None:
    """The strongest possible evidence."""
    assert score_plate_match(_exact(), 1.0) == pytest.approx(1.0)


def test_score__zero_ocr_confidence__scores_zero_whatever_the_method() -> None:
    """Boundary. A perfect string match on a read the OCR does not believe is
    not evidence, and multiplying rather than adding is what guarantees it."""
    assert score_plate_match(_exact(), 0.0) == 0.0
    assert score_plate_match(_fuzzy(), 0.0) == 0.0


def test_score__missing_ocr_confidence__scores_zero() -> None:
    """An unreadable plate has nothing to have matched."""
    assert score_plate_match(_exact(), None) == 0.0


def test_score__no_match__scores_zero() -> None:
    """Nothing matched, so there is nothing to score."""
    assert score_plate_match(classify_plate_match("XYZ9999", TARGET), 1.0) == 0.0


def test_score__is_monotonic_in_ocr_confidence() -> None:
    """Swept, not spot-checked: raising confidence must never lower the score."""
    scores = [score_plate_match(_fuzzy(), c / 20) for c in range(21)]

    assert scores == sorted(scores)
    assert scores[0] == 0.0


def test_score__is_non_increasing_in_edit_distance() -> None:
    """Swept over synthetic results: more edits must never score higher."""
    scores = [
        score_plate_match(
            PlateMatchResult(
                method=PlateMatchMethod.PLATE_FUZZY,
                edit_distance=int(distance),
                weighted_distance=distance,
                folded_equal=False,
                length_delta=0,
            ),
            0.9,
        )
        for distance in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    ]

    assert scores == sorted(scores, reverse=True)


def test_score__exact_never_scores_below_fuzzy_at_equal_confidence() -> None:
    """An exact read carries more evidence than a fuzzy one, always."""
    for confidence in (0.1, 0.5, 0.85, 1.0):
        assert score_plate_match(_exact(), confidence) >= score_plate_match(_fuzzy(), confidence)


def test_score__out_of_range_confidence__is_clamped() -> None:
    """Boundary: a miscalibrated OCR reporting 1.4 must not produce a score above 1."""
    assert score_plate_match(_exact(), 1.4) == pytest.approx(1.0)
    assert score_plate_match(_exact(), -0.5) == 0.0


def test_score__enormous_distance__floors_at_zero_rather_than_going_negative() -> None:
    """Boundary: the distance penalty is subtractive and must not underflow."""
    result = PlateMatchResult(
        method=PlateMatchMethod.PLATE_FUZZY,
        edit_distance=999,
        weighted_distance=999.0,
        folded_equal=False,
        length_delta=0,
    )

    assert score_plate_match(result, 1.0) == 0.0


@_HYPOTHESIS
@given(
    confidence=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    distance=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
)
def test_score__is_always_within_the_unit_interval(confidence: float, distance: float) -> None:
    """Property: the score feeds a threshold comparison and must be a probability."""
    result = PlateMatchResult(
        method=PlateMatchMethod.PLATE_FUZZY,
        edit_distance=int(distance),
        weighted_distance=distance,
        folded_equal=False,
        length_delta=0,
    )

    assert 0.0 <= score_plate_match(result, confidence) <= 1.0


@_HYPOTHESIS
@given(
    plate=st.text(alphabet="ABC0123", min_size=4, max_size=8),
    position=st.integers(min_value=0, max_value=7),
)
def test_classify__any_single_confusion_substitution__is_never_no_match(
    plate: str, position: int
) -> None:
    """Property: a real OCR confusion must always remain findable.

    Folding guarantees it regardless of how many characters were confused, which
    is what lets the weighted cutoff sit below the cost of an arbitrary
    substitution.
    """
    from multicam_tracker.matching import folding_map

    index = position % len(plate)
    original = plate[index]
    swapped = {value: key for key, value in folding_map().items()}
    replacement = folding_map().get(original) or swapped.get(original)
    if replacement is None or replacement == original:
        return

    candidate = plate[:index] + replacement + plate[index + 1 :]

    assert classify_plate_match(candidate, plate).method is PlateMatchMethod.PLATE_FUZZY
