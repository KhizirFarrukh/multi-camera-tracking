"""Unit tests for :mod:`multicam_tracker.models.match`.

Each invariant here stops one specific class of inconsistent record from
reaching storage. They are tested individually because a record can violate
exactly one of them while looking entirely reasonable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_tracker.models import MatchCandidate, MatchMethod, ReviewStatus
from tests.fixtures.factories import make_match_candidate

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# plate_exact
# ---------------------------------------------------------------------------


def test_match__plate_exact_with_zero_edit_distance__is_valid() -> None:
    """The definition of an exact match."""
    candidate = make_match_candidate(match_method=MatchMethod.PLATE_EXACT, plate_edit_distance=0)

    assert candidate.match_method is MatchMethod.PLATE_EXACT


@pytest.mark.parametrize("distance", [1, 2, None], ids=["one", "two", "missing"])
def test_match__plate_exact_without_zero_edit_distance__is_rejected(distance: int | None) -> None:
    """A non-zero distance means the match is fuzzy, and mislabelling it would
    inflate every downstream precision measurement."""
    with pytest.raises(ValidationError, match="plate_exact"):
        make_match_candidate(match_method=MatchMethod.PLATE_EXACT, plate_edit_distance=distance)


# ---------------------------------------------------------------------------
# plate_fuzzy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance", [1, 2, 5], ids=["one", "two", "five"])
def test_match__plate_fuzzy_with_a_positive_edit_distance__is_valid(distance: int) -> None:
    """Any non-zero distance is a fuzzy match."""
    candidate = make_match_candidate(
        match_method=MatchMethod.PLATE_FUZZY, plate_edit_distance=distance
    )

    assert candidate.plate_edit_distance == distance


@pytest.mark.parametrize("distance", [0, None], ids=["zero", "missing"])
def test_match__plate_fuzzy_without_a_positive_edit_distance__is_rejected(
    distance: int | None,
) -> None:
    """Boundary: distance 0 is exact by definition, so it cannot be fuzzy."""
    with pytest.raises(ValidationError, match="plate_fuzzy"):
        make_match_candidate(match_method=MatchMethod.PLATE_FUZZY, plate_edit_distance=distance)


def test_match__negative_edit_distance__is_rejected() -> None:
    """Edit distance is a count."""
    with pytest.raises(ValidationError):
        make_match_candidate(match_method=MatchMethod.PLATE_FUZZY, plate_edit_distance=-1)


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


def test_match__embedding_with_a_similarity__is_valid() -> None:
    """The fallback path carries a cosine similarity as its evidence."""
    candidate = make_match_candidate(
        match_method=MatchMethod.EMBEDDING,
        plate_edit_distance=None,
        embedding_similarity=0.94,
        review_status=ReviewStatus.AUTO_ACCEPTED,
    )

    assert candidate.embedding_similarity == pytest.approx(0.94)


def test_match__embedding_without_a_similarity__is_rejected() -> None:
    """Without the similarity the record asserts a match with no evidence at all."""
    with pytest.raises(ValidationError, match="embedding_similarity"):
        make_match_candidate(
            match_method=MatchMethod.EMBEDDING,
            plate_edit_distance=None,
            embedding_similarity=None,
        )


@pytest.mark.parametrize(
    "similarity", [-1.0, 0.0, 1.0], ids=["opposite", "orthogonal", "identical"]
)
def test_match__embedding_similarity_at_the_bounds__is_accepted(similarity: float) -> None:
    """Boundary: cosine similarity spans the closed interval [-1, 1]."""
    candidate = make_match_candidate(
        match_method=MatchMethod.EMBEDDING,
        plate_edit_distance=None,
        embedding_similarity=similarity,
    )

    assert candidate.embedding_similarity == similarity


@pytest.mark.parametrize("similarity", [-1.01, 1.01], ids=["below", "above"])
def test_match__embedding_similarity_out_of_range__is_rejected(similarity: float) -> None:
    """A value outside [-1, 1] is not a cosine and usually means an unnormalized dot product."""
    with pytest.raises(ValidationError):
        make_match_candidate(
            match_method=MatchMethod.EMBEDDING,
            plate_edit_distance=None,
            embedding_similarity=similarity,
        )


# ---------------------------------------------------------------------------
# manual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [ReviewStatus.CONFIRMED, ReviewStatus.REJECTED], ids=["confirmed", "rejected"]
)
def test_match__manual_with_a_resolved_status__is_valid(status: ReviewStatus) -> None:
    """A manual record exists because a human decided; both outcomes are decisions."""
    candidate = make_match_candidate(
        match_method=MatchMethod.MANUAL, plate_edit_distance=None, review_status=status
    )

    assert candidate.review_status is status


@pytest.mark.parametrize(
    "status",
    [ReviewStatus.PENDING_REVIEW, ReviewStatus.AUTO_ACCEPTED],
    ids=["pending", "auto-accepted"],
)
def test_match__manual_with_an_unresolved_status__is_rejected(status: ReviewStatus) -> None:
    """'Manual but pending' and 'manual but auto-accepted' are both contradictions."""
    with pytest.raises(ValidationError, match="manual"):
        make_match_candidate(
            match_method=MatchMethod.MANUAL, plate_edit_distance=None, review_status=status
        )


# ---------------------------------------------------------------------------
# Shared constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [0.0, 1.0], ids=["zero", "one"])
def test_match__score_at_the_bounds__is_accepted(score: float) -> None:
    """Boundary: the closed interval [0, 1]."""
    assert make_match_candidate(match_score=score).match_score == score


@pytest.mark.parametrize("score", [-0.01, 1.01], ids=["below", "above"])
def test_match__score_out_of_range__is_rejected(score: float) -> None:
    """A score outside [0, 1] cannot be compared against a threshold."""
    with pytest.raises(ValidationError):
        make_match_candidate(match_score=score)


@pytest.mark.parametrize("field", ["sighting_id", "target_id"], ids=["sighting", "target"])
def test_match__empty_identifier__is_rejected(field: str) -> None:
    """A match with a blank endpoint links nothing to nothing."""
    with pytest.raises(ValidationError):
        make_match_candidate(**{field: "  "})


def test_match__unknown_method_string__is_rejected() -> None:
    """The method set is closed; a new one requires a contract change."""
    with pytest.raises(ValidationError):
        make_match_candidate(match_method="plate_probably")


def test_match__json_round_trip__preserves_enums_as_strings() -> None:
    """Enum fields cross the wire as their string values."""
    candidate = make_match_candidate()
    payload = candidate.to_json_dict()

    assert payload["match_method"] == "plate_exact"
    assert payload["review_status"] == "auto_accepted"
    assert MatchCandidate.from_json_dict(payload) == candidate
