"""Unit tests for evidence combination, thresholding, references, and versioning.

The decisive property is the ordering guarantee: an embedding-only match can
never outrank a plate match. It is asserted across randomized inputs rather than
at a few points, because a single inversion would let appearance masquerade as
the stronger signal in exactly the ambiguous cases that matter.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching import (
    add_reference_embedding,
    classify_embedding_matches,
    classify_plate_match,
    combine_evidence,
    distinct_model_versions,
    minimum_pairwise_similarity,
    prune_references,
    require_same_model_version,
    select_diverse,
    sightings_for_version,
)
from multicam_tracker.models import MatchMethod, ReviewStatus
from tests.fixtures.factories import make_sighting, make_target, unit_vector

pytestmark = pytest.mark.unit

TARGET = "ABC1234"
DIM = 32
ACCEPT = 0.95
REVIEW = 0.75
_HYPOTHESIS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _unit(seed: int) -> list[float]:
    """Return a deterministic unit vector.

    Args:
        seed: Generator seed.

    Returns:
        A normalized vector.
    """
    return unit_vector(seed=seed, dim=DIM)


def _full(seed: int) -> list[float]:
    """Return a deterministic unit vector at the configured model dimension.

    Anything passing through :class:`Target` must use the real embedding width,
    since the model validates it.

    Args:
        seed: Generator seed.

    Returns:
        A normalized vector of the configured dimension.
    """
    return unit_vector(seed=seed)


def _exact() -> object:
    """Return an exact plate match result.

    Returns:
        The classification.
    """
    return classify_plate_match(TARGET, TARGET)


# ---------------------------------------------------------------------------
# Two-tier thresholding and the margin rule
# ---------------------------------------------------------------------------


def _scored(*similarities: float) -> list[tuple[object, float]]:
    """Build scored candidates with distinct sightings.

    Args:
        *similarities: One similarity per candidate.

    Returns:
        ``(sighting, similarity)`` pairs.
    """
    return [
        (make_sighting("cam_01", offset_sec=index * 10), value)
        for index, value in enumerate(similarities)
    ]


def test_threshold__exactly_at_auto_accept__is_auto_accepted() -> None:
    """Boundary: the threshold is inclusive."""
    results = classify_embedding_matches(
        _scored(ACCEPT), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.0
    )

    assert results[0].review_status is ReviewStatus.AUTO_ACCEPTED


def test_threshold__just_below_auto_accept__is_pending_review() -> None:
    """One notch under, and a human decides."""
    results = classify_embedding_matches(
        _scored(ACCEPT - 0.001), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.0
    )

    assert results[0].review_status is ReviewStatus.PENDING_REVIEW


def test_threshold__exactly_at_the_review_floor__is_retained() -> None:
    """Boundary: the floor is inclusive."""
    results = classify_embedding_matches(
        _scored(REVIEW), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.0
    )

    assert len(results) == 1
    assert results[0].review_status is ReviewStatus.PENDING_REVIEW


def test_threshold__just_below_the_review_floor__is_excluded_entirely() -> None:
    """Discarded, not queued: a review queue full of noise is one nobody reads."""
    results = classify_embedding_matches(
        _scored(REVIEW - 0.001), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.0
    )

    assert results == []


def test_margin_rule__close_runner_up__downgrades_the_leader() -> None:
    """Two near-identical vehicles must never be auto-resolved.

    Auto-accepting either would be a coin flip reported as a fact.
    """
    results = classify_embedding_matches(
        _scored(0.97, 0.96), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.04
    )

    assert results[0].review_status is ReviewStatus.PENDING_REVIEW
    assert results[0].downgraded_by_margin is True


def test_margin_rule__comfortable_margin__leaves_the_leader_auto_accepted() -> None:
    """The rule fires only when the decision is genuinely ambiguous."""
    results = classify_embedding_matches(
        _scored(0.97, 0.80), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.04
    )

    assert results[0].review_status is ReviewStatus.AUTO_ACCEPTED
    assert results[0].downgraded_by_margin is False


def test_margin_rule__single_candidate__is_not_downgraded() -> None:
    """With no runner-up there is no ambiguity to protect against."""
    results = classify_embedding_matches(
        _scored(0.97), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.5
    )

    assert results[0].review_status is ReviewStatus.AUTO_ACCEPTED
    assert results[0].margin is None


def test_margin_rule__only_the_leader_is_eligible_for_demotion() -> None:
    """A second-placed candidate is not being auto-accepted anyway."""
    results = classify_embedding_matches(
        _scored(0.99, 0.97, 0.96), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.04
    )

    assert results[1].downgraded_by_margin is False


def test_threshold__results_are_ranked_by_descending_similarity() -> None:
    """The review queue shows the strongest evidence first."""
    results = classify_embedding_matches(
        _scored(0.80, 0.99, 0.90), auto_accept_min=ACCEPT, review_min=REVIEW, margin_min=0.0
    )

    assert [round(entry.similarity, 2) for entry in results] == [0.99, 0.90, 0.80]


# ---------------------------------------------------------------------------
# combine_evidence
# ---------------------------------------------------------------------------


def test_combine__strong_plate_and_strong_embedding__uses_the_plate_method() -> None:
    """Corroboration boosts confidence but does not change what the evidence is."""
    verdict = combine_evidence(_exact(), 0.95, 0.97, embedding_auto_accept=ACCEPT)

    assert verdict.method is MatchMethod.PLATE_EXACT
    assert verdict.score >= 0.95
    assert verdict.score <= 1.0
    assert verdict.conflicting is False


def test_combine__no_plate_and_strong_embedding__is_capped_below_plate_evidence() -> None:
    """Appearance alone is corroborating evidence, never proof."""
    verdict = combine_evidence(None, None, 0.99, embedding_auto_accept=ACCEPT)

    assert verdict.method is MatchMethod.EMBEDDING
    assert verdict.review_status is ReviewStatus.PENDING_REVIEW
    assert verdict.score <= 0.45


def test_combine__strong_plate_and_weak_embedding__is_conflicting() -> None:
    """Two signals pointing different ways is more alarming than one weak signal."""
    verdict = combine_evidence(_exact(), 0.95, 0.10, embedding_auto_accept=ACCEPT)

    assert verdict.conflicting is True
    assert verdict.review_status is ReviewStatus.PENDING_REVIEW
    assert "one of the two" in verdict.disagreement


def test_combine__weak_plate_and_strong_embedding__is_conflicting() -> None:
    """The mirror case: appearance says yes, the plate does not corroborate."""
    fuzzy = classify_plate_match("A8C1234", TARGET)
    verdict = combine_evidence(fuzzy, 0.30, 0.99, embedding_auto_accept=ACCEPT)

    assert verdict.conflicting is True
    assert verdict.review_status is ReviewStatus.PENDING_REVIEW


def test_combine__no_evidence_at_all__returns_a_rejected_non_match() -> None:
    """Documented choice: not an exception.

    A caller scans many sightings, most of which match nothing. That is the
    ordinary case rather than an error.
    """
    verdict = combine_evidence(None, None, None)

    assert verdict.is_match is False
    assert verdict.method is None
    assert verdict.score == 0.0
    assert verdict.review_status is ReviewStatus.REJECTED


def test_combine__no_match_result_and_no_embedding__is_a_non_match() -> None:
    """A plate that failed to match is the same as no plate evidence."""
    verdict = combine_evidence(classify_plate_match("XYZ9999", TARGET), 0.9, None)

    assert verdict.is_match is False


def test_combine__embedding_only__cannot_be_converted_with_a_plate_distance() -> None:
    """The stored record must not claim a plate distance it never measured."""
    verdict = combine_evidence(None, None, 0.99, embedding_auto_accept=ACCEPT)

    candidate = verdict.to_candidate(
        sighting_id="b0000000-0000-4000-8000-000000000001",
        target_id="a0000000-0000-4000-8000-000000000001",
        edit_distance=3,
    )

    assert candidate.match_method is MatchMethod.EMBEDDING
    assert candidate.plate_edit_distance is None


def test_combine__non_match__cannot_become_a_candidate() -> None:
    """Persisting a verdict with no evidence would be a fabricated record."""
    with pytest.raises(ValueError, match="supports no match"):
        combine_evidence(None, None, None).to_candidate(
            sighting_id="s", target_id="t", edit_distance=None
        )


@_HYPOTHESIS
@given(
    similarity=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_combine__score_is_always_within_the_unit_interval(
    similarity: float, confidence: float
) -> None:
    """Property: the score feeds a threshold comparison."""
    verdict = combine_evidence(_exact(), confidence, similarity)

    assert 0.0 <= verdict.score <= 1.0


@_HYPOTHESIS
@given(similarity=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
def test_combine__embedding_only__never_outranks_any_valid_plate_match(
    similarity: float,
) -> None:
    """The exit criterion, asserted structurally rather than by tuning.

    The ceiling sits below the lowest score a retained plate match can produce,
    so no combination of inputs can invert the ordering.
    """
    from multicam_tracker.config import get_settings

    visual_only = combine_evidence(None, None, similarity)
    floor = get_settings().thresholds.plate_review_min_confidence

    assert visual_only.score <= floor


# ---------------------------------------------------------------------------
# Reference management
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [ReviewStatus.CONFIRMED, ReviewStatus.AUTO_ACCEPTED], ids=["confirmed", "auto"]
)
def test_add_reference__from_a_confirmed_match__succeeds(status: ReviewStatus) -> None:
    """The ordinary path: a human confirmed it, so it becomes reference material."""
    target = make_target(reference_embeddings=[_full(1)])
    sighting = make_sighting("cam_01", embedding=_full(2))

    updated = add_reference_embedding(target, sighting, status)

    assert len(updated.reference_embeddings or []) == 2


def test_add_reference__from_a_pending_match__is_rejected() -> None:
    """Guarding against self-confirmation.

    max aggregation never falls, so a wrong reference inflates scores
    permanently and produces more wrong confirmations.
    """
    target = make_target(reference_embeddings=[_full(1)])
    sighting = make_sighting("cam_01", embedding=_full(2))

    with pytest.raises(MatchingError, match="confirm its own guesses"):
        add_reference_embedding(target, sighting, ReviewStatus.PENDING_REVIEW)


def test_add_reference__from_a_sighting_without_an_embedding__is_rejected() -> None:
    """There is nothing to remember."""
    target = make_target(reference_embeddings=[_full(1)])

    with pytest.raises(MatchingError, match="no embedding"):
        add_reference_embedding(target, make_sighting("cam_01"), ReviewStatus.CONFIRMED)


def test_add_reference__does_not_mutate_the_input_target() -> None:
    """Domain models are values; a caller keeps whatever it passed in."""
    target = make_target(reference_embeddings=[_full(1)])
    sighting = make_sighting("cam_01", embedding=_full(2))

    add_reference_embedding(target, sighting, ReviewStatus.CONFIRMED)

    assert len(target.reference_embeddings or []) == 1


def test_prune__keeps_the_configured_maximum() -> None:
    """The cap is enforced, not merely suggested."""
    target = make_target(reference_embeddings=[_full(index) for index in range(20)])

    assert len(prune_references(target, max_references=5).reference_embeddings or []) == 5


def test_prune__a_set_already_within_the_limit__is_a_no_op() -> None:
    """Pruning must not reshuffle a set it had no reason to touch."""
    references = [_full(1), _full(2)]
    target = make_target(reference_embeddings=references)

    assert prune_references(target, max_references=8).reference_embeddings == references


def test_prune__selects_a_more_diverse_subset_than_recency_would() -> None:
    """The property that justifies greedy selection over keeping the newest.

    Constructed so recency clusters: the last four references are near-copies of
    one another, while the earlier ones span distinct directions.
    """
    clustered_base = _unit(50)
    references = [
        _unit(1),
        _unit(2),
        _unit(3),
        clustered_base,
        [component + 0.001 for component in clustered_base],
        [component + 0.002 for component in clustered_base],
    ]
    normalized = []
    for reference in references:
        magnitude = sum(value * value for value in reference) ** 0.5
        normalized.append([value / magnitude for value in reference])

    diverse = select_diverse(normalized, 3)
    recent = normalized[-3:]

    assert minimum_pairwise_similarity(diverse) < minimum_pairwise_similarity(recent)


def test_select_diverse__non_positive_limit__is_rejected() -> None:
    """Boundary."""
    with pytest.raises(MatchingError, match="at least 1"):
        select_diverse([_unit(1), _unit(2)], 0)


# ---------------------------------------------------------------------------
# Model-version safety
# ---------------------------------------------------------------------------


def test_require_same_model_version__matching_versions__is_allowed() -> None:
    """The ordinary path."""
    assert require_same_model_version("osnet@v1", "osnet@v1") == "osnet@v1"


def test_require_same_model_version__differing_versions__raises_naming_both() -> None:
    """A cross-version similarity is a plausible number with no meaning."""
    with pytest.raises(MatchingError) as excinfo:
        require_same_model_version("osnet@v1", "osnet@v2")

    assert excinfo.value.context["left_model_version"] == "osnet@v1"
    assert excinfo.value.context["right_model_version"] == "osnet@v2"


def test_require_same_model_version__one_side_unlabelled__still_raises() -> None:
    """An unlabelled vector cannot be assumed to match, or the guard is pointless."""
    with pytest.raises(MatchingError):
        require_same_model_version("osnet@v1", None)


def test_distinct_model_versions__reports_every_version_present() -> None:
    """A mixed-version table is a migration problem worth surfacing."""
    sightings = [
        make_sighting("cam_01", embedding=unit_vector(seed=1), embedding_model_version="v1"),
        make_sighting("cam_01", embedding=unit_vector(seed=2), embedding_model_version="v2"),
        make_sighting("cam_01"),
    ]

    assert distinct_model_versions(sightings) == {"v1", "v2"}


def test_sightings_for_version__filters_to_one_model() -> None:
    """The repository query has an in-memory counterpart for the same reason."""
    first = make_sighting("cam_01", embedding=unit_vector(seed=1), embedding_model_version="v1")
    second = make_sighting("cam_01", embedding=unit_vector(seed=2), embedding_model_version="v2")

    assert sightings_for_version([first, second], "v1") == [first]


def test_sightings_for_version__excludes_sightings_without_an_embedding() -> None:
    """No vector, nothing to compare."""
    assert sightings_for_version([make_sighting("cam_01")], None) == []
