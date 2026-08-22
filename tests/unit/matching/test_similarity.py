"""Unit tests for the similarity primitives and reference aggregation.

The non-normalized case is the one that matters. Silently renormalizing would
hide a broken extractor and quietly corrupt every similarity computed
afterwards, so it must raise here exactly as it does at the model boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.exceptions import MatchingError, ValidationError
from multicam_tracker.matching import (
    AggregationStrategy,
    aggregate_similarity,
    batch_cosine_similarity,
    cosine_similarity,
    top_k_similar,
)
from multicam_tracker.matching.similarity import as_matrix
from tests.fixtures.factories import unit_vector

pytestmark = pytest.mark.unit

DIM = 32
_HYPOTHESIS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _unit(seed: int) -> list[float]:
    """Return a deterministic unit vector.

    Args:
        seed: Generator seed.

    Returns:
        A normalized vector of the test dimension.
    """
    return unit_vector(seed=seed, dim=DIM)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine__vector_with_itself__is_one() -> None:
    """The upper bound."""
    vector = _unit(1)

    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-12)


def test_cosine__orthogonal_vectors__is_zero() -> None:
    """The 'unrelated vehicles' case."""
    left = [1.0, 0.0, 0.0]
    right = [0.0, 1.0, 0.0]

    assert cosine_similarity(left, right) == pytest.approx(0.0)


def test_cosine__opposite_vectors__is_minus_one() -> None:
    """The lower bound."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine__non_normalized_input__raises_and_does_not_renormalize() -> None:
    """A non-normalized embedding is an upstream bug, not something to repair.

    Renormalizing would hide a broken extractor and corrupt every similarity
    computed afterwards.
    """
    doubled = [component * 2.0 for component in _unit(2)]

    with pytest.raises(ValidationError, match="not L2-normalized"):
        cosine_similarity(doubled, _unit(3))


def test_cosine__zero_vector__raises_rather_than_producing_nan() -> None:
    """A NaN would propagate silently through a ranking."""
    with pytest.raises(ValidationError, match="zero magnitude"):
        cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine__non_finite_values__are_rejected() -> None:
    """An inf reaching the dot product would poison the whole comparison."""
    with pytest.raises(ValidationError, match="non-finite"):
        cosine_similarity([float("inf"), 0.0], [1.0, 0.0])


def test_cosine__mismatched_dimensions__are_rejected() -> None:
    """Two models produced these, and their similarity would be meaningless."""
    with pytest.raises(ValidationError, match="different dimensions"):
        cosine_similarity(_unit(1), unit_vector(seed=1, dim=DIM * 2))


def test_cosine__empty_vector__is_rejected() -> None:
    """Boundary: there is nothing to compare."""
    with pytest.raises(ValidationError, match="non-empty"):
        cosine_similarity([], [1.0])


# ---------------------------------------------------------------------------
# batch_cosine_similarity and top_k_similar
# ---------------------------------------------------------------------------


def test_batch__matches_the_elementwise_result_for_every_row() -> None:
    """The vectorized path must agree with the scalar one exactly."""
    query = _unit(1)
    candidates = [_unit(index) for index in range(2, 8)]

    batched = batch_cosine_similarity(query, as_matrix(candidates, field_name="c"))

    for index, candidate in enumerate(candidates):
        assert batched[index] == pytest.approx(cosine_similarity(query, candidate), abs=1e-12)


def test_batch__empty_matrix__returns_an_empty_array() -> None:
    """Boundary: no candidates is not an error."""
    result = batch_cosine_similarity(_unit(1), np.zeros((0, 0)))

    assert result.size == 0


def test_top_k__returns_exactly_k_sorted_descending() -> None:
    """Ranking is the whole output of this function."""
    query = _unit(1)
    candidates = [_unit(index) for index in range(2, 12)]

    hits = top_k_similar(query, candidates, k=3)

    assert len(hits) == 3
    assert [hit.similarity for hit in hits] == sorted(
        [hit.similarity for hit in hits], reverse=True
    )


def test_top_k__the_query_itself__ranks_first() -> None:
    """A sanity check that the ordering is not inverted."""
    query = _unit(5)
    candidates = [_unit(1), query, _unit(2)]

    assert top_k_similar(query, candidates, k=1)[0].position == 1


def test_top_k__k_larger_than_the_candidate_set__returns_all_without_raising() -> None:
    """A caller ranking a short list should not have to special-case it."""
    candidates = [_unit(1), _unit(2)]

    assert len(top_k_similar(_unit(3), candidates, k=99)) == 2


def test_top_k__empty_candidates__returns_empty() -> None:
    """Boundary."""
    assert top_k_similar(_unit(1), [], k=5) == []


def test_top_k__k_of_zero__returns_empty() -> None:
    """Boundary: asking for nothing returns nothing."""
    assert top_k_similar(_unit(1), [_unit(2)], k=0) == []


@_HYPOTHESIS
@given(left=st.integers(0, 500), right=st.integers(0, 500))
def test_cosine__is_always_within_the_valid_range(left: int, right: int) -> None:
    """Property: floating-point error must never push it outside [-1, 1]."""
    assert -1.0 <= cosine_similarity(_unit(left), _unit(right)) <= 1.0


# ---------------------------------------------------------------------------
# Aggregation across several references
# ---------------------------------------------------------------------------


def test_aggregate__max__returns_the_highest_individual_similarity() -> None:
    """The default strategy asks: does this look like the target from any angle?"""
    query = _unit(1)
    references = [_unit(50), query, _unit(51)]

    result = aggregate_similarity(query, references, strategy=AggregationStrategy.MAX)

    assert result == pytest.approx(1.0, abs=1e-9)


def test_aggregate__mean_over_identical_references__equals_the_single_case() -> None:
    """Averaging copies of one vector changes nothing."""
    query = _unit(1)
    reference = _unit(2)

    single = aggregate_similarity(query, [reference], strategy=AggregationStrategy.MEAN)
    repeated = aggregate_similarity(query, [reference] * 4, strategy=AggregationStrategy.MEAN)

    assert repeated == pytest.approx(single, abs=1e-9)


def test_aggregate__topk_mean_with_k_of_one__equals_max() -> None:
    """The strategies are a continuum, and k=1 is its endpoint."""
    query = _unit(1)
    references = [_unit(index) for index in range(2, 6)]

    assert aggregate_similarity(
        query, references, strategy=AggregationStrategy.TOPK_MEAN, top_k=1
    ) == pytest.approx(aggregate_similarity(query, references, strategy=AggregationStrategy.MAX))


@pytest.mark.parametrize(
    "strategy",
    [AggregationStrategy.MAX, AggregationStrategy.MEAN, AggregationStrategy.TOPK_MEAN],
)
def test_aggregate__single_reference__is_identical_under_every_strategy(
    strategy: AggregationStrategy,
) -> None:
    """A target with one confirmed sighting behaves the same however configured."""
    query = _unit(1)
    reference = _unit(2)

    assert aggregate_similarity(query, [reference], strategy=strategy) == pytest.approx(
        cosine_similarity(query, reference), abs=1e-9
    )


def test_aggregate__adding_a_poor_reference__lowers_mean_but_not_max() -> None:
    """Documents the trade-off that makes 'max' the default.

    'max' never falls as references accumulate, which is why an unconfirmed
    reference would do permanent damage -- and why references.py refuses one.
    """
    query = _unit(1)
    good = [query]
    with_poor = [query, _unit(99)]

    assert aggregate_similarity(
        query, with_poor, strategy=AggregationStrategy.MAX
    ) == pytest.approx(aggregate_similarity(query, good, strategy=AggregationStrategy.MAX))
    assert aggregate_similarity(query, with_poor, strategy=AggregationStrategy.MEAN) < (
        aggregate_similarity(query, good, strategy=AggregationStrategy.MEAN)
    )


def test_aggregate__empty_reference_set__raises() -> None:
    """Returning a neutral 0.0 would look like a legitimate weak match."""
    with pytest.raises(MatchingError, match="empty reference set"):
        aggregate_similarity(_unit(1), [])


def test_aggregate__non_positive_top_k__is_rejected() -> None:
    """Boundary."""
    with pytest.raises(MatchingError, match="top_k"):
        aggregate_similarity(_unit(1), [_unit(2)], strategy=AggregationStrategy.TOPK_MEAN, top_k=0)


def test_aggregate__mean_of_opposing_references__is_rejected() -> None:
    """Boundary: vectors that cancel to zero have no direction to compare."""
    vector = [1.0, 0.0, 0.0]
    opposite = [-1.0, 0.0, 0.0]

    with pytest.raises(MatchingError, match="zero vector"):
        aggregate_similarity(vector, [vector, opposite], strategy=AggregationStrategy.MEAN)


@_HYPOTHESIS
@given(seed=st.integers(0, 400))
def test_aggregate__result_is_always_a_valid_similarity(seed: int) -> None:
    """Property: aggregation cannot leave the valid range."""
    result = aggregate_similarity(_unit(seed), [_unit(1), _unit(2), _unit(3)])

    assert -1.0 <= result <= 1.0
    assert not math.isnan(result)
