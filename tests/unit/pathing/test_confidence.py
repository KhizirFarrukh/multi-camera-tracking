"""Unit tests for hop and trajectory confidence.

The weakest-link guarantee is asserted directly rather than inferred from the
formula: whatever the aggregation does internally, the number it reports must
never exceed the weakest hop by more than the configured allowance.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multicam_tracker.pathing import ConfidenceStrategy, aggregate_confidence, build_graph
from multicam_tracker.pathing.confidence import hop_confidence
from tests.fixtures.pathing import FAR_CAMERA, chain_topology_with_outlier, prepared

pytestmark = pytest.mark.unit

TOLERANCE = 0.05
"""The configured allowance, restated so the guarantee reads explicitly."""

_HYPOTHESIS = settings(max_examples=200, deadline=None)


# ---------------------------------------------------------------------------
# Per-hop
# ---------------------------------------------------------------------------


def _edge(entries: list[tuple[str, float, float]]) -> tuple[object, object, object]:
    """Build one edge and its endpoints.

    Args:
        entries: Exactly two ``(camera, offset_sec, score)`` triples.

    Returns:
        ``(origin, destination, edge)``.
    """
    nodes = [prepared(camera, offset, score) for camera, offset, score in entries]
    graph = build_graph(nodes, chain_topology_with_outlier())
    return nodes[0], nodes[1], graph.edges[0]


def test_hop_confidence__combines_both_endpoints_and_the_plausibility() -> None:
    """The three inputs the contract names for a hop, and nothing else."""
    origin, destination, edge = _edge([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.8)])

    assert hop_confidence(origin, destination, edge) == pytest.approx(0.9 * 0.8 * 1.0)


def test_hop_confidence__falls_when_either_endpoint_is_weaker() -> None:
    """Monotone in both endpoints."""
    strong = hop_confidence(*_edge([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)]))
    weaker = hop_confidence(*_edge([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.5)]))

    assert weaker < strong


def test_hop_confidence__a_gap_hop_is_penalised() -> None:
    """A hop the topology cannot account for is not as good as one it can."""
    origin, destination, edge = _edge([("cam_01", 0.0, 0.9), (FAR_CAMERA, 4000.0, 0.9)])

    penalised = hop_confidence(origin, destination, edge, implausible_penalty=0.5)
    unpenalised = hop_confidence(origin, destination, edge, implausible_penalty=1.0)

    assert penalised == pytest.approx(unpenalised * 0.5)


def test_hop_confidence__never_leaves_the_unit_interval() -> None:
    """The trajectory model rejects anything outside it."""
    origin, destination, edge = _edge([("cam_01", 0.0, 1.0), ("cam_02", 120.0, 1.0)])

    assert 0.0 <= hop_confidence(origin, destination, edge) <= 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_a_path_of_perfect_hops__aggregates_to_one() -> None:
    """The upper bound."""
    assert aggregate_confidence([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_one_weak_hop__dominates_the_result() -> None:
    """The whole point.

    A geometric mean alone would report 0.46 here, which reads as "maybe" for a
    route whose middle link is barely evidence at all.
    """
    result = aggregate_confidence([1.0, 1.0, 0.1])

    assert result <= 0.1 + TOLERANCE


def test_the_result__never_exceeds_the_weakest_hop_by_more_than_the_allowance() -> None:
    """The guarantee, stated as a guarantee rather than as a formula."""
    for hops in ([0.9, 0.9, 0.2], [0.5, 0.4], [0.99, 0.99, 0.99], [0.05, 0.95, 0.95]):
        assert aggregate_confidence(hops) <= min(hops) + TOLERANCE + 1e-12


def test_adding_a_weak_hop__always_lowers_the_result() -> None:
    """Monotonicity. Extending a route cannot make it more certain."""
    before = aggregate_confidence([0.9, 0.9])
    after = aggregate_confidence([0.9, 0.9, 0.3])

    assert after < before


def test_a_single_sighting_trajectory__reports_that_sighting_s_confidence() -> None:
    """There is no chain to be weaker than."""
    assert aggregate_confidence([], single_sighting_confidence=0.77) == pytest.approx(0.77)


def test_no_hops_and_no_single_sighting_confidence__is_rejected() -> None:
    """There would be nothing to report, and zero would be a lie."""
    with pytest.raises(ValueError, match="nothing to report"):
        aggregate_confidence([])


def test_a_zero_confidence_hop__drives_the_result_to_zero() -> None:
    """Boundary: a link with no support at all breaks the chain."""
    assert aggregate_confidence([0.9, 0.0, 0.9]) == pytest.approx(0.0)


def test_the_minimum_strategy__reports_exactly_the_weakest_hop() -> None:
    """Available for a caller that wants no allowance at all."""
    result = aggregate_confidence([0.9, 0.3, 0.8], strategy=ConfidenceStrategy.MINIMUM)

    assert result == pytest.approx(0.3)


def test_the_cap_applies_whatever_the_aggregation_produced() -> None:
    """The guarantee is about the number reported, not about the formula.

    An arithmetic mean would read 0.70 for this route. It is not offered as a
    strategy precisely because the cap would flatten it to the same answer, so
    it would be an option with no reachable behaviour.
    """
    assert aggregate_confidence([1.0, 1.0, 0.1]) == pytest.approx(0.1 + TOLERANCE)


@_HYPOTHESIS
@given(
    hops=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=12),
)
def test_aggregation__always_lands_in_the_unit_interval(hops: list[float]) -> None:
    """Property: whatever the inputs, the result is a valid confidence."""
    result = aggregate_confidence(hops)

    assert 0.0 <= result <= 1.0


@_HYPOTHESIS
@given(
    hops=st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=1, max_size=12),
)
def test_aggregation__never_beats_the_weakest_link_by_more_than_the_allowance(
    hops: list[float],
) -> None:
    """Property: the documented guarantee, over arbitrary routes."""
    assert aggregate_confidence(hops) <= min(hops) + TOLERANCE + 1e-12
