"""Unit tests for alternative routes and the ambiguity verdict.

Ambiguity is the safety valve: when two routes explain the evidence equally
well, saying so is the only honest answer. These tests pin both halves -- that
genuine ties are flagged, and that a clear winner is not.
"""

from __future__ import annotations

import pytest

from multicam_tracker.pathing import build_graph, k_best_paths
from tests.fixtures.pathing import (
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    prepared,
)

pytestmark = pytest.mark.unit


def _graph(entries: list[tuple[str, float, float]]) -> object:
    """Build a graph from ``(camera, offset_sec, score)`` triples.

    Args:
        entries: The candidates.

    Returns:
        The graph.
    """
    return build_graph(
        [prepared(camera, offset, score) for camera, offset, score in entries],
        chain_topology_with_outlier(),
        cameras=chain_cameras(),
    )


def test_k_best__returns_exactly_k_when_k_distinct_routes_exist() -> None:
    """The ordinary case."""
    graph = _graph(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_02", 200.0, 0.8),
            ("cam_03", 320.0, 0.9),
        ]
    )

    report = k_best_paths(graph, 3)

    assert len(report.alternatives) == 3


def test_k_best__returns_fewer_than_k_without_raising_when_fewer_exist() -> None:
    """A caller asking for ten routes through a two-node graph is not an error."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    report = k_best_paths(graph, 10)

    assert 0 < len(report.alternatives) < 10


def test_k_best__paths_are_ordered_by_descending_score() -> None:
    """The first alternative is the one the trajectory is built from."""
    graph = _graph(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_02", 200.0, 0.7),
            ("cam_03", 320.0, 0.9),
        ]
    )

    scores = [alternative.score for alternative in k_best_paths(graph, 4).alternatives]

    assert scores == sorted(scores, reverse=True)


def test_alternatives__are_competing_accounts_not_shortened_copies() -> None:
    """A route with its last sighting removed is not an alternative explanation.

    Every path has many such shadows, and letting them fill the list would make
    the margin measure how much the final hop contributed rather than whether
    another story exists -- flagging decisive trajectories as too close to call.
    """
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)])

    alternatives = k_best_paths(graph, 3).alternatives

    best = set(alternatives[0].node_indices)
    assert all(not set(other.node_indices) <= best for other in alternatives[1:])


def test_two_equally_scoring_routes__produce_a_zero_margin_and_the_ambiguous_flag() -> None:
    """The case the flag exists for.

    From one confirmed sighting the vehicle is seen at two cameras a second
    apart. It cannot have been at both -- the hop between them would need 4000
    km/h, so no edge joins them -- and each branch is equally well supported.
    Nothing in the evidence prefers either, and saying so is the only honest
    answer available.
    """
    report = k_best_paths(
        _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 121.0, 0.9)]), 2
    )

    assert report.margin == pytest.approx(0.0)
    assert report.is_ambiguous
    assert "does not distinguish" in report.describe()


def test_a_clear_winner__produces_a_large_margin_and_no_ambiguous_flag() -> None:
    """The control: an unambiguous route must not be reported as a coin flip."""
    report = k_best_paths(
        _graph([("cam_01", 0.0, 0.95), ("cam_02", 120.0, 0.95), (FAR_CAMERA, 9000.0, 0.30)]), 2
    )

    assert not report.is_ambiguous
    assert report.margin > 0.05


@pytest.mark.parametrize(
    ("margin_min", "expected"),
    [(0.05, False), (0.2, True)],
    ids=["threshold below the margin", "threshold above the margin"],
)
def test_the_ambiguity_threshold__is_tested_at_its_boundary(
    margin_min: float, expected: bool
) -> None:
    """The verdict flips exactly where the configured threshold says it should.

    Same two mutually exclusive branches, one slightly better supported than the
    other, so the margin lands between the two thresholds under test.
    """
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 121.0, 0.82)])
    report = k_best_paths(graph, 2, ambiguity_margin_min=margin_min)

    assert 0.05 < report.margin < 0.2
    assert report.is_ambiguous is expected


def test_a_single_route__is_never_ambiguous() -> None:
    """With no runner-up there is nothing to be confused with."""
    report = k_best_paths(_graph([("cam_01", 0.0, 0.9)]), 3)

    assert report.second_score is None
    assert report.margin == 1.0
    assert not report.is_ambiguous
    assert "only path" in report.describe()


def test_an_empty_graph__reports_no_alternatives_and_no_ambiguity() -> None:
    """Boundary: an empty answer is an unambiguous one."""
    report = k_best_paths(build_graph([], chain_topology_with_outlier()), 3)

    assert report.alternatives == []
    assert not report.is_ambiguous


def test_a_non_positive_k__is_rejected() -> None:
    """Asking for zero routes is a caller bug, not an empty result."""
    with pytest.raises(ValueError, match="k must be positive"):
        k_best_paths(_graph([("cam_01", 0.0, 0.9)]), 0)
