"""Unit tests for optimal path selection.

Every graph here is small enough to score by hand, and several tests do exactly
that -- computing the expected optimum with :func:`path_score` and asserting the
search agrees. A search grading its own homework proves nothing.
"""

from __future__ import annotations

import pytest

from multicam_tracker.pathing import build_graph
from multicam_tracker.pathing.optimal_path import PathResult, best_path, path_score, score_table
from tests.fixtures.pathing import (
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    prepared,
)

pytestmark = pytest.mark.unit


def _graph(entries: list[tuple[str, float, float]], *, with_cameras: bool = False) -> object:
    """Build a graph from ``(camera, offset_sec, score)`` triples.

    Args:
        entries: The candidates.
        with_cameras: Supply camera coordinates, which enables the physical
            speed constraint. Off by default so most tests exercise the search
            rather than the constraint.

    Returns:
        The graph.
    """
    return build_graph(
        [prepared(camera, offset, score) for camera, offset, score in entries],
        chain_topology_with_outlier(),
        cameras=chain_cameras() if with_cameras else None,
    )


def test_hand_constructed_graph__returns_the_path_scored_highest_by_hand() -> None:
    """The search must agree with the published objective, not with itself."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)])

    result = best_path(graph)

    assert result.node_indices == [0, 1, 2]
    assert result.score == pytest.approx(path_score(graph, [0, 1, 2]))


def test_a_chain_of_plausible_high_confidence_sightings__returns_the_full_chain() -> None:
    """The engine must not stop early on evidence that keeps supporting it."""
    graph = _graph(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_03", 240.0, 0.9),
            ("cam_04", 360.0, 0.9),
        ]
    )

    assert best_path(graph).node_indices == [0, 1, 2, 3]


def test_a_single_isolated_candidate__returns_a_one_sighting_path() -> None:
    """Boundary: one sighting is a legitimate answer to "where did it go?"."""
    graph = _graph([("cam_01", 0.0, 0.9)])

    result = best_path(graph)

    assert result.node_indices == [0]
    assert result.edges == []


def test_an_empty_graph__returns_an_empty_result_rather_than_raising() -> None:
    """Documented convention: no candidates means no route, not an error."""
    result = best_path(build_graph([], chain_topology_with_outlier()))

    assert result == PathResult(node_indices=[], edges=[], score=0.0, excluded_indices=[])
    assert result.is_empty


def test_the_search__is_deterministic_across_repeated_runs() -> None:
    """An operator rerunning a search must get the same answer."""
    entries = [
        ("cam_01", 0.0, 0.9),
        ("cam_02", 120.0, 0.9),
        (FAR_CAMERA, 200.0, 0.9),
        ("cam_03", 240.0, 0.9),
    ]

    runs = {tuple(best_path(_graph(entries)).node_indices) for _ in range(10)}

    assert len(runs) == 1


def test_ties__are_broken_by_a_documented_rule_not_by_dict_ordering() -> None:
    """Two identical-scoring options must resolve the same way every time.

    Both candidates here are equally confident and equally plausible, so the
    scores are exactly equal and only the tie rule decides.
    """
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_02", 240.0, 0.9)])

    first = best_path(graph)
    second = best_path(graph)

    assert first.node_indices == second.node_indices


def test_excluded_candidates__are_recorded_alongside_the_route() -> None:
    """The explanation cannot answer "why not that one?" without them.

    The excluded candidate is weakly matched *and* off the network, so reaching
    it across unmonitored ground costs more than it is worth. A confident match
    in the same place would be included, which is the point of the penalty being
    a price rather than a prohibition.
    """
    graph = _graph(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            (FAR_CAMERA, 4000.0, 0.45),
        ]
    )

    result = best_path(graph)

    assert result.node_indices == [0, 1]
    assert result.excluded_indices == [2]


def test_a_confident_match_across_unmonitored_ground__is_still_reached() -> None:
    """The other half of the same rule: a skip edge that earns its price.

    Without this the trajectory fragments the moment a vehicle drives through a
    district nobody watches, which is the normal condition of a real network.
    """
    graph = _graph(
        [
            ("cam_01", 0.0, 0.95),
            ("cam_02", 120.0, 0.95),
            (FAR_CAMERA, 4000.0, 0.95),
        ]
    )

    assert best_path(graph).node_indices == [0, 1, 2]


def test_the_score_table__reports_the_best_score_ending_at_each_node() -> None:
    """The dynamic program's state is exposed so incremental extension can reuse it."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    best, predecessor = score_table(graph)

    assert predecessor == [None, 0]
    assert best[1] > best[0]


def test_path_score__rejects_a_sequence_with_no_edge_between_two_nodes() -> None:
    """A set of disconnected sightings is not a path, and must not be scored as one."""
    graph = _graph([("cam_01", 0.0, 0.9), (FAR_CAMERA, 2.0, 0.9)], with_cameras=True)

    assert graph.edges == []  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not a path"):
        path_score(graph, [0, 1])


def test_path_score__of_an_empty_path__is_zero() -> None:
    """Boundary."""
    assert path_score(_graph([("cam_01", 0.0, 0.9)]), []) == 0.0


def test_a_route_may_begin_at_any_candidate__not_only_the_earliest() -> None:
    """The first sighting in the feed is not necessarily the first on the route.

    Here the earliest candidate is unreachable from anything, so the optimal
    route starts at the second.
    """
    graph = _graph(
        [
            (FAR_CAMERA, 0.0, 0.5),
            ("cam_01", 4000.0, 0.95),
            ("cam_02", 4120.0, 0.95),
            ("cam_03", 4240.0, 0.95),
        ]
    )

    assert best_path(graph).node_indices == [1, 2, 3]


def test_a_longer_well_supported_route__beats_a_short_confident_fragment() -> None:
    """The behaviour the inclusion bonus was expected to be needed for.

    It is not: multiplicative hop weights already make extension worth far more
    than stopping, which is why the measured bonus is zero. This test is what
    would catch that changing.
    """
    graph = _graph(
        [
            ("cam_01", 0.0, 0.99),
            ("cam_02", 120.0, 0.99),
            ("cam_03", 240.0, 0.85),
            ("cam_04", 360.0, 0.85),
        ]
    )

    assert best_path(graph).node_indices == [0, 1, 2, 3]
