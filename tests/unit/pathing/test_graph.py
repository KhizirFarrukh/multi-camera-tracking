"""Unit tests for trajectory graph construction.

Two properties carry the rest of the engine: every edge points forward in time
(which is what makes the linear dynamic program exact), and no edge exists for a
physically impossible transition (which is what makes a teleport unselectable
rather than merely unlikely).
"""

from __future__ import annotations

import pytest

from multicam_tracker.pathing import EdgeKind, build_graph, edge_weight, prepare_candidates
from tests.fixtures.factories import make_sighting
from tests.fixtures.pathing import (
    CHAIN_MAX_SEC,
    CHAIN_MIN_SEC,
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    prepared,
    scored_candidate,
)

pytestmark = pytest.mark.unit


def _graph(entries: list[tuple[str, float, float]], **kwargs: object) -> object:
    """Build a graph from ``(camera, offset_sec, score)`` triples.

    Args:
        entries: The candidates.
        **kwargs: Passed through to ``build_graph``.

    Returns:
        The graph.
    """
    nodes = [prepared(camera, offset, score) for camera, offset, score in entries]
    return build_graph(nodes, chain_topology_with_outlier(), **kwargs)  # type: ignore[arg-type]


def _kinds(graph: object) -> dict[tuple[int, int], EdgeKind]:
    """Return edge kinds by endpoint pair.

    Args:
        graph: The graph.

    Returns:
        ``(from, to) -> kind``.
    """
    return {(edge.from_index, edge.to_index): edge.kind for edge in graph.edges}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_edges__only_ever_point_forward_in_time() -> None:
    """The property the exact linear search depends on."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)])

    assert graph.is_acyclic  # type: ignore[attr-defined]
    assert all(edge.from_index < edge.to_index for edge in graph.edges)  # type: ignore[attr-defined]


def test_graph__is_acyclic__verified_programmatically() -> None:
    """Asserted directly rather than trusted from the construction comment."""
    graph = _graph(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_01", 300.0, 0.9),
            ("cam_03", 420.0, 0.9),
        ]
    )

    assert graph.is_acyclic  # type: ignore[attr-defined]


def test_single_candidate__produces_one_node_and_no_edges() -> None:
    """Boundary: there is nowhere to go."""
    graph = _graph([("cam_01", 0.0, 0.9)])

    assert len(graph.nodes) == 1  # type: ignore[attr-defined]
    assert graph.edges == []  # type: ignore[attr-defined]


def test_empty_candidate_set__produces_an_empty_graph() -> None:
    """Boundary."""
    graph = build_graph([], chain_topology_with_outlier())

    assert graph.nodes == []
    assert graph.edges == []


def test_simultaneous_sightings__produce_no_edge_between_them() -> None:
    """Zero elapsed time satisfies no travel window, and the model rejects it."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 0.0, 0.9)])

    assert graph.edges == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_a_link_traversed_inside_its_window__is_direct() -> None:
    """The ordinary case."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    assert _kinds(graph)[(0, 1)] is EdgeKind.DIRECT


@pytest.mark.parametrize(
    ("elapsed", "label"),
    [(CHAIN_MIN_SEC, "exactly the minimum"), (CHAIN_MAX_SEC, "exactly the maximum")],
)
def test_the_travel_window_boundaries__are_inclusive(elapsed: float, label: str) -> None:
    """Stated at the exact values, because a hop's verdict must not hinge on rounding."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", elapsed, 0.9)])

    assert _kinds(graph)[(0, 1)] is EdgeKind.DIRECT, label


def test_a_link_traversed_too_slowly__is_implausible_not_a_coverage_gap() -> None:
    """The topology accounts for this pair and says the timing does not fit.

    Different claim from unmonitored ground, and priced differently.
    """
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", CHAIN_MAX_SEC + 600.0, 0.9)])

    assert _kinds(graph)[(0, 1)] is EdgeKind.IMPLAUSIBLE


def test_a_two_hop_transition_within_the_arrival_window__is_indirect() -> None:
    """The vehicle passed a camera that did not see it. Unremarkable."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)])

    assert _kinds(graph)[(0, 1)] is EdgeKind.INDIRECT


def test_a_camera_the_topology_cannot_reach_at_all__is_a_gap() -> None:
    """Unmonitored ground: permitted, penalised, and recorded as such."""
    nodes = [prepared("cam_01", 0.0, 0.9), prepared(FAR_CAMERA, 4000.0, 0.9)]

    graph = build_graph(nodes, chain_topology_with_outlier())

    assert _kinds(graph)[(0, 1)] is EdgeKind.GAP


def test_returning_to_the_same_camera__is_a_gap_with_an_explanation() -> None:
    """A loop is real, and what happened in between was not observed."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_01", 600.0, 0.9)])

    edge = graph.edges[0]  # type: ignore[attr-defined]
    assert edge.kind is EdgeKind.GAP
    assert "cam_01" in edge.reason


def test_every_edge__carries_a_human_readable_reason() -> None:
    """The explanation output is assembled from these, so none may be blank."""
    graph = _graph([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 400.0, 0.9)])

    assert all(edge.reason for edge in graph.edges)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The physical constraint
# ---------------------------------------------------------------------------


def test_a_physically_impossible_transition__produces_no_edge_at_all() -> None:
    """A teleport must be unselectable, not merely unlikely.

    Travel-time windows come from a speed model and can be optimistic; a hop
    that merely scores badly is still one the search may take when everything
    else scores worse.
    """
    nodes = [prepared("cam_01", 0.0, 0.9), prepared(FAR_CAMERA, 2.0, 0.99)]

    graph = build_graph(nodes, chain_topology_with_outlier(), cameras=chain_cameras())

    assert graph.edges == []


def test_without_camera_coordinates__the_speed_constraint_cannot_be_applied() -> None:
    """Documents the weaker guarantee rather than pretending it is the same.

    No coordinates means no speed, and the caller should know the difference.
    """
    nodes = [prepared("cam_01", 0.0, 0.9), prepared(FAR_CAMERA, 2.0, 0.99)]

    graph = build_graph(nodes, chain_topology_with_outlier(), cameras=None)

    assert len(graph.edges) == 1


def test_a_camera_missing_from_the_map__does_not_suppress_the_edge() -> None:
    """An unknown position is not evidence of an impossible one."""
    nodes = [prepared("cam_01", 0.0, 0.9), prepared(FAR_CAMERA, 2.0, 0.99)]
    partial = {"cam_01": chain_cameras()["cam_01"]}

    graph = build_graph(nodes, chain_topology_with_outlier(), cameras=partial)

    assert len(graph.edges) == 1


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_edge_weight__increases_with_endpoint_confidence() -> None:
    """More confident endpoints, better hop -- all else equal."""
    weak = edge_weight(
        prepared("cam_01", 0.0, 0.5),
        prepared("cam_02", 120.0, 0.5),
        1.0,
        EdgeKind.DIRECT,
        gap_penalty=0.05,
        implausible_penalty=0.5,
    )
    strong = edge_weight(
        prepared("cam_01", 0.0, 0.9),
        prepared("cam_02", 120.0, 0.9),
        1.0,
        EdgeKind.DIRECT,
        gap_penalty=0.05,
        implausible_penalty=0.5,
    )

    assert strong > weak


def test_edge_weight__increases_with_topology_plausibility() -> None:
    """Same matches, better-explained transit, higher weight."""
    origin = prepared("cam_01", 0.0, 0.9)
    destination = prepared("cam_02", 120.0, 0.9)

    poor = edge_weight(
        origin, destination, 0.2, EdgeKind.DIRECT, gap_penalty=0.05, implausible_penalty=0.5
    )
    good = edge_weight(
        origin, destination, 1.0, EdgeKind.DIRECT, gap_penalty=0.05, implausible_penalty=0.5
    )

    assert good > poor


def test_a_gap_hop__is_penalised_against_an_otherwise_identical_direct_hop() -> None:
    """Crossing unmonitored ground costs something, always."""
    origin = prepared("cam_01", 0.0, 0.9)
    destination = prepared("cam_02", 120.0, 0.9)

    direct = edge_weight(
        origin, destination, 1.0, EdgeKind.DIRECT, gap_penalty=0.05, implausible_penalty=0.5
    )
    gap = edge_weight(
        origin, destination, 1.0, EdgeKind.GAP, gap_penalty=0.05, implausible_penalty=0.5
    )

    assert gap == pytest.approx(direct - 0.05)


def test_an_implausible_hop__is_discounted_multiplicatively() -> None:
    """A multiplier, so discounting a strong hop costs more than a weak one."""
    origin = prepared("cam_01", 0.0, 0.9)
    destination = prepared("cam_02", 120.0, 0.9)

    direct = edge_weight(
        origin, destination, 1.0, EdgeKind.DIRECT, gap_penalty=0.05, implausible_penalty=0.5
    )
    implausible = edge_weight(
        origin, destination, 1.0, EdgeKind.IMPLAUSIBLE, gap_penalty=0.05, implausible_penalty=0.5
    )

    assert implausible == pytest.approx(direct * 0.5)


def test_a_gap_hop_between_weak_matches__is_negative() -> None:
    """The discrimination the penalty exists for.

    A strongly supported vehicle may bridge unmonitored ground; a pair of weak
    guesses may not.
    """
    weak_gap = edge_weight(
        prepared("cam_01", 0.0, 0.45),
        prepared(FAR_CAMERA, 600.0, 0.45),
        0.1,
        EdgeKind.GAP,
        gap_penalty=0.05,
        implausible_penalty=0.5,
    )
    strong_gap = edge_weight(
        prepared("cam_01", 0.0, 0.9),
        prepared(FAR_CAMERA, 600.0, 0.9),
        0.1,
        EdgeKind.GAP,
        gap_penalty=0.05,
        implausible_penalty=0.5,
    )

    assert weak_gap < 0.0 < strong_gap


def test_rebuilding_only_a_suffix__leaves_the_earlier_edges_alone() -> None:
    """What incremental extension relies on to avoid starting over."""
    nodes = [
        prepared("cam_01", 0.0, 0.9),
        prepared("cam_02", 120.0, 0.9),
        prepared("cam_03", 240.0, 0.9),
    ]

    full = build_graph(nodes, chain_topology_with_outlier())
    suffix = build_graph(nodes, chain_topology_with_outlier(), min_to_index=2)

    assert all(edge.to_index >= 2 for edge in suffix.edges)
    assert [edge for edge in full.edges if edge.to_index >= 2] == suffix.edges


def test_prepared_candidates_feed_straight_into_the_graph() -> None:
    """The two halves of the pipeline agree about their interface."""
    sightings = [make_sighting("cam_01", offset_sec=0.0), make_sighting("cam_02", offset_sec=120.0)]
    candidates = [scored_candidate(sighting, 0.9) for sighting in sightings]

    nodes = prepare_candidates(
        candidates, {sighting.sighting_id: sighting for sighting in sightings}
    )
    graph = build_graph(nodes, chain_topology_with_outlier())

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
