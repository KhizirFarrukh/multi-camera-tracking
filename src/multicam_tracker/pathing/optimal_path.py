"""Choosing the single most plausible route through the candidate graph.

**Why not greedy.** The obvious algorithm is to walk forward in time taking the
best next sighting. It fails on exactly the case that matters: one false
positive in the middle of the timeline. Greedy chaining has already committed to
everything before it, so it takes the bad hop and every subsequent decision is
made from the wrong place. One spurious match teleports the route across the
city and the reconstruction is worthless from that point on.

Global scoring cannot be fooled that way. The false positive is only accepted if
the *whole path through it* scores better than the whole path around it, and
reaching an implausible location and returning costs two bad hops — which the
genuine route, needing none, beats.

The objective::

    score(path) = sum(edge weights along the path)
                + inclusion_bonus * (number of sightings in the path)

The inclusion bonus is the entire reason this is not "find the two most
confident sightings and stop". Without it the highest-scoring path is always the
shortest one that can be assembled from strong edges, because every additional
edge can only add a number below 1. With it, adding a sighting is worth
something in itself, so the search prefers the longer well-supported route --
but not unconditionally: a node reached across unmonitored ground pays the gap
penalty, which is set above the bonus, so weak nodes are still skipped.

Because every edge points forward in time, index order is a topological order
and one backward pass computes the exact optimum in O(V + E). No heuristic
search, no beam, no approximation.

**Determinism.** Ties are broken by preferring the path whose last hop starts at
the *earlier* node, and then by sighting id. Never by dict or set iteration
order: an operator who reruns a search must get the same answer, and a
regression test that only passes some of the time is worse than no test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from multicam_tracker.pathing.graph import TrajectoryEdge, TrajectoryGraph
from multicam_tracker.pathing.preparation import PreparedCandidate

__all__ = ["PathResult", "best_path", "path_from_table", "path_score", "score_table"]


@dataclass(frozen=True)
class PathResult:
    """One scored route through the candidate graph."""

    node_indices: list[int]
    """Positions in :attr:`TrajectoryGraph.nodes`, ascending."""

    edges: list[TrajectoryEdge]
    """The hops taken, one fewer than the nodes."""

    score: float
    """The objective value. Comparable only against paths through the same graph."""

    excluded_indices: list[int] = field(default_factory=list)
    """Candidates the search left out, ascending. Recorded rather than discarded:
    stage 08's explainability requirement is that an operator can ask why a
    sighting is *not* in the route, and that question has no answer if the
    excluded set is thrown away."""

    def nodes(self, graph: TrajectoryGraph) -> list[PreparedCandidate]:
        """Return the candidates on this path.

        Args:
            graph: The graph the indices refer to.

        Returns:
            The candidates, in chronological order.
        """
        return [graph.nodes[index] for index in self.node_indices]

    @property
    def is_empty(self) -> bool:
        """Return whether the path contains no sightings at all."""
        return not self.node_indices


def path_score(
    graph: TrajectoryGraph, node_indices: list[int], *, inclusion_bonus: float | None = None
) -> float:
    """Score an arbitrary path under the documented objective.

    Exposed so a test can state the expected optimum by hand and compare, rather
    than trusting the search to grade its own homework.

    Args:
        graph: The graph the indices refer to.
        node_indices: Ascending node positions.
        inclusion_bonus: Per-sighting bonus. Defaults to config.

    Returns:
        The objective value. ``0.0`` for an empty path.

    Raises:
        ValueError: If consecutive nodes have no edge between them, which would
            make the "path" a set of disconnected sightings rather than a route.
    """
    from multicam_tracker.config import get_settings

    bonus = (
        inclusion_bonus
        if inclusion_bonus is not None
        else get_settings().thresholds.path_node_inclusion_bonus
    )
    if not node_indices:
        return 0.0

    by_pair = {(edge.from_index, edge.to_index): edge for edge in graph.edges}
    total = bonus * len(node_indices)
    for origin, destination in pairwise(node_indices):
        edge = by_pair.get((origin, destination))
        if edge is None:
            msg = f"no edge from node {origin} to node {destination}; this is not a path"
            raise ValueError(msg)
        total += edge.weight
    return total


def score_table(
    graph: TrajectoryGraph, *, inclusion_bonus: float | None = None
) -> tuple[list[float], list[int | None]]:
    """Run the dynamic program and return its raw state.

    Args:
        graph: The graph to search.
        inclusion_bonus: Per-sighting bonus. Defaults to config.

    Returns:
        ``(best, predecessor)`` where ``best[i]`` is the score of the highest
        scoring path *ending at* node ``i``, and ``predecessor[i]`` is the node
        it arrived from, or ``None`` when the path starts there.
    """
    from multicam_tracker.config import get_settings

    bonus = (
        inclusion_bonus
        if inclusion_bonus is not None
        else get_settings().thresholds.path_node_inclusion_bonus
    )

    incoming: dict[int, list[TrajectoryEdge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.to_index, []).append(edge)

    best = [bonus] * len(graph.nodes)
    predecessor: list[int | None] = [None] * len(graph.nodes)

    # Index order is a topological order, so one forward pass suffices.
    for index in range(len(graph.nodes)):
        for edge in incoming.get(index, []):
            candidate = best[edge.from_index] + edge.weight + bonus
            # Strictly greater: on a tie the earlier predecessor already
            # recorded wins, which is the documented deterministic rule.
            if candidate > best[index]:
                best[index] = candidate
                predecessor[index] = edge.from_index

    return best, predecessor


def path_from_table(
    graph: TrajectoryGraph, best: list[float], predecessor: list[int | None]
) -> PathResult:
    """Walk a completed dynamic-programming table back into a path.

    Separated from :func:`best_path` because incremental extension recomputes
    only part of the table and must not pay for a full re-run to read the answer
    out of it.

    Args:
        graph: The searched graph.
        best: Best score ending at each node.
        predecessor: The node each best path arrived from.

    Returns:
        The optimal path the table describes.
    """
    if not graph.nodes:
        return PathResult(node_indices=[], edges=[], score=0.0, excluded_indices=[])

    # Ties on the terminal node resolve to the earliest, which combined with the
    # tie rule inside the table makes the whole search order-independent.
    end = max(range(len(best)), key=lambda index: (best[index], -index))

    reversed_indices: list[int] = []
    cursor: int | None = end
    while cursor is not None:
        reversed_indices.append(cursor)
        cursor = predecessor[cursor]
    node_indices = list(reversed(reversed_indices))

    by_pair = {(edge.from_index, edge.to_index): edge for edge in graph.edges}
    edges = [by_pair[(origin, destination)] for origin, destination in pairwise(node_indices)]
    on_path = set(node_indices)

    return PathResult(
        node_indices=node_indices,
        edges=edges,
        score=best[end],
        excluded_indices=[index for index in range(len(graph.nodes)) if index not in on_path],
    )


def best_path(graph: TrajectoryGraph, *, inclusion_bonus: float | None = None) -> PathResult:
    """Return the highest-scoring path through the graph.

    Args:
        graph: The graph to search.
        inclusion_bonus: Per-sighting bonus. Defaults to config.

    Returns:
        The optimal path, with the candidates it excluded recorded alongside.
        An empty graph yields an empty result rather than raising: a target with
        no candidates has no route, and that is an answer rather than an error.
    """
    if not graph.nodes:
        return PathResult(node_indices=[], edges=[], score=0.0, excluded_indices=[])

    best, predecessor = score_table(graph, inclusion_bonus=inclusion_bonus)
    return path_from_table(graph, best, predecessor)
