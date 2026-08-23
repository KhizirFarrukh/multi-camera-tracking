"""Alternative routes, and knowing when there is no single answer.

Sometimes two routes explain the evidence equally well — most obviously when a
cloned plate puts one identity in two places, but also whenever coverage is thin
enough that a gap hop and a real hop score alike. Returning the higher-scoring
one and calling it *the* trajectory presents a coin flip as a fact.

So the search reports the top *k* and the margin between the best two. Below the
configured margin the result is marked ambiguous and the alternatives travel
with it, for the operator to choose between.

**Normalized margin.** The raw score difference is meaningless across
trajectories of different lengths: 0.4 between two ten-hop paths is noise, while
0.4 between two two-hop paths is decisive. The margin is therefore divided by
the magnitude of the best score::

    margin = (best - second) / max(|best|, epsilon)

which makes the threshold a single number that means the same thing everywhere.

**The k-best algorithm** is the same dynamic program as
:mod:`~multicam_tracker.pathing.optimal_path`, carrying the ``k`` best partial
paths at each node instead of one. That stays exact and linear in ``k * E``,
where a naive "find best, remove an edge, search again" loop would be neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from multicam_tracker.pathing.graph import TrajectoryEdge, TrajectoryGraph
from multicam_tracker.pathing.optimal_path import PathResult

__all__ = ["AmbiguityReport", "k_best_paths"]

_EPSILON = 1e-9
"""Guards the normalization against a best score of zero."""


@dataclass(frozen=True)
class AmbiguityReport:
    """How close the runner-up came, and whether that is too close to call."""

    best_score: float
    second_score: float | None
    """``None`` when only one path exists, which is unambiguous by definition."""

    margin: float
    """Normalized separation. ``1.0`` when there is no alternative at all."""

    is_ambiguous: bool
    alternatives: list[PathResult]
    """Every path returned, best first, including the best one."""

    def describe(self) -> str:
        """Explain the verdict in one sentence.

        Returns:
            A sentence an operator can act on.
        """
        if self.second_score is None:
            return "no alternative route was found; this is the only path the evidence supports"
        if self.is_ambiguous:
            return (
                f"the two best routes score {self.best_score:.3f} and "
                f"{self.second_score:.3f} -- a normalized margin of {self.margin:.3f}. "
                f"The evidence does not distinguish them; both are shown rather than "
                f"one being presented as the answer"
            )
        return (
            f"the best route scores {self.best_score:.3f} against {self.second_score:.3f} "
            f"for the runner-up (margin {self.margin:.3f}), a clear separation"
        )


def _paths_ending_at(
    graph: TrajectoryGraph, k: int, inclusion_bonus: float
) -> list[list[tuple[float, list[int]]]]:
    """Return the ``k`` best partial paths ending at each node.

    Args:
        graph: The graph to search.
        k: How many to keep per node.
        inclusion_bonus: Per-sighting bonus.

    Returns:
        Per node, a descending list of ``(score, node_indices)``.
    """
    incoming: dict[int, list[TrajectoryEdge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.to_index, []).append(edge)

    table: list[list[tuple[float, list[int]]]] = []
    for index in range(len(graph.nodes)):
        # Starting fresh at this node is always an option, which is what lets
        # the search begin anywhere rather than only at the earliest candidate.
        options: list[tuple[float, list[int]]] = [(inclusion_bonus, [index])]
        for edge in incoming.get(index, []):
            for score, prefix in table[edge.from_index]:
                options.append((score + edge.weight + inclusion_bonus, [*prefix, index]))

        # Ties resolve toward the path that started earlier, then toward the
        # lexicographically smaller index sequence -- a total order, so the
        # result never depends on the order options were generated in.
        options.sort(key=lambda option: (-option[0], option[1]))
        table.append(options[:k])

    return table


def _competing(
    candidates: list[tuple[float, list[int]]], limit: int
) -> list[tuple[float, list[int]]]:
    """Reduce scored paths to genuinely competing explanations.

    Alternatives have to differ in *which* sightings they attribute to the
    target, not merely in how many. A path whose sightings are a subset of a
    better one -- or a superset of it -- is not a competing account of what
    happened; it is the same account with more or less of it. Every route has
    dozens of such shadows (drop the last node, add a trailing weak one), and
    letting them fill the list would make the ambiguity margin measure "how much
    did the final hop contribute" rather than "is there another story here",
    flagging perfectly decisive trajectories as too close to call.

    So two routes compete only when each contains a sighting the other does not.
    That is exactly the situation an operator has to adjudicate: not "is this
    trailing sighting included", but "which of these two vehicles was it".

    Args:
        candidates: Scored paths, already sorted best first.
        limit: How many to keep.

    Returns:
        At most ``limit`` paths, no two of which are nested.
    """
    retained: list[tuple[float, list[int]]] = []
    retained_sets: list[set[int]] = []

    for score, indices in candidates:
        as_set = set(indices)
        if any(as_set <= kept or kept <= as_set for kept in retained_sets):
            continue
        retained.append((score, indices))
        retained_sets.append(as_set)
        if len(retained) == limit:
            break

    return retained


def k_best_paths(
    graph: TrajectoryGraph,
    k: int | None = None,
    *,
    inclusion_bonus: float | None = None,
    ambiguity_margin_min: float | None = None,
) -> AmbiguityReport:
    """Return the top-``k`` distinct paths and judge whether the best is decisive.

    Args:
        graph: The graph to search.
        k: How many paths to return. Fewer are returned without raising when
            fewer exist. Defaults to config.
        inclusion_bonus: Per-sighting bonus. Defaults to config.
        ambiguity_margin_min: Normalized margin below which the result is
            ambiguous. Defaults to config.

    Returns:
        The report. For an empty graph the best score is ``0.0``, there are no
        alternatives, and the result is not ambiguous -- an empty answer is
        unambiguous.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    from multicam_tracker.config import get_settings

    settings = get_settings()
    limit = k if k is not None else settings.pathing.k_best_default
    bonus = (
        inclusion_bonus
        if inclusion_bonus is not None
        else settings.thresholds.path_node_inclusion_bonus
    )
    margin_min = (
        ambiguity_margin_min
        if ambiguity_margin_min is not None
        else settings.thresholds.path_ambiguity_margin_min
    )

    if limit <= 0:
        msg = f"k must be positive; got {limit}"
        raise ValueError(msg)

    if not graph.nodes:
        return AmbiguityReport(
            best_score=0.0, second_score=None, margin=1.0, is_ambiguous=False, alternatives=[]
        )

    # Kept per node rather than only the global k, because a route that loses
    # overall may still be the best one ending where it does, and dropping it
    # early would hide a genuine alternative.
    table = _paths_ending_at(graph, limit + 1, bonus)

    # Every partial path is also a complete path: a route may end at any node,
    # not only the last one in time.
    candidates = sorted(
        (option for options in table for option in options),
        key=lambda option: (-option[0], option[1]),
    )
    top = _competing(candidates, limit)

    by_pair = {(edge.from_index, edge.to_index): edge for edge in graph.edges}
    alternatives: list[PathResult] = []
    for score, indices in top:
        on_path = set(indices)
        alternatives.append(
            PathResult(
                node_indices=list(indices),
                edges=[by_pair[(origin, destination)] for origin, destination in pairwise(indices)],
                score=score,
                excluded_indices=[
                    index for index in range(len(graph.nodes)) if index not in on_path
                ],
            )
        )

    best_score = alternatives[0].score
    second_score = alternatives[1].score if len(alternatives) > 1 else None
    if second_score is None:
        margin = 1.0
    else:
        margin = (best_score - second_score) / max(abs(best_score), _EPSILON)

    return AmbiguityReport(
        best_score=best_score,
        second_score=second_score,
        margin=margin,
        is_ambiguous=second_score is not None and margin < margin_min,
        alternatives=alternatives,
    )
