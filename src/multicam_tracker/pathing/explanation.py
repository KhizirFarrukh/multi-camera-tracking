"""Why the route is what it is, in a form a human can interrogate.

A trajectory a human cannot question is not usable evidence. An operator has to
be able to ask two things and get a straight answer: *why is this sighting in
the route?* and — more often, and more importantly — *why is that one not?*

The second question is the reason exclusions are recorded rather than discarded.
A search that quietly drops a candidate looks identical to a search that never
saw it, and the difference matters enormously when the dropped candidate is the
one the operator was expecting.

Everything here serializes to JSON and reloads intact, because the audit trail
outlives the process that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from multicam_tracker.pathing.graph import TrajectoryGraph
from multicam_tracker.pathing.optimal_path import PathResult, path_score

__all__ = [
    "ExclusionReason",
    "InclusionRecord",
    "TrajectoryExplanation",
    "explain_path",
]

MAX_EXPLAINED_EXCLUSIONS = 10
"""How many rejected candidates carry a full explanation.

Bounded because a busy corridor produces hundreds of rejects and an explanation
nobody can read is not an explanation. The highest-confidence rejects are the
ones kept: those are the candidates an operator is most likely to ask about."""


class ExclusionReason(StrEnum):
    """Which constraint kept a candidate out of the route."""

    LOWER_SCORING_PATH = "lower_scoring_path"
    """Including it produced a worse route overall. The ordinary case, and the
    one the global objective exists to decide."""

    NO_PLAUSIBLE_CONNECTION = "no_plausible_connection"
    """Reaching it required crossing unmonitored ground in both directions --
    the geographically impossible detour that gives false positives away."""

    NOT_REACHABLE_IN_TIME = "not_reachable_in_time"
    """A route exists but not in the time available."""


@dataclass(frozen=True)
class InclusionRecord:
    """Why one sighting is part of the route."""

    sighting_id: str
    camera_id: str
    timestamp_utc: str
    match_confidence: float
    match_method: str
    arrived_via: str
    """The reason string from the edge that reached it, or a note that it starts
    the route."""

    hop_confidence: float | None
    """``None`` for the first sighting, which has no preceding hop."""


@dataclass(frozen=True)
class ExclusionRecord:
    """Why one candidate is not part of the route."""

    sighting_id: str
    camera_id: str
    timestamp_utc: str
    match_confidence: float
    reason: ExclusionReason
    detail: str
    score_if_included: float | None
    """The objective value of the best route that does include it, when one
    exists. Being able to say "including it scores 3.1 against 3.8" is the
    difference between an explanation and an assertion."""


@dataclass
class TrajectoryExplanation:
    """The complete account of one reconstruction."""

    objective: str
    """The scoring rule in words, so a reader need not have the source open."""

    score: float
    included: list[InclusionRecord] = field(default_factory=list)
    excluded: list[ExclusionRecord] = field(default_factory=list)
    excluded_total: int = 0
    """How many candidates were excluded in all, including those not explained
    individually."""

    ambiguity: str = ""
    """The ambiguity verdict, when alternatives were enumerated."""

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for storage and for the review UI.

        Returns:
            A JSON-native mapping.
        """
        return {
            "objective": self.objective,
            "score": round(self.score, 6),
            "included": [asdict(record) for record in self.included],
            "excluded": [
                {**asdict(record), "reason": record.reason.value} for record in self.excluded
            ],
            "excluded_total": self.excluded_total,
            "ambiguity": self.ambiguity,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> TrajectoryExplanation:
        """Rebuild an explanation from its serialized form.

        Args:
            payload: A mapping produced by :meth:`to_json_dict`.

        Returns:
            The reconstructed explanation.
        """
        return cls(
            objective=payload["objective"],
            score=payload["score"],
            included=[InclusionRecord(**record) for record in payload["included"]],
            excluded=[
                ExclusionRecord(**{**record, "reason": ExclusionReason(record["reason"])})
                for record in payload["excluded"]
            ],
            excluded_total=payload["excluded_total"],
            ambiguity=payload.get("ambiguity", ""),
        )


OBJECTIVE_DESCRIPTION = (
    "maximum total score over all routes, where a route scores the sum of its "
    "hop weights (origin confidence x destination confidence x topology "
    "plausibility, less a penalty for hops across unmonitored ground) plus a "
    "fixed bonus for each sighting included"
)


def _exclusion_reason(
    graph: TrajectoryGraph, index: int, path_indices: set[int]
) -> tuple[ExclusionReason, str]:
    """Work out why one candidate is absent from the route.

    Args:
        graph: The searched graph.
        index: The excluded candidate's position.
        path_indices: Positions that are on the route.

    Returns:
        ``(reason, detail)``.
    """
    connecting = [
        edge
        for edge in graph.edges
        if (edge.from_index == index and edge.to_index in path_indices)
        or (edge.to_index == index and edge.from_index in path_indices)
    ]
    non_gap = [edge for edge in connecting if not edge.is_gap]

    if not connecting:
        return (
            ExclusionReason.NO_PLAUSIBLE_CONNECTION,
            "no transition to or from any sighting on the route was possible at all",
        )

    if not non_gap:
        return (
            ExclusionReason.NO_PLAUSIBLE_CONNECTION,
            "every route through this sighting has to cross unmonitored ground both "
            "coming and going, which costs more than the sighting is worth",
        )

    weakest = min(edge.plausibility for edge in non_gap)
    if weakest < 1.0:
        return (
            ExclusionReason.NOT_REACHABLE_IN_TIME,
            f"the best supported transition to this sighting is only {weakest:.2f} "
            f"plausible against the travel-time window",
        )

    return (
        ExclusionReason.LOWER_SCORING_PATH,
        "a route through this sighting scores lower than the one returned",
    )


def _best_score_including(graph: TrajectoryGraph, index: int, bonus: float | None) -> float | None:
    """Return the best objective value of any route containing one node.

    Args:
        graph: The searched graph.
        index: The node that must be included.
        bonus: Per-sighting bonus, or ``None`` for the configured value.

    Returns:
        The score, or ``None`` when the node cannot be reached from the search
        at all.
    """
    from multicam_tracker.pathing.optimal_path import score_table

    forward, _ = score_table(graph, inclusion_bonus=bonus)

    # Best suffix starting at the node: the same dynamic program run backwards.
    suffix = [0.0] * len(graph.nodes)
    outgoing: dict[int, list[Any]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.from_index, []).append(edge)

    from multicam_tracker.config import get_settings

    resolved_bonus = (
        bonus if bonus is not None else get_settings().thresholds.path_node_inclusion_bonus
    )
    for node in range(len(graph.nodes) - 1, -1, -1):
        best_extension = 0.0
        for edge in outgoing.get(node, []):
            best_extension = max(
                best_extension, edge.weight + resolved_bonus + suffix[edge.to_index]
            )
        suffix[node] = best_extension

    if index >= len(graph.nodes):
        return None
    return forward[index] + suffix[index]


def explain_path(
    graph: TrajectoryGraph,
    result: PathResult,
    *,
    ambiguity: str = "",
    inclusion_bonus: float | None = None,
    max_exclusions: int = MAX_EXPLAINED_EXCLUSIONS,
) -> TrajectoryExplanation:
    """Build the full account of one reconstruction.

    Args:
        graph: The graph that was searched.
        result: The chosen route.
        ambiguity: The ambiguity verdict, when alternatives were enumerated.
        inclusion_bonus: Per-sighting bonus, for recomputing what an excluded
            candidate would have scored. Defaults to config.
        max_exclusions: How many rejected candidates to explain individually.

    Returns:
        The explanation, ready to serialize.
    """
    from multicam_tracker.pathing.confidence import hop_confidence

    included: list[InclusionRecord] = []
    for position, node_index in enumerate(result.node_indices):
        candidate = graph.nodes[node_index]
        if position == 0:
            arrived_via = "first sighting on the route; nothing precedes it"
            hop_score: float | None = None
        else:
            edge = result.edges[position - 1]
            arrived_via = edge.reason
            hop_score = hop_confidence(
                graph.nodes[result.node_indices[position - 1]], candidate, edge
            )
        included.append(
            InclusionRecord(
                sighting_id=candidate.sighting_id,
                camera_id=candidate.camera_id,
                timestamp_utc=candidate.sighting.timestamp_utc.isoformat(),
                match_confidence=candidate.confidence,
                match_method=candidate.candidate.match_method.value,
                arrived_via=arrived_via,
                hop_confidence=hop_score,
            )
        )

    on_path = set(result.node_indices)
    ranked = sorted(
        result.excluded_indices,
        key=lambda index: (-graph.nodes[index].confidence, graph.nodes[index].sighting_id),
    )

    excluded: list[ExclusionRecord] = []
    for index in ranked[:max_exclusions]:
        candidate = graph.nodes[index]
        reason, detail = _exclusion_reason(graph, index, on_path)
        excluded.append(
            ExclusionRecord(
                sighting_id=candidate.sighting_id,
                camera_id=candidate.camera_id,
                timestamp_utc=candidate.sighting.timestamp_utc.isoformat(),
                match_confidence=candidate.confidence,
                reason=reason,
                detail=detail,
                score_if_included=_best_score_including(graph, index, inclusion_bonus),
            )
        )

    return TrajectoryExplanation(
        objective=OBJECTIVE_DESCRIPTION,
        score=result.score,
        included=included,
        excluded=excluded,
        excluded_total=len(result.excluded_indices),
        ambiguity=ambiguity,
    )


def verify_path_score(graph: TrajectoryGraph, result: PathResult) -> bool:
    """Return whether a result's score matches the documented objective.

    A cheap self-check: the search and the published scoring rule must agree, or
    the explanation is describing an algorithm the system does not run.

    Args:
        graph: The searched graph.
        result: The route to check.

    Returns:
        ``True`` when the recomputed score matches.
    """
    return abs(path_score(graph, result.node_indices) - result.score) < 1e-9
