"""Assembling the pieces into one answer.

Everything else in this package does one job well; this puts them in order and
produces the :class:`~multicam_tracker.models.Trajectory` the rest of the system
consumes:

1. prepare the candidates,
2. build the DAG of possible routes,
3. take the highest-scoring route, and the runners-up,
4. classify the stretches it cannot account for,
5. aggregate hop confidence into one number,
6. derive movement statistics, and
7. record why every sighting is in or out.

The result carries far more than the trajectory. Stage 18's review queue needs
the alternatives, stage 19's audit trail needs the explanation, and stage 20
needs the movement flags — and every one of those is impossible to reconstruct
after the fact from the trajectory alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import (
    Camera,
    CoverageGap,
    MatchCandidate,
    Sighting,
    TemporalIntegrity,
    Trajectory,
    TrajectoryHop,
)
from multicam_tracker.pathing.confidence import (
    ConfidenceStrategy,
    aggregate_confidence,
    hop_confidence,
)
from multicam_tracker.pathing.explanation import TrajectoryExplanation, explain_path
from multicam_tracker.pathing.gaps import GapReport, detect_gaps
from multicam_tracker.pathing.graph import EdgeKind, TrajectoryGraph, build_graph
from multicam_tracker.pathing.kbest import AmbiguityReport, k_best_paths
from multicam_tracker.pathing.movement import MovementSummary, summarize_movement
from multicam_tracker.pathing.optimal_path import PathResult, best_path
from multicam_tracker.pathing.preparation import PreparedCandidate, prepare_candidates
from multicam_tracker.topology import Topology

__all__ = ["ReconstructionResult", "assemble_result", "reconstruct_trajectory"]

logger = get_logger(__name__)


@dataclass
class ReconstructionResult:
    """One reconstruction, and everything needed to defend it."""

    trajectory: Trajectory | None
    """``None`` when no candidate survived preparation. An absent trajectory is
    an answer -- "nothing supports a route for this target" -- not an error."""

    graph: TrajectoryGraph
    path: PathResult
    gaps: list[GapReport] = field(default_factory=list)
    movement: MovementSummary = field(default_factory=MovementSummary)
    ambiguity: AmbiguityReport | None = None
    explanation: TrajectoryExplanation | None = None

    @property
    def is_ambiguous(self) -> bool:
        """Return whether a runner-up route was too close to call."""
        return self.ambiguity is not None and self.ambiguity.is_ambiguous

    @property
    def has_teleport(self) -> bool:
        """Return whether any hop implies a physically impossible speed.

        The property stage 20 checks and the one the whole engine exists to keep
        false: a route containing an impossible hop is wrong however confident
        it looks.
        """
        return self.movement.has_implausible_speed


def _build_hops(
    path: PathResult, graph: TrajectoryGraph, *, implausible_penalty: float | None
) -> tuple[list[TrajectoryHop], list[float]]:
    """Build the domain hop records for a route.

    Args:
        path: The chosen route.
        graph: The graph it runs through.
        implausible_penalty: Multiplier for gap hops, or ``None`` for config.

    Returns:
        ``(hops, hop_confidences)``.
    """
    hops: list[TrajectoryHop] = []
    confidences: list[float] = []

    for position, edge in enumerate(path.edges):
        origin = graph.nodes[path.node_indices[position]]
        destination = graph.nodes[path.node_indices[position + 1]]
        confidence = hop_confidence(
            origin, destination, edge, implausible_penalty=implausible_penalty
        )
        confidences.append(confidence)
        hops.append(
            TrajectoryHop(
                from_sighting_id=origin.sighting_id,
                to_sighting_id=destination.sighting_id,
                from_camera_id=origin.camera_id,
                to_camera_id=destination.camera_id,
                elapsed_sec=edge.elapsed_sec,
                # Strictly the direct case: the model documents this flag as
                # "elapsed_sec fell inside the CameraLink travel-time window",
                # and an indirect route did not, however plausible it is.
                topology_plausible=edge.kind is EdgeKind.DIRECT,
                hop_confidence=confidence,
            )
        )

    return hops, confidences


def reconstruct_trajectory(
    target_id: str,
    candidates: list[MatchCandidate],
    sightings: dict[str, Sighting],
    topology: Topology,
    *,
    cameras: dict[str, Camera] | None = None,
    activity: list[Sighting] | None = None,
    confirmed_only: bool = False,
    k_alternatives: int | None = None,
    inclusion_bonus: float | None = None,
    gap_penalty: float | None = None,
    confidence_strategy: ConfidenceStrategy = ConfidenceStrategy.GEOMETRIC_MEAN,
    prepared: list[PreparedCandidate] | None = None,
    temporal_integrity: TemporalIntegrity | None = None,
) -> ReconstructionResult:
    """Reconstruct the most plausible route for one target.

    Args:
        target_id: The target the trajectory belongs to.
        candidates: Scored match candidates, in any order.
        sightings: Sightings by id.
        topology: The camera graph.
        cameras: Cameras by id, for movement statistics. Movement is reported
            empty when omitted rather than the reconstruction failing.
        activity: All sightings in the window for any vehicle, so a silent
            camera can be told from a broken one. See
            :func:`~multicam_tracker.pathing.gaps.detect_gaps`.
        confirmed_only: Exclude candidates still awaiting human review.
        k_alternatives: How many alternative routes to enumerate. Defaults to
            config.
        inclusion_bonus: Per-sighting bonus. Defaults to config.
        gap_penalty: Charge for a hop across unmonitored ground. Defaults to
            config.
        confidence_strategy: How hop confidences aggregate.
        prepared: Pre-prepared candidates, used by incremental extension to
            avoid redoing the filtering. Bypasses ``candidates`` entirely.
        temporal_integrity: The stage 09 verdict on whether these cameras'
            timestamps are comparable. Carried onto the trajectory rather than
            logged, because an operator reading a route has to see the caveat
            beside it. ``None`` means the route was assembled without checking,
            which is a different claim from "checked and clean".

    Returns:
        The reconstruction, including an absent trajectory when nothing
        survived preparation.
    """
    nodes = (
        prepared
        if prepared is not None
        else prepare_candidates(candidates, sightings, confirmed_only=confirmed_only)
    )
    graph = build_graph(nodes, topology, cameras=cameras, gap_penalty=gap_penalty)

    if not nodes:
        return ReconstructionResult(
            trajectory=None,
            graph=graph,
            path=PathResult(node_indices=[], edges=[], score=0.0),
        )

    return assemble_result(
        target_id,
        graph,
        best_path(graph, inclusion_bonus=inclusion_bonus),
        topology,
        cameras=cameras,
        activity=activity,
        k_alternatives=k_alternatives,
        inclusion_bonus=inclusion_bonus,
        confidence_strategy=confidence_strategy,
        temporal_integrity=temporal_integrity,
    )


def assemble_result(
    target_id: str,
    graph: TrajectoryGraph,
    path: PathResult,
    topology: Topology,
    *,
    cameras: dict[str, Camera] | None = None,
    activity: list[Sighting] | None = None,
    k_alternatives: int | None = None,
    inclusion_bonus: float | None = None,
    confidence_strategy: ConfidenceStrategy = ConfidenceStrategy.GEOMETRIC_MEAN,
    temporal_integrity: TemporalIntegrity | None = None,
) -> ReconstructionResult:
    """Turn a chosen route into the trajectory and everything that explains it.

    Separated from the search so incremental extension, which already knows the
    route, does not have to re-run the search to produce a result.

    Args:
        target_id: The target the trajectory belongs to.
        graph: The searched graph.
        path: The chosen route through it.
        topology: The camera graph.
        cameras: Cameras by id, for movement statistics.
        activity: All sightings in the window for any vehicle, for outage
            detection.
        k_alternatives: How many alternative routes to enumerate.
        inclusion_bonus: Per-sighting bonus. Defaults to config.
        confidence_strategy: How hop confidences aggregate.
        temporal_integrity: The stage 09 verdict, carried onto the trajectory.

    Returns:
        The complete reconstruction.
    """
    if not path.node_indices:
        return ReconstructionResult(trajectory=None, graph=graph, path=path)

    ambiguity = k_best_paths(graph, k_alternatives, inclusion_bonus=inclusion_bonus)
    hops, confidences = _build_hops(path, graph, implausible_penalty=None)
    gap_reports = detect_gaps(graph, path.node_indices, path.edges, topology, activity=activity)
    route = path.nodes(graph)

    overall = aggregate_confidence(
        confidences,
        single_sighting_confidence=route[0].confidence,
        strategy=confidence_strategy,
    )

    trajectory = Trajectory(
        target_id=target_id,
        sightings=[node.sighting for node in route],
        hops=hops,
        overall_confidence=overall,
        start_time_utc=route[0].sighting.timestamp_utc,
        end_time_utc=route[-1].sighting.timestamp_utc,
        gaps=[report.gap for report in gap_reports],
        temporal_integrity=temporal_integrity,
    )

    movement = summarize_movement(route, cameras) if cameras is not None else MovementSummary()

    if movement.has_implausible_speed:
        logger.warning(
            "trajectory_implausible_speed",
            target_id=target_id,
            trajectory_id=trajectory.trajectory_id,
            hop_positions=movement.implausible_hops,
            detail=(
                "a hop on the returned route implies a speed above the configured limit; "
                "the cause may be the match, the camera clock, or the topology, so the "
                "hop is flagged rather than removed"
            ),
        )

    return ReconstructionResult(
        trajectory=trajectory,
        graph=graph,
        path=path,
        gaps=gap_reports,
        movement=movement,
        ambiguity=ambiguity,
        explanation=explain_path(
            graph, path, ambiguity=ambiguity.describe(), inclusion_bonus=inclusion_bonus
        ),
    )


def gaps_as_models(reports: list[GapReport]) -> list[CoverageGap]:
    """Return just the domain models from a list of gap reports.

    Args:
        reports: Gap reports.

    Returns:
        The models, in the same order.
    """
    return [report.gap for report in reports]
