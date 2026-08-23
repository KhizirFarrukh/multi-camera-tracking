"""The DAG of possible routes through a set of candidate sightings.

Every node is a candidate sighting; every edge is a transition the vehicle could
have made. Because an edge exists only when its destination is strictly later
than its origin, the graph is acyclic by construction rather than by checking --
time provides the topological order for free, and the exact optimum is then one
linear pass (:mod:`~multicam_tracker.pathing.optimal_path`).

Three kinds of edge, and the third is the one that makes this usable on real
camera networks:

``DIRECT``
    A declared link between the two cameras, traversed within its travel-time
    window. The ordinary case.

``INDIRECT``
    No direct link, but the topology says the vehicle could have got there via
    intermediate cameras in the time available. It passed cameras that did not
    see it, or did not match it -- unremarkable at any real coverage level.

``IMPLAUSIBLE``
    A route exists and the observed time violates it -- the vehicle would have
    had to travel faster than the link allows, or dawdle far longer than its
    window. Kept, because a detour or a stop is real and a hard rejection would
    delete it, but multiplied by the configured implausibility penalty so it
    loses to any account that does not require the timetable to be wrong.

``GAP``
    None of the above. The vehicle left monitored ground entirely. This edge
    exists on purpose and is penalised rather than forbidden: refusing it would
    split one vehicle's route into disconnected fragments the moment it drove
    through an unmonitored district, which is the normal condition of every
    camera network that is not a closed campus.

The distinction between the last two matters more than it looks. Both used to be
"gap", and lumping them together priced an 18-second violation of a travel-time
window the same as a drive through an unwatched district -- so a decoy that could
not have made the trip cost the same as a genuine sighting that simply passed no
camera. They are different claims about the world and they carry different
penalties.

The penalty is what keeps ``GAP`` honest. It is set above the per-node inclusion
bonus, so reaching across unmonitored ground to collect one more sighting is a
net loss unless that sighting is itself well supported.

**No edge is ever created for a physically impossible transition.** When camera
coordinates are available, a pair implying a speed above the configured limit
produces no edge at all, so no route through it exists to be scored. This is a
hard constraint rather than a penalty on purpose: the travel-time windows are
derived from a speed model and can be optimistic, and a 700 km/h hop that merely
scores badly is still a hop the search may take when the alternative scores
worse. Stage 20 requires that no failure mode produces a confident wrong
trajectory, and a teleport is the most visible wrong trajectory there is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from multicam_tracker.models import Camera, GeoPoint
from multicam_tracker.pathing.preparation import PreparedCandidate
from multicam_tracker.topology import (
    PlausibilityReason,
    ReachableCamera,
    Topology,
    describe_gap,
    is_transition_plausible,
    plausibility_score,
    reachable_within,
)

__all__ = ["EdgeKind", "TrajectoryEdge", "TrajectoryGraph", "build_graph"]


class EdgeKind(StrEnum):
    """How a transition is supported by the topology."""

    DIRECT = "direct"
    """A declared link, traversed inside its travel-time window."""

    INDIRECT = "indirect"
    """No direct link, but reachable via intermediate cameras in the time available."""

    IMPLAUSIBLE = "implausible"
    """A route exists; the observed time violates it. Penalised, not discarded."""

    GAP = "gap"
    """Unmonitored ground. Permitted, penalised, and always recorded as such."""


@dataclass(frozen=True)
class TrajectoryEdge:
    """One possible transition between two candidate sightings."""

    from_index: int
    to_index: int
    elapsed_sec: float
    kind: EdgeKind
    plausibility: float
    """Stage 04's graded plausibility of the transition, in ``[0, 1]``."""

    weight: float
    """What the path search maximises. See :func:`edge_weight`."""

    reason: str
    """Human-readable justification, carried into the explanation output."""

    @property
    def is_gap(self) -> bool:
        """Return whether this edge crosses unmonitored ground."""
        return self.kind is EdgeKind.GAP

    @property
    def is_unaccounted(self) -> bool:
        """Return whether the topology could not fully account for this hop."""
        return self.kind in {EdgeKind.GAP, EdgeKind.IMPLAUSIBLE}


@dataclass(frozen=True)
class TrajectoryGraph:
    """Candidate sightings and every transition between them worth considering."""

    nodes: list[PreparedCandidate]
    """Strictly time-ordered, so index order is a topological order."""

    edges: list[TrajectoryEdge]

    def outgoing(self, index: int) -> list[TrajectoryEdge]:
        """Return the edges leaving one node.

        Args:
            index: Node position.

        Returns:
            Its outgoing edges, in construction order.
        """
        return [edge for edge in self.edges if edge.from_index == index]

    def incoming(self, index: int) -> list[TrajectoryEdge]:
        """Return the edges entering one node.

        Args:
            index: Node position.

        Returns:
            Its incoming edges, in construction order.
        """
        return [edge for edge in self.edges if edge.to_index == index]

    @property
    def is_acyclic(self) -> bool:
        """Return whether every edge points forward in time.

        A property rather than a validation step: this is what makes the linear
        dynamic program exact, so it is worth being able to assert directly in a
        test rather than trusting the construction comment.
        """
        return all(edge.from_index < edge.to_index for edge in self.edges)


def edge_weight(
    origin: PreparedCandidate,
    destination: PreparedCandidate,
    plausibility: float,
    kind: EdgeKind,
    *,
    gap_penalty: float,
    implausible_penalty: float,
) -> float:
    """Score one transition.

    The three inputs the contract names for hop confidence, combined
    multiplicatively rather than additively::

        weight = origin.confidence * destination.confidence * plausibility
                 x (implausible_penalty if the timetable was violated)
                 - (gap_penalty if the hop crosses unmonitored ground)

    Multiplicative because these are not independent contributions to be traded
    off: a hop between two sightings is only as good as its *weakest* component.
    A confident match at each end means nothing if the transit was impossible,
    and a perfect transit means nothing between two sightings that are probably
    not the target. A sum would let a strong plausibility paper over a weak
    match, which is exactly how a false positive enters a route.

    Args:
        origin: The earlier candidate.
        destination: The later candidate.
        plausibility: Graded topology plausibility in ``[0, 1]``.
        kind: How the transition is supported.
        gap_penalty: Charge for crossing unmonitored ground.
        implausible_penalty: Multiplier for a hop that violates a travel-time
            window. A multiplier rather than a subtraction so it scales with the
            evidence: discounting a well-supported hop costs more than
            discounting a weak one, which is the right ordering.

    Returns:
        The edge weight. May be negative for a gap hop between weak matches,
        which is the point: such an edge should lose to skipping the node.
    """
    base = origin.confidence * destination.confidence * plausibility
    if kind is EdgeKind.IMPLAUSIBLE:
        base *= implausible_penalty
    return base - (gap_penalty if kind is EdgeKind.GAP else 0.0)


def _exceeds_speed_limit(
    cameras: dict[str, Camera],
    origin: PreparedCandidate,
    destination: PreparedCandidate,
    elapsed_sec: float,
    max_speed_kph: float,
) -> bool:
    """Return whether a transition would require an impossible speed.

    Args:
        cameras: Cameras by id.
        origin: The earlier candidate.
        destination: The later candidate.
        elapsed_sec: Seconds between them.
        max_speed_kph: The limit.

    Returns:
        ``True`` when the straight-line distance over the elapsed time exceeds
        the limit. A camera missing from the map yields ``False``: an unknown
        position is not evidence of an impossible one, and refusing the edge
        would silently truncate routes through decommissioned cameras.
    """
    from_camera = cameras.get(origin.camera_id)
    to_camera = cameras.get(destination.camera_id)
    if from_camera is None or to_camera is None or elapsed_sec <= 0.0:
        return False

    distance = GeoPoint(lat=from_camera.lat, lon=from_camera.lon).haversine_distance_to(
        GeoPoint(lat=to_camera.lat, lon=to_camera.lon)
    )
    return (distance / elapsed_sec) * 3.6 > max_speed_kph


def _classify(
    topology: Topology,
    origin: PreparedCandidate,
    destination: PreparedCandidate,
    elapsed_sec: float,
    arrivals: dict[str, ReachableCamera],
) -> tuple[EdgeKind, str]:
    """Decide how a transition is supported and say why.

    Args:
        topology: The camera graph.
        origin: The earlier candidate.
        destination: The later candidate.
        elapsed_sec: Seconds between them.
        arrivals: Cameras reachable from the origin within the hop budget, by
            camera id, with the window each could have been reached in.

    Returns:
        ``(kind, reason)``.
    """
    verdict = is_transition_plausible(
        topology, origin.camera_id, destination.camera_id, elapsed_sec
    )
    if verdict.plausible:
        return (
            EdgeKind.DIRECT,
            f"direct link {origin.camera_id} -> {destination.camera_id} traversed in "
            f"{elapsed_sec:.0f}s, inside its travel-time window",
        )

    if verdict.reason is PlausibilityReason.SAME_CAMERA:
        return (
            EdgeKind.GAP,
            f"returned to {origin.camera_id} after {elapsed_sec:.0f}s; whatever route "
            f"the vehicle took between the two passes was not observed",
        )

    reachable = arrivals.get(destination.camera_id)
    if reachable is not None and reachable.arrival_window.contains(
        destination.sighting.timestamp_utc
    ):
        return (
            EdgeKind.INDIRECT,
            f"no direct link, but {destination.camera_id} is reachable from "
            f"{origin.camera_id} in {reachable.hops} hops within the arrival window "
            f"{reachable.arrival_window.start_utc.isoformat()} to "
            f"{reachable.arrival_window.end_utc.isoformat()}; {elapsed_sec:.0f}s observed",
        )

    # A declared link the vehicle could not have used in this time is a
    # different claim from unmonitored ground: the topology does account for the
    # pair, and says no.
    if topology.get_link(origin.camera_id, destination.camera_id) is not None or (
        reachable is not None
    ):
        return (
            EdgeKind.IMPLAUSIBLE,
            describe_gap(topology, origin.camera_id, destination.camera_id, elapsed_sec),
        )

    return (
        EdgeKind.GAP,
        describe_gap(topology, origin.camera_id, destination.camera_id, elapsed_sec),
    )


def build_graph(
    nodes: list[PreparedCandidate],
    topology: Topology,
    *,
    cameras: dict[str, Camera] | None = None,
    gap_penalty: float | None = None,
    max_skip_hops: int | None = None,
    max_edge_span_sec: float | None = None,
    max_speed_kph: float | None = None,
    min_to_index: int = 0,
) -> TrajectoryGraph:
    """Build the DAG of every transition worth considering.

    Args:
        nodes: Prepared candidates, strictly time-ordered.
        topology: The camera graph.
        cameras: Cameras by id. Supplying them enables the physical speed
            constraint; without coordinates no speed can be computed and the
            constraint cannot be applied, which a caller should treat as a
            weaker guarantee rather than an equivalent one.
        gap_penalty: Charge for a hop across unmonitored ground. Defaults to
            config.
        max_skip_hops: Hop budget for judging indirect reachability. Defaults to
            config.
        max_edge_span_sec: Optional cap on how far apart two candidates may be
            and still be joined. Left open by default: a vehicle can park for an
            hour, and an unjoinable pair fragments a route that really happened.
        max_speed_kph: Implied speed above which no edge is created at all.
            Defaults to the configured topology limit.
        min_to_index: Emit only edges arriving at this node or later. Used by
            incremental extension, which keeps the edges into the unchanged
            prefix and rebuilds only the rest. Zero rebuilds everything.

    Returns:
        The graph. Edges are emitted in ``(from_index, to_index)`` order, which
        is what makes the search deterministic.
    """
    from multicam_tracker.config import get_settings

    settings = get_settings()
    penalty = gap_penalty if gap_penalty is not None else settings.thresholds.path_gap_edge_penalty
    hops = max_skip_hops if max_skip_hops is not None else settings.pathing.max_skip_hops
    speed_limit = (
        max_speed_kph if max_speed_kph is not None else settings.topology.implausible_speed_kph
    )
    implausible_penalty = settings.thresholds.hop_implausible_penalty

    edges: list[TrajectoryEdge] = []
    for from_index, origin in enumerate(nodes):
        if from_index + 1 >= len(nodes):
            break
        if max(from_index + 1, min_to_index) >= len(nodes):
            continue

        # One reachability expansion per origin rather than one per pair: the
        # answer depends on the origin and the horizon, not on which later
        # candidate is being asked about.
        horizon = (nodes[-1].sighting.timestamp_utc - origin.sighting.timestamp_utc).total_seconds()
        if max_edge_span_sec is not None:
            horizon = min(horizon, max_edge_span_sec)
        arrivals = {
            entry.camera_id: entry
            for entry in reachable_within(
                topology,
                origin.camera_id,
                origin.sighting.timestamp_utc,
                max(horizon, 0.0),
                max_hops=hops,
            ).cameras
        }

        for to_index in range(max(from_index + 1, min_to_index), len(nodes)):
            destination = nodes[to_index]
            elapsed = (
                destination.sighting.timestamp_utc - origin.sighting.timestamp_utc
            ).total_seconds()
            if elapsed <= 0.0:
                # Equal timestamps carry no defined order, so no edge: a hop
                # with zero elapsed time satisfies no travel-time window and the
                # Trajectory model rejects it outright.
                continue
            if max_edge_span_sec is not None and elapsed > max_edge_span_sec:
                continue
            if cameras is not None and _exceeds_speed_limit(
                cameras, origin, destination, elapsed, speed_limit
            ):
                continue

            kind, reason = _classify(topology, origin, destination, elapsed, arrivals)
            # An indirect transit scores as plausible outright, not at the
            # unlinked floor. The floor exists for pairs the topology cannot
            # account for at all; this pair it can -- the reachability query
            # bounded both the earliest and the latest arrival, and the observed
            # time fell inside. What is uncertain about such a hop is that
            # nothing was seen in between, and that is reported as a coverage
            # gap rather than discounted twice.
            plausibility = (
                1.0
                if kind is EdgeKind.INDIRECT
                else plausibility_score(topology, origin.camera_id, destination.camera_id, elapsed)
            )
            edges.append(
                TrajectoryEdge(
                    from_index=from_index,
                    to_index=to_index,
                    elapsed_sec=elapsed,
                    kind=kind,
                    plausibility=plausibility,
                    weight=edge_weight(
                        origin,
                        destination,
                        plausibility,
                        kind,
                        gap_penalty=penalty,
                        implausible_penalty=implausible_penalty,
                    ),
                    reason=reason,
                )
            )

    return TrajectoryGraph(nodes=list(nodes), edges=edges)
