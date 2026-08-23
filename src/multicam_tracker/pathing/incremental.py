"""Extending a trajectory as new sightings arrive, without starting over.

The live case (stage 15) reconstructs the same target every few seconds. Redoing
the whole search each time is wasteful, but the cheap version — appending the
new sighting to the tail — is wrong, and wrong in a way that is easy to miss.

**Streams deliver out of order.** A camera at the edge of the network buffers,
retries, or has its clock corrected, and its sighting arrives after sightings
that happened later. Blindly appending it produces a trajectory whose timestamps
run backwards; appending it *in place* without redoing anything leaves a route
that was chosen without knowing it existed. Either way the answer differs from
what a full recomputation would have given, and a live system that disagrees
with its own batch reprocessing is worse than one that is merely slow.

So the rule here is exactness, not speed-at-any-cost: **the incremental result
is identical to a full recomputation over the same candidates**, and there is a
property test asserting it over randomized arrival orders. What incremental
extension saves is work, never correctness.

How: the dynamic program is a forward pass over time-ordered nodes, so the score
at node *i* depends only on nodes before it. Inserting a node at position *p*
invalidates exactly the suffix from *p* onward — the prefix table stays valid,
and only the edges arriving at *p* or later need rebuilding.

Sightings older than the reorder buffer are **rejected and reported**, not
quietly dropped and not silently accepted. A stream running minutes behind is a
fault the operator needs to see, and a trajectory that keeps being rewritten
hours after the fact is not one anyone can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import Camera, MatchCandidate, Sighting
from multicam_tracker.pathing.graph import TrajectoryEdge, TrajectoryGraph, build_graph
from multicam_tracker.pathing.optimal_path import path_from_table, score_table
from multicam_tracker.pathing.preparation import PreparedCandidate, prepare_candidates
from multicam_tracker.pathing.reconstruction import ReconstructionResult, assemble_result
from multicam_tracker.topology import Topology

__all__ = ["ExtensionResult", "IncrementalReconstructor", "LateArrival"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class LateArrival:
    """A sighting that arrived too late to be admitted."""

    sighting_id: str
    timestamp_utc: datetime
    behind_by_sec: float
    """How far behind the trajectory's tail it landed."""

    reason: str


@dataclass
class ExtensionResult:
    """What one extension changed."""

    result: ReconstructionResult
    accepted_sighting_ids: list[str] = field(default_factory=list)
    rejected: list[LateArrival] = field(default_factory=list)
    recomputed_from_index: int | None = None
    """Where the suffix recomputation started. ``None`` when nothing changed."""

    @property
    def was_out_of_order(self) -> bool:
        """Return whether an accepted sighting landed before the previous tail."""
        return self.recomputed_from_index is not None and self.recomputed_from_index < (
            len(self.result.graph.nodes) - len(self.accepted_sighting_ids)
        )


class IncrementalReconstructor:
    """Maintains one target's trajectory as sightings arrive.

    Args:
        target_id: The target being tracked.
        topology: The camera graph.
        cameras: Cameras by id, for movement statistics.
        confirmed_only: Exclude candidates awaiting human review.
        reorder_buffer_sec: How far behind the tail a late sighting may land and
            still be admitted. Defaults to config.
        inclusion_bonus: Per-sighting bonus. Defaults to config.
        gap_penalty: Charge for a hop across unmonitored ground. Defaults to
            config.
    """

    def __init__(
        self,
        target_id: str,
        topology: Topology,
        *,
        cameras: dict[str, Camera] | None = None,
        confirmed_only: bool = False,
        reorder_buffer_sec: float | None = None,
        inclusion_bonus: float | None = None,
        gap_penalty: float | None = None,
    ) -> None:
        from multicam_tracker.config import get_settings

        self.target_id = target_id
        self.topology = topology
        self.cameras = cameras
        self.confirmed_only = confirmed_only
        self.inclusion_bonus = inclusion_bonus
        self.gap_penalty = gap_penalty
        self.reorder_buffer_sec = (
            reorder_buffer_sec
            if reorder_buffer_sec is not None
            else get_settings().pathing.reorder_buffer_sec
        )

        self._candidates: list[MatchCandidate] = []
        self._sightings: dict[str, Sighting] = {}
        self._nodes: list[PreparedCandidate] = []
        self._edges: list[TrajectoryEdge] = []
        self._best: list[float] = []
        self._predecessor: list[int | None] = []

    @property
    def nodes(self) -> list[PreparedCandidate]:
        """Return the candidates currently on the graph, time-ordered."""
        return list(self._nodes)

    @property
    def tail_timestamp(self) -> datetime | None:
        """Return the latest admitted sighting's timestamp, if any."""
        return self._nodes[-1].sighting.timestamp_utc if self._nodes else None

    def extend(
        self,
        candidates: list[MatchCandidate],
        sightings: dict[str, Sighting],
        *,
        activity: list[Sighting] | None = None,
    ) -> ExtensionResult:
        """Admit new candidates and update the trajectory.

        Args:
            candidates: Newly arrived match candidates.
            sightings: Their sightings, by id.
            activity: All sightings in the window for any vehicle, for outage
                detection.

        Returns:
            The updated reconstruction, which sightings were admitted, and which
            arrived too late.
        """
        rejected = self._reject_late(candidates, sightings)
        rejected_ids = {entry.sighting_id for entry in rejected}

        admitted = [
            candidate for candidate in candidates if candidate.sighting_id not in rejected_ids
        ]
        for candidate in admitted:
            self._candidates.append(candidate)
        self._sightings.update(sightings)

        previous_ids = [node.sighting_id for node in self._nodes]
        self._nodes = prepare_candidates(
            self._candidates, self._sightings, confirmed_only=self.confirmed_only
        )
        changed_from = _first_difference(previous_ids, [node.sighting_id for node in self._nodes])

        if changed_from is None and not rejected:
            return ExtensionResult(result=self._materialize(activity), recomputed_from_index=None)

        if changed_from is not None:
            self._rebuild_suffix(changed_from)

        accepted = [
            candidate.sighting_id
            for candidate in admitted
            if candidate.sighting_id in {node.sighting_id for node in self._nodes}
        ]
        return ExtensionResult(
            result=self._materialize(activity),
            accepted_sighting_ids=accepted,
            rejected=rejected,
            recomputed_from_index=changed_from,
        )

    def _reject_late(
        self, candidates: list[MatchCandidate], sightings: dict[str, Sighting]
    ) -> list[LateArrival]:
        """Identify sightings that fall outside the reorder buffer.

        Args:
            candidates: Newly arrived candidates.
            sightings: Their sightings, by id.

        Returns:
            One entry per rejected candidate.
        """
        tail = self.tail_timestamp
        if tail is None:
            return []

        rejected: list[LateArrival] = []
        for candidate in candidates:
            sighting = sightings.get(candidate.sighting_id)
            if sighting is None:
                continue
            behind = (tail - sighting.timestamp_utc).total_seconds()
            if behind <= self.reorder_buffer_sec:
                continue
            rejected.append(
                LateArrival(
                    sighting_id=candidate.sighting_id,
                    timestamp_utc=sighting.timestamp_utc,
                    behind_by_sec=behind,
                    reason=(
                        f"arrived {behind:.0f}s behind the trajectory tail, outside the "
                        f"{self.reorder_buffer_sec:.0f}s reorder buffer; admitting it would "
                        f"rewrite a route already acted on, so it is reported instead"
                    ),
                )
            )
            logger.warning(
                "trajectory_late_arrival",
                target_id=self.target_id,
                sighting_id=candidate.sighting_id,
                behind_by_sec=behind,
                reorder_buffer_sec=self.reorder_buffer_sec,
            )

        return rejected

    def _rebuild_suffix(self, from_index: int) -> None:
        """Rebuild edges and dynamic-programming state from one node onward.

        Args:
            from_index: First node whose incoming edges may have changed.
        """
        retained = [edge for edge in self._edges if edge.to_index < from_index]
        rebuilt = build_graph(
            self._nodes,
            self.topology,
            cameras=self.cameras,
            gap_penalty=self.gap_penalty,
            min_to_index=from_index,
        )
        self._edges = [*retained, *rebuilt.edges]
        graph = TrajectoryGraph(nodes=self._nodes, edges=self._edges)
        # Recomputing the whole table would also be correct; the suffix version
        # is what makes this cheaper than starting over, and the prefix scores
        # are provably unaffected because every edge points forward in time.
        self._best, self._predecessor = _partial_score_table(
            graph, self._best, self._predecessor, from_index, self.inclusion_bonus
        )

    def _materialize(self, activity: list[Sighting] | None) -> ReconstructionResult:
        """Assemble the current state into a full reconstruction.

        Args:
            activity: All sightings in the window, for outage detection.

        Returns:
            The reconstruction.
        """
        graph = TrajectoryGraph(nodes=self._nodes, edges=self._edges)
        return assemble_result(
            self.target_id,
            graph,
            path_from_table(graph, self._best, self._predecessor),
            self.topology,
            cameras=self.cameras,
            activity=activity,
            inclusion_bonus=self.inclusion_bonus,
        )


def _first_difference(previous: list[str], current: list[str]) -> int | None:
    """Return the first position at which two node sequences differ.

    Args:
        previous: Sighting ids before the update.
        current: Sighting ids after it.

    Returns:
        The index, or ``None`` when the sequences are identical.
    """
    for index, (before, after) in enumerate(zip(previous, current, strict=False)):
        if before != after:
            return index
    if len(previous) != len(current):
        return min(len(previous), len(current))
    return None


def _partial_score_table(
    graph: TrajectoryGraph,
    best: list[float],
    predecessor: list[int | None],
    from_index: int,
    inclusion_bonus: float | None,
) -> tuple[list[float], list[int | None]]:
    """Recompute the dynamic program from one node onward.

    Args:
        graph: The graph, with edges already rebuilt.
        best: The previous table.
        predecessor: The previous back-pointers.
        from_index: First node to recompute.
        inclusion_bonus: Per-sighting bonus, or ``None`` for config.

    Returns:
        The updated ``(best, predecessor)``.
    """
    from multicam_tracker.config import get_settings

    if from_index <= 0 or from_index > len(best):
        return score_table(graph, inclusion_bonus=inclusion_bonus)

    bonus = (
        inclusion_bonus
        if inclusion_bonus is not None
        else get_settings().thresholds.path_node_inclusion_bonus
    )

    updated_best = [*best[:from_index], *([bonus] * (len(graph.nodes) - from_index))]
    updated_predecessor: list[int | None] = [
        *predecessor[:from_index],
        *([None] * (len(graph.nodes) - from_index)),
    ]

    incoming: dict[int, list[TrajectoryEdge]] = {}
    for edge in graph.edges:
        if edge.to_index >= from_index:
            incoming.setdefault(edge.to_index, []).append(edge)

    for index in range(from_index, len(graph.nodes)):
        for edge in sorted(incoming.get(index, []), key=lambda item: item.from_index):
            candidate = updated_best[edge.from_index] + edge.weight + bonus
            if candidate > updated_best[index]:
                updated_best[index] = candidate
                updated_predecessor[index] = edge.from_index

    return updated_best, updated_predecessor
