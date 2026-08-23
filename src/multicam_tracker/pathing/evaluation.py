"""Measuring reconstructed routes against ground truth.

Trajectory quality is not one number. A route can contain every true sighting
and three false ones; it can contain half the truth in perfect order; it can be
entirely correct except for one hop that crosses the city in ninety seconds.
Those are different failures with different consequences, so they are counted
separately:

``exact_path``
    The reconstructed sequence equals the true one, in order. The strictest
    measure, and the one that falls fastest as data degrades.

``sighting precision / recall``
    Per-sighting agreement, ignoring order. What an operator actually sees.

``hop accuracy``
    Share of consecutive true pairs that appear as consecutive pairs in the
    reconstruction. Catches a route that has the right sightings in the wrong
    arrangement.

``teleport rate``
    Share of reconstructions containing a physically impossible hop. **The
    target is exactly zero**, and it is the only metric here that is not a
    trade-off: a route can be incomplete, or uncertain, and still be useful; a
    route that teleports is evidence of nothing and looks like evidence of
    something.

``gap detection``
    Whether the stretches the route could not account for were reported.
    Unreported gaps are the quiet failure -- a trajectory that silently bridges
    unmonitored ground reads as continuous observation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from multicam_tracker.models import Camera, MatchCandidate, ReviewStatus, Sighting
from multicam_tracker.pathing.reconstruction import ReconstructionResult, reconstruct_trajectory
from multicam_tracker.synth.generator import SyntheticDataset
from multicam_tracker.synth.ground_truth import GroundTruth
from multicam_tracker.topology import Topology

__all__ = ["PathingMetrics", "candidates_from_matches", "evaluate_pathing"]


@dataclass
class PathingMetrics:
    """What reconstruction scored on one dataset."""

    scenario_name: str

    exact_path_match: bool = False
    true_sightings: int = 0
    reconstructed_sightings: int = 0
    correct_sightings: int = 0

    true_hops: int = 0
    correct_hops: int = 0

    gaps_reported: int = 0
    gaps_expected: int = 0

    teleport_hops: int = 0
    """Hops implying a speed above the configured limit. Target: zero."""

    is_ambiguous: bool = False
    overall_confidence: float = 0.0
    decoys_included: int = 0
    """Sightings in the route that ground truth attributes to another vehicle."""

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        """Return a ratio, treating an empty denominator as perfect.

        Args:
            numerator: The count of interest.
            denominator: The population.

        Returns:
            ``1.0`` when the population is empty.
        """
        return 1.0 if denominator == 0 else numerator / denominator

    @property
    def sighting_precision(self) -> float:
        """Return the share of the route's sightings that were really the target."""
        return self._ratio(self.correct_sightings, self.reconstructed_sightings)

    @property
    def sighting_recall(self) -> float:
        """Return the share of the target's sightings the route contains."""
        return self._ratio(self.correct_sightings, self.true_sightings)

    @property
    def hop_accuracy(self) -> float:
        """Return the share of true consecutive pairs the route reproduces."""
        return self._ratio(self.correct_hops, self.true_hops)

    @property
    def teleport_rate(self) -> float:
        """Return the share of reconstructions containing an impossible hop.

        One dataset yields one reconstruction, so this is 0.0 or 1.0 per
        scenario and is averaged across scenarios by the evaluation script.
        """
        return 1.0 if self.teleport_hops > 0 else 0.0

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for a regression baseline.

        Returns:
            A JSON-native mapping with rates rounded so a diff shows a real
            change rather than float noise.
        """
        return {
            "scenario_name": self.scenario_name,
            "exact_path_match": self.exact_path_match,
            "sighting_precision": round(self.sighting_precision, 4),
            "sighting_recall": round(self.sighting_recall, 4),
            "hop_accuracy": round(self.hop_accuracy, 4),
            "teleport_hops": self.teleport_hops,
            "teleport_rate": round(self.teleport_rate, 4),
            "true_sightings": self.true_sightings,
            "reconstructed_sightings": self.reconstructed_sightings,
            "correct_sightings": self.correct_sightings,
            "decoys_included": self.decoys_included,
            "gaps_reported": self.gaps_reported,
            "is_ambiguous": self.is_ambiguous,
            "overall_confidence": round(self.overall_confidence, 4),
        }


def candidates_from_matches(
    dataset: SyntheticDataset,
    ground_truth: GroundTruth,
    target_id: str,
    *,
    review_min: float | None = None,
    use_embeddings: bool = True,
    topology: Topology | None = None,
    max_horizon_sec: float = 3600.0,
) -> list[MatchCandidate]:
    """Run stages 06 and 07 over a dataset and return the candidates they produce.

    Reconstruction is measured on the matchers real output rather than on ground
    truth, because a path engine that only works on perfect input has not been
    measured at all -- the false positives are the whole problem.

    Both evidence paths are used, and that is not a detail. On ``degraded``,
    plate matching alone returns **one** sighting of the five: nothing to
    reconstruct, and any pathing metric computed from it would be measuring
    stage 06. Appearance recovers the rest, which is what stage 07 was built
    for.

    The search starts from one confirmed anchor -- the targets first sighting
    carrying an embedding -- exactly as a real search does. It is included as a
    confirmed candidate in its own right rather than being treated as external
    knowledge.

    Args:
        dataset: The generated sightings.
        ground_truth: The answer key, for the targets plate and anchor.
        target_id: Identifier to record on each candidate.
        review_min: Plate score below which a match is discarded. Defaults to
            config.
        use_embeddings: Include the appearance path. ``False`` measures what
            reconstruction can do on plate evidence alone.
        topology: Constrains which sightings the appearance path compares
            against. Strongly recommended: unconstrained re-id on adversarial
            data returns decoys the path search then has to reject.
        max_horizon_sec: How far ahead of the anchor the appearance path looks.

    Returns:
        One candidate per sighting either path returned.

    Raises:
        ValueError: If the ground truth declares no target.
    """
    from multicam_tracker.config import get_settings
    from multicam_tracker.matching import combine_evidence
    from multicam_tracker.matching.aggregation import aggregate_similarity
    from multicam_tracker.matching.embedding_search import classify_embedding_matches
    from multicam_tracker.matching.plate_match import classify_plate_match
    from multicam_tracker.models import MatchMethod

    target = ground_truth.target
    if target is None:
        msg = f"scenario {ground_truth.scenario_name} declares no target vehicle"
        raise ValueError(msg)

    thresholds = get_settings().thresholds
    discard_below = review_min if review_min is not None else thresholds.plate_review_min_confidence

    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    embedded = [
        by_id[sid]
        for sid in target.sighting_ids
        if sid in by_id and by_id[sid].embedding is not None
    ]
    anchor = embedded[0] if (use_embeddings and embedded) else None

    similarity_by_id: dict[str, float] = {}
    appearance_ids: set[str] = set()
    if anchor is not None:
        references = [list(anchor.embedding or [])]
        admissible = _admissible_filter(topology, anchor, max_horizon_sec)
        scored = [
            (sighting, aggregate_similarity(sighting.embedding, references))
            for sighting in dataset.sightings
            if sighting.embedding is not None
            and sighting.sighting_id != anchor.sighting_id
            and admissible(sighting)
        ]
        similarity_by_id = {sighting.sighting_id: score for sighting, score in scored}
        appearance_ids = {
            entry.sighting.sighting_id for entry in classify_embedding_matches(scored)
        }

    candidates: list[MatchCandidate] = []
    if anchor is not None:
        # The anchor is evidence in its own right: a human confirmed it, which
        # is the strongest match state the system has.
        candidates.append(
            MatchCandidate(
                sighting_id=anchor.sighting_id,
                target_id=target_id,
                match_method=MatchMethod.MANUAL,
                match_score=1.0,
                review_status=ReviewStatus.CONFIRMED,
            )
        )

    for sighting in dataset.sightings:
        if anchor is not None and sighting.sighting_id == anchor.sighting_id:
            continue

        plate_result = (
            classify_plate_match(sighting.plate_text_normalized, target.true_plate)
            if sighting.plate_text_normalized
            else None
        )
        verdict = combine_evidence(
            plate_result,
            sighting.plate_confidence,
            similarity_by_id.get(sighting.sighting_id),
        )
        if not verdict.is_match:
            continue
        # The same retention rule the stage 07 harness uses: a plate score above
        # the review floor, or a candidate the appearance path kept. Requiring
        # the plate floor of everything would discard every visual-only match by
        # construction, since the visual ceiling sits below it on purpose.
        if verdict.score < discard_below and sighting.sighting_id not in appearance_ids:
            continue

        candidates.append(
            verdict.to_candidate(
                sighting_id=sighting.sighting_id,
                target_id=target_id,
                edit_distance=plate_result.edit_distance if plate_result is not None else None,
            )
        )

    return candidates


def _admissible_filter(
    topology: Topology | None, anchor: Sighting, max_horizon_sec: float
) -> Callable[[Sighting], bool]:
    """Build the predicate deciding which sightings appearance may compare.

    Args:
        topology: The graph, or ``None`` for an unconstrained comparison.
        anchor: The confirmed sighting to search outward from.
        max_horizon_sec: How far ahead to look.

    Returns:
        A predicate over sightings.
    """
    if topology is None:
        return lambda _sighting: True

    from multicam_tracker.topology import reachable_within

    result = reachable_within(
        topology, anchor.camera_id, anchor.timestamp_utc, max_horizon_sec, max_hops=3
    )
    arrival = {entry.camera_id: entry.arrival_window for entry in result.cameras}

    def admissible(sighting: Sighting) -> bool:
        if sighting.camera_id == anchor.camera_id:
            return sighting.timestamp_utc >= anchor.timestamp_utc
        window = arrival.get(sighting.camera_id)
        return window is not None and window.contains(sighting.timestamp_utc)

    return admissible


def _true_route(dataset: SyntheticDataset, ground_truth: GroundTruth) -> list[Sighting]:
    """Return the target's real sightings in chronological order.

    Args:
        dataset: The generated sightings.
        ground_truth: The answer key.

    Returns:
        The true route.

    Raises:
        ValueError: If the ground truth declares no target.
    """
    target = ground_truth.target
    if target is None:
        msg = f"scenario {ground_truth.scenario_name} declares no target vehicle"
        raise ValueError(msg)

    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    route = [by_id[sid] for sid in target.sighting_ids if sid in by_id]
    return sorted(route, key=lambda sighting: (sighting.timestamp_utc, sighting.sighting_id))


def evaluate_pathing(
    dataset: SyntheticDataset,
    ground_truth: GroundTruth,
    topology: Topology,
    *,
    cameras: dict[str, Camera] | None = None,
    target_id: str = "a0000000-0000-4000-8000-000000000001",
    result: ReconstructionResult | None = None,
) -> PathingMetrics:
    """Score one reconstruction against ground truth.

    Args:
        dataset: The generated sightings.
        ground_truth: The answer key, loaded from its file where possible.
        topology: The camera graph.
        cameras: Cameras by id. Required for the teleport check -- without
            coordinates no speed can be computed, and the metric would silently
            report zero teleports because it never looked.
        target_id: Identifier recorded on the generated candidates.
        result: A reconstruction to score instead of computing one.

    Returns:
        The metrics.
    """
    truth_route = _true_route(dataset, ground_truth)
    resolved_cameras = cameras if cameras is not None else _cameras_from_topology(topology)

    if result is None:
        candidates = candidates_from_matches(dataset, ground_truth, target_id, topology=topology)
        sightings = {sighting.sighting_id: sighting for sighting in dataset.sightings}
        result = reconstruct_trajectory(
            target_id,
            candidates,
            sightings,
            topology,
            cameras=resolved_cameras,
            activity=list(dataset.sightings),
        )

    metrics = PathingMetrics(scenario_name=ground_truth.scenario_name)
    metrics.true_sightings = len(truth_route)
    metrics.true_hops = max(0, len(truth_route) - 1)

    truth_ids = [sighting.sighting_id for sighting in truth_route]
    truth_set = set(truth_ids)

    trajectory = result.trajectory
    if trajectory is None:
        return metrics

    route_ids = [sighting.sighting_id for sighting in trajectory.sightings]
    metrics.reconstructed_sightings = len(route_ids)
    metrics.correct_sightings = sum(1 for sid in route_ids if sid in truth_set)
    metrics.decoys_included = sum(
        1 for sid in route_ids if sid not in truth_set and ground_truth.vehicle_for(sid) is not None
    )
    metrics.exact_path_match = route_ids == truth_ids

    true_pairs = set(pairwise(truth_ids))
    route_pairs = set(pairwise(route_ids))
    metrics.correct_hops = len(true_pairs & route_pairs)

    metrics.gaps_reported = len(trajectory.gaps)
    metrics.teleport_hops = len(result.movement.implausible_hops)
    metrics.is_ambiguous = result.is_ambiguous
    metrics.overall_confidence = trajectory.overall_confidence

    return metrics


def _cameras_from_topology(topology: Topology) -> dict[str, Camera]:
    """Return the topology's cameras by id.

    Args:
        topology: The graph.

    Returns:
        Cameras keyed by id.
    """
    return dict(topology.cameras_by_id)
