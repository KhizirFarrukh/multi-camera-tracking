"""Measuring the embedding path, alone and combined with plate evidence.

Stage 06 measured what plate matching can do. This measures what appearance
recovers **on top of it** — and the headline number is
``plate_failed_recovered``: sightings whose plate was completely unreadable that
appearance found anyway. That number is the entire justification for the stage.
If it were zero, re-id would be complexity with no return.

The evaluation mirrors real use rather than an idealised setup. The target's
*first* sighting becomes the reference, exactly as a real search starts from one
confirmed anchor, and every other sighting is a candidate. Seeding the reference
set with all of the target's sightings would measure a system that already knows
the answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from multicam_tracker.matching.aggregation import AggregationStrategy, aggregate_similarity
from multicam_tracker.matching.evidence import combine_evidence
from multicam_tracker.matching.plate_match import classify_plate_match
from multicam_tracker.models import ReviewStatus, Sighting
from multicam_tracker.synth.generator import SyntheticDataset
from multicam_tracker.synth.ground_truth import GroundTruth

__all__ = ["ReidMetrics", "evaluate_reid_matching"]

CMC_RANKS = (1, 5, 10)
"""Ranks reported in the CMC curve."""


@dataclass
class ReidMetrics:
    """What the embedding and combined paths scored on one dataset."""

    scenario_name: str

    embedding_true_positives: int = 0
    embedding_false_positives: int = 0
    embedding_false_negatives: int = 0
    embedding_auto_accepted_false: int = 0

    combined_true_positives: int = 0
    combined_false_positives: int = 0
    combined_false_negatives: int = 0

    plate_failed_total: int = 0
    """Target sightings the plate path could not find at all."""

    plate_failed_recovered: int = 0
    """Of those, how many appearance found. The reason this stage exists."""

    cmc: dict[int, float] = field(default_factory=dict)
    """Rank -> share of queries whose true match appeared by that rank."""

    margin_downgrades: int = 0
    """Candidates demoted from auto-accept because a runner-up was too close."""

    candidates_considered: int = 0
    """How many sightings the search space admitted for comparison.

    The size of the space is the point of the topology constraint, and on a
    scenario with no visual decoys it is the only thing the constraint changes:
    precision is already perfect, and what improves is how much work the query
    does to get there.
    """

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
    def embedding_precision(self) -> float:
        """Return precision of the appearance path alone."""
        return self._ratio(
            self.embedding_true_positives,
            self.embedding_true_positives + self.embedding_false_positives,
        )

    @property
    def embedding_recall(self) -> float:
        """Return recall of the appearance path alone."""
        return self._ratio(
            self.embedding_true_positives,
            self.embedding_true_positives + self.embedding_false_negatives,
        )

    @property
    def embedding_false_positive_rate(self) -> float:
        """Return the share of returned appearance matches that were wrong.

        Reported separately from precision because the documented target for
        hard negatives is stated as a false-positive rate.
        """
        returned = self.embedding_true_positives + self.embedding_false_positives
        return 0.0 if returned == 0 else self.embedding_false_positives / returned

    @property
    def combined_precision(self) -> float:
        """Return precision when both paths are combined."""
        return self._ratio(
            self.combined_true_positives,
            self.combined_true_positives + self.combined_false_positives,
        )

    @property
    def combined_recall(self) -> float:
        """Return recall when both paths are combined."""
        return self._ratio(
            self.combined_true_positives,
            self.combined_true_positives + self.combined_false_negatives,
        )

    @property
    def plate_failure_recovery_rate(self) -> float:
        """Return the share of plate-path misses that appearance recovered."""
        return (
            0.0
            if self.plate_failed_total == 0
            else (self.plate_failed_recovered / self.plate_failed_total)
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for a regression baseline.

        Returns:
            A JSON-native mapping with rates rounded so a diff shows a real
            change rather than float noise.
        """
        return {
            "scenario_name": self.scenario_name,
            "embedding_precision": round(self.embedding_precision, 4),
            "embedding_recall": round(self.embedding_recall, 4),
            "embedding_false_positive_rate": round(self.embedding_false_positive_rate, 4),
            "embedding_auto_accepted_false": self.embedding_auto_accepted_false,
            "combined_precision": round(self.combined_precision, 4),
            "combined_recall": round(self.combined_recall, 4),
            "plate_failed_total": self.plate_failed_total,
            "plate_failed_recovered": self.plate_failed_recovered,
            "plate_failure_recovery_rate": round(self.plate_failure_recovery_rate, 4),
            "cmc": {str(rank): round(value, 4) for rank, value in sorted(self.cmc.items())},
            "margin_downgrades": self.margin_downgrades,
            "candidates_considered": self.candidates_considered,
        }


def _plate_found(sighting: Sighting, target_plate: str, review_min: float) -> bool:
    """Return whether the plate path would have returned this sighting.

    Args:
        sighting: The candidate.
        target_plate: The plate being searched for.
        review_min: Score below which the plate path discards a candidate.

    Returns:
        ``True`` when plate matching alone finds it.
    """
    from multicam_tracker.matching.search import score_sighting

    match = score_sighting(sighting, target_plate)
    return match is not None and match.score >= review_min


def _reachable_filter(
    topology: object | None,
    anchor: Sighting,
    max_horizon_sec: float,
    max_hops: int,
) -> Callable[[Sighting], bool]:
    """Build the predicate deciding which sightings are worth comparing.

    With a topology, only cameras the vehicle could have reached from the anchor
    -- within the window it could have arrived in -- are admissible. A decoy that
    looks identical to the target is harmless if it was never anywhere the target
    could have been, which is why constraining the space beats raising the
    threshold.

    Args:
        topology: The graph, or ``None`` for the unconstrained baseline.
        anchor: The confirmed sighting to search outward from.
        max_horizon_sec: How far ahead to look.
        max_hops: Hop budget for reachability.

    Returns:
        A predicate over sightings.
    """
    if topology is None:
        return lambda _sighting: True

    from multicam_tracker.topology import Topology, reachable_within

    graph = cast("Topology", topology)
    result = reachable_within(
        graph, anchor.camera_id, anchor.timestamp_utc, max_horizon_sec, max_hops=max_hops
    )
    arrival = {entry.camera_id: entry.arrival_window for entry in result.cameras}

    def admissible(sighting: Sighting) -> bool:
        if sighting.camera_id == anchor.camera_id:
            return sighting.timestamp_utc >= anchor.timestamp_utc
        window = arrival.get(sighting.camera_id)
        return window is not None and window.contains(sighting.timestamp_utc)

    return admissible


def evaluate_reid_matching(
    dataset: SyntheticDataset,
    ground_truth: GroundTruth,
    *,
    auto_accept_similarity: float | None = None,
    review_similarity: float | None = None,
    margin_min: float | None = None,
    strategy: AggregationStrategy = AggregationStrategy.MAX,
    topology: object | None = None,
    max_horizon_sec: float = 3600.0,
    max_hops: int = 3,
) -> ReidMetrics:
    """Score the embedding and combined paths on one generated dataset.

    Args:
        dataset: The generated sightings.
        ground_truth: The answer key.
        auto_accept_similarity: Similarity at or above which appearance alone is
            accepted. Defaults to config.
        review_similarity: Similarity below which a candidate is discarded.
            Defaults to config.
        margin_min: Required gap to the runner-up. Defaults to config.
        strategy: Reference aggregation strategy.
        topology: When supplied, candidates are restricted to the cameras and
            arrival window reachable from the anchor. This is the precision
            lever the stage rests on; omitting it measures the unconstrained
            baseline the constraint is compared against.
        max_horizon_sec: How far ahead of the anchor to search.
        max_hops: Hop budget for reachability.

    Returns:
        The metrics, including the plate-failure recovery that justifies the
        stage.

    Raises:
        ValueError: If the ground truth declares no target, or the target has no
            embedded sighting to use as a reference.
    """
    from multicam_tracker.config import get_settings
    from multicam_tracker.matching.embedding_search import classify_embedding_matches

    target = ground_truth.target
    if target is None:
        msg = f"scenario {ground_truth.scenario_name} declares no target vehicle"
        raise ValueError(msg)

    thresholds = get_settings().thresholds
    accept_at = (
        auto_accept_similarity
        if auto_accept_similarity is not None
        else thresholds.embedding_auto_accept_min_similarity
    )
    discard_below = (
        review_similarity
        if review_similarity is not None
        else thresholds.embedding_review_min_similarity
    )
    plate_review_min = thresholds.plate_review_min_confidence

    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    target_sightings = [by_id[sid] for sid in target.sighting_ids if sid in by_id]
    embedded = [s for s in target_sightings if s.embedding is not None]
    if not embedded:
        msg = f"target in {ground_truth.scenario_name} has no embedded sighting to anchor on"
        raise ValueError(msg)

    # One reference, from the first confirmed sighting -- exactly how a real
    # search begins. Seeding with all of them would measure a system that
    # already knows the answer.
    anchor = embedded[0]
    references = [list(anchor.embedding or [])]
    truth_ids = set(target.sighting_ids) - {anchor.sighting_id}

    admissible = _reachable_filter(topology, anchor, max_horizon_sec, max_hops)

    scored: list[tuple[Sighting, float]] = [
        (sighting, aggregate_similarity(sighting.embedding, references, strategy=strategy))
        for sighting in dataset.sightings
        if sighting.embedding is not None
        and sighting.sighting_id != anchor.sighting_id
        and admissible(sighting)
    ]

    metrics = ReidMetrics(
        scenario_name=ground_truth.scenario_name, candidates_considered=len(scored)
    )

    classified = classify_embedding_matches(
        scored,
        auto_accept_min=accept_at,
        review_min=discard_below,
        margin_min=margin_min,
    )
    metrics.margin_downgrades = sum(1 for entry in classified if entry.downgraded_by_margin)

    returned_ids = {entry.sighting.sighting_id for entry in classified}
    for entry in classified:
        if entry.sighting.sighting_id in truth_ids:
            metrics.embedding_true_positives += 1
        else:
            metrics.embedding_false_positives += 1
            if entry.review_status is ReviewStatus.AUTO_ACCEPTED:
                metrics.embedding_auto_accepted_false += 1
    metrics.embedding_false_negatives = len(truth_ids - returned_ids)

    # --- combined evidence -------------------------------------------------
    similarity_by_id = {sighting.sighting_id: score for sighting, score in scored}
    for sighting in dataset.sightings:
        if sighting.sighting_id == anchor.sighting_id:
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
            embedding_auto_accept=accept_at,
        )
        # A combined positive is the union of what each path would return on its
        # own: a plate score at or above stage 06's review floor, or a candidate
        # the embedding path retained. Requiring the plate floor of everything
        # would discard every visual-only match by construction -- the ceiling
        # sits below that floor deliberately -- and measure plate matching over
        # again under a new name.
        if not verdict.is_match:
            continue
        if verdict.score < plate_review_min and sighting.sighting_id not in returned_ids:
            continue
        if sighting.sighting_id in truth_ids:
            metrics.combined_true_positives += 1
        else:
            metrics.combined_false_positives += 1
    metrics.combined_false_negatives = len(truth_ids) - metrics.combined_true_positives

    # --- the number this stage exists for ----------------------------------
    for sighting in target_sightings:
        if sighting.sighting_id == anchor.sighting_id:
            continue
        if _plate_found(sighting, target.true_plate, plate_review_min):
            continue
        metrics.plate_failed_total += 1
        if sighting.sighting_id in returned_ids:
            metrics.plate_failed_recovered += 1

    # --- CMC ---------------------------------------------------------------
    ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0].sighting_id))
    order = [sighting.sighting_id for sighting, _ in ranked]
    for rank in CMC_RANKS:
        hit = any(sighting_id in truth_ids for sighting_id in order[:rank])
        metrics.cmc[rank] = 1.0 if hit else 0.0

    return metrics
