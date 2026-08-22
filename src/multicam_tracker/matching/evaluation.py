"""Measuring plate matching against synthetic ground truth.

This is what stage 05 was built for. Precision and recall stop being adjectives
and become numbers, thresholds get chosen from a sweep instead of intuition, and
a change that degrades matching fails a test instead of going unnoticed.

Definitions, stated explicitly because a subtly different one would make every
number here incomparable with the next person's:

* A **positive** is a sighting the matcher returned at or above the review floor
  -- everything it put in front of a human or accepted outright.
* A **true positive** is a positive the ground truth attributes to the target
  vehicle.
* A **false negative** is a target sighting the matcher did not return.

Sightings of the target whose plate was *completely unreadable* still count as
false negatives. Plate matching genuinely cannot find them, and excluding them
would flatter the recall number and hide exactly the gap that justifies re-id in
stage 07.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from multicam_tracker.matching.search import score_sighting
from multicam_tracker.models import ReviewStatus
from multicam_tracker.synth.generator import SyntheticDataset
from multicam_tracker.synth.ground_truth import GroundTruth

__all__ = ["MatchingMetrics", "evaluate_plate_matching"]


@dataclass
class MatchingMetrics:
    """What a matcher scored on one dataset."""

    scenario_name: str
    target_vehicle_id: str
    target_plate: str

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    auto_accepted_true: int = 0
    auto_accepted_false: int = 0
    """Decoy sightings accepted without human review. The number that matters
    most: an auto-accepted false positive reaches an operator as fact."""

    misses_by_corruption: dict[str, int] = field(default_factory=dict)
    """Which OCR failure mode caused each missed target sighting. ``clean`` means
    the plate was read correctly and the matcher still missed it, which would be
    a matcher bug rather than a data problem."""

    false_positive_vehicles: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Return the share of returned matches that were really the target.

        Returns:
            ``1.0`` when nothing was returned -- a matcher that answers nothing
            has told no lies, and recall is the number that penalises it.
        """
        positives = self.true_positives + self.false_positives
        return 1.0 if positives == 0 else self.true_positives / positives

    @property
    def recall(self) -> float:
        """Return the share of the target's sightings that were found.

        Returns:
            ``1.0`` when the target has no sightings at all.
        """
        actual = self.true_positives + self.false_negatives
        return 1.0 if actual == 0 else self.true_positives / actual

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def auto_accept_precision(self) -> float:
        """Return precision restricted to auto-accepted matches.

        Returns:
            The share of unreviewed acceptances that were correct. This is the
            number the human-review gate exists to protect, so it is reported
            separately from overall precision.
        """
        accepted = self.auto_accepted_true + self.auto_accepted_false
        return 1.0 if accepted == 0 else self.auto_accepted_true / accepted

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the metrics for a baseline file.

        Returns:
            A JSON-native mapping. Rates are rounded to four decimals so a
            baseline diff shows a real change rather than float noise.
        """
        return {
            "scenario_name": self.scenario_name,
            "target_vehicle_id": self.target_vehicle_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "auto_accepted_true": self.auto_accepted_true,
            "auto_accepted_false": self.auto_accepted_false,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auto_accept_precision": round(self.auto_accept_precision, 4),
            "misses_by_corruption": dict(sorted(self.misses_by_corruption.items())),
        }


def evaluate_plate_matching(
    dataset: SyntheticDataset,
    ground_truth: GroundTruth,
    *,
    auto_accept_min: float | None = None,
    review_min: float | None = None,
    max_weighted_distance: float | None = None,
) -> MatchingMetrics:
    """Score plate matching on one generated dataset.

    Runs the matcher over the dataset in memory rather than through a
    repository, so the measurement isolates the matching logic from storage.
    Candidate selection is the same classification and scoring the indexed path
    applies; only the prefilter is replaced by a full pass, which is sound here
    because the prefilter is a superset of what scoring would accept.

    Args:
        dataset: The generated sightings.
        ground_truth: The answer key. Passed separately rather than taken from
            the dataset so a caller can score against the ground truth as
            *loaded from its file*, proving the file is the real record.
        auto_accept_min: Score at or above which a match is auto-accepted.
            Defaults to config.
        review_min: Score below which a match is discarded. Defaults to config.

    Returns:
        The metrics, including which corruption modes caused the misses.

    Raises:
        ValueError: If the ground truth has no target vehicle to score against.
    """
    from multicam_tracker.config import get_settings

    target = ground_truth.target
    if target is None:
        msg = f"scenario {ground_truth.scenario_name} declares no target vehicle"
        raise ValueError(msg)

    thresholds = get_settings().thresholds
    accept_at = (
        auto_accept_min
        if auto_accept_min is not None
        else thresholds.plate_auto_accept_min_confidence
    )
    discard_below = review_min if review_min is not None else thresholds.plate_review_min_confidence

    corruption_by_sighting = {event.sighting_id: event.kind for event in ground_truth.corruptions}
    target_sightings = set(target.sighting_ids)

    metrics = MatchingMetrics(
        scenario_name=ground_truth.scenario_name,
        target_vehicle_id=target.vehicle_id,
        target_plate=target.true_plate,
    )

    returned: set[str] = set()
    false_positive_vehicles: Counter[str] = Counter()

    for sighting in dataset.sightings:
        match = score_sighting(
            sighting, target.true_plate, max_weighted_distance=max_weighted_distance
        )
        if match is None or match.score < discard_below:
            continue

        returned.add(sighting.sighting_id)
        status = (
            ReviewStatus.AUTO_ACCEPTED if match.score >= accept_at else ReviewStatus.PENDING_REVIEW
        )
        is_target = sighting.sighting_id in target_sightings

        if is_target:
            metrics.true_positives += 1
            if status is ReviewStatus.AUTO_ACCEPTED:
                metrics.auto_accepted_true += 1
        else:
            metrics.false_positives += 1
            owner = ground_truth.vehicle_for(sighting.sighting_id) or "unknown"
            false_positive_vehicles[owner] += 1
            if status is ReviewStatus.AUTO_ACCEPTED:
                metrics.auto_accepted_false += 1

    misses: Counter[str] = Counter()
    for sighting_id in target_sightings:
        if sighting_id in returned:
            continue
        metrics.false_negatives += 1
        misses[corruption_by_sighting.get(sighting_id, "clean")] += 1

    metrics.misses_by_corruption = dict(misses)
    metrics.false_positive_vehicles = sorted(false_positive_vehicles)
    return metrics
