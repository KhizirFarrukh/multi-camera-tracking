"""Candidate search: finding the sightings that might be the target.

Two stages, and the split is the whole point.

**Stage one is indexed and database-side.** ``find_by_plate_exact`` and
``find_by_plate_folded`` both hit partial B-tree indexes, so the database returns
a handful of rows rather than the table. The folded lookup is what makes fuzzy
matching cheap: collapsing confusable characters turns "find plates that might
be misreads of this one" into an equality query.

**Stage two scores those candidates in Python.** Exact classification and
confidence weighting are string work on a small set, which no index can do.

A full scan is never performed. Query cost scales with the number of
*prefiltered* candidates, not with dataset size -- which is why this still works
when the sightings table holds millions of rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from multicam_tracker.db.repositories.protocols import SightingRepository
from multicam_tracker.logging_config import get_logger
from multicam_tracker.matching.plate_match import (
    PlateMatchMethod,
    PlateMatchResult,
    classify_plate_match,
)
from multicam_tracker.matching.scoring import score_plate_match
from multicam_tracker.models import MatchCandidate, MatchMethod, ReviewStatus, Sighting, TimeWindow

__all__ = ["ScoredMatch", "find_plate_matches", "score_sighting"]

logger = get_logger(__name__)

_METHOD_MAP = {
    PlateMatchMethod.PLATE_EXACT: MatchMethod.PLATE_EXACT,
    PlateMatchMethod.PLATE_FUZZY: MatchMethod.PLATE_FUZZY,
}


@dataclass(frozen=True)
class ScoredMatch:
    """One candidate sighting, its classification, and its score."""

    sighting: Sighting
    result: PlateMatchResult
    score: float

    def to_candidate(self, target_id: str, review_status: ReviewStatus) -> MatchCandidate:
        """Convert to the persisted match-candidate model.

        Args:
            target_id: The target this match belongs to.
            review_status: The adjudication state assigned by threshold.

        Returns:
            The domain model, ready to store.
        """
        return MatchCandidate(
            sighting_id=self.sighting.sighting_id,
            target_id=target_id,
            match_method=_METHOD_MAP[self.result.method],
            match_score=self.score,
            plate_edit_distance=self.result.edit_distance,
            embedding_similarity=None,
            review_status=review_status,
        )


def score_sighting(
    sighting: Sighting,
    target_plate: str,
    *,
    max_weighted_distance: float | None = None,
) -> ScoredMatch | None:
    """Classify and score one sighting against a target plate.

    Args:
        sighting: The candidate.
        target_plate: The normalized plate being searched for.
        max_weighted_distance: Fuzzy cutoff override. Exposed so the threshold
            sweep can vary it without mutating global config.

    Returns:
        The scored match, or ``None`` when the sighting carries no plate or does
        not match at all.
    """
    if sighting.plate_text_normalized is None:
        return None

    result = classify_plate_match(
        sighting.plate_text_normalized,
        target_plate,
        max_weighted_distance=max_weighted_distance,
    )
    if result.method is PlateMatchMethod.NO_MATCH:
        return None

    return ScoredMatch(
        sighting=sighting,
        result=result,
        score=score_plate_match(result, sighting.plate_confidence),
    )


def _review_status(score: float, auto_accept: float) -> ReviewStatus:
    """Assign an adjudication state from a score.

    Args:
        score: The match score.
        auto_accept: Threshold at or above which no human is needed.

    Returns:
        ``AUTO_ACCEPTED`` at or above the threshold, ``PENDING_REVIEW`` below it.
    """
    return ReviewStatus.AUTO_ACCEPTED if score >= auto_accept else ReviewStatus.PENDING_REVIEW


def find_plate_matches(
    target_plate: str,
    sighting_repo: SightingRepository,
    *,
    target_id: str,
    time_window: TimeWindow | None = None,
    camera_ids: list[str] | None = None,
    auto_accept_min: float | None = None,
    review_min: float | None = None,
    max_weighted_distance: float | None = None,
) -> list[MatchCandidate]:
    """Find and rank the sightings that match a target plate.

    Args:
        target_plate: The normalized plate to search for.
        sighting_repo: Repository to query. Only the indexed prefilter methods
            are called.
        target_id: Target the resulting candidates belong to.
        time_window: Restrict to this half-open window.
        camera_ids: Restrict to these cameras. Applied in Python rather than by
            an extra query, because the prefilter has already reduced the set to
            a handful of rows and a second round trip would cost more than the
            filter saves.
        auto_accept_min: Score at or above which a match is auto-accepted.
            Defaults to config.
        review_min: Score below which a match is discarded entirely. Defaults to
            config.

    Returns:
        Candidates ranked by descending score, each with a review status. Empty
        list when nothing matched, never ``None``.

    Raises:
        StorageError: If a repository read fails.
    """
    from multicam_tracker.config import get_settings

    thresholds = get_settings().thresholds
    accept_at = (
        auto_accept_min
        if auto_accept_min is not None
        else thresholds.plate_auto_accept_min_confidence
    )
    discard_below = review_min if review_min is not None else thresholds.plate_review_min_confidence

    # Both lookups are indexed. The folded set is a superset of the exact set --
    # an exact match folds identically -- but both are issued because the exact
    # index is the cheaper path and the union is de-duplicated by id anyway.
    candidates: dict[str, Sighting] = {}
    for sighting in sighting_repo.find_by_plate_exact(target_plate, time_window):
        candidates[sighting.sighting_id] = sighting
    for sighting in sighting_repo.find_by_plate_folded(target_plate, time_window):
        candidates[sighting.sighting_id] = sighting

    allowed = set(camera_ids) if camera_ids is not None else None

    scored: list[ScoredMatch] = []
    for sighting in candidates.values():
        if allowed is not None and sighting.camera_id not in allowed:
            continue
        match = score_sighting(sighting, target_plate, max_weighted_distance=max_weighted_distance)
        if match is not None and match.score >= discard_below:
            scored.append(match)

    # Ties broken by id so a ranking is reproducible across runs; an unstable
    # order would make a review queue reshuffle itself between refreshes.
    scored.sort(key=lambda item: (-item.score, item.sighting.sighting_id))

    logger.info(
        "plate_search_completed",
        target_plate=target_plate,
        prefiltered=len(candidates),
        retained=len(scored),
    )

    return [
        match.to_candidate(target_id, _review_status(match.score, accept_at)) for match in scored
    ]
