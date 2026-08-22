"""Topology-constrained embedding search.

**Restricting the search space beats raising the threshold.** That is the whole
design. Given an anchor -- a sighting already confirmed to be the target -- the
topology says which cameras the vehicle could have reached and when, and
searching only those turns a global similarity contest into a local one. A decoy
that looks identical to the target is harmless if it was never anywhere the
target could have been.

Raising the similarity threshold instead trades recall for precision on every
sighting equally. Constraining the space costs nothing on the true match while
removing most opportunities to be wrong, which is why stage 07 leans on it.

Without an anchor the search falls back to a global time window and says so in
the logs: it is a materially weaker query, and a caller should know when they are
getting one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from multicam_tracker.db.repositories.protocols import SightingRepository
from multicam_tracker.logging_config import get_logger
from multicam_tracker.matching.aggregation import AggregationStrategy, aggregate_similarity
from multicam_tracker.matching.versioning import require_same_model_version
from multicam_tracker.models import ReviewStatus, Sighting, Target, TimeWindow
from multicam_tracker.topology import Topology, reachable_within

__all__ = ["EmbeddingMatchResult", "classify_embedding_matches", "find_embedding_matches"]

logger = get_logger(__name__)

_ALL_TIME = TimeWindow(
    start_utc=datetime(1970, 1, 1, tzinfo=UTC),
    end_utc=datetime(2200, 1, 1, tzinfo=UTC),
)
"""Stands in for "no time constraint" on the unanchored fallback path. A bounded
window rather than datetime.min/max, which overflow on conversion.
"""


@dataclass(frozen=True)
class EmbeddingMatchResult:
    """One candidate scored by appearance."""

    sighting: Sighting
    similarity: float
    review_status: ReviewStatus
    margin: float | None
    """Gap to the runner-up. ``None`` when there was no runner-up."""

    downgraded_by_margin: bool = False
    """Whether the margin rule demoted an otherwise auto-acceptable match."""


def _search_space(
    topology: Topology,
    anchor: Sighting,
    max_horizon_sec: float,
    max_hops: int,
) -> tuple[list[str], TimeWindow]:
    """Derive the reachable cameras and arrival window from an anchor.

    Args:
        topology: The graph.
        anchor: The confirmed sighting to search outward from.
        max_horizon_sec: How far ahead to look.
        max_hops: Hop budget for the reachability search.

    Returns:
        ``(camera_ids, window)``. The camera list includes the anchor's own
        camera, because a vehicle can be seen twice by one camera and excluding
        it would drop genuine repeat passes.
    """
    result = reachable_within(
        topology, anchor.camera_id, anchor.timestamp_utc, max_horizon_sec, max_hops=max_hops
    )

    cameras = [anchor.camera_id, *result.camera_ids()]
    latest = max(
        (entry.latest_arrival for entry in result.cameras),
        default=anchor.timestamp_utc + timedelta(seconds=max_horizon_sec),
    )
    window = TimeWindow(
        start_utc=anchor.timestamp_utc,
        end_utc=max(latest, anchor.timestamp_utc + timedelta(milliseconds=1)),
    )
    return sorted(set(cameras)), window


def classify_embedding_matches(
    scored: list[tuple[Sighting, float]],
    *,
    auto_accept_min: float | None = None,
    review_min: float | None = None,
    margin_min: float | None = None,
) -> list[EmbeddingMatchResult]:
    """Apply the two-tier thresholds and the margin rule to scored candidates.

    The margin rule is the safety property that matters. A candidate may clear
    the auto-accept threshold outright and still be demoted, because clearing it
    while a *second* candidate is almost as similar means the two are
    indistinguishable by appearance. Auto-accepting either would be a coin flip
    reported as a fact.

    Args:
        scored: ``(sighting, similarity)`` pairs, in any order.
        auto_accept_min: Similarity at or above which no human is needed.
            Defaults to config.
        review_min: Similarity below which a candidate is discarded. Defaults to
            config.
        margin_min: Required gap to the runner-up. Defaults to config.

    Returns:
        Retained candidates ranked by descending similarity, each with its
        adjudication state. Candidates below the review floor are omitted.
    """
    from multicam_tracker.config import get_settings

    thresholds = get_settings().thresholds
    accept_at = (
        auto_accept_min
        if auto_accept_min is not None
        else thresholds.embedding_auto_accept_min_similarity
    )
    discard_below = (
        review_min if review_min is not None else thresholds.embedding_review_min_similarity
    )
    required_margin = margin_min if margin_min is not None else thresholds.embedding_margin_min

    ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0].sighting_id))
    retained = [(sighting, score) for sighting, score in ranked if score >= discard_below]

    results: list[EmbeddingMatchResult] = []
    for position, (sighting, score) in enumerate(retained):
        margin = score - retained[position + 1][1] if position + 1 < len(retained) else None

        status = ReviewStatus.AUTO_ACCEPTED if score >= accept_at else ReviewStatus.PENDING_REVIEW
        downgraded = False

        # Only the leader can be demoted by the margin rule: a candidate ranked
        # second is not being auto-accepted anyway, so its own gap to the next
        # one says nothing about whether the decision is ambiguous.
        if (
            position == 0
            and status is ReviewStatus.AUTO_ACCEPTED
            and margin is not None
            and margin < required_margin
        ):
            status = ReviewStatus.PENDING_REVIEW
            downgraded = True

        results.append(
            EmbeddingMatchResult(
                sighting=sighting,
                similarity=score,
                review_status=status,
                margin=margin,
                downgraded_by_margin=downgraded,
            )
        )

    return results


def find_embedding_matches(
    target: Target,
    sighting_repo: SightingRepository,
    topology: Topology,
    *,
    anchor_sighting: Sighting | None = None,
    time_window: TimeWindow | None = None,
    camera_ids: list[str] | None = None,
    model_version: str | None = None,
    k: int = 50,
    max_horizon_sec: float = 3600.0,
    max_hops: int = 3,
    strategy: AggregationStrategy = AggregationStrategy.MAX,
) -> list[EmbeddingMatchResult]:
    """Find sightings that look like the target.

    Args:
        target: The vehicle being searched for. Must carry reference embeddings.
        sighting_repo: Repository to query.
        topology: Graph used to constrain the search when an anchor is given.
        anchor_sighting: A sighting already believed to be the target. Supplying
            one is what makes this query precise.
        time_window: Explicit window. Ignored when an anchor is supplied, since
            the anchor derives a tighter one.
        camera_ids: Additional camera restriction, intersected with the
            topology-derived set rather than replacing it.
        model_version: Only compare against embeddings from this model. Defaults
            to the anchor's version when one is supplied.
        k: Maximum candidates to retrieve from the vector index.
        max_horizon_sec: How far ahead of the anchor to search.
        max_hops: Hop budget for reachability.
        strategy: How to aggregate across the target's references.

    Returns:
        Ranked candidates with adjudication states. Empty when nothing cleared
        the review floor.

    Raises:
        MatchingError: If the target has no reference embeddings, or a candidate
            was embedded by a different model version than the query.
        StorageError: If the repository read fails.
    """
    from multicam_tracker.exceptions import MatchingError

    references = target.reference_embeddings
    if not references:
        raise MatchingError(
            "Cannot search by appearance for a target with no reference embeddings",
            {"target_id": target.target_id},
        )

    resolved_version = model_version
    if resolved_version is None and anchor_sighting is not None:
        resolved_version = anchor_sighting.embedding_model_version

    if anchor_sighting is not None:
        cameras, window = _search_space(topology, anchor_sighting, max_horizon_sec, max_hops)
        if camera_ids is not None:
            cameras = sorted(set(cameras) & set(camera_ids))
        logger.info(
            "embedding_search_constrained",
            target_id=target.target_id,
            anchor_camera=anchor_sighting.camera_id,
            reachable_cameras=len(cameras),
        )
    else:
        cameras = list(camera_ids) if camera_ids is not None else []
        window = time_window or _ALL_TIME
        logger.warning(
            "embedding_search_unconstrained",
            target_id=target.target_id,
            detail=(
                "no anchor sighting supplied, so the search is global rather than "
                "topology-constrained; precision is materially lower"
            ),
        )

    # The version filter goes into the query, not into a check afterwards. A
    # table part-way through a re-embedding migration holds both versions on
    # purpose, and a search that merely rejected the wrong ones would return k
    # rows of which most were unusable -- or raise, and be unusable itself.
    matches = sighting_repo.find_nearest_by_embedding(
        references[0],
        k,
        camera_ids=cameras or None,
        window=window,
        model_version=resolved_version,
    )

    scored: list[tuple[Sighting, float]] = []
    for match in matches:
        sighting = match.sighting
        if sighting.embedding is None:
            continue
        if anchor_sighting is not None and sighting.sighting_id == anchor_sighting.sighting_id:
            continue
        # Comparing across model versions would produce a plausible number with
        # no meaning, so it raises rather than being filtered away silently --
        # a mixed-version table is a migration problem, not a query problem.
        require_same_model_version(resolved_version, sighting.embedding_model_version)
        scored.append(
            (sighting, aggregate_similarity(sighting.embedding, references, strategy=strategy))
        )

    return classify_embedding_matches(scored)
