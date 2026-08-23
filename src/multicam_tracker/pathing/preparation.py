"""Turning scored match candidates into the input the path search can use.

Three jobs, in order: drop what must not be considered, collapse what is really
one observation, and put the rest in time order.

The collapsing step matters more than it looks. A camera sampling at 3 fps sees
a vehicle in a dozen consecutive frames; that is one transit, not twelve. Left
in, those twelve become twelve nodes joined by same-camera hops the topology
cannot explain, and the path search spends its budget deciding which frame of
one pass to prefer. The rule for what counts as one pass is stage 04's
``min_redetection_gap_sec``, read from there rather than re-invented here.
"""

from __future__ import annotations

from dataclasses import dataclass

from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import MatchCandidate, ReviewStatus, Sighting
from multicam_tracker.topology.plausibility import is_same_pass

__all__ = ["PreparedCandidate", "prepare_candidates"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedCandidate:
    """One sighting, with the match evidence that put it in the running."""

    sighting: Sighting
    candidate: MatchCandidate

    @property
    def sighting_id(self) -> str:
        """Return the sighting's id."""
        return self.sighting.sighting_id

    @property
    def camera_id(self) -> str:
        """Return the observing camera."""
        return self.sighting.camera_id

    @property
    def confidence(self) -> float:
        """Return the match score that admitted this sighting."""
        return self.candidate.match_score

    @property
    def timestamp(self) -> object:
        """Return the sighting's corrected UTC timestamp."""
        return self.sighting.timestamp_utc


def _sort_key(prepared: PreparedCandidate) -> tuple[object, str]:
    """Return the deterministic ordering key.

    Args:
        prepared: The candidate.

    Returns:
        ``(timestamp, sighting_id)``. The id breaks ties, so two sightings
        sharing a timestamp order identically on every run rather than by
        whatever order the repository happened to return.
    """
    return (prepared.sighting.timestamp_utc, prepared.sighting_id)


def prepare_candidates(
    candidates: list[MatchCandidate],
    sightings: dict[str, Sighting],
    *,
    confirmed_only: bool = False,
    min_redetection_gap_sec: float | None = None,
) -> list[PreparedCandidate]:
    """Filter, deduplicate, and time-order match candidates.

    Args:
        candidates: Scored candidates for one target, in any order.
        sightings: Sightings by id. A candidate whose sighting is absent is
            skipped with a warning rather than raising: a purge under the
            retention TTL can legitimately remove the row a candidate points at,
            and one missing sighting must not abort a reconstruction.
        confirmed_only: When ``True``, only confirmed and auto-accepted
            candidates are kept. When ``False`` (the default) candidates awaiting
            human review are included too, because excluding them would silently
            answer the review question as "no".
        min_redetection_gap_sec: Override for stage 04's same-pass gap.

    Returns:
        Prepared candidates, strictly ordered by timestamp then id, with
        near-simultaneous sightings on one camera collapsed to their
        highest-confidence representative.
    """
    admissible: set[ReviewStatus] = (
        {ReviewStatus.CONFIRMED, ReviewStatus.AUTO_ACCEPTED}
        if confirmed_only
        else {ReviewStatus.CONFIRMED, ReviewStatus.AUTO_ACCEPTED, ReviewStatus.PENDING_REVIEW}
    )

    prepared: list[PreparedCandidate] = []
    for candidate in candidates:
        if candidate.review_status not in admissible:
            continue
        sighting = sightings.get(candidate.sighting_id)
        if sighting is None:
            logger.warning(
                "path_candidate_without_sighting",
                sighting_id=candidate.sighting_id,
                target_id=candidate.target_id,
                detail="candidate references a sighting that is not present; skipping it",
            )
            continue
        prepared.append(PreparedCandidate(sighting=sighting, candidate=candidate))

    prepared.sort(key=_sort_key)
    return _collapse_same_pass(prepared, min_redetection_gap_sec)


def _collapse_same_pass(
    prepared: list[PreparedCandidate], min_redetection_gap_sec: float | None
) -> list[PreparedCandidate]:
    """Collapse repeat detections of one pass to a single representative.

    Comparison is against the last *kept* sighting on that camera rather than
    the previous one, so a burst of detections at 3 fps collapses to one rather
    than to one per pair of adjacent frames.

    Args:
        prepared: Time-ordered candidates.
        min_redetection_gap_sec: Override for the configured gap.

    Returns:
        The surviving candidates, still time-ordered. Where a pass produced
        several detections the highest-confidence one is kept -- it is the
        clearest view of the vehicle, and its timestamp is as good as any other
        from the same pass.
    """
    kept: list[PreparedCandidate] = []
    # camera -> (index of the pass representative, timestamp of the last
    # detection seen in that pass). The two are tracked separately because the
    # representative can be replaced by a higher-confidence frame, and the
    # window must keep running from the last detection rather than jumping to
    # whichever frame currently holds the title.
    open_pass: dict[str, tuple[int, object]] = {}

    for entry in prepared:
        state = open_pass.get(entry.camera_id)
        if state is not None:
            index, last_seen = state
            elapsed = (entry.sighting.timestamp_utc - last_seen).total_seconds()  # type: ignore[operator]
            if is_same_pass(elapsed, min_redetection_gap_sec=min_redetection_gap_sec):
                if entry.confidence > kept[index].confidence:
                    kept[index] = entry
                open_pass[entry.camera_id] = (index, entry.sighting.timestamp_utc)
                continue

        open_pass[entry.camera_id] = (len(kept), entry.sighting.timestamp_utc)
        kept.append(entry)

    kept.sort(key=_sort_key)
    return kept
