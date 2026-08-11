"""Repository interfaces and their result types.

These are shaped around the queries the system actually runs, not around generic
CRUD. ``find_by_camera_and_window`` exists because trajectory assembly asks that
exact question; there is no ``find_by`` taking arbitrary filters, because a
generic query surface cannot be indexed deliberately and would hide which access
patterns the schema is tuned for.

Every implementation must satisfy the same behavioural contract, which is stated
here in the docstrings and enforced by the shared conformance suite. Two
conventions run through all of them:

* **Time windows are half-open** ``[start, end)``, matching
  :class:`~multicam_tracker.models.TimeWindow`.
* **Failures raise** :class:`~multicam_tracker.exceptions.StorageError`. No
  driver exception escapes this layer, because a caller that had to catch
  ``psycopg.errors.UniqueViolation`` would be coupled to the database in use.
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple, Protocol, runtime_checkable

from multicam_tracker.models import (
    Camera,
    CameraLink,
    MatchCandidate,
    ReviewStatus,
    Sighting,
    Target,
    TimeWindow,
    Trajectory,
)

__all__ = [
    "CameraHourCount",
    "CameraLinkRepository",
    "CameraRepository",
    "EmbeddingMatch",
    "MatchRepository",
    "PurgeResult",
    "SightingRepository",
    "TargetRepository",
    "TrajectoryRepository",
]


class EmbeddingMatch(NamedTuple):
    """A sighting returned by vector search, with its similarity to the query."""

    sighting: Sighting
    similarity: float
    """Cosine similarity in [-1, 1]. Higher is more similar."""


class PurgeResult(NamedTuple):
    """What a retention purge removed."""

    deleted_count: int
    thumbnail_paths: list[str]
    """Paths of thumbnails belonging to the deleted rows, for stage 18's job to
    unlink from disk. Collected before deletion, since afterwards the rows that
    named them are gone."""


class CameraHourCount(NamedTuple):
    """Sightings observed by one camera in one UTC hour.

    The field is ``sighting_count`` rather than ``count`` because ``NamedTuple``
    inherits ``tuple.count``, and shadowing it would break the tuple protocol.
    """

    camera_id: str
    hour_start_utc: datetime
    sighting_count: int


@runtime_checkable
class CameraRepository(Protocol):
    """Persistence for cameras."""

    def upsert(self, camera: Camera) -> Camera:
        """Insert ``camera``, or replace the existing row with the same id.

        Args:
            camera: The camera to store.

        Returns:
            The stored camera.

        Raises:
            StorageError: If the write fails.
        """
        ...

    def get(self, camera_id: str) -> Camera | None:
        """Return one camera by id.

        Args:
            camera_id: The identifier to look up.

        Returns:
            The camera, or ``None`` if no such row exists.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def list_enabled(self) -> list[Camera]:
        """Return every enabled camera, ordered by ``camera_id``.

        Returns:
            The enabled cameras. Empty list when none are enabled.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def delete(self, camera_id: str) -> bool:
        """Delete one camera.

        Args:
            camera_id: The identifier to delete.

        Returns:
            ``True`` if a row was removed, ``False`` if none existed.

        Raises:
            StorageError: If the camera still has sightings. Deleting it would
                orphan evidence, so the foreign key restricts it; retire the
                camera with ``enabled=False`` instead.
        """
        ...


@runtime_checkable
class CameraLinkRepository(Protocol):
    """Persistence for the topology's edges.

    Links are stored exactly as declared. ``get_links_from`` does not synthesise
    the reverse of a bidirectional link -- expanding the declared edges into a
    traversable graph is stage 04's job, and doing it here would make the stored
    and returned data disagree.
    """

    def upsert(self, link: CameraLink) -> CameraLink:
        """Insert ``link``, or replace the existing row for the same pair.

        Args:
            link: The link to store.

        Returns:
            The stored link.

        Raises:
            StorageError: If either endpoint does not exist, or the write fails.
        """
        ...

    def get_link(self, from_camera_id: str, to_camera_id: str) -> CameraLink | None:
        """Return the link declared for one ordered pair.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            The link, or ``None`` if the pair is not declared in this direction.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def get_links_from(self, camera_id: str) -> list[CameraLink]:
        """Return every link declared with ``camera_id`` as its origin.

        Args:
            camera_id: Origin camera.

        Returns:
            The links, ordered by destination id. Empty list when none exist.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def list_all(self) -> list[CameraLink]:
        """Return every link, ordered by origin then destination.

        Returns:
            All declared links.

        Raises:
            StorageError: If the read fails.
        """
        ...


@runtime_checkable
class SightingRepository(Protocol):
    """Persistence and query surface for sightings."""

    def add(self, sighting: Sighting) -> Sighting:
        """Insert one sighting.

        Args:
            sighting: The sighting to store.

        Returns:
            The stored sighting.

        Raises:
            StorageError: If the camera does not exist, the id already exists,
                or the write fails.
        """
        ...

    def add_batch(self, sightings: list[Sighting], *, ignore_conflicts: bool = False) -> int:
        """Insert many sightings in one round trip.

        Args:
            sightings: The sightings to store.
            ignore_conflicts: When ``True``, rows whose ``sighting_id`` already
                exists are skipped instead of raising. This is what makes
                re-processing a video idempotent.

        Returns:
            The number of rows actually inserted, which is less than
            ``len(sightings)`` when conflicts were skipped.

        Raises:
            StorageError: If the write fails, or a conflict occurs while
                ``ignore_conflicts`` is ``False``.
        """
        ...

    def get(self, sighting_id: str) -> Sighting | None:
        """Return one sighting by id.

        Args:
            sighting_id: The identifier to look up.

        Returns:
            The sighting, or ``None`` if no such row exists.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def find_by_plate_exact(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose normalized plate equals ``plate_normalized``.

        Args:
            plate_normalized: The normalized plate to match exactly.
            window: Optional half-open time window on ``timestamp_utc``.

        Returns:
            Matching sightings ordered by ascending ``timestamp_utc``.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def find_by_plate_folded(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose folded plate equals the folded form of the query.

        This is the fuzzy-match prefilter: it catches reads differing only by
        confusable characters. Stage 06 scores the returned candidates; this
        method only narrows the search.

        Args:
            plate_normalized: The normalized plate to fold and match.
            window: Optional half-open time window on ``timestamp_utc``.

        Returns:
            Matching sightings ordered by ascending ``timestamp_utc``. Includes
            exact matches, since an exact match also folds identically.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def find_by_camera_and_window(self, camera_id: str, window: TimeWindow) -> list[Sighting]:
        """Return one camera's sightings within a time window.

        Args:
            camera_id: The observing camera.
            window: Half-open window on ``timestamp_utc``.

        Returns:
            Sightings ordered by ascending ``timestamp_utc``. Empty list when
            the camera saw nothing, never ``None``.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def find_by_window(self, window: TimeWindow) -> list[Sighting]:
        """Return every camera's sightings within a time window.

        Args:
            window: Half-open window on ``timestamp_utc``.

        Returns:
            Sightings ordered by ascending ``timestamp_utc`` across all cameras.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def find_nearest_by_embedding(
        self,
        embedding: list[float],
        k: int,
        *,
        camera_ids: list[str] | None = None,
        window: TimeWindow | None = None,
    ) -> list[EmbeddingMatch]:
        """Return the ``k`` sightings most similar to ``embedding``.

        Sightings with no embedding are excluded: they cannot be compared, and
        including them at similarity 0 would let them displace real candidates.

        Args:
            embedding: The query vector, L2-normalized and of the configured
                dimension.
            k: Maximum number of results.
            camera_ids: Restrict to these cameras. ``None`` searches all.
            window: Optional half-open window on ``timestamp_utc``.

        Returns:
            At most ``k`` matches, ordered by descending cosine similarity.

        Raises:
            StorageError: If the read fails or the query vector has the wrong
                dimension.
        """
        ...

    def delete_older_than(self, cutoff_utc: datetime) -> PurgeResult:
        """Delete sightings created strictly before ``cutoff_utc``.

        Cuts on ``created_at`` rather than ``timestamp_utc`` so retention counts
        from when the system learned of a sighting, not from when it happened --
        otherwise footage ingested late would age out immediately.

        Match candidates cascade with their sightings. Saved trajectories do
        **not**: they are historical conclusions and must not be rewritten by a
        purge. A trajectory whose sightings have been purged becomes
        unresolvable, and :meth:`TrajectoryRepository.get` says so rather than
        returning a silently shortened route.

        Args:
            cutoff_utc: Aware UTC instant. Rows with ``created_at < cutoff_utc``
                are removed; a row exactly at the cutoff is kept.

        Returns:
            The number deleted and the thumbnail paths that are now orphaned.

        Raises:
            StorageError: If the delete fails.
        """
        ...

    def count_by_camera_hour(self, window: TimeWindow) -> list[CameraHourCount]:
        """Return per-camera, per-UTC-hour sighting counts.

        Args:
            window: Half-open window on ``timestamp_utc``.

        Returns:
            One entry per (camera, hour) that has at least one sighting, ordered
            by camera then hour. Hours with no sightings are omitted rather than
            returned as zeroes.

        Raises:
            StorageError: If the read fails.
        """
        ...


@runtime_checkable
class TargetRepository(Protocol):
    """Persistence for search targets."""

    def create(self, target: Target) -> Target:
        """Insert a new target.

        Args:
            target: The target to store.

        Returns:
            The stored target.

        Raises:
            StorageError: If the id already exists or the write fails.
        """
        ...

    def get(self, target_id: str) -> Target | None:
        """Return one target by id.

        Args:
            target_id: The identifier to look up.

        Returns:
            The target, or ``None`` if no such row exists.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def list_active(self) -> list[Target]:
        """Return every active target, newest first.

        Returns:
            The active targets.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def deactivate(self, target_id: str) -> bool:
        """Mark a target inactive, keeping the row for the audit trail.

        Args:
            target_id: The identifier to deactivate.

        Returns:
            ``True`` if a row changed, ``False`` if none existed.

        Raises:
            StorageError: If the write fails.
        """
        ...


@runtime_checkable
class MatchRepository(Protocol):
    """Persistence for match candidates."""

    def upsert_candidate(self, candidate: MatchCandidate) -> MatchCandidate:
        """Insert a candidate, or update the existing one for the same pair.

        Keyed on ``(target_id, sighting_id)``, so re-running the matcher revises
        its verdict rather than stacking duplicates.

        Args:
            candidate: The candidate to store.

        Returns:
            The stored candidate.

        Raises:
            StorageError: If the write fails.
        """
        ...

    def bulk_upsert(self, candidates: list[MatchCandidate]) -> int:
        """Upsert many candidates in one round trip.

        Args:
            candidates: The candidates to store.

        Returns:
            The number of rows written.

        Raises:
            StorageError: If the write fails.
        """
        ...

    def list_for_target(
        self, target_id: str, review_status: ReviewStatus | None = None
    ) -> list[MatchCandidate]:
        """Return a target's candidates, optionally filtered by review status.

        Args:
            target_id: The target whose candidates to list.
            review_status: Restrict to this status. ``None`` returns all.

        Returns:
            Candidates ordered by descending ``match_score``, so the strongest
            evidence is reviewed first.

        Raises:
            StorageError: If the read fails.
        """
        ...

    def update_review_status(
        self, target_id: str, sighting_id: str, review_status: ReviewStatus
    ) -> bool:
        """Set the review status of exactly one candidate.

        Args:
            target_id: The candidate's target.
            sighting_id: The candidate's sighting.
            review_status: The new status.

        Returns:
            ``True`` if a row changed, ``False`` if no such candidate exists.

        Raises:
            StorageError: If the write fails.
        """
        ...


@runtime_checkable
class TrajectoryRepository(Protocol):
    """Persistence for reconstructed trajectories."""

    def save(self, trajectory: Trajectory) -> Trajectory:
        """Insert a trajectory and its hops, replacing any prior version.

        Args:
            trajectory: The trajectory to store.

        Returns:
            The stored trajectory.

        Raises:
            StorageError: If the write fails.
        """
        ...

    def get(self, trajectory_id: str) -> Trajectory | None:
        """Return one trajectory, with its sightings resolved in order.

        Args:
            trajectory_id: The identifier to look up.

        Returns:
            The trajectory, or ``None`` if no such row exists.

        Raises:
            StorageError: If any of its sightings have been purged. The
                trajectory cannot be rebuilt faithfully, and returning a
                shortened route would misrepresent the evidence, so the error
                names the missing ids.
        """
        ...

    def list_for_target(self, target_id: str) -> list[Trajectory]:
        """Return a target's trajectories, newest first.

        Args:
            target_id: The target whose trajectories to list.

        Returns:
            The trajectories, ordered by descending ``start_time_utc``.

        Raises:
            StorageError: If the read fails, including when a stored trajectory
                references purged sightings.
        """
        ...
