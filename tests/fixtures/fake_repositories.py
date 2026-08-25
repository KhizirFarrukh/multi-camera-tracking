"""In-memory repository implementations for use in unit tests.

Every stage from 04 onward tests its logic against these instead of a database.
That only works if they behave *identically* to the Postgres implementations --
same ordering, same window inclusivity, same conflict handling, same errors --
which is why the conformance suite runs one set of behavioural tests against
both. A fake that quietly differs would make every unit test above it a
prediction about a system that does not exist.

Notable places where matching Postgres took deliberate care:

* ``find_*`` results are ordered by ``(timestamp_utc, sighting_id)``, mirroring
  the explicit ORDER BY in the SQL rather than relying on insertion order.
* ``delete_older_than`` cascades to match candidates, because the foreign key
  does.
* ``count_by_camera_hour`` truncates to the hour in UTC, matching
  ``date_trunc('hour', ...)`` on a ``timestamptz`` column.
* A sighting referencing an unknown camera raises ``StorageError``, because the
  foreign key would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from multicam_tracker.db.folding import fold_plate
from multicam_tracker.db.repositories.protocols import (
    CameraHourCount,
    CameraLinkRepository,
    CameraRepository,
    EmbeddingMatch,
    MatchRepository,
    PurgeResult,
    SightingRepository,
    TargetRepository,
    TrajectoryRepository,
)
from multicam_tracker.exceptions import StorageError
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
    "InMemoryCameraLinkRepository",
    "InMemoryCameraRepository",
    "InMemoryMatchRepository",
    "InMemorySightingRepository",
    "InMemoryStore",
    "InMemoryTargetRepository",
    "InMemoryTrajectoryRepository",
    "RepositorySet",
    "build_in_memory_repositories",
]


@dataclass
class InMemoryStore:
    """Shared state behind the in-memory repositories.

    One store stands in for one database, so repositories built from it see each
    other's writes exactly as they would through a shared session.
    """

    cameras: dict[str, Camera] = field(default_factory=dict)
    links: dict[tuple[str, str], CameraLink] = field(default_factory=dict)
    sightings: dict[str, Sighting] = field(default_factory=dict)
    targets: dict[str, Target] = field(default_factory=dict)
    matches: dict[tuple[str, str], MatchCandidate] = field(default_factory=dict)
    trajectories: dict[str, Trajectory] = field(default_factory=dict)


@dataclass(frozen=True)
class RepositorySet:
    """One repository of each kind, backed by the same storage.

    The conformance suite takes this so a single test body can run against
    either backend.
    """

    cameras: CameraRepository
    links: CameraLinkRepository
    sightings: SightingRepository
    targets: TargetRepository
    matches: MatchRepository
    trajectories: TrajectoryRepository


def _in_window(moment: datetime, window: TimeWindow | None) -> bool:
    """Return whether ``moment`` falls in a half-open window.

    Args:
        moment: The instant to test.
        window: The window, or ``None`` to accept everything.

    Returns:
        ``True`` when ``window`` is ``None`` or contains ``moment``.
    """
    return window is None or window.contains(moment)


def _time_ordered(sightings: list[Sighting]) -> list[Sighting]:
    """Sort sightings the way the SQL ORDER BY does.

    Args:
        sightings: The sightings to order.

    Returns:
        Sorted by ascending timestamp, then by id to break ties deterministically.
    """
    return sorted(sightings, key=lambda item: (item.timestamp_utc, item.sighting_id))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two vectors.

    Computed in full rather than as a dot product: although stored embeddings
    are L2-normalized to within a tolerance, pgvector computes the true cosine,
    and the fake must agree with it.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Cosine similarity, or ``0.0`` if either vector has zero magnitude.
    """
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryCameraRepository:
    """In-memory :class:`CameraRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def upsert(self, camera: Camera) -> Camera:
        """Insert or replace a camera.

        Args:
            camera: The camera to store.

        Returns:
            The stored camera.
        """
        self._store.cameras[camera.camera_id] = camera
        return camera

    def get(self, camera_id: str) -> Camera | None:
        """Return one camera by id.

        Args:
            camera_id: The identifier to look up.

        Returns:
            The camera, or ``None``.
        """
        return self._store.cameras.get(camera_id)

    def list_enabled(self) -> list[Camera]:
        """Return enabled cameras ordered by id.

        Returns:
            The enabled cameras.
        """
        return sorted(
            (camera for camera in self._store.cameras.values() if camera.enabled),
            key=lambda camera: camera.camera_id,
        )

    def delete(self, camera_id: str) -> bool:
        """Delete a camera unless it still has sightings.

        Args:
            camera_id: The identifier to delete.

        Returns:
            ``True`` if a row was removed.

        Raises:
            StorageError: If sightings reference it, mirroring ON DELETE
                RESTRICT.
        """
        if camera_id not in self._store.cameras:
            return False

        referencing = sum(
            1 for sighting in self._store.sightings.values() if sighting.camera_id == camera_id
        )
        if referencing:
            raise StorageError(
                "camera delete violated a database constraint",
                {
                    "camera_id": camera_id,
                    "constraint": "fk_sightings_camera",
                    "referencing_sightings": referencing,
                },
            )

        del self._store.cameras[camera_id]
        self._store.links = {
            key: link
            for key, link in self._store.links.items()
            if camera_id not in (link.from_camera_id, link.to_camera_id)
        }
        return True


class InMemoryCameraLinkRepository:
    """In-memory :class:`CameraLinkRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def upsert(self, link: CameraLink) -> CameraLink:
        """Insert or replace the link for one ordered pair.

        Args:
            link: The link to store.

        Returns:
            The stored link.

        Raises:
            StorageError: If either endpoint is unknown, mirroring the foreign
                keys.
        """
        for camera_id in (link.from_camera_id, link.to_camera_id):
            if camera_id not in self._store.cameras:
                raise StorageError(
                    "camera link upsert violated a database constraint",
                    {"constraint": "fk_camera_links_camera", "camera_id": camera_id},
                )
        self._store.links[link.from_camera_id, link.to_camera_id] = link
        return link

    def get_link(self, from_camera_id: str, to_camera_id: str) -> CameraLink | None:
        """Return the link for one ordered pair.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            The link, or ``None``.
        """
        return self._store.links.get((from_camera_id, to_camera_id))

    def get_links_from(self, camera_id: str) -> list[CameraLink]:
        """Return links whose origin is ``camera_id``, ordered by destination.

        Args:
            camera_id: Origin camera.

        Returns:
            The links.
        """
        return sorted(
            (link for link in self._store.links.values() if link.from_camera_id == camera_id),
            key=lambda link: link.to_camera_id,
        )

    def list_all(self) -> list[CameraLink]:
        """Return every link, ordered by origin then destination.

        Returns:
            All links.
        """
        return sorted(
            self._store.links.values(),
            key=lambda link: (link.from_camera_id, link.to_camera_id),
        )


class InMemorySightingRepository:
    """In-memory :class:`SightingRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    # -- writes -------------------------------------------------------------

    def add(self, sighting: Sighting) -> Sighting:
        """Insert one sighting.

        Args:
            sighting: The sighting to store.

        Returns:
            The stored sighting.

        Raises:
            StorageError: If the camera is unknown or the id already exists.
        """
        self._require_known_camera(sighting)
        if sighting.sighting_id in self._store.sightings:
            raise StorageError(
                "sighting insert violated a database constraint",
                {"constraint": "pk_sightings", "sighting_id": sighting.sighting_id},
            )
        self._store.sightings[sighting.sighting_id] = sighting
        return sighting

    def add_batch(self, sightings: list[Sighting], *, ignore_conflicts: bool = False) -> int:
        """Insert many sightings, optionally skipping duplicates.

        Args:
            sightings: The sightings to store.
            ignore_conflicts: Skip ids that already exist.

        Returns:
            The number actually inserted.

        Raises:
            StorageError: On an unknown camera, or on a conflict when
                ``ignore_conflicts`` is ``False``. Nothing is written in that
                case, mirroring the atomicity of a single INSERT statement.
        """
        for sighting in sightings:
            self._require_known_camera(sighting)

        if not ignore_conflicts:
            clashes = [s.sighting_id for s in sightings if s.sighting_id in self._store.sightings]
            if clashes:
                raise StorageError(
                    "sighting batch insert violated a database constraint",
                    {"constraint": "pk_sightings", "sighting_ids": clashes},
                )

        inserted = 0
        for sighting in sightings:
            if sighting.sighting_id in self._store.sightings:
                continue
            self._store.sightings[sighting.sighting_id] = sighting
            inserted += 1
        return inserted

    def _require_known_camera(self, sighting: Sighting) -> None:
        """Reject a sighting whose camera does not exist.

        Args:
            sighting: The sighting to check.

        Raises:
            StorageError: Mirroring ``fk_sightings_camera``.
        """
        if sighting.camera_id not in self._store.cameras:
            raise StorageError(
                "sighting insert violated a database constraint",
                {"constraint": "fk_sightings_camera", "camera_id": sighting.camera_id},
            )

    # -- reads --------------------------------------------------------------

    def get(self, sighting_id: str) -> Sighting | None:
        """Return one sighting by id.

        Args:
            sighting_id: The identifier to look up.

        Returns:
            The sighting, or ``None``.
        """
        return self._store.sightings.get(sighting_id)

    def find_by_plate_exact(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose normalized plate matches exactly.

        Args:
            plate_normalized: The plate to match.
            window: Optional half-open window.

        Returns:
            Matches ordered by ascending timestamp.
        """
        return _time_ordered(
            [
                sighting
                for sighting in self._store.sightings.values()
                if sighting.plate_text_normalized == plate_normalized
                and _in_window(sighting.timestamp_utc, window)
            ]
        )

    def find_by_plate_folded(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose folded plate matches the folded query.

        Args:
            plate_normalized: The plate to fold and match.
            window: Optional half-open window.

        Returns:
            Matches ordered by ascending timestamp.
        """
        wanted = fold_plate(plate_normalized)
        return _time_ordered(
            [
                sighting
                for sighting in self._store.sightings.values()
                if sighting.plate_text_normalized is not None
                and fold_plate(sighting.plate_text_normalized) == wanted
                and _in_window(sighting.timestamp_utc, window)
            ]
        )

    def find_by_camera_and_window(self, camera_id: str, window: TimeWindow) -> list[Sighting]:
        """Return one camera's sightings within a window.

        Args:
            camera_id: The observing camera.
            window: Half-open window.

        Returns:
            Matches ordered by ascending timestamp; empty list when none.
        """
        return _time_ordered(
            [
                sighting
                for sighting in self._store.sightings.values()
                if sighting.camera_id == camera_id and window.contains(sighting.timestamp_utc)
            ]
        )

    def find_by_window(self, window: TimeWindow) -> list[Sighting]:
        """Return all cameras' sightings within a window.

        Args:
            window: Half-open window.

        Returns:
            Matches ordered by ascending timestamp.
        """
        return _time_ordered(
            [
                sighting
                for sighting in self._store.sightings.values()
                if window.contains(sighting.timestamp_utc)
            ]
        )

    def find_nearest_by_embedding(
        self,
        embedding: list[float],
        k: int,
        *,
        camera_ids: list[str] | None = None,
        window: TimeWindow | None = None,
        model_version: str | None = None,
    ) -> list[EmbeddingMatch]:
        """Return the ``k`` most similar sightings.

        Args:
            embedding: Query vector.
            k: Maximum results.
            camera_ids: Restrict to these cameras.
            window: Optional half-open window.
            model_version: Restrict to embeddings from this model.

        Returns:
            At most ``k`` matches, most similar first.

        Raises:
            StorageError: If the query vector has the wrong dimension.
        """
        expected = _expected_embedding_dim()
        if len(embedding) != expected:
            raise StorageError(
                "Query embedding has the wrong dimension",
                {"expected": expected, "actual": len(embedding)},
            )
        if k <= 0:
            return []

        allowed = set(camera_ids) if camera_ids is not None else None
        scored = [
            EmbeddingMatch(
                sighting=sighting,
                similarity=_cosine_similarity(embedding, sighting.embedding),
            )
            for sighting in self._store.sightings.values()
            if sighting.embedding is not None
            and (allowed is None or sighting.camera_id in allowed)
            and (model_version is None or sighting.embedding_model_version == model_version)
            and _in_window(sighting.timestamp_utc, window)
        ]
        # Ties broken by id so the fake is deterministic; Postgres orders by
        # distance alone, but no two distinct vectors tie in practice.
        scored.sort(key=lambda match: (-match.similarity, match.sighting.sighting_id))
        return scored[:k]

    def apply_clock_offset(self, camera_id: str, new_offset_ms: int) -> int:
        """Recompute every stored sighting for one camera under a new offset.

        Args:
            camera_id: The camera whose clock is being corrected.
            new_offset_ms: Milliseconds to add to its raw timestamps.

        Returns:
            How many sightings were rewritten.
        """
        from multicam_tracker.timesync import correct_sighting

        rewritten = 0
        for sighting_id, sighting in list(self._store.sightings.items()):
            if sighting.camera_id != camera_id:
                continue
            self._store.sightings[sighting_id] = correct_sighting(sighting, new_offset_ms)
            rewritten += 1
        return rewritten

    def count_by_camera_hour(self, window: TimeWindow) -> list[CameraHourCount]:
        """Return per-camera, per-UTC-hour counts.

        Args:
            window: Half-open window.

        Returns:
            One entry per populated bucket, ordered by camera then hour.
        """
        buckets: dict[tuple[str, datetime], int] = {}
        for sighting in self._store.sightings.values():
            if not window.contains(sighting.timestamp_utc):
                continue
            hour = sighting.timestamp_utc.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
            buckets[sighting.camera_id, hour] = buckets.get((sighting.camera_id, hour), 0) + 1

        return [
            CameraHourCount(camera_id=camera_id, hour_start_utc=hour, sighting_count=count)
            for (camera_id, hour), count in sorted(buckets.items())
        ]

    # -- retention ----------------------------------------------------------

    def delete_older_than(self, cutoff_utc: datetime) -> PurgeResult:
        """Delete sightings created before ``cutoff_utc``.

        Args:
            cutoff_utc: Aware UTC cutoff; rows exactly at it are kept.

        Returns:
            Deleted count and orphaned thumbnail paths.
        """
        doomed = [
            sighting
            for sighting in self._store.sightings.values()
            if sighting.created_at < cutoff_utc
        ]
        paths = [s.thumbnail_path for s in doomed if s.thumbnail_path is not None]
        doomed_ids = {sighting.sighting_id for sighting in doomed}

        for sighting_id in doomed_ids:
            del self._store.sightings[sighting_id]

        # Mirrors ON DELETE CASCADE on match_candidates.sighting_id. Saved
        # trajectories are deliberately left alone; see the protocol docstring.
        self._store.matches = {
            key: candidate
            for key, candidate in self._store.matches.items()
            if candidate.sighting_id not in doomed_ids
        }

        return PurgeResult(deleted_count=len(doomed_ids), thumbnail_paths=paths)


class InMemoryTargetRepository:
    """In-memory :class:`TargetRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def create(self, target: Target) -> Target:
        """Insert a new target.

        Args:
            target: The target to store.

        Returns:
            The stored target.

        Raises:
            StorageError: If the id already exists.
        """
        if target.target_id in self._store.targets:
            raise StorageError(
                "target insert violated a database constraint",
                {"constraint": "pk_targets", "target_id": target.target_id},
            )
        self._store.targets[target.target_id] = target
        return target

    def get(self, target_id: str) -> Target | None:
        """Return one target by id.

        Args:
            target_id: The identifier to look up.

        Returns:
            The target, or ``None``.
        """
        return self._store.targets.get(target_id)

    def list_active(self) -> list[Target]:
        """Return active targets, newest first.

        Returns:
            The active targets.
        """
        return sorted(
            (target for target in self._store.targets.values() if target.active),
            key=lambda target: (target.created_at, target.target_id),
            reverse=True,
        )

    def deactivate(self, target_id: str) -> bool:
        """Mark a target inactive.

        Args:
            target_id: The identifier to deactivate.

        Returns:
            ``True`` if a row changed.
        """
        target = self._store.targets.get(target_id)
        if target is None or not target.active:
            return False
        self._store.targets[target_id] = target.model_copy(update={"active": False})
        return True


class InMemoryMatchRepository:
    """In-memory :class:`MatchRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def upsert_candidate(self, candidate: MatchCandidate) -> MatchCandidate:
        """Insert or update the candidate for one (target, sighting) pair.

        Args:
            candidate: The candidate to store.

        Returns:
            The stored candidate.
        """
        self._store.matches[candidate.target_id, candidate.sighting_id] = candidate
        return candidate

    def bulk_upsert(self, candidates: list[MatchCandidate]) -> int:
        """Upsert many candidates.

        Args:
            candidates: The candidates to store.

        Returns:
            The number written.
        """
        for candidate in candidates:
            self.upsert_candidate(candidate)
        return len(candidates)

    def list_for_target(
        self, target_id: str, review_status: ReviewStatus | None = None
    ) -> list[MatchCandidate]:
        """Return a target's candidates, strongest first.

        Args:
            target_id: The target whose candidates to list.
            review_status: Restrict to this status, or ``None`` for all.

        Returns:
            Candidates ordered by descending score.
        """
        selected = [
            candidate
            for candidate in self._store.matches.values()
            if candidate.target_id == target_id
            and (review_status is None or candidate.review_status is review_status)
        ]
        return sorted(selected, key=lambda c: (-c.match_score, c.sighting_id))

    def update_review_status(
        self, target_id: str, sighting_id: str, review_status: ReviewStatus
    ) -> bool:
        """Set one candidate's review status.

        Args:
            target_id: The candidate's target.
            sighting_id: The candidate's sighting.
            review_status: The new status.

        Returns:
            ``True`` if a row changed.
        """
        existing = self._store.matches.get((target_id, sighting_id))
        if existing is None:
            return False
        self._store.matches[target_id, sighting_id] = existing.model_copy(
            update={"review_status": review_status}
        )
        return True


class InMemoryTrajectoryRepository:
    """In-memory :class:`TrajectoryRepository`."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def save(self, trajectory: Trajectory) -> Trajectory:
        """Insert or replace a trajectory.

        Only the sighting ids are retained, mirroring the schema: the row stores
        references, not copies, so a purge is visible on the way back out.

        Args:
            trajectory: The trajectory to store.

        Returns:
            The stored trajectory.
        """
        self._store.trajectories[trajectory.trajectory_id] = trajectory
        return trajectory

    def get(self, trajectory_id: str) -> Trajectory | None:
        """Return one trajectory with its sightings re-resolved.

        Args:
            trajectory_id: The identifier to look up.

        Returns:
            The trajectory, or ``None``.

        Raises:
            StorageError: If any referenced sighting has been purged.
        """
        trajectory = self._store.trajectories.get(trajectory_id)
        if trajectory is None:
            return None
        return self._resolve(trajectory)

    def flag_for_recomputation(self, camera_id: str) -> list[str]:
        """Mark every trajectory that used one camera as needing recomputation.

        Args:
            camera_id: The camera whose clock was corrected.

        Returns:
            The ids of the trajectories that were flagged.
        """
        flagged: list[str] = []
        for trajectory_id, trajectory in list(self._store.trajectories.items()):
            if not any(sighting.camera_id == camera_id for sighting in trajectory.sightings):
                continue
            self._store.trajectories[trajectory_id] = trajectory.model_copy(
                update={"requires_recomputation": True}
            )
            flagged.append(trajectory_id)
        return sorted(flagged)

    def list_for_target(self, target_id: str) -> list[Trajectory]:
        """Return a target's trajectories, newest first.

        Args:
            target_id: The target whose trajectories to list.

        Returns:
            The trajectories.

        Raises:
            StorageError: If a stored trajectory references purged sightings.
        """
        selected = [
            trajectory
            for trajectory in self._store.trajectories.values()
            if trajectory.target_id == target_id
        ]
        ordered = sorted(selected, key=lambda t: (t.start_time_utc, t.trajectory_id), reverse=True)
        return [self._resolve(trajectory) for trajectory in ordered]

    def _resolve(self, trajectory: Trajectory) -> Trajectory:
        """Re-read the trajectory's sightings from the store.

        Args:
            trajectory: The stored trajectory.

        Returns:
            The trajectory with current sighting rows.

        Raises:
            StorageError: If any sighting has been purged, matching the Postgres
                implementation rather than returning a shortened route.
        """
        missing = [
            sighting.sighting_id
            for sighting in trajectory.sightings
            if sighting.sighting_id not in self._store.sightings
        ]
        if missing:
            raise StorageError(
                "Trajectory references sightings that no longer exist",
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "missing_sighting_ids": missing,
                    "hint": "the sightings were removed by the retention purge",
                },
            )
        return trajectory


def _expected_embedding_dim() -> int:
    """Return the configured embedding dimension.

    Read lazily so the fakes do not import the ORM, which would drag SQLAlchemy
    into every unit test that only wants a fake.

    Returns:
        The configured dimension.
    """
    from multicam_tracker.config import get_settings

    return get_settings().vision.embedding_dim


def build_in_memory_repositories(store: InMemoryStore | None = None) -> RepositorySet:
    """Build a full set of in-memory repositories over one store.

    Args:
        store: Existing store to reuse. A fresh one is created when omitted.

    Returns:
        One repository of each kind, all sharing the same state.
    """
    resolved = store if store is not None else InMemoryStore()
    return RepositorySet(
        cameras=InMemoryCameraRepository(resolved),
        links=InMemoryCameraLinkRepository(resolved),
        sightings=InMemorySightingRepository(resolved),
        targets=InMemoryTargetRepository(resolved),
        matches=InMemoryMatchRepository(resolved),
        trajectories=InMemoryTrajectoryRepository(resolved),
    )
