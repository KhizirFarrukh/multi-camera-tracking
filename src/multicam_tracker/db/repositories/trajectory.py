"""Postgres implementation of the trajectory repository."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects import postgresql

from multicam_tracker.db.mappers import (
    sighting_to_domain,
    trajectory_to_domain,
    trajectory_to_orm,
)
from multicam_tracker.db.orm import SightingORM, TrajectoryHopORM, TrajectoryORM
from multicam_tracker.db.repositories._base import PostgresRepositoryBase, storage_errors
from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import Sighting, Trajectory

__all__ = ["PostgresTrajectoryRepository"]


class PostgresTrajectoryRepository(PostgresRepositoryBase):
    """Trajectory persistence backed by Postgres."""

    def save(self, trajectory: Trajectory) -> Trajectory:
        """Insert a trajectory and its hops, replacing any prior version.

        Delete-then-insert rather than a merge: hops are positional, and an
        in-place update would have to reconcile a changed hop count against
        existing positions. Removing the old row first makes the write
        unambiguous, and the cascade takes its hops with it.

        Args:
            trajectory: The trajectory to store.

        Returns:
            The stored trajectory.

        Raises:
            StorageError: If the write fails.
        """
        row, hops = trajectory_to_orm(trajectory)

        with storage_errors("trajectory save", trajectory_id=trajectory.trajectory_id):
            self._session.execute(
                delete(TrajectoryORM).where(TrajectoryORM.trajectory_id == row.trajectory_id)
            )
            self._session.flush()
            self._session.add(row)
            self._session.add_all(hops)
            self._session.flush()

        return trajectory

    def get(self, trajectory_id: str) -> Trajectory | None:
        """Return one trajectory with its sightings resolved in stored order.

        Args:
            trajectory_id: The identifier to look up.

        Returns:
            The trajectory, or ``None`` if no such row exists.

        Raises:
            StorageError: If any referenced sighting has been purged.
        """
        with storage_errors("trajectory lookup", trajectory_id=trajectory_id):
            row = self._session.get(TrajectoryORM, _as_uuid(trajectory_id))
            if row is None:
                return None
            return self._rebuild(row)

    def flag_for_recomputation(self, camera_id: str) -> list[str]:
        """Mark every trajectory that used one camera as needing recomputation.

        Args:
            camera_id: The camera whose clock was corrected.

        Returns:
            The ids of the trajectories that were flagged.

        Raises:
            StorageError: If the write fails.
        """
        # The camera's sightings, aggregated into one array so the overlap test
        # is a single expression. coalesce keeps a camera with no sightings from
        # producing NULL, which would match nothing *and* look like an error.
        affected = (
            select(func.coalesce(func.array_agg(SightingORM.sighting_id), postgresql.array([])))
            .where(SightingORM.camera_id == camera_id)
            .scalar_subquery()
        )
        statement = (
            update(TrajectoryORM)
            # Array overlap: a trajectory is affected if any sighting it names
            # came from this camera. One statement, so the flag lands atomically
            # with whatever else the caller's transaction is doing.
            .where(TrajectoryORM.sighting_ids.overlap(affected))
            .values(requires_recomputation=True)
            .returning(TrajectoryORM.trajectory_id)
        )

        with storage_errors("flagging trajectories for recomputation", camera_id=camera_id):
            rows = self._session.execute(statement).scalars().all()

        return [str(value) for value in rows]

    def list_for_target(self, target_id: str) -> list[Trajectory]:
        """Return a target's trajectories, newest first.

        Args:
            target_id: The target whose trajectories to list.

        Returns:
            The trajectories, ordered by descending ``start_time_utc``.

        Raises:
            StorageError: If the read fails or a stored trajectory references
                purged sightings.
        """
        statement = (
            select(TrajectoryORM)
            .where(TrajectoryORM.target_id == _as_uuid(target_id))
            .order_by(TrajectoryORM.start_time_utc.desc(), TrajectoryORM.trajectory_id)
        )
        with storage_errors("trajectory listing", target_id=target_id):
            rows = self._session.execute(statement).scalars().all()
            return [self._rebuild(row) for row in rows]

    def _rebuild(self, row: TrajectoryORM) -> Trajectory:
        """Reassemble a trajectory from its row, hops, and sightings.

        Args:
            row: The trajectory row.

        Returns:
            The domain trajectory.

        Raises:
            StorageError: If any sighting named in ``sighting_ids`` is missing,
                which means retention purged it. The trajectory is a historical
                conclusion and is not rewritten to fit what survives, so this is
                reported rather than papered over.
        """
        hop_rows = list(
            self._session.execute(
                select(TrajectoryHopORM)
                .where(TrajectoryHopORM.trajectory_id == row.trajectory_id)
                .order_by(TrajectoryHopORM.position)
            )
            .scalars()
            .all()
        )

        found = {
            sighting.sighting_id: sighting
            for sighting in self._session.execute(
                select(SightingORM).where(SightingORM.sighting_id.in_(row.sighting_ids))
            )
            .scalars()
            .all()
        }

        missing = [str(identifier) for identifier in row.sighting_ids if identifier not in found]
        if missing:
            raise StorageError(
                "Trajectory references sightings that no longer exist",
                {
                    "trajectory_id": str(row.trajectory_id),
                    "missing_sighting_ids": missing,
                    "hint": "the sightings were removed by the retention purge",
                },
            )

        sightings: list[Sighting] = [
            sighting_to_domain(found[identifier]) for identifier in row.sighting_ids
        ]
        return trajectory_to_domain(row, hop_rows, sightings)


def _as_uuid(value: str) -> uuid.UUID:
    """Parse an identifier, reporting a malformed one as a storage error.

    Args:
        value: The identifier string.

    Returns:
        The parsed UUID.

    Raises:
        StorageError: If the value is not a UUID.
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise StorageError("Identifier is not a valid UUID", {"value": str(value)}) from exc
