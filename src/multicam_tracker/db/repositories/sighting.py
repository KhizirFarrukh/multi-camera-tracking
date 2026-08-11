"""Postgres implementation of the sighting repository.

This is the hot path of the system: the pipeline writes thousands of rows per
minute and every search reads from here. Each query below maps to one of the
indexes declared in :mod:`multicam_tracker.db.orm`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from multicam_tracker.db.folding import fold_plate
from multicam_tracker.db.mappers import sighting_to_domain, sighting_to_orm
from multicam_tracker.db.orm import EMBEDDING_DIM, SightingORM
from multicam_tracker.db.repositories._base import (
    PostgresRepositoryBase,
    rowcount_of,
    storage_errors,
)
from multicam_tracker.db.repositories.protocols import (
    CameraHourCount,
    EmbeddingMatch,
    PurgeResult,
)
from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import Sighting, TimeWindow

__all__ = ["BATCH_CHUNK_SIZE", "PostgresSightingRepository"]

BATCH_CHUNK_SIZE = 1000
"""Rows per INSERT during a batch write.

Chunking keeps the statement and its parameter list bounded -- Postgres caps
bind parameters at 65535, and a sighting has 18 of them, so an unchunked
10,000-row insert would exceed the limit outright.
"""


class PostgresSightingRepository(PostgresRepositoryBase):
    """Sighting persistence and queries backed by Postgres."""

    # -- writes -------------------------------------------------------------

    def add(self, sighting: Sighting) -> Sighting:
        """Insert one sighting.

        Args:
            sighting: The sighting to store.

        Returns:
            The stored sighting.

        Raises:
            StorageError: If the camera is unknown, the id exists, or the write
                fails.
        """
        with storage_errors("sighting insert", sighting_id=sighting.sighting_id):
            self._session.add(sighting_to_orm(sighting))
            self._session.flush()
        return sighting

    def add_batch(self, sightings: list[Sighting], *, ignore_conflicts: bool = False) -> int:
        """Bulk-insert sightings, chunked, optionally skipping duplicates.

        Args:
            sightings: The sightings to store.
            ignore_conflicts: Skip rows whose ``sighting_id`` already exists.

        Returns:
            The number of rows actually inserted.

        Raises:
            StorageError: If the write fails, or on a conflict when
                ``ignore_conflicts`` is ``False``.
        """
        if not sightings:
            return 0

        payloads = [self._row_values(sighting) for sighting in sightings]
        inserted = 0

        with storage_errors("sighting batch insert", batch_size=len(sightings)):
            for start in range(0, len(payloads), BATCH_CHUNK_SIZE):
                chunk = payloads[start : start + BATCH_CHUNK_SIZE]
                statement = pg_insert(SightingORM).values(chunk)
                if ignore_conflicts:
                    statement = statement.on_conflict_do_nothing(
                        index_elements=[SightingORM.sighting_id]
                    )
                affected = rowcount_of(self._session.execute(statement))
                # A driver that declines to report a count did not fail: the
                # statement either inserted the whole chunk or raised.
                inserted += affected if affected >= 0 else len(chunk)
            self._session.flush()

        return inserted

    @staticmethod
    def _row_values(sighting: Sighting) -> dict[str, Any]:
        """Return a sighting as a column-keyed dict for bulk insertion.

        ``plate_folded`` is excluded: it is a generated column, and naming it in
        an INSERT is an error in Postgres.

        Args:
            sighting: The sighting to flatten.

        Returns:
            Column name -> value.
        """
        row = sighting_to_orm(sighting)
        return {
            column.key: getattr(row, column.key)
            for column in SightingORM.__table__.columns
            if column.computed is None
        }

    # -- reads --------------------------------------------------------------

    def get(self, sighting_id: str) -> Sighting | None:
        """Return one sighting by id.

        Args:
            sighting_id: The identifier to look up.

        Returns:
            The sighting, or ``None``.

        Raises:
            StorageError: If the id is malformed or the read fails.
        """
        with storage_errors("sighting lookup", sighting_id=sighting_id):
            row = self._session.get(SightingORM, _as_uuid(sighting_id))
        return sighting_to_domain(row) if row is not None else None

    def find_by_plate_exact(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose normalized plate matches exactly.

        Args:
            plate_normalized: The plate to match.
            window: Optional half-open time window.

        Returns:
            Matches ordered by ascending ``timestamp_utc``.

        Raises:
            StorageError: If the read fails.
        """
        statement = select(SightingORM).where(SightingORM.plate_text_normalized == plate_normalized)
        return self._ordered_by_time(
            _apply_window(statement, window), "plate exact lookup", plate=plate_normalized
        )

    def find_by_plate_folded(
        self, plate_normalized: str, window: TimeWindow | None = None
    ) -> list[Sighting]:
        """Return sightings whose folded plate matches the folded query.

        Args:
            plate_normalized: The plate to fold and match.
            window: Optional half-open time window.

        Returns:
            Matches ordered by ascending ``timestamp_utc``.

        Raises:
            StorageError: If the read fails.
        """
        statement = select(SightingORM).where(
            SightingORM.plate_folded == fold_plate(plate_normalized)
        )
        return self._ordered_by_time(
            _apply_window(statement, window), "plate folded lookup", plate=plate_normalized
        )

    def find_by_camera_and_window(self, camera_id: str, window: TimeWindow) -> list[Sighting]:
        """Return one camera's sightings within a window.

        Args:
            camera_id: The observing camera.
            window: Half-open time window.

        Returns:
            Matches ordered by ascending ``timestamp_utc``.

        Raises:
            StorageError: If the read fails.
        """
        statement = _apply_window(
            select(SightingORM).where(SightingORM.camera_id == camera_id), window
        )
        return self._ordered_by_time(statement, "camera window scan", camera_id=camera_id)

    def find_by_window(self, window: TimeWindow) -> list[Sighting]:
        """Return all cameras' sightings within a window.

        Args:
            window: Half-open time window.

        Returns:
            Matches ordered by ascending ``timestamp_utc``.

        Raises:
            StorageError: If the read fails.
        """
        return self._ordered_by_time(
            _apply_window(select(SightingORM), window), "cross-camera window scan"
        )

    def find_nearest_by_embedding(
        self,
        embedding: list[float],
        k: int,
        *,
        camera_ids: list[str] | None = None,
        window: TimeWindow | None = None,
    ) -> list[EmbeddingMatch]:
        """Return the ``k`` most similar sightings by cosine similarity.

        Args:
            embedding: Query vector of the configured dimension.
            k: Maximum results.
            camera_ids: Restrict to these cameras.
            window: Optional half-open time window.

        Returns:
            At most ``k`` matches, most similar first.

        Raises:
            StorageError: If the vector has the wrong dimension or the read
                fails.
        """
        _check_dimension(embedding)
        if k <= 0:
            return []

        distance = SightingORM.embedding.cosine_distance(embedding)
        statement = (
            select(SightingORM, distance.label("distance"))
            .where(SightingORM.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        if camera_ids is not None:
            statement = statement.where(SightingORM.camera_id.in_(camera_ids))
        statement = _apply_window(statement, window)

        with storage_errors("embedding nearest-neighbour search", k=k):
            rows = self._session.execute(statement).all()

        return [
            EmbeddingMatch(sighting=sighting_to_domain(row[0]), similarity=1.0 - float(row[1]))
            for row in rows
        ]

    def count_by_camera_hour(self, window: TimeWindow) -> list[CameraHourCount]:
        """Return per-camera, per-UTC-hour counts within a window.

        Args:
            window: Half-open time window.

        Returns:
            One entry per populated (camera, hour) bucket.

        Raises:
            StorageError: If the read fails.
        """
        hour = func.date_trunc("hour", SightingORM.timestamp_utc).label("hour_start")
        statement = (
            _apply_window(select(SightingORM.camera_id, hour, func.count().label("total")), window)
            .group_by(SightingORM.camera_id, hour)
            .order_by(SightingORM.camera_id, hour)
        )

        with storage_errors("per-camera hourly count"):
            rows = self._session.execute(statement).all()

        return [
            CameraHourCount(camera_id=row[0], hour_start_utc=row[1], sighting_count=int(row[2]))
            for row in rows
        ]

    # -- retention ----------------------------------------------------------

    def delete_older_than(self, cutoff_utc: datetime) -> PurgeResult:
        """Delete sightings created before ``cutoff_utc``.

        Thumbnail paths are collected before the delete, since afterwards the
        rows naming them no longer exist.

        Args:
            cutoff_utc: Aware UTC cutoff. Rows exactly at it are kept.

        Returns:
            Deleted count and orphaned thumbnail paths.

        Raises:
            StorageError: If the delete fails.
        """
        doomed = select(SightingORM.thumbnail_path).where(
            SightingORM.created_at < cutoff_utc, SightingORM.thumbnail_path.is_not(None)
        )
        removal = delete(SightingORM).where(SightingORM.created_at < cutoff_utc)

        with storage_errors("retention purge", cutoff=cutoff_utc.isoformat()):
            paths = [path for path in self._session.execute(doomed).scalars().all() if path]
            result = self._session.execute(removal)
            self._session.flush()

        return PurgeResult(deleted_count=max(0, rowcount_of(result)), thumbnail_paths=paths)

    # -- helpers ------------------------------------------------------------

    def _ordered_by_time(
        self, statement: Select[Any], operation: str, **context: Any
    ) -> list[Sighting]:
        """Execute a sighting query ordered by ascending timestamp.

        Args:
            statement: The select to run.
            operation: Name used if it fails.
            **context: Extra error context.

        Returns:
            The matching sightings.

        Raises:
            StorageError: If the read fails.
        """
        ordered = statement.order_by(SightingORM.timestamp_utc, SightingORM.sighting_id)
        with storage_errors(operation, **context):
            rows = self._session.execute(ordered).scalars().all()
        return [sighting_to_domain(row) for row in rows]


def _apply_window(statement: Select[Any], window: TimeWindow | None) -> Select[Any]:
    """Add a half-open ``[start, end)`` filter on ``timestamp_utc``.

    Args:
        statement: The select to constrain.
        window: The window, or ``None`` for no constraint.

    Returns:
        The constrained select.
    """
    if window is None:
        return statement
    return statement.where(
        SightingORM.timestamp_utc >= window.start_utc,
        SightingORM.timestamp_utc < window.end_utc,
    )


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


def _check_dimension(embedding: list[float]) -> None:
    """Reject a query vector of the wrong width before it reaches the driver.

    Args:
        embedding: The query vector.

    Raises:
        StorageError: If its length is not the configured dimension. Postgres
            would reject it too, but with a message that does not name the
            expected width.
    """
    if len(embedding) != EMBEDDING_DIM:
        raise StorageError(
            "Query embedding has the wrong dimension",
            {"expected": EMBEDDING_DIM, "actual": len(embedding)},
        )
