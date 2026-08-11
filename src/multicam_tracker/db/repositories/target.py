"""Postgres implementation of the target repository."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update

from multicam_tracker.db.mappers import target_to_domain, target_to_orm
from multicam_tracker.db.orm import TargetORM
from multicam_tracker.db.repositories._base import (
    PostgresRepositoryBase,
    rowcount_of,
    storage_errors,
)
from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import Target

__all__ = ["PostgresTargetRepository"]


class PostgresTargetRepository(PostgresRepositoryBase):
    """Target persistence backed by Postgres."""

    def create(self, target: Target) -> Target:
        """Insert a new target.

        Args:
            target: The target to store.

        Returns:
            The stored target.

        Raises:
            StorageError: If the id already exists or the write fails.
        """
        with storage_errors("target insert", target_id=target.target_id):
            self._session.add(target_to_orm(target))
            self._session.flush()
        return target

    def get(self, target_id: str) -> Target | None:
        """Return one target by id.

        Args:
            target_id: The identifier to look up.

        Returns:
            The target, or ``None``.

        Raises:
            StorageError: If the id is malformed or the read fails.
        """
        with storage_errors("target lookup", target_id=target_id):
            row = self._session.get(TargetORM, _as_uuid(target_id))
        return target_to_domain(row) if row is not None else None

    def list_active(self) -> list[Target]:
        """Return every active target, newest first.

        Returns:
            The active targets.

        Raises:
            StorageError: If the read fails.
        """
        statement = (
            select(TargetORM)
            .where(TargetORM.active.is_(True))
            .order_by(TargetORM.created_at.desc(), TargetORM.target_id)
        )
        with storage_errors("active target listing"):
            rows = self._session.execute(statement).scalars().all()
        return [target_to_domain(row) for row in rows]

    def deactivate(self, target_id: str) -> bool:
        """Mark a target inactive without deleting it.

        The row is kept because the audit trail references it: a confirmed match
        must remain explicable after the search that produced it is closed.

        Args:
            target_id: The identifier to deactivate.

        Returns:
            ``True`` if a row changed.

        Raises:
            StorageError: If the write fails.
        """
        statement = (
            update(TargetORM)
            .where(TargetORM.target_id == _as_uuid(target_id), TargetORM.active.is_(True))
            .values(active=False)
        )
        with storage_errors("target deactivation", target_id=target_id):
            result = self._session.execute(statement)
            self._session.flush()
        return rowcount_of(result) > 0


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
