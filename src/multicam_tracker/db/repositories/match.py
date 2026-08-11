"""Postgres implementation of the match-candidate repository."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from multicam_tracker.db.mappers import match_candidate_to_domain, match_candidate_to_orm
from multicam_tracker.db.orm import MatchCandidateORM
from multicam_tracker.db.repositories._base import (
    PostgresRepositoryBase,
    rowcount_of,
    storage_errors,
)
from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import MatchCandidate, ReviewStatus

__all__ = ["PostgresMatchRepository"]

_KEY_COLUMNS = ("target_id", "sighting_id")


class PostgresMatchRepository(PostgresRepositoryBase):
    """Match-candidate persistence backed by Postgres."""

    def upsert_candidate(self, candidate: MatchCandidate) -> MatchCandidate:
        """Insert or update the candidate for one (target, sighting) pair.

        Args:
            candidate: The candidate to store.

        Returns:
            The stored candidate.

        Raises:
            StorageError: If the write fails.
        """
        with storage_errors(
            "match candidate upsert",
            target_id=candidate.target_id,
            sighting_id=candidate.sighting_id,
        ):
            self._session.execute(_upsert_statement([_row_values(candidate)]))
            self._session.flush()
        return candidate

    def bulk_upsert(self, candidates: list[MatchCandidate]) -> int:
        """Upsert many candidates in one statement.

        Args:
            candidates: The candidates to store.

        Returns:
            The number of rows written.

        Raises:
            StorageError: If the write fails.
        """
        if not candidates:
            return 0

        with storage_errors("match candidate bulk upsert", batch_size=len(candidates)):
            self._session.execute(_upsert_statement([_row_values(c) for c in candidates]))
            self._session.flush()
        return len(candidates)

    def list_for_target(
        self, target_id: str, review_status: ReviewStatus | None = None
    ) -> list[MatchCandidate]:
        """Return a target's candidates, strongest evidence first.

        Args:
            target_id: The target whose candidates to list.
            review_status: Restrict to this status, or ``None`` for all.

        Returns:
            The candidates, ordered by descending ``match_score``.

        Raises:
            StorageError: If the read fails.
        """
        statement = select(MatchCandidateORM).where(
            MatchCandidateORM.target_id == _as_uuid(target_id)
        )
        if review_status is not None:
            statement = statement.where(MatchCandidateORM.review_status == review_status.value)
        statement = statement.order_by(
            MatchCandidateORM.match_score.desc(), MatchCandidateORM.sighting_id
        )

        with storage_errors("match candidate listing", target_id=target_id):
            rows = self._session.execute(statement).scalars().all()
        return [match_candidate_to_domain(row) for row in rows]

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

        Raises:
            StorageError: If the write fails.
        """
        statement = (
            update(MatchCandidateORM)
            .where(
                MatchCandidateORM.target_id == _as_uuid(target_id),
                MatchCandidateORM.sighting_id == _as_uuid(sighting_id),
            )
            .values(review_status=review_status.value)
        )
        with storage_errors(
            "match review status update", target_id=target_id, sighting_id=sighting_id
        ):
            result = self._session.execute(statement)
            self._session.flush()
        return rowcount_of(result) > 0


def _row_values(candidate: MatchCandidate) -> dict[str, Any]:
    """Return a candidate as a column-keyed dict.

    Args:
        candidate: The candidate to flatten.

    Returns:
        Column name -> value.
    """
    row = match_candidate_to_orm(candidate)
    return {column.key: getattr(row, column.key) for column in MatchCandidateORM.__table__.columns}


def _upsert_statement(payloads: list[dict[str, Any]]) -> Any:
    """Build an ON CONFLICT DO UPDATE statement keyed on the candidate pair.

    Args:
        payloads: Column-keyed row values.

    Returns:
        The insert statement.
    """
    statement = pg_insert(MatchCandidateORM).values(payloads)
    return statement.on_conflict_do_update(
        index_elements=[MatchCandidateORM.target_id, MatchCandidateORM.sighting_id],
        set_={
            column.key: getattr(statement.excluded, column.key)
            for column in MatchCandidateORM.__table__.columns
            if column.key not in _KEY_COLUMNS
        },
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
