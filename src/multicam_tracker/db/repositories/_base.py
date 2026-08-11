"""Shared plumbing for the Postgres repository implementations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from sqlalchemy import CursorResult, Result
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from multicam_tracker.exceptions import StorageError

__all__ = ["PostgresRepositoryBase", "rowcount_of", "storage_errors"]

UNKNOWN_ROWCOUNT = -1
"""Returned when the driver cannot report how many rows a statement touched."""


def rowcount_of(result: Result[Any]) -> int:
    """Return how many rows a DML statement affected.

    ``Session.execute`` is typed as returning :class:`Result`, but every DML
    statement actually returns a :class:`CursorResult`, which is the only one
    carrying ``rowcount``. The cast is confined here rather than repeated at
    each call site.

    Args:
        result: The result of an INSERT, UPDATE, or DELETE.

    Returns:
        The affected row count, or :data:`UNKNOWN_ROWCOUNT` when the driver does
        not report one -- which some drivers do for executemany.
    """
    raw = cast("CursorResult[Any]", result).rowcount
    return int(raw) if raw is not None and raw >= 0 else UNKNOWN_ROWCOUNT


@contextmanager
def storage_errors(operation: str, **context: Any) -> Iterator[None]:
    """Translate SQLAlchemy and driver errors into :class:`StorageError`.

    No driver exception may escape the repository layer. A caller forced to
    catch ``psycopg.errors.UniqueViolation`` would be coupled to the database in
    use, and the in-memory fakes could not raise the same thing, which would
    make the conformance suite meaningless.

    Integrity errors are given a distinguishable ``constraint`` context entry
    where the driver supplies one, since "which constraint" is the first thing
    anyone asks.

    Args:
        operation: Short name of the failing operation, used as the message.
        **context: Extra detail attached to the raised error.

    Yields:
        ``None``.

    Raises:
        StorageError: Wrapping any :class:`~sqlalchemy.exc.SQLAlchemyError`.
    """
    try:
        yield
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        raise StorageError(
            f"{operation} violated a database constraint",
            {**context, "constraint": constraint, "reason": str(exc.orig or exc)},
        ) from exc
    except DBAPIError as exc:
        raise StorageError(
            f"{operation} failed at the database driver",
            {**context, "reason": str(exc.orig or exc)},
        ) from exc
    except SQLAlchemyError as exc:
        raise StorageError(f"{operation} failed", {**context, "reason": str(exc)}) from exc


class PostgresRepositoryBase:
    """Base for repositories backed by a SQLAlchemy session.

    The session is injected rather than created. Transaction boundaries belong
    to the caller (``session_scope``), so a single request or pipeline step can
    span several repositories and still commit atomically.

    Args:
        session: An open SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Return the underlying session."""
        return self._session
