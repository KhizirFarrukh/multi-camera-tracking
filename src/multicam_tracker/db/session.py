"""Engine, sessionmaker, and scoped-session helpers.

**Sync is the primary API.** The two heaviest consumers are the batch pipeline
(stages 14-15), which is GPU- and CPU-bound rather than IO-bound, and Alembic,
which is sync-only. Async would add colouring to every call path in exchange for
concurrency the workload does not need. FastAPI runs sync dependencies in a
threadpool, so the API layer pays nothing for the choice.

An async engine and session scope are provided alongside for stage 15's live
ingestion, where many camera streams are genuinely IO-concurrent. Both share the
same DSN and pool settings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from multicam_tracker.config import Settings, get_settings
from multicam_tracker.exceptions import StorageError
from multicam_tracker.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = [
    "SessionFactory",
    "build_engine",
    "build_session_factory",
    "dispose_engines",
    "get_async_engine",
    "get_async_session_factory",
    "get_engine",
    "get_session",
    "get_session_factory",
    "session_scope",
]

logger = get_logger(__name__)

SessionFactory = Callable[[], Session]
"""Anything that produces a :class:`~sqlalchemy.orm.Session` when called."""


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    """Return pool and connection options shared by the sync and async engines.

    Args:
        settings: Source of the pool configuration.

    Returns:
        Keyword arguments for ``create_engine`` / ``create_async_engine``.
    """
    return {
        "pool_size": settings.database.pool_size,
        "max_overflow": settings.database.pool_max_overflow,
        # Recycle before a typical idle-connection reaper closes the socket, so
        # the pool never hands out a connection the server has already dropped.
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "connect_args": {"connect_timeout": settings.database.connect_timeout_sec},
    }


def build_engine(settings: Settings | None = None) -> Engine:
    """Create a new sync engine.

    Args:
        settings: Configuration to build from. Defaults to the process settings.

    Returns:
        A fresh engine. Callers are responsible for disposing it; prefer
        :func:`get_engine` for the shared process-wide instance.
    """
    resolved = settings if settings is not None else get_settings()
    return create_engine(resolved.database.dsn, **_engine_kwargs(resolved))


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide sync engine, creating it on first call.

    Returns:
        The shared engine. One engine per process is correct: it owns the
        connection pool, and a second engine would silently double the
        connection count against the server's limit.
    """
    engine = build_engine()
    logger.info("database_engine_created", dsn=get_settings().database.safe_dsn)
    return engine


def build_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Create a sessionmaker bound to ``engine``.

    Args:
        engine: Engine to bind. Defaults to the shared engine.

    Returns:
        A configured sessionmaker.
    """
    return sessionmaker(
        bind=engine if engine is not None else get_engine(),
        # Attributes stay usable after commit; repositories return detached
        # domain objects, so a refresh-on-access would issue pointless queries.
        expire_on_commit=False,
        autoflush=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide sessionmaker.

    Returns:
        The shared sessionmaker, bound to the shared engine.
    """
    return build_session_factory()


@contextmanager
def session_scope(factory: SessionFactory | None = None) -> Iterator[Session]:
    """Provide a transactional session that commits on success, rolls back on error.

    On the failure path the original exception always wins. A rollback that
    itself fails -- a dropped connection is the common cause -- is logged and
    swallowed, because replacing a ``UniqueViolation`` with an
    ``InterfaceError`` from the cleanup would send the reader chasing the wrong
    problem entirely. The same applies to a failing ``close``.

    Args:
        factory: Session factory to use. Defaults to the shared sessionmaker.
            Injectable so tests can exercise the commit and rollback paths
            without a database.

    Yields:
        An open session.

    Raises:
        Exception: Whatever the body raised, unchanged.
    """
    make_session = factory if factory is not None else get_session_factory()
    session = make_session()
    failed = False
    try:
        yield session
        session.commit()
    except BaseException:
        failed = True
        try:
            session.rollback()
        except Exception:
            logger.warning("session_rollback_failed", exc_info=True)
        raise
    finally:
        try:
            session.close()
        except Exception:
            if not failed:
                raise
            logger.warning("session_close_failed", exc_info=True)


def get_session() -> Iterator[Session]:
    """Yield a session for use as a FastAPI dependency (stage 16).

    Usage::

        @router.get("/sightings")
        def list_sightings(session: Session = Depends(get_session)) -> ...:

    Yields:
        An open session, committed or rolled back when the request ends.
    """
    with session_scope() as session:
        yield session


# ---------------------------------------------------------------------------
# Async, for stage 15's concurrent stream ingestion
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first call.

    Returns:
        The shared async engine, using the same psycopg driver and DSN as the
        sync engine.
    """
    settings = get_settings()
    return create_async_engine(settings.database.dsn, **_engine_kwargs(settings))


@lru_cache(maxsize=1)
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async sessionmaker.

    Returns:
        A sessionmaker producing :class:`AsyncSession` instances.
    """
    return async_sessionmaker(bind=get_async_engine(), expire_on_commit=False, autoflush=True)


async def async_session_scope() -> AsyncIterator[AsyncSession]:
    """Async counterpart of :func:`session_scope`.

    Yields:
        An open async session, committed on clean exit and rolled back on error.

    Raises:
        Exception: Whatever the body raised, unchanged.
    """
    session = get_async_session_factory()()
    try:
        yield session
        await session.commit()
    except BaseException:
        try:
            await session.rollback()
        except Exception:
            logger.warning("async_session_rollback_failed", exc_info=True)
        raise
    finally:
        await session.close()


def dispose_engines() -> None:
    """Dispose the cached engines and clear the caches.

    Closes every pooled connection. Called at shutdown, and by tests that swap
    the configured database between cases.

    Raises:
        StorageError: If disposal fails.
    """
    try:
        if get_engine.cache_info().currsize:
            get_engine().dispose()
    except SQLAlchemyError as exc:
        raise StorageError("Failed to dispose the database engine", {"reason": str(exc)}) from exc
    finally:
        get_engine.cache_clear()
        get_session_factory.cache_clear()
        get_async_engine.cache_clear()
        get_async_session_factory.cache_clear()
