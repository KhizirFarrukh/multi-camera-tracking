"""Integration tests for the local Postgres dependency.

These prove the database half of the environment before stage 03 writes a single
migration against it. The pgvector extension in particular is worth checking
now: it is the one non-standard requirement, and discovering mid-stage-03 that
the image lacks it would be a confusing failure.

Testcontainers is used rather than the compose stack so CI does not depend on a
developer having run ``make db-up`` (global contract, stage 01 tests substage).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text

from multicam_tracker.config import DatabaseSettings

# testcontainers moved the Postgres module under .community and the old path
# emits a DeprecationWarning -- which this suite escalates to an error. Prefer
# the new location and fall back quietly for older versions of the library.
try:
    from testcontainers.community.postgres import PostgresContainer
except ImportError:  # pragma: no cover - depends on the installed version
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from testcontainers.postgres import PostgresContainer
    except ImportError:
        PostgresContainer = None  # type: ignore[assignment, misc]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(PostgresContainer is None, reason="testcontainers is not installed"),
]

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
"""Must match the image in docker-compose.yml, or CI and local diverge silently."""

TEST_USER = "multicam_test"
# Throwaway credential for an ephemeral container that never outlives the test run.
TEST_PASSWORD = "multicam_test_pw"
TEST_DBNAME = "multicam_test"


@pytest.fixture(scope="module")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start a pgvector-enabled Postgres container for the module.

    Yields:
        The running container handle.

    Raises:
        pytest.skip.Exception: If Docker is unavailable on this machine, which
            is expected on a developer box without Docker Desktop running.
    """
    # Construction, not just start(), talks to the Docker daemon -- so both go
    # inside the guard. pytest.skip raises a BaseException subclass, which this
    # `except Exception` deliberately does not swallow.
    try:
        container = PostgresContainer(
            image=POSTGRES_IMAGE,
            username=TEST_USER,
            password=TEST_PASSWORD,
            dbname=TEST_DBNAME,
            driver="psycopg",
        )
        container.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Docker is not available for integration tests: {exc}")

    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def database_settings(postgres_container: PostgresContainer) -> DatabaseSettings:
    """Build settings pointing at the running container.

    Routing through :class:`DatabaseSettings` rather than the container's own
    connection URL is the point of the test: it proves the DSN this project
    builds is a DSN Postgres accepts.

    Returns:
        Settings whose ``dsn`` addresses the container.
    """
    host = postgres_container.get_container_host_ip()
    port = int(postgres_container.get_exposed_port(5432))
    return DatabaseSettings(
        host=host,
        port=port,
        user=TEST_USER,
        password=SecretStr(TEST_PASSWORD),
        dbname=TEST_DBNAME,
    )


def test_postgres_container__starts__reports_a_mapped_port(
    postgres_container: PostgresContainer,
) -> None:
    """The container is up and its port is published to the host."""
    port = int(postgres_container.get_exposed_port(5432))

    assert port > 0


def test_database_settings_dsn__against_running_container__connects(
    database_settings: DatabaseSettings,
) -> None:
    """The project's own DSN construction produces a connectable URL."""
    engine = create_engine(database_settings.dsn)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        engine.dispose()


def test_database__server_version__is_postgres_16(
    database_settings: DatabaseSettings,
) -> None:
    """Pinned major version; a drift here changes migration behaviour."""
    engine = create_engine(database_settings.dsn)
    try:
        with engine.connect() as connection:
            version = connection.execute(text("SHOW server_version")).scalar_one()
    finally:
        engine.dispose()

    assert str(version).startswith("16")


def test_database__vector_extension__can_be_created(
    database_settings: DatabaseSettings,
) -> None:
    """pgvector is present in the image and installable into the database."""
    engine = create_engine(database_settings.dsn)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        with engine.connect() as connection:
            installed = connection.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    finally:
        engine.dispose()

    assert installed == "vector"


def test_database__vector_extension__creation_is_idempotent(
    database_settings: DatabaseSettings,
) -> None:
    """Migrations re-run on every deploy; IF NOT EXISTS must actually hold."""
    engine = create_engine(database_settings.dsn)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        with engine.connect() as connection:
            count = connection.execute(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one()
    finally:
        engine.dispose()

    assert count == 1
