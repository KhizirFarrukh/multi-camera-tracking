"""Shared pytest fixtures for every stage.

Fixtures live here (and in ``tests/fixtures/``) rather than being redefined per
test module, per the global contract's ``testing_standards.fixtures``.

The autouse fixtures below exist to make the suite hermetic. Configuration reads
from the process environment, the current working directory's ``.env``, and a
YAML file on disk -- three channels through which a developer's machine could
silently change a test result. Each is neutralised before every test.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog

from multicam_tracker.clock import FixedClock
from multicam_tracker.config import Settings, reset_settings_cache

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root, independent of the working directory."""

REPO_THRESHOLDS_FILE = REPO_ROOT / "config" / "thresholds.yaml"
"""The real, committed thresholds file. Tests assert against its actual values."""

FROZEN_INSTANT = datetime(2026, 8, 10, 14, 22, 11, 500000, tzinfo=UTC)
"""A fixed reference instant. Matches the example sighting in the project plan."""


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Neutralise every host-specific configuration channel.

    Clears all ``MCT_`` environment variables, moves the working directory to a
    scratch path so a developer's ``.env`` is not discovered, and clears the
    settings cache on both sides of the test.

    Yields:
        ``None``; the isolation is the effect.
    """
    for name in [key for key in os.environ if key.startswith("MCT_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Restore root logger handlers and structlog context around each test.

    ``configure_logging`` replaces the root handlers by design. Without this,
    the first test to configure logging would leak its handler into every
    subsequent test's output.

    Yields:
        ``None``; the restoration is the effect.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


@pytest.fixture
def thresholds_file() -> Path:
    """Return the path to the committed ``config/thresholds.yaml``.

    Returns:
        Absolute path to the real thresholds file.
    """
    return REPO_THRESHOLDS_FILE


@pytest.fixture
def custom_thresholds_file(tmp_path: Path) -> Path:
    """Write a thresholds YAML with non-default values and return its path.

    Values differ from the committed defaults so a test can prove the file was
    actually read rather than coincidentally matching.

    Returns:
        Path to the temporary YAML file.
    """
    path = tmp_path / "custom_thresholds.yaml"
    path.write_text(
        "\n".join(
            [
                "plate_auto_accept_min_confidence: 0.70",
                "plate_fuzzy_max_edit_distance: 1",
                "plate_fuzzy_max_weighted_distance: 0.8",
                "plate_max_length_delta: 1",
                "plate_confusion_substitution_cost: 0.4",
                "plate_exact_method_weight: 0.95",
                "plate_fuzzy_method_weight: 0.8",
                "plate_distance_penalty_per_unit: 0.2",
                "plate_review_min_confidence: 0.45",
                "embedding_auto_accept_min_similarity: 0.88",
                "embedding_review_min_similarity: 0.60",
                "embedding_margin_min: 0.06",
                "embedding_only_score_ceiling: 0.30",
                "embedding_agreement_boost: 0.10",
                "embedding_disagreement_similarity: 0.35",
                "embedding_max_references: 4",
                "hop_implausible_penalty: 0.25",
                "path_node_inclusion_bonus: 0.20",
                "path_gap_edge_penalty: 0.40",
                "path_ambiguity_margin_min: 0.10",
                "path_weakest_link_tolerance: 0.10",
                "detection_min_bbox_area_px: 200",
                "detection_max_bbox_area_fraction: 0.4",
                "detection_min_aspect_ratio: 0.3",
                "detection_max_aspect_ratio: 4.0",
                "detection_roi_min_overlap: 0.6",
                "track_association_min_iou: 0.4",
                "track_max_age_frames: 3",
                "track_min_hits_to_confirm: 2",
                "best_frame_confidence_weight: 0.4",
                "best_frame_sharpness_weight: 0.3",
                "best_frame_area_weight: 0.2",
                "best_frame_centrality_weight: 0.1",
                "best_frame_edge_penalty: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def build_settings(thresholds_file: Path) -> Callable[..., Settings]:
    """Return a factory that builds :class:`Settings` in a hermetic way.

    The factory pins the thresholds file to the committed one and disables
    ``.env`` discovery, so a test only sees the overrides it passes explicitly.

    Returns:
        A callable accepting keyword overrides and returning a ``Settings``.
    """

    def _build(**overrides: Any) -> Settings:
        overrides.setdefault("thresholds_file", thresholds_file)
        return Settings(_env_file=None, **overrides)

    return _build


@pytest.fixture
def fixed_clock() -> FixedClock:
    """Return a clock pinned to :data:`FROZEN_INSTANT`.

    Returns:
        A :class:`~multicam_tracker.clock.FixedClock`.
    """
    return FixedClock(FROZEN_INSTANT)


@pytest.fixture
def log_stream() -> io.StringIO:
    """Return an in-memory stream for capturing configured log output.

    Returns:
        An empty :class:`io.StringIO` to pass to ``configure_logging``.
    """
    return io.StringIO()


# ---------------------------------------------------------------------------
# Database fixtures, shared by tests/conformance/ and tests/integration/db/
# ---------------------------------------------------------------------------

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
"""Must match docker-compose.yml, or CI and local development diverge silently."""


def _postgres_container_class() -> type | None:
    """Return testcontainers' PostgresContainer, or ``None`` if unavailable.

    testcontainers moved the module under ``.community``; the old path emits a
    DeprecationWarning, which this suite escalates to an error. Prefer the new
    location and fall back quietly for older versions.

    Returns:
        The container class, or ``None`` when testcontainers is not installed.
    """
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - depends on the installed version
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from testcontainers.postgres import PostgresContainer
        except ImportError:
            return None
    return PostgresContainer


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Any]:
    """Start Postgres, run every migration, and yield an engine bound to it.

    Session-scoped: starting a container and building an HNSW index costs
    seconds, and every database test can share one schema because each isolates
    itself in a rolled-back transaction.

    Migrations are run rather than ``metadata.create_all`` so the tests exercise
    the schema that will actually be deployed -- generated column, partial
    indexes, and HNSW index included.

    Yields:
        A SQLAlchemy engine connected to the migrated database.

    Raises:
        pytest.skip.Exception: If testcontainers or Docker is unavailable.
    """
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine

    container_class = _postgres_container_class()
    if container_class is None:  # pragma: no cover - dev extra missing
        pytest.skip("testcontainers is not installed")

    # Construction, not just start(), contacts the Docker daemon.
    try:
        container = container_class(
            image=POSTGRES_IMAGE,
            username="multicam_test",
            password="multicam_test",
            dbname="multicam_test",
            driver="psycopg",
        )
        container.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Docker is not available for integration tests: {exc}")

    url = container.get_connection_url()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)

    engine = create_engine(url)
    try:
        command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        container.stop()


@pytest.fixture
def postgres_repositories(migrated_engine: Any) -> Iterator[Any]:
    """Yield Postgres repositories inside a transaction that is rolled back.

    Rolling back rather than truncating keeps each test isolated without paying
    for a schema rebuild, and guarantees no test leaks state into the next even
    if it fails partway through. ``join_transaction_mode="create_savepoint"``
    means a repository's own flush cannot end the outer transaction early.

    Yields:
        A ``RepositorySet`` of Postgres-backed repositories sharing one session.
    """
    from sqlalchemy.orm import Session

    from multicam_tracker.db.repositories import (
        PostgresCameraLinkRepository,
        PostgresCameraRepository,
        PostgresMatchRepository,
        PostgresSightingRepository,
        PostgresTargetRepository,
        PostgresTrajectoryRepository,
    )
    from tests.fixtures.fake_repositories import RepositorySet

    connection = migrated_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )

    try:
        yield RepositorySet(
            cameras=PostgresCameraRepository(session),
            links=PostgresCameraLinkRepository(session),
            sightings=PostgresSightingRepository(session),
            targets=PostgresTargetRepository(session),
            matches=PostgresMatchRepository(session),
            trajectories=PostgresTrajectoryRepository(session),
        )
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="session")
def alembic_config(migrated_engine: Any) -> Any:
    """Return an Alembic config pointed at the running test database.

    Args:
        migrated_engine: The migrated engine, which owns the container.

    Returns:
        A configured :class:`alembic.config.Config`.
    """
    from alembic.config import Config

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", str(migrated_engine.url.render_as_string(False)))
    return config
