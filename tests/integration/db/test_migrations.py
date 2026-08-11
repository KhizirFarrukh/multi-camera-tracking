"""Integration tests for the Alembic migrations.

A migration that cannot be reversed is a migration you cannot safely deploy, so
the reversibility check compares the schema before and after a full
downgrade-then-upgrade cycle by introspection rather than by trusting that the
DDL "looks symmetric".
"""

from __future__ import annotations

from typing import Any

import pytest
from alembic import command
from sqlalchemy import Engine, inspect, text

pytestmark = [pytest.mark.integration, pytest.mark.slow]

EXPECTED_TABLES = {
    "cameras",
    "camera_links",
    "sightings",
    "targets",
    "match_candidates",
    "trajectories",
    "trajectory_hops",
    "audit_log",
}

EXPECTED_SIGHTING_INDEXES = {
    "ix_sightings_plate_exact",
    "ix_sightings_plate_folded",
    "ix_sightings_camera_time",
    "ix_sightings_timestamp",
    "ix_sightings_created_at",
    "ix_sightings_embedding_hnsw",
}

EXPECTED_CHECK_CONSTRAINTS = {
    "ck_cameras_lat_range",
    "ck_cameras_lon_range",
    "ck_cameras_heading_range",
    "ck_camera_links_window_ordered",
    "ck_camera_links_not_self",
    "ck_sightings_detection_confidence",
    "ck_sightings_plate_confidence",
    "ck_sightings_bbox_length",
    "ck_sightings_bbox_area",
    "ck_sightings_plate_needs_confidence",
    "ck_match_score_range",
    "ck_trajectories_confidence",
    "ck_hops_confidence_range",
}

EXPECTED_FOREIGN_KEYS = {
    "fk_camera_links_from_camera",
    "fk_camera_links_to_camera",
    "fk_sightings_camera",
    "fk_match_target",
    "fk_match_sighting",
    "fk_trajectories_target",
    "fk_hops_trajectory",
}


def _schema_snapshot(engine: Engine) -> dict[str, Any]:
    """Capture the schema by introspection, for before/after comparison.

    Args:
        engine: Engine connected to the database.

    Returns:
        Tables, their column names and types, indexes, and constraints.
    """
    inspector = inspect(engine)
    snapshot: dict[str, Any] = {}
    for table in sorted(inspector.get_table_names()):
        if table == "alembic_version":
            continue
        snapshot[table] = {
            "columns": sorted(
                (column["name"], str(column["type"]), bool(column["nullable"]))
                for column in inspector.get_columns(table)
            ),
            "indexes": sorted(index["name"] or "" for index in inspector.get_indexes(table)),
            "primary_key": sorted(
                inspector.get_pk_constraint(table).get("constrained_columns", [])
            ),
            "foreign_keys": sorted(fk["name"] or "" for fk in inspector.get_foreign_keys(table)),
            "checks": sorted(
                check["name"] or "" for check in inspector.get_check_constraints(table)
            ),
            "unique": sorted(uq["name"] or "" for uq in inspector.get_unique_constraints(table)),
        }
    return snapshot


def test_migrations__upgrade_head__creates_every_table(migrated_engine: Engine) -> None:
    """The session fixture already ran ``upgrade head``; verify what it produced."""
    tables = set(inspect(migrated_engine).get_table_names())

    assert tables >= EXPECTED_TABLES


def test_migrations__vector_extension__is_installed(migrated_engine: Engine) -> None:
    """pgvector must exist before any vector column can."""
    with migrated_engine.connect() as connection:
        found = connection.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()

    assert found == "vector"


def test_migrations__all_declared_indexes_exist(migrated_engine: Engine) -> None:
    """Every index the schema promises is present, HNSW included."""
    with migrated_engine.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'sightings'")
            )
        }

    assert names >= EXPECTED_SIGHTING_INDEXES


def test_migrations__hnsw_index__uses_cosine_ops(migrated_engine: Engine) -> None:
    """A euclidean-ops index would silently rank results by the wrong metric."""
    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_sightings_embedding_hnsw'")
        ).scalar_one()

    assert "hnsw" in definition
    assert "vector_cosine_ops" in definition


def test_migrations__plate_folded__is_a_stored_generated_column(
    migrated_engine: Engine,
) -> None:
    """The folded form must be maintained by Postgres, not by the application."""
    with migrated_engine.connect() as connection:
        generated = connection.execute(
            text(
                "SELECT is_generated, generation_expression FROM information_schema.columns "
                "WHERE table_name = 'sightings' AND column_name = 'plate_folded'"
            )
        ).one()

    assert generated[0] == "ALWAYS"
    assert "translate" in generated[1]


def test_migrations__all_declared_check_constraints_exist(migrated_engine: Engine) -> None:
    """CHECK constraints guard the paths the Pydantic models cannot reach."""
    with migrated_engine.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                text("SELECT conname FROM pg_constraint WHERE contype = 'c'")
            )
        }

    assert names >= EXPECTED_CHECK_CONSTRAINTS


def test_migrations__all_declared_foreign_keys_exist(migrated_engine: Engine) -> None:
    """Referential integrity is declared, not merely intended."""
    with migrated_engine.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                text("SELECT conname FROM pg_constraint WHERE contype = 'f'")
            )
        }

    assert names >= EXPECTED_FOREIGN_KEYS


def test_migrations__sightings_camera_fk__restricts_deletion(migrated_engine: Engine) -> None:
    """The delete rule is the difference between retiring a camera and losing evidence."""
    with migrated_engine.connect() as connection:
        rule = connection.execute(
            text("SELECT confdeltype FROM pg_constraint WHERE conname = 'fk_sightings_camera'")
        ).scalar_one()

    # 'r' is RESTRICT; 'c' would be CASCADE.
    assert rule == "r"


def test_migrations__downgrade_then_upgrade__reproduces_an_identical_schema(
    migrated_engine: Engine, alembic_config: Any
) -> None:
    """Reversibility, verified by introspection rather than by reading the DDL.

    Runs last-ish in the module but is order-independent: it restores the schema
    before returning, so the session-scoped database is left as it was found.
    """
    before = _schema_snapshot(migrated_engine)

    command.downgrade(alembic_config, "base")
    remaining = set(inspect(migrated_engine).get_table_names()) - {"alembic_version"}
    assert remaining == set(), "downgrade left tables behind"

    command.upgrade(alembic_config, "head")
    after = _schema_snapshot(migrated_engine)

    assert after == before
