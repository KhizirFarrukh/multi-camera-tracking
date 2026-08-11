"""Initial schema: extension, tables, constraints, and indexes.

Revision ID: 0001
Revises:
Create Date: 2026-08-11
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from multicam_tracker.db.folding import FOLD_REPLACEMENT, FOLD_SOURCE
from multicam_tracker.db.orm import (
    CAMERA_ID_LENGTH,
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ID_TEXT_LENGTH,
    PLATE_LENGTH,
)

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the extension, tables, constraints, and indexes."""
    # pgvector must exist before any table declares a vector column.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -- cameras ------------------------------------------------------------
    op.create_table(
        "cameras",
        sa.Column("camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("name", sa.String(ID_TEXT_LENGTH), nullable=False),
        sa.Column("lat", sa.Double(), nullable=False),
        sa.Column("lon", sa.Double(), nullable=False),
        sa.Column("heading_degrees", sa.Double(), nullable=True),
        sa.Column("clock_offset_ms", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("camera_id", name="pk_cameras"),
        sa.CheckConstraint("lat >= -90 AND lat <= 90", name="ck_cameras_lat_range"),
        sa.CheckConstraint("lon >= -180 AND lon <= 180", name="ck_cameras_lon_range"),
        sa.CheckConstraint(
            "heading_degrees IS NULL OR (heading_degrees >= 0 AND heading_degrees < 360)",
            name="ck_cameras_heading_range",
        ),
    )

    # -- camera_links -------------------------------------------------------
    op.create_table(
        "camera_links",
        sa.Column("from_camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("to_camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("min_travel_time_sec", sa.Double(), nullable=False),
        sa.Column("max_travel_time_sec", sa.Double(), nullable=False),
        sa.Column("distance_meters", sa.Double(), nullable=True),
        sa.Column("bidirectional", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.PrimaryKeyConstraint("from_camera_id", "to_camera_id", name="pk_camera_links"),
        sa.ForeignKeyConstraint(
            ["from_camera_id"],
            ["cameras.camera_id"],
            name="fk_camera_links_from_camera",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["to_camera_id"],
            ["cameras.camera_id"],
            name="fk_camera_links_to_camera",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("from_camera_id", "to_camera_id", name="uq_camera_links_pair"),
        sa.CheckConstraint("min_travel_time_sec >= 0", name="ck_camera_links_min_non_negative"),
        sa.CheckConstraint(
            "max_travel_time_sec > min_travel_time_sec", name="ck_camera_links_window_ordered"
        ),
        sa.CheckConstraint("from_camera_id <> to_camera_id", name="ck_camera_links_not_self"),
        sa.CheckConstraint(
            "distance_meters IS NULL OR distance_meters >= 0", name="ck_camera_links_distance"
        ),
    )

    # -- sightings ----------------------------------------------------------
    op.create_table(
        "sightings",
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "clock_offset_applied_ms", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("object_class", sa.String(32), nullable=False),
        sa.Column("detection_confidence", sa.Double(), nullable=False),
        sa.Column("bbox", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("plate_text_raw", sa.String(PLATE_LENGTH), nullable=True),
        sa.Column("plate_text_normalized", sa.String(PLATE_LENGTH), nullable=True),
        sa.Column("plate_confidence", sa.Double(), nullable=True),
        sa.Column(
            "plate_folded",
            sa.String(PLATE_LENGTH),
            sa.Computed(
                f"translate(plate_text_normalized, '{FOLD_SOURCE}', '{FOLD_REPLACEMENT}')",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model_version", sa.String(128), nullable=True),
        sa.Column("thumbnail_path", sa.Text(), nullable=True),
        sa.Column("source_id", sa.String(ID_TEXT_LENGTH), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("sighting_id", name="pk_sightings"),
        # RESTRICT: deleting a camera must never orphan the evidence it recorded.
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.camera_id"],
            name="fk_sightings_camera",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "detection_confidence >= 0 AND detection_confidence <= 1",
            name="ck_sightings_detection_confidence",
        ),
        sa.CheckConstraint(
            "plate_confidence IS NULL OR (plate_confidence >= 0 AND plate_confidence <= 1)",
            name="ck_sightings_plate_confidence",
        ),
        sa.CheckConstraint("array_length(bbox, 1) = 4", name="ck_sightings_bbox_length"),
        sa.CheckConstraint(
            "bbox[3] > bbox[1] AND bbox[4] > bbox[2]", name="ck_sightings_bbox_area"
        ),
        sa.CheckConstraint("frame_index >= 0", name="ck_sightings_frame_index"),
        sa.CheckConstraint(
            "plate_text_normalized IS NULL OR plate_confidence IS NOT NULL",
            name="ck_sightings_plate_needs_confidence",
        ),
    )

    # Query pattern: exact plate lookup. Partial, because most sightings carry
    # no readable plate.
    op.create_index(
        "ix_sightings_plate_exact",
        "sightings",
        ["plate_text_normalized"],
        postgresql_where=sa.text("plate_text_normalized IS NOT NULL"),
    )
    # Query pattern: fuzzy candidate prefilter on the folded form.
    op.create_index(
        "ix_sightings_plate_folded",
        "sightings",
        ["plate_folded"],
        postgresql_where=sa.text("plate_folded IS NOT NULL"),
    )
    # Query pattern: per-camera time-windowed scan.
    op.create_index("ix_sightings_camera_time", "sightings", ["camera_id", "timestamp_utc"])
    # Query pattern: cross-camera windowed scan for trajectory assembly.
    op.create_index("ix_sightings_timestamp", "sightings", ["timestamp_utc"])
    # Query pattern: retention purge, which cuts on insertion time.
    op.create_index("ix_sightings_created_at", "sightings", ["created_at"])
    # Query pattern: K-nearest re-id search over L2-normalized vectors.
    op.create_index(
        "ix_sightings_embedding_hnsw",
        "sightings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # -- targets ------------------------------------------------------------
    op.create_table(
        "targets",
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("label", sa.String(ID_TEXT_LENGTH), nullable=False),
        sa.Column("plate_query", sa.String(PLATE_LENGTH), nullable=True),
        sa.Column("reference_embeddings", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.PrimaryKeyConstraint("target_id", name="pk_targets"),
        sa.CheckConstraint(
            "plate_query IS NOT NULL OR reference_embeddings IS NOT NULL",
            name="ck_targets_searchable",
        ),
    )
    op.create_index("ix_targets_active", "targets", ["active"], postgresql_where=sa.text("active"))

    # -- match_candidates ---------------------------------------------------
    op.create_table(
        "match_candidates",
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("match_method", sa.String(32), nullable=False),
        sa.Column("match_score", sa.Double(), nullable=False),
        sa.Column("plate_edit_distance", sa.Integer(), nullable=True),
        sa.Column("embedding_similarity", sa.Double(), nullable=True),
        sa.Column("review_status", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("target_id", "sighting_id", name="pk_match_candidates"),
        sa.ForeignKeyConstraint(
            ["target_id"], ["targets.target_id"], name="fk_match_target", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["sighting_id"],
            ["sightings.sighting_id"],
            name="fk_match_sighting",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("target_id", "sighting_id", name="uq_match_candidates_pair"),
        sa.CheckConstraint("match_score >= 0 AND match_score <= 1", name="ck_match_score_range"),
        sa.CheckConstraint(
            "plate_edit_distance IS NULL OR plate_edit_distance >= 0",
            name="ck_match_edit_distance",
        ),
        sa.CheckConstraint(
            "embedding_similarity IS NULL OR "
            "(embedding_similarity >= -1 AND embedding_similarity <= 1)",
            name="ck_match_similarity_range",
        ),
    )
    # Query pattern: the review queue for one target in one status.
    op.create_index(
        "ix_match_candidates_target_status", "match_candidates", ["target_id", "review_status"]
    )

    # -- trajectories -------------------------------------------------------
    op.create_table(
        "trajectories",
        sa.Column("trajectory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sighting_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column("overall_confidence", sa.Double(), nullable=False),
        sa.Column("start_time_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "gaps", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.PrimaryKeyConstraint("trajectory_id", name="pk_trajectories"),
        sa.ForeignKeyConstraint(
            ["target_id"], ["targets.target_id"], name="fk_trajectories_target", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "overall_confidence >= 0 AND overall_confidence <= 1",
            name="ck_trajectories_confidence",
        ),
        sa.CheckConstraint("end_time_utc >= start_time_utc", name="ck_trajectories_time_ordered"),
        sa.CheckConstraint("array_length(sighting_ids, 1) >= 1", name="ck_trajectories_non_empty"),
    )
    op.create_index("ix_trajectories_target", "trajectories", ["target_id"])

    # -- trajectory_hops ----------------------------------------------------
    op.create_table(
        "trajectory_hops",
        sa.Column("trajectory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("from_sighting_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("to_sighting_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("from_camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("to_camera_id", sa.String(CAMERA_ID_LENGTH), nullable=False),
        sa.Column("elapsed_sec", sa.Double(), nullable=False),
        sa.Column("topology_plausible", sa.Boolean(), nullable=False),
        sa.Column("hop_confidence", sa.Double(), nullable=False),
        sa.PrimaryKeyConstraint("trajectory_id", "position", name="pk_trajectory_hops"),
        sa.ForeignKeyConstraint(
            ["trajectory_id"],
            ["trajectories.trajectory_id"],
            name="fk_hops_trajectory",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("position >= 0", name="ck_hops_position"),
        sa.CheckConstraint("elapsed_sec >= 0", name="ck_hops_elapsed_non_negative"),
        sa.CheckConstraint(
            "hop_confidence >= 0 AND hop_confidence <= 1", name="ck_hops_confidence_range"
        ),
        sa.CheckConstraint("from_sighting_id <> to_sighting_id", name="ck_hops_endpoints_differ"),
    )

    # -- audit_log ----------------------------------------------------------
    # No foreign keys: the audit trail must outlive the rows it describes,
    # including after a retention purge.
    op.create_table(
        "audit_log",
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(ID_TEXT_LENGTH), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "query_parameters",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("audit_id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_occurred_at", "audit_log", ["occurred_at"])
    op.create_index("ix_audit_log_actor", "audit_log", ["actor"])
    op.create_index("ix_audit_log_target", "audit_log", ["target_id"])


def downgrade() -> None:
    """Drop everything this migration created, in dependency order.

    The ``vector`` extension is dropped last and only if nothing else uses it.
    Indexes are not dropped explicitly: ``DROP TABLE`` takes its own indexes
    with it, and listing them separately would only create a second place to
    forget one.
    """
    op.drop_table("audit_log")
    op.drop_table("trajectory_hops")
    op.drop_table("trajectories")
    op.drop_table("match_candidates")
    op.drop_table("targets")
    op.drop_table("sightings")
    op.drop_table("camera_links")
    op.drop_table("cameras")
    op.execute("DROP EXTENSION IF EXISTS vector")
