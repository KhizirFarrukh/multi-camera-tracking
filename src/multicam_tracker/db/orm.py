"""SQLAlchemy ORM models.

Kept deliberately separate from the Pydantic domain models in
:mod:`multicam_tracker.models`, with explicit conversion in
:mod:`multicam_tracker.db.mappers`. Merging the two would tie the storage schema
to the wire format: a column rename would become an API change, and a domain
invariant would have to be expressible as a database constraint to exist at all.

Every CHECK constraint here mirrors a Pydantic validator. The duplication is
intentional -- the model guards the application path, the constraint guards
everything else (a migration, a manual ``psql`` session, a future service in
another language). Neither is redundant with the other.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from multicam_tracker.config import get_settings
from multicam_tracker.db.folding import FOLD_REPLACEMENT, FOLD_SOURCE

__all__ = [
    "EMBEDDING_DIM",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_M",
    "AuditLogORM",
    "Base",
    "CameraLinkORM",
    "CameraORM",
    "MatchCandidateORM",
    "SightingORM",
    "TargetORM",
    "TrajectoryHopORM",
    "TrajectoryORM",
]

EMBEDDING_DIM: int = get_settings().vision.embedding_dim
"""Vector column width, read from settings so it tracks the model stage 13 picks.

Bound at import time because a column type must be fixed when the class is
defined. Changing the dimension therefore requires a migration, which is correct:
existing vectors would be meaningless at a different width.
"""

HNSW_M = 16
"""HNSW graph connectivity. 16 is pgvector's default and the usual recommendation
for datasets up to a few million vectors; higher values improve recall at the
cost of index size and build time."""

HNSW_EF_CONSTRUCTION = 64
"""HNSW build-time search width. pgvector's default. Raising it improves recall
for the same query-time ``ef_search`` but makes index construction slower."""

CAMERA_ID_LENGTH = 64
PLATE_LENGTH = 32
ID_TEXT_LENGTH = 255


class Base(DeclarativeBase):
    """Declarative base for every table in the schema."""


class CameraORM(Base):
    """A fixed camera. Referenced by sightings and links."""

    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(CAMERA_ID_LENGTH), primary_key=True)
    name: Mapped[str] = mapped_column(String(ID_TEXT_LENGTH), nullable=False)
    lat: Mapped[float] = mapped_column(Double, nullable=False)
    lon: Mapped[float] = mapped_column(Double, nullable=False)
    heading_degrees: Mapped[float | None] = mapped_column(Double, nullable=True)
    clock_offset_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("lat >= -90 AND lat <= 90", name="ck_cameras_lat_range"),
        CheckConstraint("lon >= -180 AND lon <= 180", name="ck_cameras_lon_range"),
        CheckConstraint(
            "heading_degrees IS NULL OR (heading_degrees >= 0 AND heading_degrees < 360)",
            name="ck_cameras_heading_range",
        ),
    )


class CameraLinkORM(Base):
    """A directed transition between two cameras with a plausible travel window."""

    __tablename__ = "camera_links"

    from_camera_id: Mapped[str] = mapped_column(
        String(CAMERA_ID_LENGTH),
        ForeignKey("cameras.camera_id", ondelete="CASCADE"),
        primary_key=True,
    )
    to_camera_id: Mapped[str] = mapped_column(
        String(CAMERA_ID_LENGTH),
        ForeignKey("cameras.camera_id", ondelete="CASCADE"),
        primary_key=True,
    )
    min_travel_time_sec: Mapped[float] = mapped_column(Double, nullable=False)
    max_travel_time_sec: Mapped[float] = mapped_column(Double, nullable=False)
    distance_meters: Mapped[float | None] = mapped_column(Double, nullable=True)
    bidirectional: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )

    __table_args__ = (
        # The composite primary key already enforces uniqueness; the named unique
        # constraint is declared explicitly because the stage spec requires one
        # that error handling can identify by name.
        UniqueConstraint("from_camera_id", "to_camera_id", name="uq_camera_links_pair"),
        CheckConstraint("min_travel_time_sec >= 0", name="ck_camera_links_min_non_negative"),
        CheckConstraint(
            "max_travel_time_sec > min_travel_time_sec", name="ck_camera_links_window_ordered"
        ),
        CheckConstraint("from_camera_id <> to_camera_id", name="ck_camera_links_not_self"),
        CheckConstraint(
            "distance_meters IS NULL OR distance_meters >= 0", name="ck_camera_links_distance"
        ),
    )


class SightingORM(Base):
    """One detection of one object by one camera at one instant."""

    __tablename__ = "sightings"

    sighting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    camera_id: Mapped[str] = mapped_column(
        String(CAMERA_ID_LENGTH),
        # RESTRICT, never CASCADE: deleting a camera must not silently erase the
        # evidence it recorded. Retire the camera with enabled=false instead.
        ForeignKey("cameras.camera_id", ondelete="RESTRICT"),
        nullable=False,
    )

    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    clock_offset_applied_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    object_class: Mapped[str] = mapped_column(String(32), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Double, nullable=False)
    bbox: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)

    plate_text_raw: Mapped[str | None] = mapped_column(String(PLATE_LENGTH), nullable=True)
    plate_text_normalized: Mapped[str | None] = mapped_column(String(PLATE_LENGTH), nullable=True)
    plate_confidence: Mapped[float | None] = mapped_column(Double, nullable=True)

    # Maintained by Postgres, never written by the application, so the folded
    # form cannot drift from the normalized value it is derived from. The
    # translate() arguments come from multicam_tracker.db.folding.
    plate_folded: Mapped[str | None] = mapped_column(
        String(PLATE_LENGTH),
        Computed(
            f"translate(plate_text_normalized, '{FOLD_SOURCE}', '{FOLD_REPLACEMENT}')",
            persisted=True,
        ),
        nullable=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_id: Mapped[str] = mapped_column(String(ID_TEXT_LENGTH), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "detection_confidence >= 0 AND detection_confidence <= 1",
            name="ck_sightings_detection_confidence",
        ),
        CheckConstraint(
            "plate_confidence IS NULL OR (plate_confidence >= 0 AND plate_confidence <= 1)",
            name="ck_sightings_plate_confidence",
        ),
        CheckConstraint("array_length(bbox, 1) = 4", name="ck_sightings_bbox_length"),
        CheckConstraint("bbox[3] > bbox[1] AND bbox[4] > bbox[2]", name="ck_sightings_bbox_area"),
        CheckConstraint("frame_index >= 0", name="ck_sightings_frame_index"),
        CheckConstraint(
            "plate_text_normalized IS NULL OR plate_confidence IS NOT NULL",
            name="ck_sightings_plate_needs_confidence",
        ),
        # Pattern: exact plate lookup. Partial because most sightings have no
        # readable plate, which keeps the index a fraction of the table's size.
        Index(
            "ix_sightings_plate_exact",
            "plate_text_normalized",
            postgresql_where=text("plate_text_normalized IS NOT NULL"),
        ),
        # Pattern: fuzzy candidate prefilter. Same partial rationale.
        Index(
            "ix_sightings_plate_folded",
            "plate_folded",
            postgresql_where=text("plate_folded IS NOT NULL"),
        ),
        # Pattern: per-camera time-windowed scan. Column order matters --
        # camera_id first makes the index usable for both the two-column lookup
        # and a camera-only filter.
        Index("ix_sightings_camera_time", "camera_id", "timestamp_utc"),
        # Pattern: cross-camera windowed scan during trajectory assembly.
        Index("ix_sightings_timestamp", "timestamp_utc"),
        # Pattern: retention purge, which scans by insertion time rather than by
        # observation time so a late-arriving old sighting still ages out.
        Index("ix_sightings_created_at", "created_at"),
        # Pattern: K-nearest re-id search. Cosine ops because embeddings are
        # L2-normalized and matching is defined in terms of cosine similarity.
        Index(
            "ix_sightings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class TargetORM(Base):
    """An object under search."""

    __tablename__ = "targets"

    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    label: Mapped[str] = mapped_column(String(ID_TEXT_LENGTH), nullable=False)
    plate_query: Mapped[str | None] = mapped_column(String(PLATE_LENGTH), nullable=True)
    # JSONB rather than ARRAY(Vector): the count varies per target and these are
    # never searched by similarity, only loaded with the target.
    reference_embeddings: Mapped[list[list[float]] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        CheckConstraint(
            "plate_query IS NOT NULL OR reference_embeddings IS NOT NULL",
            name="ck_targets_searchable",
        ),
        Index("ix_targets_active", "active", postgresql_where=text("active")),
    )


class MatchCandidateORM(Base):
    """A scored link between one sighting and one target."""

    __tablename__ = "match_candidates"

    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("targets.target_id", ondelete="CASCADE"),
        primary_key=True,
    )
    sighting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sightings.sighting_id", ondelete="CASCADE"),
        primary_key=True,
    )
    match_method: Mapped[str] = mapped_column(String(32), nullable=False)
    match_score: Mapped[float] = mapped_column(Double, nullable=False)
    plate_edit_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_similarity: Mapped[float | None] = mapped_column(Double, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        # A sighting may be matched to a target exactly once. Without this, a
        # re-run of the matcher would stack duplicate candidates and inflate
        # every count taken over the table.
        UniqueConstraint("target_id", "sighting_id", name="uq_match_candidates_pair"),
        CheckConstraint("match_score >= 0 AND match_score <= 1", name="ck_match_score_range"),
        CheckConstraint(
            "plate_edit_distance IS NULL OR plate_edit_distance >= 0",
            name="ck_match_edit_distance",
        ),
        CheckConstraint(
            "embedding_similarity IS NULL OR "
            "(embedding_similarity >= -1 AND embedding_similarity <= 1)",
            name="ck_match_similarity_range",
        ),
        # Pattern: the review queue, which lists one target's candidates in a
        # given status.
        Index("ix_match_candidates_target_status", "target_id", "review_status"),
    )


class TrajectoryORM(Base):
    """A reconstructed route: the ordered sightings plus their hops."""

    __tablename__ = "trajectories"

    trajectory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("targets.target_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Ordered, and deliberately NOT a foreign key array: a trajectory is a
    # historical record of what was concluded at a point in time. Retention may
    # purge the sightings underneath it, and that must not rewrite or delete the
    # conclusion. See SightingRepository.delete_older_than.
    sighting_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=False)
    overall_confidence: Mapped[float] = mapped_column(Double, nullable=False)
    start_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    gaps: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # Nullable on purpose: NULL means "never checked", which is a different
    # claim from "checked and clean" and must not be collapsed into it.
    temporal_integrity: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    requires_recomputation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    __table_args__ = (
        CheckConstraint(
            "overall_confidence >= 0 AND overall_confidence <= 1",
            name="ck_trajectories_confidence",
        ),
        CheckConstraint("end_time_utc >= start_time_utc", name="ck_trajectories_time_ordered"),
        CheckConstraint("array_length(sighting_ids, 1) >= 1", name="ck_trajectories_non_empty"),
        Index("ix_trajectories_target", "target_id"),
    )


class TrajectoryHopORM(Base):
    """One transition within a trajectory, ordered by ``position``."""

    __tablename__ = "trajectory_hops"

    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trajectories.trajectory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_sighting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    to_sighting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    from_camera_id: Mapped[str] = mapped_column(String(CAMERA_ID_LENGTH), nullable=False)
    to_camera_id: Mapped[str] = mapped_column(String(CAMERA_ID_LENGTH), nullable=False)
    elapsed_sec: Mapped[float] = mapped_column(Double, nullable=False)
    topology_plausible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hop_confidence: Mapped[float] = mapped_column(Double, nullable=False)

    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_hops_position"),
        CheckConstraint("elapsed_sec >= 0", name="ck_hops_elapsed_non_negative"),
        CheckConstraint(
            "hop_confidence >= 0 AND hop_confidence <= 1", name="ck_hops_confidence_range"
        ),
        CheckConstraint("from_sighting_id <> to_sighting_id", name="ck_hops_endpoints_differ"),
    )


class AuditLogORM(Base):
    """Append-only record of every search, confirmation, and rejection.

    Required by the global contract's operational constraints. The table is
    created here so it exists from the first migration; stage 18 owns writing to
    it and the retention rules that apply to it.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor: Mapped[str] = mapped_column(String(ID_TEXT_LENGTH), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    sighting_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    query_parameters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # No foreign keys on purpose: the audit trail must outlive the rows it
        # describes, including after a retention purge.
        Index("ix_audit_log_occurred_at", "occurred_at"),
        Index("ix_audit_log_actor", "actor"),
        Index("ix_audit_log_target", "target_id"),
    )
