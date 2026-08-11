"""Pydantic domain models for the canonical data contracts.

These models are the shared vocabulary of the whole system. Every later stage
consumes them, and none of them redefines a contract field under a different
name or type.

Import from this package rather than the individual modules::

    from multicam_tracker.models import Camera, Sighting, Trajectory
"""

from __future__ import annotations

from multicam_tracker.models.base import (
    EMBEDDING_NORM_TOLERANCE,
    MCTBaseModel,
    UtcDatetime,
    default_embedding_dim,
    validate_embedding,
)
from multicam_tracker.models.camera import CAMERA_ID_PATTERN, Camera, CameraLink
from multicam_tracker.models.enums import MatchMethod, ObjectClass, ReviewStatus, SourceType
from multicam_tracker.models.geo import EARTH_RADIUS_METERS, GeoPoint
from multicam_tracker.models.match import MatchCandidate
from multicam_tracker.models.sighting import (
    BBOX_LENGTH,
    TIMESTAMP_CONSISTENCY_TOLERANCE_MS,
    Sighting,
)
from multicam_tracker.models.target import Target
from multicam_tracker.models.timewindow import TimeWindow
from multicam_tracker.models.trajectory import CoverageGap, Trajectory, TrajectoryHop

__all__ = [
    "BBOX_LENGTH",
    "CAMERA_ID_PATTERN",
    "EARTH_RADIUS_METERS",
    "EMBEDDING_NORM_TOLERANCE",
    "TIMESTAMP_CONSISTENCY_TOLERANCE_MS",
    "Camera",
    "CameraLink",
    "CoverageGap",
    "GeoPoint",
    "MCTBaseModel",
    "MatchCandidate",
    "MatchMethod",
    "ObjectClass",
    "ReviewStatus",
    "Sighting",
    "SourceType",
    "Target",
    "TimeWindow",
    "Trajectory",
    "TrajectoryHop",
    "UtcDatetime",
    "default_embedding_dim",
    "validate_embedding",
]
