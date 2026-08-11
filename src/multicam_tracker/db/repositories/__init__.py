"""Repository interfaces and their Postgres implementations.

Import the Protocol when declaring a dependency and the concrete class only at
the composition root. Everything above this layer -- matching, pathing, the API
-- depends on the interface, which is what lets stage 04 onward run its unit
tests against the in-memory fakes with no database at all.
"""

from __future__ import annotations

from multicam_tracker.db.repositories.camera import (
    PostgresCameraLinkRepository,
    PostgresCameraRepository,
)
from multicam_tracker.db.repositories.match import PostgresMatchRepository
from multicam_tracker.db.repositories.protocols import (
    CameraHourCount,
    CameraLinkRepository,
    CameraRepository,
    EmbeddingMatch,
    MatchRepository,
    PurgeResult,
    SightingRepository,
    TargetRepository,
    TrajectoryRepository,
)
from multicam_tracker.db.repositories.sighting import PostgresSightingRepository
from multicam_tracker.db.repositories.target import PostgresTargetRepository
from multicam_tracker.db.repositories.trajectory import PostgresTrajectoryRepository

__all__ = [
    "CameraHourCount",
    "CameraLinkRepository",
    "CameraRepository",
    "EmbeddingMatch",
    "MatchRepository",
    "PostgresCameraLinkRepository",
    "PostgresCameraRepository",
    "PostgresMatchRepository",
    "PostgresSightingRepository",
    "PostgresTargetRepository",
    "PostgresTrajectoryRepository",
    "PurgeResult",
    "SightingRepository",
    "TargetRepository",
    "TrajectoryRepository",
]
