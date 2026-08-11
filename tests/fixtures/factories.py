"""Deterministic builders for domain models.

Every builder returns a *valid* model with sensible defaults, so a test only has
to state the one field it is actually about. That keeps the intent of a test
visible instead of buried under twelve lines of required arguments, and it means
a new required field added in a later stage is fixed in one place.

Nothing here reads the wall clock or an unseeded RNG: instants derive from
:data:`BASE_INSTANT` and embeddings from a seeded generator, so a failure is
always reproducible.
"""

from __future__ import annotations

import itertools
import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from multicam_tracker.models import (
    Camera,
    CameraLink,
    MatchCandidate,
    MatchMethod,
    ObjectClass,
    ReviewStatus,
    Sighting,
    Target,
    Trajectory,
    TrajectoryHop,
)

__all__ = [
    "BASE_INSTANT",
    "DEFAULT_EMBEDDING_DIM",
    "PLACEHOLDER_SIGHTING_ID",
    "PLACEHOLDER_TARGET_ID",
    "make_camera",
    "make_link",
    "make_match_candidate",
    "make_sighting",
    "make_target",
    "make_trajectory",
    "unit_vector",
]

BASE_INSTANT = datetime(2026, 8, 10, 14, 22, 11, 500000, tzinfo=UTC)
"""Reference instant. Matches the example sighting in the project plan."""

DEFAULT_EMBEDDING_DIM = 512
"""Mirrors the ``vision.embedding_dim`` default. Overridable per builder call."""

PLACEHOLDER_TARGET_ID = "a0000000-0000-4000-8000-000000000001"
"""Stand-in target id. A real UUID, not a readable slug: the ``targets`` table
keys on a UUID column, so a slug would fail at the mapper rather than in the
test that meant to exercise it."""

PLACEHOLDER_SIGHTING_ID = "b0000000-0000-4000-8000-000000000001"
"""Stand-in sighting id, for the same reason."""


def unit_vector(seed: int = 0, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Return a deterministic L2-normalized vector.

    Args:
        seed: Seeds the generator, so the same seed always yields the same
            vector and different seeds yield different directions.
        dim: Number of components.

    Returns:
        A list of ``dim`` floats whose L2 norm is 1.0.
    """
    rng = random.Random(seed)
    components = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(math.fsum(value * value for value in components))
    return [value / norm for value in components]


def make_camera(camera_id: str = "cam_01", **overrides: Any) -> Camera:
    """Build a valid :class:`Camera`.

    Args:
        camera_id: The camera identifier.
        **overrides: Field overrides applied on top of the defaults.

    Returns:
        A validated camera.
    """
    fields: dict[str, Any] = {
        "camera_id": camera_id,
        "name": f"Camera {camera_id}",
        "lat": 40.7128,
        "lon": -74.0060,
    }
    fields.update(overrides)
    return Camera(**fields)


def make_link(
    from_camera_id: str = "cam_01", to_camera_id: str = "cam_02", **overrides: Any
) -> CameraLink:
    """Build a valid :class:`CameraLink`.

    Args:
        from_camera_id: Origin camera.
        to_camera_id: Destination camera.
        **overrides: Field overrides applied on top of the defaults.

    Returns:
        A validated camera link.
    """
    fields: dict[str, Any] = {
        "from_camera_id": from_camera_id,
        "to_camera_id": to_camera_id,
        "min_travel_time_sec": 45.0,
        "max_travel_time_sec": 180.0,
        "distance_meters": 1200.0,
    }
    fields.update(overrides)
    return CameraLink(**fields)


def make_sighting(
    camera_id: str = "cam_01",
    *,
    offset_sec: float = 0.0,
    **overrides: Any,
) -> Sighting:
    """Build a valid :class:`Sighting`.

    ``timestamp_utc``, ``raw_timestamp``, and ``clock_offset_applied_ms`` are
    kept mutually consistent by default; override them together when testing the
    consistency validator itself.

    Args:
        camera_id: The observing camera.
        offset_sec: Seconds after :data:`BASE_INSTANT` at which the sighting
            occurred.
        **overrides: Field overrides applied on top of the defaults.

    Returns:
        A validated sighting.
    """
    moment = BASE_INSTANT + timedelta(seconds=offset_sec)
    fields: dict[str, Any] = {
        "camera_id": camera_id,
        "timestamp_utc": moment,
        "raw_timestamp": moment,
        "clock_offset_applied_ms": 0,
        "object_class": ObjectClass.CAR,
        "detection_confidence": 0.92,
        "bbox": [100, 200, 300, 460],
        "frame_index": 42,
        "source_id": f"{camera_id}_clip",
        "created_at": moment,
    }
    fields.update(overrides)
    return Sighting(**fields)


def make_target(**overrides: Any) -> Target:
    """Build a valid :class:`Target` searching for a plate.

    Args:
        **overrides: Field overrides applied on top of the defaults.

    Returns:
        A validated target.
    """
    fields: dict[str, Any] = {
        "label": "Stolen Blue Sedan",
        "plate_query": "ABC1234",
        "created_at": BASE_INSTANT,
    }
    fields.update(overrides)
    return Target(**fields)


def make_match_candidate(**overrides: Any) -> MatchCandidate:
    """Build a valid exact-plate :class:`MatchCandidate`.

    Args:
        **overrides: Field overrides applied on top of the defaults.

    Returns:
        A validated match candidate.
    """
    fields: dict[str, Any] = {
        "sighting_id": PLACEHOLDER_SIGHTING_ID,
        "target_id": PLACEHOLDER_TARGET_ID,
        "match_method": MatchMethod.PLATE_EXACT,
        "match_score": 0.97,
        "plate_edit_distance": 0,
        "review_status": ReviewStatus.AUTO_ACCEPTED,
    }
    fields.update(overrides)
    return MatchCandidate(**fields)


def make_trajectory(sightings: list[Sighting] | None = None, **overrides: Any) -> Trajectory:
    """Build a :class:`Trajectory`, deriving hops and bounds from the sightings.

    The builder never raises on its own. Given deliberately invalid input -- an
    empty list, sightings out of order -- it still assembles arguments and lets
    ``Trajectory`` produce the error. A helper that crashed first would mask the
    very validator the test is there to exercise, which is why ``elapsed_sec`` is
    clamped at zero and the bounds fall back to the base instant.

    Args:
        sightings: Sightings in ascending time order. Defaults to a two-camera
            route 90 seconds apart.
        **overrides: Field overrides applied on top of the derived values.

    Returns:
        A validated trajectory whose hops bridge consecutive sightings.
    """
    if sightings is None:
        sightings = [
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_02", offset_sec=90),
        ]

    hops = [
        TrajectoryHop(
            from_sighting_id=origin.sighting_id,
            to_sighting_id=destination.sighting_id,
            from_camera_id=origin.camera_id,
            to_camera_id=destination.camera_id,
            elapsed_sec=max(
                0.0, (destination.timestamp_utc - origin.timestamp_utc).total_seconds()
            ),
            topology_plausible=True,
            hop_confidence=0.88,
        )
        for origin, destination in itertools.pairwise(sightings)
    ]

    fields: dict[str, Any] = {
        "target_id": PLACEHOLDER_TARGET_ID,
        "sightings": sightings,
        "hops": hops,
        "overall_confidence": 0.9,
        "start_time_utc": sightings[0].timestamp_utc if sightings else BASE_INSTANT,
        "end_time_utc": (
            sightings[-1].timestamp_utc if sightings else BASE_INSTANT + timedelta(seconds=1)
        ),
    }
    fields.update(overrides)
    return Trajectory(**fields)
