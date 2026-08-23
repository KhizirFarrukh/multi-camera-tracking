"""Builders for hand-constructed path-reconstruction cases.

Every test in ``tests/unit/pathing/`` states its graph explicitly rather than
generating one, because the point of those tests is that the correct answer is
known by hand before the algorithm runs. A helper that hid the camera layout
would defeat that.

The layout here is a straight chain -- ``cam_01 -> cam_02 -> ... -> cam_0n`` --
with the cameras spaced along the equator, plus ``cam_far`` which is linked to
nothing and sits a long way off. ``cam_far`` is where a false positive goes when
a test needs one at a geographically impossible location.
"""

from __future__ import annotations

from typing import Any

from multicam_tracker.models import Camera, MatchCandidate, MatchMethod, ReviewStatus, Sighting
from multicam_tracker.pathing.preparation import PreparedCandidate
from multicam_tracker.topology import Topology
from tests.fixtures.factories import make_camera, make_sighting

__all__ = [
    "CHAIN_MAX_SEC",
    "CHAIN_MIN_SEC",
    "FAR_CAMERA",
    "chain_cameras",
    "chain_topology_with_outlier",
    "prepared",
    "scored_candidate",
]

CHAIN_MIN_SEC = 60.0
CHAIN_MAX_SEC = 300.0
FAR_CAMERA = "cam_far"

_DEGREE_SPACING = 0.01
"""About 1.1 km between neighbours at the equator -- close enough that the chain
travel times are physically sensible, far enough that a two-second hop between
them is not."""


def chain_cameras(length: int = 4) -> dict[str, Camera]:
    """Return the chain's cameras plus the unlinked outlier.

    Args:
        length: How many chain cameras to build.

    Returns:
        Cameras by id. ``cam_far`` sits half a degree away -- roughly 55 km --
        so any hop to it inside a few minutes exceeds the speed limit.
    """
    cameras = {
        f"cam_{index + 1:02d}": make_camera(
            camera_id=f"cam_{index + 1:02d}", lat=0.0, lon=round(_DEGREE_SPACING * index, 6)
        )
        for index in range(length)
    }
    cameras[FAR_CAMERA] = make_camera(camera_id=FAR_CAMERA, lat=0.5, lon=0.5)
    return cameras


def chain_topology_with_outlier(length: int = 4) -> Topology:
    """Return a one-way chain plus an unreachable camera.

    Args:
        length: How many chain cameras to build.

    Returns:
        ``cam_01 -> cam_02 -> ... -> cam_0n``, with ``cam_far`` present in the
        graph but linked to nothing.
    """
    from multicam_tracker.models import CameraLink
    from multicam_tracker.topology import TopologyEdge

    cameras = chain_cameras(length)
    edges = [
        TopologyEdge(
            CameraLink(
                from_camera_id=f"cam_{index + 1:02d}",
                to_camera_id=f"cam_{index + 2:02d}",
                min_travel_time_sec=CHAIN_MIN_SEC,
                max_travel_time_sec=CHAIN_MAX_SEC,
            )
        )
        for index in range(length - 1)
    ]
    return Topology(list(cameras.values()), edges)


def scored_candidate(
    sighting: Sighting,
    score: float,
    *,
    target_id: str = "t-0001",
    method: MatchMethod = MatchMethod.PLATE_EXACT,
    review_status: ReviewStatus = ReviewStatus.AUTO_ACCEPTED,
    **overrides: Any,
) -> MatchCandidate:
    """Build a match candidate for one sighting.

    Args:
        sighting: The sighting being matched.
        score: The match score.
        target_id: The target it is matched to.
        method: How it was matched.
        review_status: Its adjudication state.
        **overrides: Further field overrides.

    Returns:
        A validated candidate whose evidence fields suit the declared method.
    """
    fields: dict[str, Any] = {
        "sighting_id": sighting.sighting_id,
        "target_id": target_id,
        "match_method": method,
        "match_score": score,
        "review_status": review_status,
    }
    if method is MatchMethod.PLATE_EXACT:
        fields["plate_edit_distance"] = 0
    elif method is MatchMethod.PLATE_FUZZY:
        fields["plate_edit_distance"] = 1
    elif method is MatchMethod.EMBEDDING:
        fields["embedding_similarity"] = 0.8
    fields.update(overrides)
    return MatchCandidate(**fields)


def prepared(
    camera_id: str, offset_sec: float, score: float, **overrides: Any
) -> PreparedCandidate:
    """Build one prepared candidate directly.

    Args:
        camera_id: The observing camera.
        offset_sec: Seconds after the base instant.
        score: The match score.
        **overrides: Candidate field overrides.

    Returns:
        The prepared candidate.
    """
    sighting = make_sighting(camera_id, offset_sec=offset_sec)
    return PreparedCandidate(
        sighting=sighting, candidate=scored_candidate(sighting, score, **overrides)
    )
