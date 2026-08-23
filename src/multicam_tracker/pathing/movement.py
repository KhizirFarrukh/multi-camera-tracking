"""Direction, speed, and what they say about a reconstructed route.

Derived quantities, but they answer questions an operator actually asks --
"which way was it heading?", "did it double back?" — and one of them is a
correctness check the rest of the engine cannot perform: an implied speed of
400 km/h means the route is wrong, whatever its confidence says.

Implausible speed is *flagged*, never used to silently drop a hop. The cause
might be the match (a false positive), the clock (drift, which stage 09
corrects), or the topology (a link with an optimistic minimum). Deleting the hop
would hide all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from multicam_tracker.models import Camera, GeoPoint
from multicam_tracker.pathing.preparation import PreparedCandidate

__all__ = [
    "HopMovement",
    "MovementSummary",
    "bearing_degrees",
    "compass_point",
    "summarize_movement",
]

_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_REVERSAL_DEGREES = 135.0
"""Change of heading beyond which the vehicle is treated as doubling back.

Not 180: a U-turn observed through a sparse camera network rarely reads as an
exact reversal, and requiring one would report no reversals at all."""


def bearing_degrees(origin: GeoPoint, destination: GeoPoint) -> float:
    """Return the initial great-circle bearing from one point to another.

    Args:
        origin: Start coordinate.
        destination: End coordinate.

    Returns:
        Degrees clockwise from true north, in ``[0, 360)``. Two identical points
        return ``0.0``; there is no meaningful heading, and raising would make
        every caller special-case a repeat detection.
    """
    lat1 = math.radians(origin.lat)
    lat2 = math.radians(destination.lat)
    delta_lon = math.radians(destination.lon - origin.lon)

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    if x == 0.0 and y == 0.0:
        return 0.0
    return math.degrees(math.atan2(x, y)) % 360.0


def compass_point(bearing: float) -> str:
    """Return the eight-point compass label for a bearing.

    Args:
        bearing: Degrees clockwise from north.

    Returns:
        One of N, NE, E, SE, S, SW, W, NW.
    """
    index = int((bearing % 360.0) / 45.0 + 0.5) % 8
    return _COMPASS[index]


@dataclass(frozen=True)
class HopMovement:
    """What one hop implies about the vehicle's movement."""

    from_sighting_id: str
    to_sighting_id: str
    distance_meters: float
    elapsed_sec: float
    speed_kph: float
    bearing: float
    implausible_speed: bool
    """True when the implied speed exceeds the configured limit. The hop is kept
    and marked, not removed."""

    @property
    def heading(self) -> str:
        """Return the eight-point compass label for this hop."""
        return compass_point(self.bearing)


@dataclass(frozen=True)
class MovementSummary:
    """Trajectory-level movement statistics."""

    hops: list[HopMovement] = field(default_factory=list)
    total_distance_meters: float = 0.0
    total_elapsed_sec: float = 0.0
    average_speed_kph: float = 0.0
    dominant_direction: str = ""
    """Compass label of the net displacement from first sighting to last.

    Net displacement rather than an average of headings: a vehicle that drives
    east, turns around, and drives west has an average heading of nothing in
    particular, but a net displacement that says where it actually ended up."""

    reversals: list[int] = field(default_factory=list)
    """Hop positions where the heading changed by more than the reversal
    threshold -- the vehicle doubled back."""

    implausible_hops: list[int] = field(default_factory=list)
    """Hop positions whose implied speed exceeds the limit."""

    @property
    def has_implausible_speed(self) -> bool:
        """Return whether any hop implies an impossible speed."""
        return bool(self.implausible_hops)


def summarize_movement(
    path: list[PreparedCandidate],
    cameras: dict[str, Camera],
    *,
    implausible_speed_kph: float | None = None,
) -> MovementSummary:
    """Derive per-hop and trajectory-level movement statistics.

    Args:
        path: The route's candidates, in chronological order.
        cameras: Cameras by id, for their coordinates. A hop whose camera is
            unknown is skipped rather than raising -- a camera decommissioned
            after a sighting was recorded is an ordinary occurrence, and it must
            not make an old trajectory unreadable.
        implausible_speed_kph: Speed above which a hop is flagged. Defaults to
            the configured topology limit.

    Returns:
        The summary. Empty for a path of fewer than two sightings, which implies
        no movement at all.
    """
    from multicam_tracker.config import get_settings

    limit = (
        implausible_speed_kph
        if implausible_speed_kph is not None
        else get_settings().topology.implausible_speed_kph
    )

    if len(path) < 2:
        return MovementSummary()

    hops: list[HopMovement] = []
    for position in range(len(path) - 1):
        origin = path[position]
        destination = path[position + 1]
        from_camera = cameras.get(origin.camera_id)
        to_camera = cameras.get(destination.camera_id)
        if from_camera is None or to_camera is None:
            continue

        start = GeoPoint(lat=from_camera.lat, lon=from_camera.lon)
        end = GeoPoint(lat=to_camera.lat, lon=to_camera.lon)
        distance = start.haversine_distance_to(end)
        elapsed = (
            destination.sighting.timestamp_utc - origin.sighting.timestamp_utc
        ).total_seconds()

        # Zero elapsed time cannot happen on a prepared path -- the graph emits
        # no edge for it -- but this function is public and a caller may hand it
        # anything, so the division is guarded rather than assumed safe.
        speed_kph = (distance / elapsed) * 3.6 if elapsed > 0.0 else 0.0

        hops.append(
            HopMovement(
                from_sighting_id=origin.sighting_id,
                to_sighting_id=destination.sighting_id,
                distance_meters=distance,
                elapsed_sec=elapsed,
                speed_kph=speed_kph,
                bearing=bearing_degrees(start, end),
                implausible_speed=speed_kph > limit,
            )
        )

    if not hops:
        return MovementSummary()

    total_distance = sum(hop.distance_meters for hop in hops)
    total_elapsed = sum(hop.elapsed_sec for hop in hops)
    average_speed = (total_distance / total_elapsed) * 3.6 if total_elapsed > 0.0 else 0.0

    reversals = [
        position
        for position in range(1, len(hops))
        if _heading_change(hops[position - 1].bearing, hops[position].bearing) > _REVERSAL_DEGREES
    ]

    first_camera = cameras.get(path[0].camera_id)
    last_camera = cameras.get(path[-1].camera_id)
    dominant = ""
    if first_camera is not None and last_camera is not None:
        dominant = compass_point(
            bearing_degrees(
                GeoPoint(lat=first_camera.lat, lon=first_camera.lon),
                GeoPoint(lat=last_camera.lat, lon=last_camera.lon),
            )
        )

    return MovementSummary(
        hops=hops,
        total_distance_meters=total_distance,
        total_elapsed_sec=total_elapsed,
        average_speed_kph=average_speed,
        dominant_direction=dominant,
        reversals=reversals,
        implausible_hops=[position for position, hop in enumerate(hops) if hop.implausible_speed],
    )


def _heading_change(first: float, second: float) -> float:
    """Return the absolute change between two bearings.

    Args:
        first: Earlier bearing in degrees.
        second: Later bearing in degrees.

    Returns:
        The smaller of the two arcs between them, in ``[0, 180]``. Taking the
        smaller arc is what makes 350 degrees and 10 degrees a 20-degree change
        rather than a 340-degree one.
    """
    delta = abs(second - first) % 360.0
    return min(delta, 360.0 - delta)
