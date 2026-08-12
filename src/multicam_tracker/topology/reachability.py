"""Time-constrained reachability over the camera graph.

This is what turns "where else did this car appear?" into a bounded question.
Given a sighting at one camera, it answers which cameras the vehicle could have
reached, and *when* it could plausibly have arrived at each -- so the search for
the next sighting is a small indexed range scan rather than a full table scan.

Multi-hop matters because coverage has holes. A vehicle that leaves cam_01,
drives through three unmonitored streets, and reappears at cam_05 produces no
direct hop; only a two- or three-hop reachability query connects them.

Compounding rule: along a path, minima sum with minima and maxima with maxima.
The earliest possible arrival is every leg driven at its fastest; the latest is
every leg at its slowest. Anything in between is a mixture, and is covered by
that range.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from multicam_tracker.clock import ensure_utc
from multicam_tracker.models import TimeWindow
from multicam_tracker.topology.graph import Topology

__all__ = ["ReachabilityResult", "ReachableCamera", "reachable_from", "reachable_within"]


@dataclass(frozen=True)
class ReachableCamera:
    """A camera the vehicle could have reached, and when."""

    camera_id: str
    arrival_window: TimeWindow
    """Absolute ``[earliest, latest)`` plausible arrival.

    Not clipped to the search horizon. The horizon decides *whether* a camera is
    included -- its earliest arrival must fall inside -- but the window itself
    reports the full plausible range, because a caller narrowing a database
    query needs the true upper bound, not the one the search happened to stop at.
    """

    hops: int
    """Number of links traversed. ``1`` for a direct neighbour."""

    @property
    def earliest_arrival(self) -> datetime:
        """Return the earliest plausible arrival instant."""
        return self.arrival_window.start_utc

    @property
    def latest_arrival(self) -> datetime:
        """Return the latest plausible arrival instant."""
        return self.arrival_window.end_utc


@dataclass(frozen=True)
class ReachabilityResult:
    """Everything a reachability search found, and whether it finished."""

    cameras: list[ReachableCamera] = field(default_factory=list)
    truncated: bool = False
    """``True`` when the visit cap stopped the search early.

    A truncated result is a *lower bound* on reachability: everything listed is
    genuinely reachable, but something reachable may be missing. Callers that
    treat absence as proof of absence must check this flag.
    """

    visited_nodes: int = 0

    def camera_ids(self) -> list[str]:
        """Return the reachable camera ids, in result order."""
        return [entry.camera_id for entry in self.cameras]


def _window(departure: datetime, min_offset: float, max_offset: float) -> TimeWindow:
    """Build an absolute arrival window from relative offsets.

    Args:
        departure: When the vehicle left the origin camera.
        min_offset: Earliest arrival, in seconds after departure.
        max_offset: Latest arrival, in seconds after departure.

    Returns:
        The absolute window.
    """
    return TimeWindow(
        start_utc=departure + timedelta(seconds=min_offset),
        end_utc=departure + timedelta(seconds=max_offset),
    )


def reachable_from(
    topology: Topology,
    camera_id: str,
    departure_time: datetime,
    max_horizon_sec: float,
) -> list[ReachableCamera]:
    """Return the direct neighbours reachable within the horizon.

    Args:
        topology: The graph to search.
        camera_id: Camera the vehicle departed from.
        departure_time: Aware datetime of departure.
        max_horizon_sec: How far ahead to look. A neighbour is excluded when
            even its *fastest* transit lands beyond this.

    Returns:
        One entry per reachable neighbour, ordered by earliest arrival then
        camera id. Empty for an isolated camera.

    Raises:
        TopologyError: If the camera is not in the graph.
        ValidationError: If ``departure_time`` is naive.
    """
    departure = ensure_utc(departure_time, field_name="departure_time")
    topology.require_camera(camera_id)

    found = [
        ReachableCamera(
            camera_id=link.to_camera_id,
            arrival_window=_window(departure, link.min_travel_time_sec, link.max_travel_time_sec),
            hops=1,
        )
        for link in topology.neighbors(camera_id)
        if link.min_travel_time_sec <= max_horizon_sec
    ]
    found.sort(key=lambda entry: (entry.earliest_arrival, entry.camera_id))
    return found


def reachable_within(
    topology: Topology,
    camera_id: str,
    departure_time: datetime,
    max_horizon_sec: float,
    max_hops: int = 3,
    *,
    max_visited_nodes: int | None = None,
) -> ReachabilityResult:
    """Return every camera reachable within a horizon and a hop budget.

    A Dijkstra-style search ordered by earliest arrival. Two guards keep it
    bounded on a cyclic graph:

    * a camera is only re-expanded when a path reaches it *earlier* than any
      path found so far, which is what terminates cycles -- going around a loop
      can only ever take longer;
    * an expansion counter caps total work, and the result says so when it bites.

    The latest-arrival bound is widened as alternative paths are found: if a
    vehicle could arrive at cam_05 via a fast two-hop route or a slow three-hop
    one, the plausible arrival range covers both.

    Args:
        topology: The graph to search.
        camera_id: Camera the vehicle departed from.
        departure_time: Aware datetime of departure.
        max_horizon_sec: Latest earliest-arrival considered, in seconds.
        max_hops: Maximum links to traverse. ``1`` reduces this to
            :func:`reachable_from`.
        max_visited_nodes: Expansion cap. Defaults to the configured value.

    Returns:
        The reachable cameras, ordered by earliest arrival then camera id, plus
        whether the search was truncated.

    Raises:
        TopologyError: If the camera is not in the graph.
        ValidationError: If ``departure_time`` is naive.
    """
    from multicam_tracker.config import get_settings

    departure = ensure_utc(departure_time, field_name="departure_time")
    topology.require_camera(camera_id)

    if max_hops < 1 or max_horizon_sec < 0:
        return ReachabilityResult(cameras=[], truncated=False, visited_nodes=0)

    cap = (
        max_visited_nodes
        if max_visited_nodes is not None
        else get_settings().topology.max_visited_nodes
    )

    # camera_id -> [earliest offset, latest offset, fewest hops]
    best: dict[str, list[float]] = {}
    # (earliest_offset, camera_id, latest_offset, hops); camera_id keeps the
    # ordering total so heap comparisons never fall through to a float tie.
    frontier: list[tuple[float, str, float, int]] = [(0.0, camera_id, 0.0, 0)]
    visited = 0
    truncated = False

    while frontier:
        if visited >= cap:
            truncated = True
            break

        min_offset, current, max_offset, hops = heapq.heappop(frontier)
        visited += 1

        if hops >= max_hops:
            continue

        for link in topology.neighbors(current):
            neighbour = link.to_camera_id
            next_min = min_offset + link.min_travel_time_sec
            next_max = max_offset + link.max_travel_time_sec

            if next_min > max_horizon_sec:
                continue
            if neighbour == camera_id:
                # A loop back to the origin is a real possibility for a vehicle,
                # but it is not a *destination*: reporting it would make the
                # origin its own reachable target and confuse trajectory
                # assembly. Same-camera repeats are handled by is_same_pass.
                continue

            known = best.get(neighbour)
            if known is None:
                best[neighbour] = [next_min, next_max, hops + 1]
                heapq.heappush(frontier, (next_min, neighbour, next_max, hops + 1))
                continue

            improved = next_min < known[0]
            known[0] = min(known[0], next_min)
            known[1] = max(known[1], next_max)
            known[2] = min(known[2], hops + 1)
            if improved:
                heapq.heappush(frontier, (next_min, neighbour, next_max, hops + 1))

    cameras = [
        ReachableCamera(
            camera_id=name,
            arrival_window=_window(departure, bounds[0], bounds[1]),
            hops=int(bounds[2]),
        )
        for name, bounds in best.items()
    ]
    cameras.sort(key=lambda entry: (entry.earliest_arrival, entry.camera_id))

    return ReachabilityResult(cameras=cameras, truncated=truncated, visited_nodes=visited)
