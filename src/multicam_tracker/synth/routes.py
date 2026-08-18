"""Route simulation: turning a list of cameras into timed passes.

Timestamps come from the topology's own travel-time windows, so generated data
is **topology-consistent by construction**. That is the whole trick: the correct
answer is not asserted afterwards, it is what the generator built. A test can
then check that every consecutive pair satisfies ``is_transition_plausible``,
and a failure means the generator is broken rather than the matcher.

A route may name cameras that are not directly linked. That is the
missed-detection case -- the vehicle drove past a camera that failed to see it,
or through an uncovered area -- and the window for such a leg is summed along the
fastest connecting path, min with min and max with max, exactly as multi-hop
reachability compounds them.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.topology import Topology

__all__ = ["LegWindow", "RoutePass", "leg_window", "simulate_route"]


@dataclass(frozen=True)
class LegWindow:
    """The plausible transit window for one leg of a route."""

    min_sec: float
    max_sec: float
    hops: int
    """Links traversed. Greater than one when the leg crosses uncovered cameras."""


@dataclass(frozen=True)
class RoutePass:
    """One camera pass on a simulated route."""

    camera_id: str
    timestamp_utc: datetime
    elapsed_sec: float | None
    """Seconds since the previous pass. ``None`` for the first."""

    boundary: str | None = None
    """``min`` or ``max`` when this hop was pinned to a window boundary."""


def leg_window(topology: Topology, origin: str, destination: str) -> LegWindow:
    """Return the plausible transit window between two cameras.

    Uses the direct link when one exists, and otherwise the fastest connecting
    path, summing minima and maxima along it.

    Args:
        topology: The graph to search.
        origin: Camera the vehicle leaves.
        destination: Camera it arrives at.

    Returns:
        The window and the number of links it spans.

    Raises:
        TopologyError: If no directed path connects the two, which means the
            scenario asks for a drive the road network does not permit.
    """
    direct = topology.get_link(origin, destination)
    if direct is not None:
        return LegWindow(direct.min_travel_time_sec, direct.max_travel_time_sec, hops=1)

    # Dijkstra on minimum transit, keeping predecessors so the maxima can be
    # summed along the same path rather than along a different, faster one.
    best: dict[str, float] = {origin: 0.0}
    previous: dict[str, str] = {}
    frontier: list[tuple[float, str]] = [(0.0, origin)]
    settled: set[str] = set()

    while frontier:
        cost, current = heapq.heappop(frontier)
        if current in settled:
            continue
        settled.add(current)
        if current == destination:
            break
        for link in topology.neighbors(current):
            candidate = cost + link.min_travel_time_sec
            if candidate < best.get(link.to_camera_id, float("inf")):
                best[link.to_camera_id] = candidate
                previous[link.to_camera_id] = current
                heapq.heappush(frontier, (candidate, link.to_camera_id))

    if destination not in best:
        raise TopologyError(
            "Scenario route requires a transit the topology does not permit",
            {"from_camera_id": origin, "to_camera_id": destination},
        )

    path = [destination]
    while path[-1] != origin:
        path.append(previous[path[-1]])
    path.reverse()

    minimum = 0.0
    maximum = 0.0
    for step_from, step_to in itertools.pairwise(path):
        step = topology.get_link(step_from, step_to)
        if step is None:  # pragma: no cover - the path was built from real links
            raise TopologyError(
                "Path step has no link", {"from_camera_id": step_from, "to_camera_id": step_to}
            )
        minimum += step.min_travel_time_sec
        maximum += step.max_travel_time_sec

    return LegWindow(minimum, maximum, hops=len(path) - 1)


def simulate_route(
    rng: random.Random,
    topology: Topology,
    route: list[str],
    departure_utc: datetime,
    *,
    speed_profile: float = 0.5,
    jitter_sec: float = 5.0,
    boundary_hops: bool = False,
) -> list[RoutePass]:
    """Generate timed passes for one vehicle's route.

    Each leg's elapsed time is the vehicle's nominal position within the window
    -- ``min + speed_profile * (max - min)`` -- plus uniform jitter, then clamped
    back inside the window. Clamping is what keeps the output plausible by
    construction: jitter can never push a hop outside the constraint the matcher
    will later check it against.

    Args:
        rng: Seeded generator.
        topology: The graph the route runs over.
        route: Ordered camera ids. Consecutive entries need not be directly
            linked.
        departure_utc: When the vehicle passes the first camera.
        speed_profile: Where in each window this vehicle sits, ``0`` fastest.
        jitter_sec: Uniform jitter magnitude, in seconds.
        boundary_hops: Pin the first hop to exactly the minimum and the second to
            exactly the maximum, so the inclusive-boundary convention is
            exercised by real data rather than only by a unit test.

    Returns:
        One pass per route entry, with strictly increasing timestamps.

    Raises:
        TopologyError: If any camera is unknown or any leg is unreachable.
    """
    for camera_id in route:
        topology.require_camera(camera_id)

    passes = [RoutePass(camera_id=route[0], timestamp_utc=departure_utc, elapsed_sec=None)]
    current = departure_utc

    for index, (origin, destination) in enumerate(itertools.pairwise(route)):
        window = leg_window(topology, origin, destination)

        boundary: str | None = None
        if boundary_hops and index == 0:
            elapsed = window.min_sec
            boundary = "min"
        elif boundary_hops and index == 1:
            elapsed = window.max_sec
            boundary = "max"
        else:
            nominal = window.min_sec + speed_profile * (window.max_sec - window.min_sec)
            elapsed = nominal + rng.uniform(-jitter_sec, jitter_sec)
            elapsed = max(window.min_sec, min(window.max_sec, elapsed))

        # Millisecond resolution matches the Sighting contract, so the value
        # stored is exactly the value generated and a self-consistency check
        # cannot fail on a rounding difference.
        elapsed = round(elapsed, 3)
        current = current + timedelta(seconds=elapsed)
        passes.append(
            RoutePass(
                camera_id=destination,
                timestamp_utc=current,
                elapsed_sec=elapsed,
                boundary=boundary,
            )
        )

    return passes


def random_route(rng: random.Random, topology: Topology, length: int) -> list[str]:
    """Draw a plausible route by walking the graph.

    Used for background traffic, which must be topology-plausible for the same
    reason the target is: a decoy on an impossible route would be trivially
    rejected and would not add any difficulty.

    Args:
        rng: Seeded generator.
        topology: The graph to walk.
        length: Desired number of cameras. A shorter route is returned when the
            walk reaches a node with no outgoing links.

    Returns:
        The camera ids in order. Empty when the graph has no cameras.

    Raises:
        TopologyError: If the graph is empty of connected cameras.
    """
    candidates = [camera_id for camera_id in topology.camera_ids if topology.neighbors(camera_id)]
    if not candidates:
        raise TopologyError(
            "Cannot generate traffic on a topology with no outgoing links",
            {"cameras": len(topology)},
        )

    route = [rng.choice(candidates)]
    while len(route) < length:
        options = topology.neighbors(route[-1])
        if not options:
            break
        route.append(rng.choice(options).to_camera_id)
    return route
