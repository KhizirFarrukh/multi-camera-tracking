"""Coverage analysis and human-readable gap explanations.

Stage 08 records a ``CoverageGap`` whenever a stretch of a trajectory cannot be
explained by the topology, and stage 17 shows it to an operator. The explanation
is not decoration: "no route is known between these two cameras" and "it arrived
four minutes faster than anything plausible" call for completely different
follow-up, and a bare "implausible" tells the operator neither.
"""

from __future__ import annotations

import heapq

from multicam_tracker.topology.graph import Topology
from multicam_tracker.topology.plausibility import (
    PlausibilityReason,
    is_transition_plausible,
)

__all__ = ["describe_gap", "find_isolated_cameras", "graph_diameter_sec", "shortest_transit_sec"]


def find_isolated_cameras(topology: Topology) -> list[str]:
    """Return cameras with no links in either direction.

    An isolated camera can record sightings but can never appear in the middle
    of a reconstructed route, because nothing connects it to anything. That is
    usually a surveying gap rather than a real property of the network.

    Args:
        topology: The graph to inspect.

    Returns:
        The isolated camera ids, sorted.
    """
    return [
        camera.camera_id
        for camera in topology.list_cameras()
        if not topology.neighbors(camera.camera_id) and not topology.incoming(camera.camera_id)
    ]


def shortest_transit_sec(
    topology: Topology, from_camera_id: str, to_camera_id: str
) -> float | None:
    """Return the fastest plausible transit between two cameras, over any path.

    Args:
        topology: The graph to search.
        from_camera_id: Origin camera.
        to_camera_id: Destination camera.

    Returns:
        The summed minimum travel times along the fastest path, or ``None`` when
        no directed path exists.

    Raises:
        TopologyError: If either camera is not in the graph.
    """
    topology.require_camera(from_camera_id)
    topology.require_camera(to_camera_id)

    if from_camera_id == to_camera_id:
        return 0.0

    best: dict[str, float] = {from_camera_id: 0.0}
    frontier: list[tuple[float, str]] = [(0.0, from_camera_id)]

    while frontier:
        cost, current = heapq.heappop(frontier)
        if current == to_camera_id:
            return cost
        if cost > best.get(current, float("inf")):
            continue
        for link in topology.neighbors(current):
            candidate = cost + link.min_travel_time_sec
            if candidate < best.get(link.to_camera_id, float("inf")):
                best[link.to_camera_id] = candidate
                heapq.heappush(frontier, (candidate, link.to_camera_id))

    return None


def graph_diameter_sec(topology: Topology) -> float | None:
    """Return the longest fastest-path transit between any two cameras.

    A rough measure of how spread out the network is, used to pick sensible
    default search horizons. Unreachable pairs are skipped rather than treated
    as infinite: a disconnected graph has no single diameter, and reporting
    infinity would make the number useless for the one thing it is for.

    Args:
        topology: The graph to measure.

    Returns:
        The largest finite fastest-path transit in seconds, or ``None`` when no
        camera can reach any other.
    """
    longest: float | None = None

    for origin in topology.camera_ids:
        for destination in topology.camera_ids:
            if origin == destination:
                continue
            transit = shortest_transit_sec(topology, origin, destination)
            if transit is None:
                continue
            if longest is None or transit > longest:
                longest = transit

    return longest


def describe_gap(
    topology: Topology,
    from_camera_id: str,
    to_camera_id: str,
    elapsed_sec: float,
) -> str:
    """Explain, in one sentence, why a hop was or was not plausible.

    Args:
        topology: The graph to consult.
        from_camera_id: Camera the vehicle left.
        to_camera_id: Camera it arrived at.
        elapsed_sec: Seconds between the two sightings.

    Returns:
        A sentence naming the cameras, the observed elapsed time, and the bound
        it violated -- enough for an operator to decide whether to investigate
        the vehicle or the topology.
    """
    verdict = is_transition_plausible(topology, from_camera_id, to_camera_id, elapsed_sec)
    link = topology.get_link(from_camera_id, to_camera_id)

    if verdict.reason is PlausibilityReason.SAME_CAMERA:
        return (
            f"both sightings are on {from_camera_id}; this is a repeat detection, "
            f"not a transition between cameras"
        )

    if verdict.reason is PlausibilityReason.NO_LINK:
        indirect = shortest_transit_sec(topology, from_camera_id, to_camera_id)
        if indirect is None:
            return (
                f"no known route between {from_camera_id} and {to_camera_id}; "
                f"the topology declares no path in this direction"
            )
        return (
            f"no direct link between {from_camera_id} and {to_camera_id}; the fastest "
            f"known route runs via other cameras and takes at least {indirect:.0f}s, "
            f"against {elapsed_sec:.0f}s observed"
        )

    if link is None:  # pragma: no cover - unreachable: NO_LINK is handled above
        return f"no known route between {from_camera_id} and {to_camera_id}"

    if verdict.reason is PlausibilityReason.TOO_FAST:
        return (
            f"arrived at {to_camera_id} {verdict.margin_sec:.0f}s faster than the minimum "
            f"plausible transit of {link.min_travel_time_sec:.0f}s from {from_camera_id} "
            f"({elapsed_sec:.0f}s observed)"
        )

    if verdict.reason is PlausibilityReason.TOO_SLOW:
        return (
            f"arrived at {to_camera_id} {verdict.margin_sec:.0f}s later than the maximum "
            f"plausible transit of {link.max_travel_time_sec:.0f}s from {from_camera_id} "
            f"({elapsed_sec:.0f}s observed); a stop or a detour would explain it"
        )

    return (
        f"{elapsed_sec:.0f}s from {from_camera_id} to {to_camera_id} is within the "
        f"plausible window of {link.min_travel_time_sec:.0f}-{link.max_travel_time_sec:.0f}s"
    )
