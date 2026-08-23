"""Saying plainly where the system does not know what happened.

A gap is not a defect in the reconstruction. It is the reconstruction being
honest: two sightings, and no observation of what the vehicle did in between.
Interpolating a route through cameras that saw nothing would look better and be
worse.

Three kinds, distinguished because an operator does something different about
each:

``COVERAGE``
    No camera was watching the ground between the two sightings. Nothing to
    investigate — this is the network's shape.

``TEMPORAL``
    A route existed and the vehicle took far longer than any plausible transit.
    It stopped, parked, or left the network and came back. Worth investigating.

``OUTAGE``
    A camera on the plausible route recorded nothing at all during the window,
    while its neighbours kept working. The vehicle may well have passed it. This
    is a *system* problem, and reporting it as an absence of evidence rather
    than evidence of absence is the difference between an honest answer and a
    misleading one.

The outage case is the one that needs a cross-reference. "This camera saw
nothing" and "this camera was not working" are indistinguishable from the
trajectory alone, and only the wider sighting activity separates them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from multicam_tracker.models import CoverageGap, Sighting, TimeWindow
from multicam_tracker.pathing.graph import EdgeKind, TrajectoryEdge, TrajectoryGraph
from multicam_tracker.topology import Topology, describe_gap, shortest_transit_sec

__all__ = ["GapKind", "GapReport", "detect_gaps", "find_camera_outages"]


class GapKind(StrEnum):
    """Why a stretch of the route is unaccounted for."""

    COVERAGE = "coverage"
    TEMPORAL = "temporal"
    OUTAGE = "outage"


@dataclass(frozen=True)
class GapReport:
    """One unaccounted-for stretch, with everything needed to explain it."""

    kind: GapKind
    gap: CoverageGap
    """The domain model carried on the trajectory."""

    hop_count: int
    """Cameras between the endpoints on the fastest known route. ``0`` when the
    topology declares no route at all."""

    silent_cameras: list[str]
    """Cameras on the plausible route that recorded nothing during the window.
    Populated only for :attr:`GapKind.OUTAGE`."""

    def describe(self) -> str:
        """Return the human-readable reason.

        Returns:
            The sentence stored on the gap.
        """
        return self.gap.reason


def find_camera_outages(
    activity: Iterable[Sighting],
    window: TimeWindow,
    candidate_cameras: Iterable[str],
) -> list[str]:
    """Return cameras that recorded nothing during a window.

    Args:
        activity: Every sighting in the window, for any vehicle -- not only the
            target's. The distinction is the whole point: a camera that saw
            *other* traffic was working and genuinely did not see the target.
        window: The half-open window to check.
        candidate_cameras: Cameras to check.

    Returns:
        The silent ones, sorted. A camera absent from the activity feed entirely
        is silent; one that logged any vehicle is not.
    """
    seen = {sighting.camera_id for sighting in activity if window.contains(sighting.timestamp_utc)}
    return sorted(camera for camera in candidate_cameras if camera not in seen)


def _route_cameras(topology: Topology, origin: str, destination: str) -> list[str]:
    """Return the intermediate cameras on the fastest known route.

    Args:
        topology: The graph.
        origin: Start camera.
        destination: End camera.

    Returns:
        Cameras strictly between the endpoints, or an empty list when no route
        exists or the two are directly linked.
    """
    if topology.get_link(origin, destination) is not None:
        return []

    direct_time = shortest_transit_sec(topology, origin, destination)
    if direct_time is None:
        return []

    # Any camera that lies on *a* fastest route: reaching it and continuing
    # costs no more than the route itself, within a second of slack for the
    # float arithmetic.
    between: list[str] = []
    for camera in topology.camera_ids:
        if camera in {origin, destination}:
            continue
        leg_in = shortest_transit_sec(topology, origin, camera)
        leg_out = shortest_transit_sec(topology, camera, destination)
        if leg_in is None or leg_out is None:
            continue
        if leg_in + leg_out <= direct_time + 1.0:
            between.append(camera)
    return sorted(between)


def _temporal_reason(
    topology: Topology, origin: str, destination: str, elapsed_sec: float, expected_max: float
) -> str:
    """Explain an unusually slow transit.

    Args:
        topology: The graph.
        origin: Camera left.
        destination: Camera arrived at.
        elapsed_sec: Observed seconds.
        expected_max: Longest plausible transit.

    Returns:
        A sentence naming the excess.
    """
    del topology
    return (
        f"took {elapsed_sec:.0f}s from {origin} to {destination}, against a plausible "
        f"maximum of {expected_max:.0f}s; the vehicle most likely stopped or left the "
        f"monitored area and returned"
    )


def detect_gaps(
    graph: TrajectoryGraph,
    node_indices: list[int],
    edges: list[TrajectoryEdge],
    topology: Topology,
    *,
    activity: Iterable[Sighting] | None = None,
    stop_gap_multiplier: float | None = None,
) -> list[GapReport]:
    """Find and classify every unaccounted-for stretch of a route.

    Args:
        graph: The graph the path runs through.
        node_indices: The path's nodes, ascending.
        edges: The path's hops, one fewer than the nodes.
        topology: The camera graph.
        activity: All sightings in the trajectory window, for any vehicle. When
            supplied, a coverage gap whose route cameras were entirely silent is
            reclassified as an outage. When omitted, outages cannot be
            distinguished from genuine absences and none are reported -- silence
            is not evidence either way.
        stop_gap_multiplier: Multiple of the plausible maximum beyond which a
            slow transit is reported as a stop. Defaults to config.

    Returns:
        One report per flagged hop, in path order.
    """
    from multicam_tracker.config import get_settings

    multiplier = (
        stop_gap_multiplier
        if stop_gap_multiplier is not None
        else get_settings().pathing.stop_gap_multiplier
    )

    reports: list[GapReport] = []
    for position, edge in enumerate(edges):
        origin = graph.nodes[node_indices[position]]
        destination = graph.nodes[node_indices[position + 1]]
        link = topology.get_link(origin.camera_id, destination.camera_id)
        fastest = shortest_transit_sec(topology, origin.camera_id, destination.camera_id)

        expected_max = link.max_travel_time_sec if link is not None else (fastest or 0.0)
        route_cameras = _route_cameras(topology, origin.camera_id, destination.camera_id)

        kind: GapKind | None = None
        reason = ""
        silent: list[str] = []

        if edge.kind is EdgeKind.IMPLAUSIBLE:
            # The topology accounts for this pair and says the observed time
            # does not fit. Reported as temporal rather than coverage: nothing
            # is missing from the network, the timetable simply disagrees.
            kind = GapKind.TEMPORAL
            reason = describe_gap(
                topology, origin.camera_id, destination.camera_id, edge.elapsed_sec
            )
        elif edge.kind is EdgeKind.GAP:
            kind = GapKind.COVERAGE
            reason = describe_gap(
                topology, origin.camera_id, destination.camera_id, edge.elapsed_sec
            )
        elif expected_max > 0.0 and edge.elapsed_sec > expected_max * multiplier:
            kind = GapKind.TEMPORAL
            reason = _temporal_reason(
                topology,
                origin.camera_id,
                destination.camera_id,
                edge.elapsed_sec,
                expected_max,
            )
        elif edge.kind is EdgeKind.INDIRECT:
            kind = GapKind.COVERAGE
            reason = (
                f"no camera between {origin.camera_id} and {destination.camera_id} recorded "
                f"the vehicle during the {edge.elapsed_sec:.0f}s transit; the route is "
                f"plausible but unobserved"
            )

        if kind is None:
            continue

        if activity is not None and route_cameras:
            window = TimeWindow(
                start_utc=origin.sighting.timestamp_utc,
                end_utc=_at_least_one_millisecond_after(
                    origin.sighting.timestamp_utc, destination.sighting.timestamp_utc
                ),
            )
            silent = find_camera_outages(activity, window, route_cameras)
            if silent and len(silent) == len(route_cameras):
                kind = GapKind.OUTAGE
                reason = (
                    f"{', '.join(silent)} lie on the route from {origin.camera_id} to "
                    f"{destination.camera_id} and recorded no vehicles at all between "
                    f"{window.start_utc.isoformat()} and {window.end_utc.isoformat()}; "
                    f"this is an outage, not evidence the vehicle was elsewhere"
                )

        reports.append(
            GapReport(
                kind=kind,
                gap=CoverageGap(
                    from_sighting_id=origin.sighting_id,
                    to_sighting_id=destination.sighting_id,
                    elapsed_sec=edge.elapsed_sec,
                    expected_max_sec=expected_max,
                    reason=reason,
                ),
                hop_count=len(route_cameras),
                silent_cameras=silent if kind is GapKind.OUTAGE else [],
            )
        )

    return reports


def _at_least_one_millisecond_after(start: datetime, end: datetime) -> datetime:
    """Return an end bound strictly after the start.

    ``TimeWindow`` requires a non-empty half-open interval, and two sightings
    one millisecond apart would otherwise produce an empty one.

    Args:
        start: Window start.
        end: Proposed window end.

    Returns:
        ``end``, or one millisecond past ``start`` when that would be empty.
    """
    from datetime import timedelta

    return end if end > start else start + timedelta(milliseconds=1)
