"""Builders for in-memory topologies.

The graph is pure, so most of its tests need no file at all -- constructing the
exact shape a test is about is clearer than maintaining a YAML fixture per case
and hoping the reader finds the relevant three lines.
"""

from __future__ import annotations

from multicam_tracker.models import Camera, CameraLink
from multicam_tracker.topology import Topology, TopologyEdge

__all__ = ["chain_topology", "make_topology"]


def make_topology(
    camera_ids: list[str],
    links: list[tuple[str, str, float, float]],
    *,
    bidirectional: bool = False,
) -> Topology:
    """Build a topology from ids and ``(from, to, min, max)`` tuples.

    Args:
        camera_ids: Cameras to create. Positions are spread along the equator so
            they are distinct without any test having to care where they are.
        links: Directed links as ``(from_id, to_id, min_sec, max_sec)``.
        bidirectional: Expand each declared link into both directions.

    Returns:
        The constructed graph.
    """
    cameras = [
        Camera(
            camera_id=camera_id,
            name=f"Camera {camera_id}",
            lat=0.0,
            lon=round(0.01 * index, 6),
        )
        for index, camera_id in enumerate(camera_ids)
    ]

    edges: list[TopologyEdge] = []
    for origin, destination, minimum, maximum in links:
        link = CameraLink(
            from_camera_id=origin,
            to_camera_id=destination,
            min_travel_time_sec=minimum,
            max_travel_time_sec=maximum,
            bidirectional=bidirectional,
        )
        edges.append(TopologyEdge(link))
        if bidirectional:
            edges.append(
                TopologyEdge(
                    CameraLink(
                        from_camera_id=destination,
                        to_camera_id=origin,
                        min_travel_time_sec=minimum,
                        max_travel_time_sec=maximum,
                        bidirectional=True,
                    ),
                    reversed_from_bidirectional=True,
                )
            )

    return Topology(cameras, edges)


def chain_topology(length: int = 3, minimum: float = 60.0, maximum: float = 300.0) -> Topology:
    """Build a one-way chain ``cam_0 -> cam_1 -> ... -> cam_n``.

    Args:
        length: Number of cameras.
        minimum: Minimum travel time on each link.
        maximum: Maximum travel time on each link.

    Returns:
        The chain.
    """
    camera_ids = [f"cam_{index}" for index in range(length)]
    links = [
        (camera_ids[index], camera_ids[index + 1], minimum, maximum) for index in range(length - 1)
    ]
    return make_topology(camera_ids, links)
