"""The camera graph.

Cameras are nodes, links are directed edges carrying a plausible travel-time
window. This module is deliberately pure: it holds no database handle, opens no
file, and reads no clock. Everything it answers is a function of the graph it
was constructed with, which is what lets stages 06-08 test their search logic
without any infrastructure at all.

Bidirectional links are expanded into two directed edges at construction, so no
downstream code ever has to remember to check the reverse direction. That
asymmetry is a real modelling case, not a formality -- a one-way street is a
link that exists in one direction only, and forgetting it would let path
reconstruction propose a route no vehicle could drive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.models import Camera, CameraLink, GeoPoint

__all__ = ["Topology", "TopologyEdge"]


class TopologyEdge:
    """A directed edge, with provenance.

    ``CameraLink`` is a canonical contract entity and does not carry a
    ``derived`` flag, so provenance is tracked here instead of by widening the
    shared schema. Every constraint the system applies must be traceable to
    whether a human wrote it or the speed model inferred it.

    Args:
        link: The directed link.
        derived: Whether the travel times were computed from the speed model
            rather than authored explicitly.
        reversed_from_bidirectional: Whether this edge is the synthesized
            reverse of a bidirectional declaration.
    """

    __slots__ = ("derived", "link", "reversed_from_bidirectional")

    def __init__(
        self,
        link: CameraLink,
        *,
        derived: bool = False,
        reversed_from_bidirectional: bool = False,
    ) -> None:
        self.link = link
        self.derived = derived
        self.reversed_from_bidirectional = reversed_from_bidirectional

    @property
    def key(self) -> tuple[str, str]:
        """Return the ``(from, to)`` pair identifying this edge."""
        return self.link.from_camera_id, self.link.to_camera_id

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return (
            f"TopologyEdge({self.link.from_camera_id} -> {self.link.to_camera_id}, "
            f"window=[{self.link.min_travel_time_sec}, {self.link.max_travel_time_sec}], "
            f"derived={self.derived})"
        )


class Topology:
    """An immutable directed graph of cameras and travel-time windows.

    Args:
        cameras: The cameras, which become the graph's nodes.
        edges: Directed edges. Bidirectional links must already be expanded by
            the caller (:mod:`multicam_tracker.topology.loader` does this).

    Raises:
        TopologyError: If two cameras share an id, if an edge references an
            unknown camera, or if two edges describe the same ordered pair.
    """

    def __init__(self, cameras: Iterable[Camera], edges: Iterable[TopologyEdge]) -> None:
        self._cameras: dict[str, Camera] = {}
        for camera in cameras:
            if camera.camera_id in self._cameras:
                raise TopologyError(
                    "Duplicate camera_id in topology",
                    {"camera_id": camera.camera_id},
                )
            self._cameras[camera.camera_id] = camera

        self._edges: dict[tuple[str, str], TopologyEdge] = {}
        self._outgoing: dict[str, list[TopologyEdge]] = {cid: [] for cid in self._cameras}
        self._incoming: dict[str, list[TopologyEdge]] = {cid: [] for cid in self._cameras}

        for edge in edges:
            self._add_edge(edge)

        # Sorted once at construction so every traversal is deterministic. An
        # unstable neighbour order would make reachability results differ
        # between runs on the same data, which is untestable.
        for adjacency in (self._outgoing, self._incoming):
            for edge_list in adjacency.values():
                edge_list.sort(key=lambda item: item.key)

    def _add_edge(self, edge: TopologyEdge) -> None:
        """Register one directed edge.

        Args:
            edge: The edge to add.

        Raises:
            TopologyError: If an endpoint is unknown or the pair is already
                declared.
        """
        origin, destination = edge.key

        for camera_id, role in ((origin, "from_camera_id"), (destination, "to_camera_id")):
            if camera_id not in self._cameras:
                raise TopologyError(
                    "Link references an unknown camera",
                    {
                        "link": f"{origin} -> {destination}",
                        "field": role,
                        "missing_camera_id": camera_id,
                    },
                )

        # No self-link check here: CameraLink already rejects equal endpoints at
        # construction, so an edge with them cannot be built. The loader checks
        # the raw YAML separately, before a CameraLink is attempted, so the file
        # error names the offending entry rather than reporting a schema failure.

        if edge.key in self._edges:
            raise TopologyError(
                "Duplicate link for the same ordered camera pair",
                {"link": f"{origin} -> {destination}"},
            )

        self._edges[edge.key] = edge
        self._outgoing[origin].append(edge)
        self._incoming[destination].append(edge)

    # -- lookups ------------------------------------------------------------

    def get_camera(self, camera_id: str) -> Camera | None:
        """Return one camera by id.

        Args:
            camera_id: The identifier to look up.

        Returns:
            The camera, or ``None`` if the graph does not contain it.
        """
        return self._cameras.get(camera_id)

    def require_camera(self, camera_id: str) -> Camera:
        """Return one camera by id, insisting it exists.

        Args:
            camera_id: The identifier to look up.

        Returns:
            The camera.

        Raises:
            TopologyError: If no such camera is in the graph.
        """
        camera = self._cameras.get(camera_id)
        if camera is None:
            raise TopologyError("Unknown camera", {"camera_id": camera_id})
        return camera

    def list_cameras(self) -> list[Camera]:
        """Return every camera, ordered by id.

        Returns:
            The cameras.
        """
        return [self._cameras[camera_id] for camera_id in sorted(self._cameras)]

    @property
    def camera_ids(self) -> list[str]:
        """Return every camera id, sorted."""
        return sorted(self._cameras)

    def neighbors(self, camera_id: str) -> list[CameraLink]:
        """Return the links leaving ``camera_id``.

        Args:
            camera_id: Origin camera.

        Returns:
            Outgoing links ordered by destination id. Empty for an isolated
            camera.

        Raises:
            TopologyError: If the camera is not in the graph.
        """
        self.require_camera(camera_id)
        return [edge.link for edge in self._outgoing[camera_id]]

    def incoming(self, camera_id: str) -> list[CameraLink]:
        """Return the links arriving at ``camera_id``.

        Args:
            camera_id: Destination camera.

        Returns:
            Incoming links ordered by origin id.

        Raises:
            TopologyError: If the camera is not in the graph.
        """
        self.require_camera(camera_id)
        return [edge.link for edge in self._incoming[camera_id]]

    def get_link(self, from_camera_id: str, to_camera_id: str) -> CameraLink | None:
        """Return the link for one ordered pair.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            The link, or ``None`` when the pair has no edge in this direction.
        """
        edge = self._edges.get((from_camera_id, to_camera_id))
        return edge.link if edge is not None else None

    def get_edge(self, from_camera_id: str, to_camera_id: str) -> TopologyEdge | None:
        """Return the edge for one ordered pair, including its provenance.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            The edge, or ``None``.
        """
        return self._edges.get((from_camera_id, to_camera_id))

    def has_link(self, from_camera_id: str, to_camera_id: str) -> bool:
        """Return whether an edge exists in this direction.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            ``True`` when the ordered pair has an edge. A bidirectional link
            yields ``True`` both ways; a one-way link only in its own direction.
        """
        return (from_camera_id, to_camera_id) in self._edges

    def is_derived(self, from_camera_id: str, to_camera_id: str) -> bool:
        """Return whether an edge's travel times came from the speed model.

        Args:
            from_camera_id: Origin camera.
            to_camera_id: Destination camera.

        Returns:
            ``True`` when the window was computed rather than authored.
            ``False`` for an authored link and for an absent one.
        """
        edge = self._edges.get((from_camera_id, to_camera_id))
        return edge is not None and edge.derived

    def list_edges(self) -> list[TopologyEdge]:
        """Return every directed edge, ordered by ``(from, to)``.

        Returns:
            The edges.
        """
        return [self._edges[key] for key in sorted(self._edges)]

    def list_links(self) -> list[CameraLink]:
        """Return every directed link, ordered by ``(from, to)``.

        Returns:
            The links.
        """
        return [edge.link for edge in self.list_edges()]

    def location_of(self, camera_id: str) -> GeoPoint:
        """Return a camera's coordinate.

        Args:
            camera_id: The camera to locate.

        Returns:
            Its position.

        Raises:
            TopologyError: If the camera is not in the graph.
        """
        camera = self.require_camera(camera_id)
        return GeoPoint(lat=camera.lat, lon=camera.lon)

    @property
    def cameras_by_id(self) -> Mapping[str, Camera]:
        """Return a read-only view of the camera index."""
        return self._cameras

    def __len__(self) -> int:
        """Return the number of cameras."""
        return len(self._cameras)

    def __contains__(self, camera_id: object) -> bool:
        """Return whether a camera id is in the graph."""
        return camera_id in self._cameras

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"Topology(cameras={len(self._cameras)}, edges={len(self._edges)})"
