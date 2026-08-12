"""Unit tests for :mod:`multicam_tracker.topology.graph`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.models import Camera, CameraLink
from multicam_tracker.topology import Topology, TopologyEdge
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

TOPOLOGY = make_topology(
    ["cam_a", "cam_b", "cam_c"],
    [("cam_a", "cam_b", 60.0, 300.0), ("cam_b", "cam_c", 30.0, 120.0)],
)


def _camera(camera_id: str) -> Camera:
    """Build a camera at the origin.

    Args:
        camera_id: The identifier.

    Returns:
        The camera.
    """
    return Camera(camera_id=camera_id, name=camera_id, lat=0.0, lon=0.0)


def _edge(origin: str, destination: str) -> TopologyEdge:
    """Build a directed edge with an arbitrary valid window.

    Args:
        origin: Origin camera id.
        destination: Destination camera id.

    Returns:
        The edge.
    """
    return TopologyEdge(
        CameraLink(
            from_camera_id=origin,
            to_camera_id=destination,
            min_travel_time_sec=10.0,
            max_travel_time_sec=20.0,
        )
    )


def test_graph__lookups__resolve_declared_entities() -> None:
    """The happy path for every accessor."""
    assert TOPOLOGY.get_camera("cam_a") is not None
    assert TOPOLOGY.get_camera("cam_missing") is None
    assert TOPOLOGY.camera_ids == ["cam_a", "cam_b", "cam_c"]
    assert len(TOPOLOGY) == 3
    assert "cam_b" in TOPOLOGY


def test_graph__neighbors__returns_outgoing_links_only() -> None:
    """Adjacency is directed; incoming links are a separate question."""
    assert [link.to_camera_id for link in TOPOLOGY.neighbors("cam_b")] == ["cam_c"]
    assert [link.from_camera_id for link in TOPOLOGY.incoming("cam_b")] == ["cam_a"]


def test_graph__neighbors_of_a_leaf__is_empty() -> None:
    """Boundary: a terminal node has no outgoing edges, and that is not an error."""
    assert TOPOLOGY.neighbors("cam_c") == []


def test_graph__neighbors_of_an_unknown_camera__raises() -> None:
    """A typo must not look like an isolated camera."""
    with pytest.raises(TopologyError, match="Unknown camera"):
        TOPOLOGY.neighbors("cam_ghost")


def test_graph__get_link_and_has_link__agree() -> None:
    """The two accessors must never disagree about whether an edge exists."""
    for origin in TOPOLOGY.camera_ids:
        for destination in TOPOLOGY.camera_ids:
            assert TOPOLOGY.has_link(origin, destination) == (
                TOPOLOGY.get_link(origin, destination) is not None
            )


def test_graph__list_edges__is_sorted_and_complete() -> None:
    """Deterministic order; an unstable one would make traversals irreproducible."""
    keys = [edge.key for edge in TOPOLOGY.list_edges()]

    assert keys == sorted(keys)
    assert keys == [("cam_a", "cam_b"), ("cam_b", "cam_c")]


def test_graph__location_of__returns_the_cameras_coordinate() -> None:
    """Used by the derivation model and the map UI."""
    point = TOPOLOGY.location_of("cam_a")

    assert point.lat == 0.0


def test_graph__duplicate_camera__is_rejected() -> None:
    """Two nodes with one id would make every lookup ambiguous."""
    with pytest.raises(TopologyError, match="Duplicate camera_id"):
        Topology([_camera("cam_a"), _camera("cam_a")], [])


def test_graph__edge_to_an_unknown_camera__is_rejected() -> None:
    """An edge into nothing is not a route."""
    with pytest.raises(TopologyError) as excinfo:
        Topology([_camera("cam_a")], [_edge("cam_a", "cam_ghost")])

    assert excinfo.value.context["missing_camera_id"] == "cam_ghost"


def test_graph__self_edge__cannot_be_constructed_at_all() -> None:
    """A self-link never reaches the graph: the model rejects it first.

    The guarantee lives in ``CameraLink`` rather than being re-checked here, so
    this asserts the guarantee rather than duplicating it. The loader checks the
    raw YAML separately, before a ``CameraLink`` is attempted, so a bad file
    reports the offending entry instead of a schema error.
    """
    with pytest.raises(PydanticValidationError, match="must differ"):
        _edge("cam_a", "cam_a")


def test_graph__duplicate_edge__is_rejected() -> None:
    """Two windows for one ordered pair leave no way to choose."""
    cameras = [_camera("cam_a"), _camera("cam_b")]

    with pytest.raises(TopologyError, match="Duplicate link"):
        Topology(cameras, [_edge("cam_a", "cam_b"), _edge("cam_a", "cam_b")])


def test_graph__bidirectional_expansion__is_idempotent_by_rejection() -> None:
    """Property from the spec: expanding twice must not create duplicate edges.

    The graph enforces this by refusing the second copy outright, which is
    stronger than silently de-duplicating -- a double expansion is a bug in the
    loader, and swallowing it would hide that.
    """
    cameras = [_camera("cam_a"), _camera("cam_b")]
    forward, backward = _edge("cam_a", "cam_b"), _edge("cam_b", "cam_a")

    once = Topology(cameras, [forward, backward])
    assert len(once.list_edges()) == 2

    with pytest.raises(TopologyError, match="Duplicate link"):
        Topology(cameras, [forward, backward, forward])


def test_graph__empty_graph__is_valid() -> None:
    """Boundary: a graph with no cameras is degenerate but not malformed."""
    empty = Topology([], [])

    assert len(empty) == 0
    assert empty.list_cameras() == []


def test_graph__repr__summarises_its_size() -> None:
    """Failure output should say how big the graph under test was."""
    assert repr(TOPOLOGY) == "Topology(cameras=3, edges=2)"


def test_edge__repr__shows_the_window_and_provenance() -> None:
    """Provenance is the point of the wrapper, so it belongs in the repr."""
    rendered = repr(TOPOLOGY.get_edge("cam_a", "cam_b"))

    assert "cam_a -> cam_b" in rendered
    assert "derived=False" in rendered
