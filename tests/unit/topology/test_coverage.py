"""Unit tests for :mod:`multicam_tracker.topology.coverage`.

The explanations are asserted on content, not just on being non-empty: "no route
exists" and "arrived four minutes early" call for completely different follow-up
from an operator, and a message that conflates them is worse than none.
"""

from __future__ import annotations

import pytest

from multicam_tracker.topology import (
    describe_gap,
    find_isolated_cameras,
    graph_diameter_sec,
    shortest_transit_sec,
)
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

# cam_a -> cam_b -> cam_c, one-way, plus two unconnected cameras.
TOPOLOGY = make_topology(
    ["cam_a", "cam_b", "cam_c", "cam_iso", "cam_orphan"],
    [("cam_a", "cam_b", 60.0, 300.0), ("cam_b", "cam_c", 30.0, 120.0)],
)


# ---------------------------------------------------------------------------
# find_isolated_cameras
# ---------------------------------------------------------------------------


def test_find_isolated_cameras__identifies_exactly_the_unlinked_ones() -> None:
    """An isolated camera can record sightings but never appear mid-route."""
    assert find_isolated_cameras(TOPOLOGY) == ["cam_iso", "cam_orphan"]


def test_find_isolated_cameras__a_camera_with_only_incoming_links__is_not_isolated() -> None:
    """A terminal node is connected, even though nothing leaves it."""
    assert "cam_c" not in find_isolated_cameras(TOPOLOGY)


def test_find_isolated_cameras__fully_connected_graph__returns_empty() -> None:
    """Boundary: nothing to report is an empty list, not None."""
    connected = make_topology(["cam_a", "cam_b"], [("cam_a", "cam_b", 10.0, 20.0)])

    assert find_isolated_cameras(connected) == []


# ---------------------------------------------------------------------------
# shortest_transit_sec / graph_diameter_sec
# ---------------------------------------------------------------------------


def test_shortest_transit__direct_link__is_its_minimum() -> None:
    """One hop is the link's own minimum."""
    assert shortest_transit_sec(TOPOLOGY, "cam_a", "cam_b") == pytest.approx(60.0)


def test_shortest_transit__two_hops__sums_the_minima() -> None:
    """The fastest route is every leg driven at its fastest."""
    assert shortest_transit_sec(TOPOLOGY, "cam_a", "cam_c") == pytest.approx(90.0)


def test_shortest_transit__prefers_the_faster_of_two_routes() -> None:
    """The search must actually search, not take the first path it finds."""
    diamond = make_topology(
        ["cam_a", "cam_slow", "cam_fast", "cam_z"],
        [
            ("cam_a", "cam_slow", 500.0, 900.0),
            ("cam_slow", "cam_z", 500.0, 900.0),
            ("cam_a", "cam_fast", 10.0, 60.0),
            ("cam_fast", "cam_z", 10.0, 60.0),
        ],
    )

    assert shortest_transit_sec(diamond, "cam_a", "cam_z") == pytest.approx(20.0)


def test_shortest_transit__unreachable_pair__is_none() -> None:
    """Direction matters: the reverse of a one-way chain has no route."""
    assert shortest_transit_sec(TOPOLOGY, "cam_c", "cam_a") is None


def test_shortest_transit__same_camera__is_zero() -> None:
    """Boundary: no travel is required to stay put."""
    assert shortest_transit_sec(TOPOLOGY, "cam_a", "cam_a") == 0.0


def test_graph_diameter__is_the_longest_fastest_path() -> None:
    """cam_a to cam_c at 90s is the widest separation in this graph."""
    assert graph_diameter_sec(TOPOLOGY) == pytest.approx(90.0)


def test_graph_diameter__ignores_unreachable_pairs() -> None:
    """A disconnected graph has no single diameter; infinity would be useless."""
    assert graph_diameter_sec(TOPOLOGY) is not None


def test_graph_diameter__graph_with_no_links__is_none() -> None:
    """Boundary: nothing can reach anything, so there is no diameter."""
    assert graph_diameter_sec(make_topology(["cam_a", "cam_b"], [])) is None


# ---------------------------------------------------------------------------
# describe_gap
# ---------------------------------------------------------------------------


def test_describe_gap__no_route_at_all__says_so() -> None:
    """The operator should investigate the topology, not the vehicle."""
    message = describe_gap(TOPOLOGY, "cam_c", "cam_a", 120.0)

    assert "no known route" in message
    assert "cam_c" in message
    assert "cam_a" in message


def test_describe_gap__no_direct_link_but_an_indirect_route__reports_the_detour() -> None:
    """Distinguishes 'unreachable' from 'reachable, but not the way you assumed'."""
    message = describe_gap(TOPOLOGY, "cam_a", "cam_c", 40.0)

    assert "no direct link" in message
    assert "90s" in message


def test_describe_gap__too_fast__quantifies_how_much_too_fast() -> None:
    """A vehicle 30s faster than possible is a data problem worth naming."""
    message = describe_gap(TOPOLOGY, "cam_a", "cam_b", 30.0)

    assert "30s faster" in message
    assert "60s" in message


def test_describe_gap__too_slow__suggests_the_ordinary_explanation() -> None:
    """A slow transit is usually a stop, not an error."""
    message = describe_gap(TOPOLOGY, "cam_a", "cam_b", 900.0)

    assert "later than" in message
    assert "stop or a detour" in message


def test_describe_gap__same_camera__is_reported_distinctly() -> None:
    """Not a gap at all: a repeat detection on one camera."""
    message = describe_gap(TOPOLOGY, "cam_a", "cam_a", 5.0)

    assert "repeat detection" in message


def test_describe_gap__plausible_hop__describes_the_window_it_fell_inside() -> None:
    """The same helper explains why a hop was accepted, not only why it was not."""
    message = describe_gap(TOPOLOGY, "cam_a", "cam_b", 120.0)

    assert "within the" in message
    assert "60-300s" in message


def test_describe_gap__every_reason__produces_a_distinct_message() -> None:
    """Four situations, four explanations; conflating any two misleads."""
    messages = {
        describe_gap(TOPOLOGY, "cam_c", "cam_a", 120.0),
        describe_gap(TOPOLOGY, "cam_a", "cam_b", 10.0),
        describe_gap(TOPOLOGY, "cam_a", "cam_b", 900.0),
        describe_gap(TOPOLOGY, "cam_a", "cam_a", 5.0),
    }

    assert len(messages) == 4
