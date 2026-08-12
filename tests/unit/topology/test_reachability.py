"""Unit tests for :mod:`multicam_tracker.topology.reachability`.

The compounding rule and the cycle guard get the most attention. A reachability
search that loops forever hangs the pipeline; one that compounds windows wrongly
returns a bounded-looking answer that silently excludes the real next sighting.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.exceptions import TopologyError, ValidationError
from multicam_tracker.topology import reachable_from, reachable_within
from tests.fixtures.topologies import chain_topology, make_topology

pytestmark = pytest.mark.unit

DEPARTURE = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
MIN_TRAVEL = 60.0
MAX_TRAVEL = 300.0

# cam_0 -> cam_1 -> cam_2, one-way, plus an unconnected camera.
CHAIN = make_topology(
    ["cam_0", "cam_1", "cam_2", "cam_iso"],
    [("cam_0", "cam_1", MIN_TRAVEL, MAX_TRAVEL), ("cam_1", "cam_2", MIN_TRAVEL, MAX_TRAVEL)],
)

_HYPOTHESIS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# reachable_from: single hop
# ---------------------------------------------------------------------------


def test_reachable_from__returns_exactly_the_direct_neighbours() -> None:
    """Single-hop reachability is the neighbour list, filtered by the horizon."""
    reached = reachable_from(CHAIN, "cam_0", DEPARTURE, max_horizon_sec=600)

    assert [entry.camera_id for entry in reached] == ["cam_1"]
    assert reached[0].hops == 1


def test_reachable_from__arrival_window__spans_the_links_travel_times() -> None:
    """The window is absolute, so a caller can hand it straight to a range scan."""
    reached = reachable_from(CHAIN, "cam_0", DEPARTURE, max_horizon_sec=600)

    assert reached[0].earliest_arrival == DEPARTURE + timedelta(seconds=MIN_TRAVEL)
    assert reached[0].latest_arrival == DEPARTURE + timedelta(seconds=MAX_TRAVEL)


def test_reachable_from__neighbour_beyond_the_horizon__is_excluded() -> None:
    """Boundary: exclusion is decided by the fastest possible transit."""
    reached = reachable_from(CHAIN, "cam_0", DEPARTURE, max_horizon_sec=MIN_TRAVEL - 1)

    assert reached == []


def test_reachable_from__neighbour_exactly_at_the_horizon__is_included() -> None:
    """Boundary: the horizon is inclusive of a transit that just fits."""
    reached = reachable_from(CHAIN, "cam_0", DEPARTURE, max_horizon_sec=MIN_TRAVEL)

    assert [entry.camera_id for entry in reached] == ["cam_1"]


def test_reachable_from__isolated_camera__returns_an_empty_list() -> None:
    """Empty list, never None: callers iterate the result unconditionally."""
    assert reachable_from(CHAIN, "cam_iso", DEPARTURE, max_horizon_sec=10_000) == []


def test_reachable_from__unknown_camera__raises() -> None:
    """A typo'd camera id must not silently look like an isolated one."""
    with pytest.raises(TopologyError, match="Unknown camera"):
        reachable_from(CHAIN, "cam_ghost", DEPARTURE, max_horizon_sec=600)


def test_reachable_from__naive_departure_time__is_rejected() -> None:
    """A naive instant would silently assume a timezone and shift every window."""
    with pytest.raises(ValidationError, match="Naive datetime"):
        reachable_from(CHAIN, "cam_0", DEPARTURE.replace(tzinfo=None), max_horizon_sec=600)


def test_reachable_from__results_are_ordered_by_earliest_arrival() -> None:
    """A stable, meaningful order lets a caller stop at the first hit."""
    topology = make_topology(
        ["cam_0", "cam_far", "cam_near"],
        [("cam_0", "cam_far", 200.0, 400.0), ("cam_0", "cam_near", 10.0, 60.0)],
    )

    reached = reachable_from(topology, "cam_0", DEPARTURE, max_horizon_sec=600)

    assert [entry.camera_id for entry in reached] == ["cam_near", "cam_far"]


# ---------------------------------------------------------------------------
# reachable_within: multi-hop
# ---------------------------------------------------------------------------


def test_reachable_within__max_hops_one__excludes_the_two_hop_camera() -> None:
    """A hop budget of one is single-hop reachability."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, 10_000, max_hops=1)

    assert result.camera_ids() == ["cam_1"]


def test_reachable_within__max_hops_two__includes_the_two_hop_camera() -> None:
    """Coverage has holes; a vehicle can reappear two hops away."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, 10_000, max_hops=2)

    assert result.camera_ids() == ["cam_1", "cam_2"]


def test_reachable_within__two_hop_window__compounds_min_with_min_and_max_with_max() -> None:
    """Earliest is every leg at its fastest; latest is every leg at its slowest."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, 10_000, max_hops=2)
    two_hop = next(entry for entry in result.cameras if entry.camera_id == "cam_2")

    assert two_hop.earliest_arrival == DEPARTURE + timedelta(seconds=2 * MIN_TRAVEL)
    assert two_hop.latest_arrival == DEPARTURE + timedelta(seconds=2 * MAX_TRAVEL)


def test_reachable_within__reports_hop_count_for_every_camera() -> None:
    """Hop count tells the caller how much of the route was unobserved."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, 10_000, max_hops=3)

    assert {entry.camera_id: entry.hops for entry in result.cameras} == {
        "cam_1": 1,
        "cam_2": 2,
    }


def test_reachable_within__horizon__prunes_the_far_hop() -> None:
    """The horizon bounds the search, not just the returned list."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, max_horizon_sec=100, max_hops=3)

    assert result.camera_ids() == ["cam_1"]


def test_reachable_within__isolated_camera__returns_nothing() -> None:
    """Nothing leaves an isolated node."""
    result = reachable_within(CHAIN, "cam_iso", DEPARTURE, 10_000, max_hops=3)

    assert result.cameras == []
    assert result.truncated is False


def test_reachable_within__cycle__terminates() -> None:
    """A ring graph must not loop forever; going round again only takes longer."""
    ring = make_topology(
        ["cam_0", "cam_1", "cam_2"],
        [
            ("cam_0", "cam_1", 60.0, 120.0),
            ("cam_1", "cam_2", 60.0, 120.0),
            ("cam_2", "cam_0", 60.0, 120.0),
        ],
    )

    result = reachable_within(ring, "cam_0", DEPARTURE, 100_000, max_hops=10)

    assert result.camera_ids() == ["cam_1", "cam_2"]
    assert result.truncated is False


def test_reachable_within__origin_is_never_reported_as_its_own_destination() -> None:
    """A loop back to the start is a real drive but not a search target."""
    ring = make_topology(
        ["cam_0", "cam_1"],
        [("cam_0", "cam_1", 60.0, 120.0), ("cam_1", "cam_0", 60.0, 120.0)],
    )

    result = reachable_within(ring, "cam_0", DEPARTURE, 100_000, max_hops=5)

    assert "cam_0" not in result.camera_ids()


def test_reachable_within__visit_cap__truncates_and_says_so() -> None:
    """A truncated result is a lower bound, and callers must be able to tell."""
    long_chain = chain_topology(length=30, minimum=10.0, maximum=20.0)

    result = reachable_within(
        long_chain, "cam_0", DEPARTURE, 100_000, max_hops=30, max_visited_nodes=3
    )

    assert result.truncated is True
    assert result.visited_nodes <= 3
    assert len(result.cameras) < 29


def test_reachable_within__generous_cap__is_not_truncated() -> None:
    """The flag must mean something: it stays false when the search completes."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, 10_000, max_hops=5)

    assert result.truncated is False


def test_reachable_within__alternative_paths__widen_the_arrival_range() -> None:
    """A camera reachable by a fast route and a slow one is plausible across both."""
    diamond = make_topology(
        ["cam_0", "cam_fast", "cam_slow", "cam_end"],
        [
            ("cam_0", "cam_fast", 10.0, 20.0),
            ("cam_0", "cam_slow", 100.0, 200.0),
            ("cam_fast", "cam_end", 10.0, 20.0),
            ("cam_slow", "cam_end", 100.0, 200.0),
        ],
    )

    result = reachable_within(diamond, "cam_0", DEPARTURE, 10_000, max_hops=3)
    end = next(entry for entry in result.cameras if entry.camera_id == "cam_end")

    assert end.earliest_arrival == DEPARTURE + timedelta(seconds=20)
    assert end.latest_arrival == DEPARTURE + timedelta(seconds=400)
    assert end.hops == 2


@pytest.mark.parametrize(
    ("max_hops", "horizon"),
    [(0, 1000.0), (-1, 1000.0), (3, -1.0)],
    ids=["zero", "negative", "past"],
)
def test_reachable_within__degenerate_budgets__return_nothing(
    max_hops: int, horizon: float
) -> None:
    """Boundary: a budget that permits no travel finds nothing, and does not raise."""
    result = reachable_within(CHAIN, "cam_0", DEPARTURE, horizon, max_hops=max_hops)

    assert result.cameras == []


def test_reachable_within__naive_departure_time__is_rejected() -> None:
    """Same timezone guarantee as the single-hop query."""
    with pytest.raises(ValidationError, match="Naive datetime"):
        reachable_within(CHAIN, "cam_0", DEPARTURE.replace(tzinfo=None), 600)


# ---------------------------------------------------------------------------
# Property-based
# ---------------------------------------------------------------------------


@_HYPOTHESIS
@given(
    edges=st.lists(
        st.tuples(
            st.sampled_from(["cam_0", "cam_1", "cam_2", "cam_3"]),
            st.sampled_from(["cam_0", "cam_1", "cam_2", "cam_3"]),
        ),
        max_size=8,
    )
)
def test_reachable_from__every_result__is_a_declared_neighbour(
    edges: list[tuple[str, str]],
) -> None:
    """Property: single-hop reachability never invents an edge.

    Directly guards the pruning contract -- a camera reported reachable in one
    hop that has no link would let path reconstruction propose a route the
    topology does not support.
    """
    unique = {(a, b) for a, b in edges if a != b}
    topology = make_topology(
        ["cam_0", "cam_1", "cam_2", "cam_3"],
        [(a, b, 60.0, 300.0) for a, b in sorted(unique)],
    )

    for entry in reachable_from(topology, "cam_0", DEPARTURE, 10_000):
        assert topology.has_link("cam_0", entry.camera_id)
        assert entry.hops == 1
