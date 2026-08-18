"""Unit tests for :mod:`multicam_tracker.synth.routes`.

Generated data is topology-consistent *by construction*: jitter is clamped
inside each link's window rather than checked afterwards. The plausibility test
below is therefore a test of the generator, not of the matcher -- if it fails,
the answer key is wrong and every measurement built on it would be too.
"""

from __future__ import annotations

import itertools
import random
from datetime import UTC, datetime

import pytest

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.synth import leg_window, random_route, simulate_route
from multicam_tracker.topology import Topology, is_transition_plausible

pytestmark = pytest.mark.unit

DEPARTURE = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)


def test_leg_window__direct_link__is_that_links_window(topology: Topology) -> None:
    """One hop needs no path search."""
    link = topology.get_link("cam_01", "cam_02")
    window = leg_window(topology, "cam_01", "cam_02")

    assert link is not None
    assert window.min_sec == link.min_travel_time_sec
    assert window.max_sec == link.max_travel_time_sec
    assert window.hops == 1


def test_leg_window__no_direct_link__sums_along_the_connecting_path(
    topology: Topology,
) -> None:
    """A missed detection means the leg spans more than one link."""
    window = leg_window(topology, "cam_01", "cam_03")

    first = topology.get_link("cam_01", "cam_02")
    second = topology.get_link("cam_02", "cam_03")
    assert first is not None
    assert second is not None
    assert window.hops == 2
    assert window.min_sec == pytest.approx(first.min_travel_time_sec + second.min_travel_time_sec)
    assert window.max_sec == pytest.approx(first.max_travel_time_sec + second.max_travel_time_sec)


def test_leg_window__unreachable_pair__is_rejected(topology: Topology) -> None:
    """A scenario asking for an impossible drive is a scenario bug."""
    with pytest.raises(TopologyError, match="does not permit"):
        leg_window(topology, "cam_01", "cam_06")


def test_simulate_route__produces_one_pass_per_camera(topology: Topology) -> None:
    """The route is the specification."""
    passes = simulate_route(random.Random(1), topology, ["cam_01", "cam_02", "cam_03"], DEPARTURE)

    assert [entry.camera_id for entry in passes] == ["cam_01", "cam_02", "cam_03"]


def test_simulate_route__first_pass__is_at_the_departure_instant(topology: Topology) -> None:
    """Boundary: the vehicle is at its origin when it departs."""
    passes = simulate_route(random.Random(1), topology, ["cam_01", "cam_02"], DEPARTURE)

    assert passes[0].timestamp_utc == DEPARTURE
    assert passes[0].elapsed_sec is None


def test_simulate_route__timestamps__strictly_increase(topology: Topology) -> None:
    """A trajectory requires strictly ascending sightings."""
    passes = simulate_route(
        random.Random(2), topology, ["cam_01", "cam_02", "cam_03", "cam_04"], DEPARTURE
    )
    times = [entry.timestamp_utc for entry in passes]

    assert all(later > earlier for earlier, later in itertools.pairwise(times))


def test_simulate_route__every_hop__is_plausible_even_with_absurd_jitter(
    topology: Topology,
) -> None:
    """Clamping is what makes the data plausible by construction.

    Jitter far wider than any window still cannot push a hop outside the
    constraint the matcher will later check it against.
    """
    passes = simulate_route(
        random.Random(3),
        topology,
        ["cam_01", "cam_02", "cam_03", "cam_04"],
        DEPARTURE,
        jitter_sec=10_000.0,
    )

    for earlier, later in itertools.pairwise(passes):
        elapsed = (later.timestamp_utc - earlier.timestamp_utc).total_seconds()
        assert is_transition_plausible(
            topology, earlier.camera_id, later.camera_id, elapsed
        ).plausible


def test_simulate_route__jitter__stays_within_the_configured_bound(
    topology: Topology,
) -> None:
    """Jitter perturbs the nominal transit; it does not replace it."""
    jitter = 4.0
    link = topology.get_link("cam_01", "cam_02")
    assert link is not None
    nominal = link.min_travel_time_sec + 0.5 * (link.max_travel_time_sec - link.min_travel_time_sec)

    for seed in range(30):
        passes = simulate_route(
            random.Random(seed), topology, ["cam_01", "cam_02"], DEPARTURE, jitter_sec=jitter
        )
        elapsed = passes[1].elapsed_sec
        assert elapsed is not None
        assert abs(elapsed - nominal) <= jitter + 1e-6


@pytest.mark.parametrize(
    ("profile", "edge"), [(0.0, "min"), (1.0, "max")], ids=["fastest", "slowest"]
)
def test_simulate_route__speed_profile__positions_the_vehicle_in_the_window(
    topology: Topology, profile: float, edge: str
) -> None:
    """Boundary: profile 0 sits at the window minimum, profile 1 at its maximum."""
    link = topology.get_link("cam_01", "cam_02")
    assert link is not None
    target = link.min_travel_time_sec if edge == "min" else link.max_travel_time_sec

    passes = simulate_route(
        random.Random(4),
        topology,
        ["cam_01", "cam_02"],
        DEPARTURE,
        speed_profile=profile,
        jitter_sec=0.0,
    )

    assert passes[1].elapsed_sec == pytest.approx(target, abs=0.01)


def test_simulate_route__revisited_camera__produces_a_pass_each_time(
    topology: Topology,
) -> None:
    """A vehicle circling back is a real pattern, not a duplicate to collapse."""
    passes = simulate_route(
        random.Random(5), topology, ["cam_01", "cam_02", "cam_01", "cam_02"], DEPARTURE
    )

    assert [entry.camera_id for entry in passes].count("cam_01") == 2
    assert len(passes) == 4


def test_simulate_route__skipped_camera__omits_it_and_stays_plausible(
    topology: Topology,
) -> None:
    """Simulating a missed detection: cam_02 is driven past but never recorded."""
    passes = simulate_route(random.Random(6), topology, ["cam_01", "cam_03"], DEPARTURE)
    window = leg_window(topology, "cam_01", "cam_03")

    assert [entry.camera_id for entry in passes] == ["cam_01", "cam_03"]
    elapsed = passes[1].elapsed_sec
    assert elapsed is not None
    assert window.min_sec <= elapsed <= window.max_sec


def test_simulate_route__boundary_hops__pin_the_first_and_second_legs(
    topology: Topology,
) -> None:
    """Exercises the inclusive-boundary convention with real data, not just a unit test."""
    passes = simulate_route(
        random.Random(7),
        topology,
        ["cam_01", "cam_02", "cam_03"],
        DEPARTURE,
        boundary_hops=True,
    )

    first_window = leg_window(topology, "cam_01", "cam_02")
    second_window = leg_window(topology, "cam_02", "cam_03")

    assert passes[1].elapsed_sec == pytest.approx(first_window.min_sec, abs=0.01)
    assert passes[1].boundary == "min"
    assert passes[2].elapsed_sec == pytest.approx(second_window.max_sec, abs=0.01)
    assert passes[2].boundary == "max"


def test_simulate_route__single_camera_route__produces_one_pass(topology: Topology) -> None:
    """Boundary: a vehicle seen once has a route of length one."""
    assert len(simulate_route(random.Random(8), topology, ["cam_01"], DEPARTURE)) == 1


def test_simulate_route__unknown_camera__is_rejected(topology: Topology) -> None:
    """A typo in a scenario must not silently produce a shorter route."""
    with pytest.raises(TopologyError, match="Unknown camera"):
        simulate_route(random.Random(9), topology, ["cam_01", "cam_ghost"], DEPARTURE)


def test_simulate_route__is_deterministic_for_a_given_seed(topology: Topology) -> None:
    """Reproducibility applies to route timing too."""
    route = ["cam_01", "cam_02", "cam_03"]
    first = simulate_route(random.Random(10), topology, route, DEPARTURE)
    second = simulate_route(random.Random(10), topology, route, DEPARTURE)

    assert [entry.timestamp_utc for entry in first] == [entry.timestamp_utc for entry in second]


def test_random_route__walks_real_edges(topology: Topology) -> None:
    """Decoys must be as plausible as the target, or they add no difficulty."""
    route = random_route(random.Random(11), topology, length=4)

    for origin, destination in itertools.pairwise(route):
        assert topology.has_link(origin, destination)


def test_random_route__never_starts_at_an_isolated_camera(topology: Topology) -> None:
    """A route from a dead end would be a single pass, which is not traffic."""
    for seed in range(30):
        assert random_route(random.Random(seed), topology, length=3)[0] != "cam_06"
