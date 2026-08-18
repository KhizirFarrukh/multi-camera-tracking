"""Unit tests for :mod:`multicam_tracker.synth.adversarial` and the injections
as they appear in a fully generated dataset.

Each case corresponds to something that genuinely happens to camera networks and
that a system built only against clean data gets wrong.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.unit.synth.conftest import DEPARTURE

from multicam_tracker.synth import Scenario, generate
from multicam_tracker.synth.scenario import (
    AdversarialSpec,
    ClockDriftInjection,
    OutageInjection,
    TrafficSpec,
)
from multicam_tracker.topology import Topology

pytestmark = pytest.mark.unit


def _with(scenario: Scenario, **updates: object) -> Scenario:
    """Return a copy of a scenario with fields replaced.

    Args:
        scenario: The scenario to copy.
        **updates: Fields to override.

    Returns:
        The modified copy.
    """
    return scenario.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Clock drift
# ---------------------------------------------------------------------------


def test_clock_drift__offsets_exactly_the_named_camera(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Only the drifting camera moves; its neighbours stay on true time."""
    drifted = generate(
        _with(
            base_scenario,
            adversarial=AdversarialSpec(
                clock_drift=[ClockDriftInjection(camera_id="cam_03", offset_sec=45.0)]
            ),
        ),
        seed=3,
        topology=topology,
    )
    clean = generate(base_scenario, seed=3, topology=topology)

    truth = drifted.ground_truth
    for sighting in drifted.sightings:
        expected = truth.true_timestamps[sighting.sighting_id]
        shift = (sighting.timestamp_utc - expected).total_seconds()
        assert shift == pytest.approx(45.0 if sighting.camera_id == "cam_03" else 0.0, abs=1e-6)

    assert len(drifted.sightings) == len(clean.sightings)


def test_clock_drift__leaves_the_data_uncorrected(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The point is to give stage 09 something to find.

    ``clock_offset_applied_ms`` stays zero, so the drift is present in the
    recorded timestamps rather than already compensated for.
    """
    dataset = generate(
        _with(
            base_scenario,
            adversarial=AdversarialSpec(
                clock_drift=[ClockDriftInjection(camera_id="cam_03", offset_sec=45.0)]
            ),
        ),
        seed=3,
        topology=topology,
    )

    drifting = [s for s in dataset.sightings if s.camera_id == "cam_03"]
    assert drifting
    assert all(s.clock_offset_applied_ms == 0 for s in drifting)
    assert all(s.raw_timestamp == s.timestamp_utc for s in drifting)


def test_clock_drift__is_recorded_in_ground_truth(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Stage 09 is scored against the recorded truth, so it has to be there."""
    dataset = generate(
        _with(
            base_scenario,
            adversarial=AdversarialSpec(
                clock_drift=[ClockDriftInjection(camera_id="cam_03", offset_sec=-12.5)]
            ),
        ),
        seed=3,
        topology=topology,
    )

    events = dataset.ground_truth.injections_of_kind("clock_drift")
    assert len(events) == 1
    assert events[0].context == {"camera_id": "cam_03", "offset_sec": -12.5}


# ---------------------------------------------------------------------------
# Outage
# ---------------------------------------------------------------------------


def test_outage__window_contains_zero_sightings_for_that_camera(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Path reconstruction must produce a gap here rather than invent a route."""
    start = DEPARTURE - timedelta(hours=1)
    end = DEPARTURE + timedelta(hours=6)
    dataset = generate(
        _with(
            base_scenario,
            traffic=TrafficSpec(decoy_vehicles=10),
            adversarial=AdversarialSpec(
                outages=[OutageInjection(camera_id="cam_02", start_utc=start, end_utc=end)]
            ),
        ),
        seed=4,
        topology=topology,
    )

    blacked_out = [
        s for s in dataset.sightings if s.camera_id == "cam_02" and start <= s.timestamp_utc < end
    ]
    assert blacked_out == []


def test_outage__removes_the_sightings_from_ground_truth_too(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The answer key must describe what was emitted, not what was planned."""
    dataset = generate(
        _with(
            base_scenario,
            adversarial=AdversarialSpec(
                outages=[
                    OutageInjection(
                        camera_id="cam_02",
                        start_utc=DEPARTURE - timedelta(hours=1),
                        end_utc=DEPARTURE + timedelta(hours=6),
                    )
                ]
            ),
        ),
        seed=4,
        topology=topology,
    )

    emitted = {s.sighting_id for s in dataset.sightings}
    assert set(dataset.ground_truth.sighting_to_vehicle) == emitted
    target_truth = dataset.ground_truth.target
    assert target_truth is not None
    assert set(target_truth.sighting_ids) <= emitted


def test_outage__records_how_many_it_removed(base_scenario: Scenario, topology: Topology) -> None:
    """An outage that removed nothing did not test anything."""
    dataset = generate(
        _with(
            base_scenario,
            adversarial=AdversarialSpec(
                outages=[
                    OutageInjection(
                        camera_id="cam_02",
                        start_utc=DEPARTURE - timedelta(hours=1),
                        end_utc=DEPARTURE + timedelta(hours=6),
                    )
                ]
            ),
        ),
        seed=4,
        topology=topology,
    )

    events = dataset.ground_truth.injections_of_kind("outage")
    assert len(events) == 1
    assert events[0].context["removed"] >= 1


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_duplicates__share_the_vehicle_and_camera_but_not_the_id(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Dedup must collapse these without collapsing a genuine revisit."""
    dataset = generate(
        _with(base_scenario, adversarial=AdversarialSpec(duplicate_passes=2)),
        seed=5,
        topology=topology,
    )

    events = dataset.ground_truth.injections_of_kind("duplicate")
    assert len(events) == 2

    by_id = {s.sighting_id: s for s in dataset.sightings}
    for event in events:
        original = by_id[str(event.context["original_sighting_id"])]
        copy = by_id[str(event.context["duplicate_sighting_id"])]
        assert original.sighting_id != copy.sighting_id
        assert original.camera_id == copy.camera_id
        gap = abs((copy.timestamp_utc - original.timestamp_utc).total_seconds())
        assert gap < 1.0


def test_duplicates__are_attributed_to_the_same_vehicle_in_ground_truth(
    base_scenario: Scenario, topology: Topology
) -> None:
    """A duplicate is the same vehicle seen twice, not a second vehicle."""
    dataset = generate(
        _with(base_scenario, adversarial=AdversarialSpec(duplicate_passes=2)),
        seed=5,
        topology=topology,
    )

    truth = dataset.ground_truth
    for event in truth.injections_of_kind("duplicate"):
        original = str(event.context["original_sighting_id"])
        copy = str(event.context["duplicate_sighting_id"])
        assert truth.vehicle_for(copy) == truth.vehicle_for(original)


# ---------------------------------------------------------------------------
# Plate cloning
# ---------------------------------------------------------------------------


def test_plate_clone__two_vehicles_share_one_plate(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Plate matching alone cannot separate them; only appearance and topology can."""
    dataset = generate(
        _with(
            base_scenario,
            traffic=TrafficSpec(decoy_vehicles=3),
            adversarial=AdversarialSpec(clone_target_plate=True),
        ),
        seed=6,
        topology=topology,
    )

    wearing = [
        vehicle for vehicle in dataset.ground_truth.vehicles if vehicle.true_plate == "ABC1234"
    ]

    assert len({vehicle.vehicle_id for vehicle in wearing}) == 2
    assert dataset.ground_truth.injections_of_kind("plate_clone")


def test_plate_clone__disabled__leaves_the_plate_unique(
    base_scenario: Scenario, topology: Topology
) -> None:
    """No decoy may accidentally collide with the target's plate."""
    dataset = generate(
        _with(base_scenario, traffic=TrafficSpec(decoy_vehicles=40)), seed=6, topology=topology
    )

    wearing = [v for v in dataset.ground_truth.vehicles if v.true_plate == "ABC1234"]

    assert len(wearing) == 1


# ---------------------------------------------------------------------------
# Boundary hops and emission order
# ---------------------------------------------------------------------------


def test_boundary_hops__produce_one_hop_at_each_window_edge(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Exercises the inclusive-boundary convention with generated data."""
    from multicam_tracker.synth import leg_window

    dataset = generate(
        _with(base_scenario, adversarial=AdversarialSpec(boundary_hops=True)),
        seed=7,
        topology=topology,
    )
    sightings = dataset.sightings_of("target")

    first_gap = (sightings[1].timestamp_utc - sightings[0].timestamp_utc).total_seconds()
    second_gap = (sightings[2].timestamp_utc - sightings[1].timestamp_utc).total_seconds()

    assert first_gap == pytest.approx(
        leg_window(topology, sightings[0].camera_id, sightings[1].camera_id).min_sec, abs=0.01
    )
    assert second_gap == pytest.approx(
        leg_window(topology, sightings[1].camera_id, sightings[2].camera_id).max_sec, abs=0.01
    )
    assert dataset.ground_truth.injections_of_kind("boundary_hop")


def test_shuffle_emission_order__emits_out_of_chronological_order(
    base_scenario: Scenario, topology: Topology
) -> None:
    """A real ingest queue does not deliver sorted; nothing may assume it does."""
    scenario = _with(
        base_scenario,
        traffic=TrafficSpec(decoy_vehicles=15),
        adversarial=AdversarialSpec(shuffle_emission_order=True),
    )
    dataset = generate(scenario, seed=8, topology=topology)

    times = [s.timestamp_utc for s in dataset.sightings]

    assert times != sorted(times)
    assert dataset.ground_truth.injections_of_kind("out_of_order")


def test_no_injections__leaves_the_log_empty(base_scenario: Scenario, topology: Topology) -> None:
    """Boundary: a clean scenario must report no adversarial events at all."""
    assert generate(base_scenario, seed=9, topology=topology).ground_truth.injections == []
