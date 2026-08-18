"""Unit tests for :mod:`multicam_tracker.synth.ground_truth` and the integrity of
the answer key a generated dataset emits.

A ground truth with an orphan or a phantom would make every precision and recall
number downstream quietly wrong, and nothing else in the system would notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.synth.conftest import committed_scenario

from multicam_tracker.synth import GroundTruth, Scenario, generate
from multicam_tracker.synth.scenario import NoiseProfile, TrafficSpec
from multicam_tracker.topology import Topology

pytestmark = pytest.mark.unit


@pytest.fixture
def populated(base_scenario: Scenario, topology: Topology) -> object:
    """Return a dataset with noise and traffic, so the truth has something in it.

    Returns:
        The generated dataset.
    """
    scenario = base_scenario.model_copy(
        update={
            "noise": NoiseProfile(corruption_probability=0.6),
            "traffic": TrafficSpec(decoy_vehicles=8, near_miss_edit_distances=[1, 2]),
        }
    )
    return generate(scenario, seed=21, topology=topology)


def test_ground_truth__every_sighting__appears_exactly_once_in_the_assignment(
    populated: object,
) -> None:
    """No orphans: a sighting with no owner cannot be scored."""
    emitted = [s.sighting_id for s in populated.sightings]  # type: ignore[attr-defined]
    assignment = populated.ground_truth.sighting_to_vehicle  # type: ignore[attr-defined]

    assert len(emitted) == len(set(emitted)), "duplicate sighting ids"
    assert set(emitted) == set(assignment)


def test_ground_truth__has_no_phantom_sightings(populated: object) -> None:
    """No phantoms: a truth entry with no sighting would inflate recall."""
    emitted = {s.sighting_id for s in populated.sightings}  # type: ignore[attr-defined]

    for vehicle in populated.ground_truth.vehicles:  # type: ignore[attr-defined]
        assert set(vehicle.sighting_ids) <= emitted


def test_ground_truth__counts_agree_with_what_was_generated(populated: object) -> None:
    """The two totals must match exactly, not approximately."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    from_vehicles = sum(len(vehicle.sighting_ids) for vehicle in truth.vehicles)

    assert from_vehicles == len(populated.sightings)  # type: ignore[attr-defined]
    assert len(truth.sighting_to_vehicle) == len(populated.sightings)  # type: ignore[attr-defined]


def test_ground_truth__each_vehicles_trajectory__is_time_ordered(populated: object) -> None:
    """Trajectories are compared against this ordering, so it must be correct."""
    truth = populated.ground_truth  # type: ignore[attr-defined]

    for vehicle in truth.vehicles:
        times = [truth.true_timestamps[sid] for sid in vehicle.sighting_ids]
        assert times == sorted(times), vehicle.vehicle_id


def test_ground_truth__each_trajectory__contains_only_that_vehicles_sightings(
    populated: object,
) -> None:
    """A misattributed sighting would make a correct matcher look wrong."""
    truth = populated.ground_truth  # type: ignore[attr-defined]

    for vehicle in truth.vehicles:
        for sighting_id in vehicle.sighting_ids:
            assert truth.vehicle_for(sighting_id) == vehicle.vehicle_id


def test_ground_truth__records_the_true_plate_of_every_vehicle(populated: object) -> None:
    """Corrupted reads are only measurable against the plate that was really there."""
    truth = populated.ground_truth  # type: ignore[attr-defined]

    assert all(vehicle.true_plate for vehicle in truth.vehicles)
    target = truth.target
    assert target is not None
    assert target.true_plate == "ABC1234"


def test_ground_truth__every_corruption__names_a_real_sighting(populated: object) -> None:
    """The corruption log is asserted on by later stages; dangling ids break that."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    emitted = {s.sighting_id for s in populated.sightings}  # type: ignore[attr-defined]

    for event in truth.corruptions:
        assert event.sighting_id in emitted
        assert truth.vehicle_for(event.sighting_id) == event.vehicle_id


def test_ground_truth__corruption_log__matches_the_observed_plates(
    populated: object,
) -> None:
    """The log has to describe what was actually emitted, not what was intended."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    by_id = {s.sighting_id: s for s in populated.sightings}  # type: ignore[attr-defined]

    for event in truth.corruptions:
        assert by_id[event.sighting_id].plate_text_normalized == event.observed_plate


def test_ground_truth__uncorrupted_sightings__carry_the_true_plate(
    populated: object,
) -> None:
    """Anything not in the log must be a clean read, or the log is incomplete."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    corrupted = {event.sighting_id for event in truth.corruptions}

    for sighting in populated.sightings:  # type: ignore[attr-defined]
        if sighting.sighting_id in corrupted:
            continue
        vehicle = truth.truth_for(truth.vehicle_for(sighting.sighting_id))
        assert vehicle is not None
        assert sighting.plate_text_normalized == vehicle.true_plate


def test_ground_truth__round_trips_through_json(populated: object, tmp_path: Path) -> None:
    """The artifact is a file later stages read; it has to survive being one."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    path = truth.write_json(tmp_path / "truth.json")

    assert GroundTruth.read_json(path) == truth


def test_ground_truth__json_is_stable_across_writes(populated: object, tmp_path: Path) -> None:
    """A ground-truth file that churned would make every diff unreadable."""
    truth = populated.ground_truth  # type: ignore[attr-defined]
    first = truth.write_json(tmp_path / "a.json").read_text(encoding="utf-8")
    second = truth.write_json(tmp_path / "b.json").read_text(encoding="utf-8")

    assert first == second


def test_read_failure_rate__clean_scenario__is_zero(topology: Topology) -> None:
    """Boundary: the 'clean' scenario asserts on this directly."""
    dataset = generate(committed_scenario("clean"), topology=topology)

    assert dataset.ground_truth.read_failure_rate == 0.0
    assert dataset.ground_truth.corruptions == []


def test_read_failure_rate__empty_dataset__is_zero_rather_than_dividing_by_zero() -> None:
    """Boundary: an empty answer key has no rate, and must not raise."""
    from multicam_tracker.synth.generator import GENERATION_EPOCH

    empty = GroundTruth(scenario_name="empty", seed=0, generated_at_utc=GENERATION_EPOCH)

    assert empty.read_failure_rate == 0.0


def test_ground_truth__unknown_sighting__resolves_to_none(populated: object) -> None:
    """Lookups return None rather than raising, since callers scan freely."""
    assert populated.ground_truth.vehicle_for("not-a-sighting") is None  # type: ignore[attr-defined]
    assert populated.ground_truth.truth_for("not-a-vehicle") is None  # type: ignore[attr-defined]
