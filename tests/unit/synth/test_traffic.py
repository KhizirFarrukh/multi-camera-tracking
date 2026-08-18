"""Unit tests for :mod:`multicam_tracker.synth.traffic` and :mod:`~.scenario`.

Background traffic is what makes a measurement mean anything. A target that is
the only vehicle in the dataset is found by any implementation at all.
"""

from __future__ import annotations

import random
import statistics

import pytest
from pydantic import ValidationError as PydanticValidationError
from tests.unit.synth.conftest import DEPARTURE, SCENARIO_DIR, TARGET_ROUTE

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.synth import (
    Scenario,
    cosine_similarity,
    edit_distance,
    generate,
    generate_decoys,
    load_scenario,
    random_plate,
    save_scenario,
)
from multicam_tracker.synth.scenario import EmbeddingProfile, NoiseProfile, TrafficSpec, VehicleSpec
from multicam_tracker.topology import Topology

pytestmark = pytest.mark.unit


def _with_traffic(scenario: Scenario, **traffic: object) -> Scenario:
    """Return a copy of a scenario with a modified traffic spec.

    Args:
        scenario: The scenario to copy.
        **traffic: Traffic fields to set.

    Returns:
        The modified copy.
    """
    return scenario.model_copy(update={"traffic": TrafficSpec(**traffic)})


# ---------------------------------------------------------------------------
# Volume and plausibility
# ---------------------------------------------------------------------------


def test_generate_decoys__produces_the_requested_volume(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The requested haystack is the one you get."""
    scenario = _with_traffic(base_scenario, decoy_vehicles=17)

    decoys = generate_decoys(
        random.Random(1),
        topology,
        scenario,
        target_plate="ABC1234",
        span_start=DEPARTURE,
        span_seconds=3600.0,
    )

    assert len(decoys) == 17


def test_generate_decoys__routes_are_topology_plausible(
    base_scenario: Scenario, topology: Topology
) -> None:
    """A decoy on an impossible route would be rejected trivially and add nothing."""
    decoys = generate_decoys(
        random.Random(2),
        topology,
        _with_traffic(base_scenario, decoy_vehicles=20, decoy_route_length=4),
        target_plate="ABC1234",
        span_start=DEPARTURE,
        span_seconds=3600.0,
    )

    for decoy in decoys:
        for origin, destination in zip(decoy.route, decoy.route[1:], strict=False):
            assert topology.has_link(origin, destination)


def test_generate_decoys__zero_requested__produces_none(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Boundary: the 'clean' scenario has no traffic at all."""
    decoys = generate_decoys(
        random.Random(3),
        topology,
        _with_traffic(base_scenario, decoy_vehicles=0),
        target_plate="ABC1234",
        span_start=DEPARTURE,
        span_seconds=3600.0,
    )

    assert decoys == []


# ---------------------------------------------------------------------------
# Near-miss plates
# ---------------------------------------------------------------------------


def test_generate_decoys__near_miss_plates__sit_at_exactly_the_requested_distance(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Fuzzy matching is stressed at a known threshold, not wherever chance lands."""
    scenario = _with_traffic(base_scenario, decoy_vehicles=0, near_miss_edit_distances=[1, 2, 3])

    decoys = generate_decoys(
        random.Random(4),
        topology,
        scenario,
        target_plate="ABC1234",
        span_start=DEPARTURE,
        span_seconds=3600.0,
    )

    measured = {decoy.near_miss_distance: edit_distance("ABC1234", decoy.plate) for decoy in decoys}
    assert measured == {1: 1, 2: 2, 3: 3}


def test_generate_decoys__no_decoy_collides_with_the_target_plate(
    base_scenario: Scenario, topology: Topology
) -> None:
    """An accidental clone would look like a matching bug rather than a scenario."""
    decoys = generate_decoys(
        random.Random(5),
        topology,
        _with_traffic(base_scenario, decoy_vehicles=200),
        target_plate="ABC1234",
        span_start=DEPARTURE,
        span_seconds=3600.0,
    )

    assert all(decoy.plate != "ABC1234" for decoy in decoys)
    assert len({decoy.plate for decoy in decoys}) == len(decoys), "decoy plates are unique"


def test_random_plate__has_the_expected_shape() -> None:
    """Three letters and four digits, the common European format."""
    plate = random_plate(random.Random(6))

    assert len(plate) == 7
    assert plate[:3].isalpha()
    assert plate[3:].isdigit()


# ---------------------------------------------------------------------------
# Hard negatives, end to end
# ---------------------------------------------------------------------------


def test_hard_negatives__are_measurably_closer_to_the_target_than_ordinary_decoys(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The 'hard_negatives' scenario must be genuinely hard, not decorative."""
    scenario = base_scenario.model_copy(
        update={
            "traffic": TrafficSpec(decoy_vehicles=20),
            "embeddings": EmbeddingProfile(
                dimension=512, hard_negative_fraction=0.5, hard_negative_sigma=0.15
            ),
        }
    )
    dataset = generate(scenario, seed=7, topology=topology)
    truth = dataset.ground_truth

    target_sightings = dataset.sightings_of("target")
    assert target_sightings
    reference = target_sightings[0].embedding
    assert reference is not None

    def _mean_similarity(hard: bool) -> float:
        scores = [
            cosine_similarity(reference, sighting.embedding)
            for vehicle in truth.vehicles
            if vehicle.is_decoy and vehicle.is_hard_negative is hard
            for sighting in dataset.sightings_of(vehicle.vehicle_id)
            if sighting.embedding is not None
        ]
        return statistics.mean(scores)

    assert _mean_similarity(hard=True) > _mean_similarity(hard=False) + 0.4


def test_hard_negatives__at_least_one_clears_the_default_auto_accept_threshold(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Otherwise the scenario would not exercise the threshold it exists to attack."""
    scenario = base_scenario.model_copy(
        update={
            "traffic": TrafficSpec(decoy_vehicles=20),
            "embeddings": EmbeddingProfile(
                dimension=512, hard_negative_fraction=1.0, hard_negative_sigma=0.15
            ),
        }
    )
    dataset = generate(scenario, seed=8, topology=topology)

    reference = dataset.sightings_of("target")[0].embedding
    assert reference is not None

    best = max(
        cosine_similarity(reference, sighting.embedding)
        for vehicle in dataset.ground_truth.vehicles
        if vehicle.is_hard_negative
        for sighting in dataset.sightings_of(vehicle.vehicle_id)
        if sighting.embedding is not None
    )

    assert best > 0.92, "no hard negative reaches the contract's auto-accept similarity"


# ---------------------------------------------------------------------------
# Scenario model and serialization
# ---------------------------------------------------------------------------


def test_scenario__round_trips_through_yaml(base_scenario: Scenario, tmp_path: object) -> None:
    """Scenarios are committed fixtures, so the file must reload to the same object."""
    path = save_scenario(base_scenario, tmp_path / "scenario.yaml")  # type: ignore[operator]

    assert load_scenario(path) == base_scenario


@pytest.mark.parametrize(
    "name",
    ["clean", "realistic", "degraded", "hard_negatives", "sparse_coverage", "adversarial"],
)
def test_scenario__every_committed_fixture__loads(name: str) -> None:
    """All six regression scenarios must remain loadable."""
    scenario = load_scenario(SCENARIO_DIR / f"{name}.yaml")

    assert scenario.name == name
    assert scenario.target is not None


def test_scenario__duplicate_vehicle_ids__are_rejected() -> None:
    """Two vehicles with one id would make the assignment map ambiguous."""
    spec = VehicleSpec(vehicle_id="v", plate="ABC1234", route=TARGET_ROUTE, departure_utc=DEPARTURE)

    with pytest.raises(PydanticValidationError, match="duplicate vehicle_id"):
        Scenario(name="dup", vehicles=[spec, spec])


def test_scenario__near_miss_distance_zero__is_rejected() -> None:
    """Distance zero is plate cloning, which has its own explicit switch."""
    with pytest.raises(PydanticValidationError, match="plate cloning"):
        TrafficSpec(near_miss_edit_distances=[0])


def test_noise_profile__all_weights_zero_with_corruption_enabled__is_rejected() -> None:
    """An unusable profile would silently produce clean data labelled as noisy."""
    with pytest.raises(PydanticValidationError, match="non-zero failure-mode weight"):
        NoiseProfile(
            corruption_probability=0.5,
            substitution_weight=0.0,
            dropout_weight=0.0,
            read_failure_weight=0.0,
            spurious_weight=0.0,
        )


def test_noise_profile__inverted_confidence_range__is_rejected() -> None:
    """Boundary check on the clean-confidence band."""
    with pytest.raises(PydanticValidationError, match="clean_confidence_max"):
        NoiseProfile(clean_confidence_min=0.9, clean_confidence_max=0.5)


def test_load_scenario__missing_file__raises_a_project_error() -> None:
    """Callers catch one hierarchy, not a mix of OSError and pydantic errors."""
    with pytest.raises(ValidationError, match="not found"):
        load_scenario(SCENARIO_DIR / "no_such_scenario.yaml")


def test_load_scenario__malformed_yaml__raises_a_project_error(tmp_path: object) -> None:
    """A raw parser exception must not escape."""
    path = tmp_path / "bad.yaml"  # type: ignore[operator]
    path.write_text('name: "unterminated\n  bad: [\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="could not be parsed"):
        load_scenario(path)
