"""Unit tests for :mod:`multicam_tracker.synth.generator`.

Determinism gets the most attention. Stages 06-08 tune thresholds against these
datasets, so a generator that drifted between runs would turn every regression
test above it into noise -- and the drift would be invisible, because each run
would look internally consistent.
"""

from __future__ import annotations

import itertools
import json

import pytest
from tests.unit.synth.conftest import committed_scenario

from multicam_tracker.synth import Scenario, generate, stream_seed
from multicam_tracker.topology import Topology, is_transition_plausible

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generate__same_scenario_and_seed__produces_identical_output(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Byte-for-byte, comparing the full serialized dataset rather than a summary."""
    first = generate(base_scenario, seed=1234, topology=topology)
    second = generate(base_scenario, seed=1234, topology=topology)

    assert json.dumps(first.to_json_dict(), sort_keys=True) == json.dumps(
        second.to_json_dict(), sort_keys=True
    )


def test_generate__different_seeds__produce_different_output(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The seed must actually reach the generators, not be recorded and ignored."""
    first = generate(base_scenario, seed=1, topology=topology)
    second = generate(base_scenario, seed=2, topology=topology)

    assert first.to_json_dict() != second.to_json_dict()


def test_generate__seed_is_recorded_and_matches_the_request(
    base_scenario: Scenario, topology: Topology
) -> None:
    """A dataset you cannot regenerate is not reproducible, however deterministic."""
    dataset = generate(base_scenario, seed=99, topology=topology)

    assert dataset.seed == 99
    assert dataset.ground_truth.seed == 99


def test_generate__no_seed_override__uses_the_scenarios_own_seed(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The scenario file is self-contained by default."""
    assert generate(base_scenario, topology=topology).seed == base_scenario.seed


def test_generate__is_reproducible_across_committed_scenarios(topology: Topology) -> None:
    """The property has to hold for the scenarios that are actually used."""
    for name in ("realistic", "adversarial"):
        scenario = committed_scenario(name)
        first = generate(scenario, topology=topology)
        second = generate(scenario, topology=topology)
        assert first.to_json_dict() == second.to_json_dict(), name


def test_stream_seed__is_stable_across_processes() -> None:
    """CRC32, not hash().

    Python randomizes string hashing per process, so a hash-derived stream seed
    would make output differ between runs of the same command.
    """
    assert stream_seed(42, "plate_noise") == stream_seed(42, "plate_noise")
    assert stream_seed(42, "plate_noise") != stream_seed(42, "embeddings")


def test_stream_seed__separates_components(base_scenario: Scenario, topology: Topology) -> None:
    """Each component draws from its own stream.

    Turning on traffic must not change the target's route timings -- otherwise
    every scenario edit would silently perturb everything else.
    """
    quiet = generate(base_scenario, seed=5, topology=topology)
    busy_scenario = base_scenario.model_copy(
        update={"traffic": base_scenario.traffic.model_copy(update={"decoy_vehicles": 5})}
    )
    busy = generate(busy_scenario, seed=5, topology=topology)

    quiet_times = [s.timestamp_utc for s in quiet.sightings_of("target")]
    busy_times = [s.timestamp_utc for s in busy.sightings_of("target")]

    assert quiet_times == busy_times


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


def test_generate__every_hop__is_plausible_against_the_topology(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The correct answer is built, not asserted afterwards.

    If this fails the generator is broken, not the matcher -- which is exactly
    the separation that makes stages 06-08 measurable.
    """
    dataset = generate(base_scenario, seed=3, topology=topology)
    sightings = dataset.sightings_of("target")

    for earlier, later in itertools.pairwise(sightings):
        elapsed = (later.timestamp_utc - earlier.timestamp_utc).total_seconds()
        verdict = is_transition_plausible(topology, earlier.camera_id, later.camera_id, elapsed)
        assert verdict.plausible, f"{earlier.camera_id} -> {later.camera_id}: {verdict}"


def test_generate__target_sightings__appear_on_the_specified_cameras_in_order(
    base_scenario: Scenario, topology: Topology
) -> None:
    """The route is the specification; the data must follow it exactly."""
    dataset = generate(base_scenario, seed=3, topology=topology)

    assert [s.camera_id for s in dataset.sightings_of("target")] == base_scenario.vehicles[0].route


def test_generate__timestamps_along_a_route__strictly_increase(
    base_scenario: Scenario, topology: Topology
) -> None:
    """A trajectory requires strictly ascending sightings; the data must supply them."""
    times = [
        s.timestamp_utc
        for s in generate(base_scenario, seed=3, topology=topology).sightings_of("target")
    ]

    assert all(later > earlier for earlier, later in itertools.pairwise(times))


def test_generate__embeddings__are_present_and_normalized(
    base_scenario: Scenario, topology: Topology
) -> None:
    """Every sighting carries a usable vector, including where the plate failed."""
    import math

    for sighting in generate(base_scenario, seed=3, topology=topology).sightings:
        assert sighting.embedding is not None
        norm = math.sqrt(sum(component**2 for component in sighting.embedding))
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_generate__multi_hop_route__is_plausible_over_the_connecting_path(
    topology: Topology,
) -> None:
    """A route across uncovered ground still has to obey the summed windows."""
    scenario = committed_scenario("sparse_coverage")
    dataset = generate(scenario, topology=topology)
    sightings = dataset.sightings_of("target")

    assert [s.camera_id for s in sightings] == ["cam_01", "cam_03", "cam_05"]
    # No direct link exists for either leg, so a naive single-hop check must fail
    # -- which is the point of the scenario.
    assert topology.has_link("cam_01", "cam_03") is False
