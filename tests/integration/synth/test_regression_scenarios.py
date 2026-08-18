"""Integration tests for the six committed regression scenarios.

Each scenario exists to exercise a specific difficulty, and each assertion here
checks that it *actually does*. A "degraded" scenario whose read-failure rate
had quietly drifted to 2% would still pass every generic self-consistency check
while measuring nothing, and stage 06 would be tuned against a problem that had
silently disappeared.

Marked ``integration`` because the largest scenarios take a noticeable moment;
they need no database.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import pytest

from multicam_tracker.synth import cosine_similarity, generate, load_scenario
from multicam_tracker.topology import Topology, is_transition_plausible, load_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
SCENARIO_NAMES = [
    "clean",
    "realistic",
    "degraded",
    "hard_negatives",
    "sparse_coverage",
    "adversarial",
]
GENERATION_BUDGET_SEC = 30.0
"""Loose on purpose: this guards against a change that made generation
quadratic, not against CI hardware being slow."""


@pytest.fixture(scope="module")
def topology() -> Topology:
    """Return the shipped topology the scenarios run over.

    Returns:
        The graph.
    """
    return load_topology(REPO_ROOT / "config" / "topology.yaml")


def _generate(name: str, topology: Topology) -> object:
    """Generate one committed scenario.

    Args:
        name: Scenario file stem.
        topology: Pre-loaded graph.

    Returns:
        The generated dataset.
    """
    return generate(load_scenario(SCENARIO_DIR / f"{name}.yaml"), topology=topology)


# ---------------------------------------------------------------------------
# All six generate and are self-consistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario__generates_without_error(name: str, topology: Topology) -> None:
    """The floor: a scenario that cannot generate cannot regress-test anything."""
    dataset = _generate(name, topology)

    assert dataset.sightings  # type: ignore[attr-defined]
    assert dataset.ground_truth.vehicles  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario__data_is_self_consistent_with_its_ground_truth(
    name: str, topology: Topology
) -> None:
    """Every sighting has exactly one owner, and no owner claims a phantom."""
    dataset = _generate(name, topology)
    emitted = {s.sighting_id for s in dataset.sightings}  # type: ignore[attr-defined]
    truth = dataset.ground_truth  # type: ignore[attr-defined]

    assert set(truth.sighting_to_vehicle) == emitted
    assert set(truth.true_timestamps) == emitted

    claimed = [sid for vehicle in truth.vehicles for sid in vehicle.sighting_ids]
    assert sorted(claimed) == sorted(emitted)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario__target_route__is_plausible_against_true_time(
    name: str, topology: Topology
) -> None:
    """Measured on *true* instants, so injected clock drift does not mask a real bug.

    The adversarial scenario deliberately reports the wrong time for one camera;
    checking against the recorded timestamps there would fail for the right
    reason and hide anything failing for the wrong one.
    """
    dataset = _generate(name, topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]
    target = truth.target
    assert target is not None

    by_id = {s.sighting_id: s for s in dataset.sightings}  # type: ignore[attr-defined]
    ordered = [by_id[sid] for sid in target.sighting_ids if sid in by_id]

    duplicate_ids = {
        str(event.context["duplicate_sighting_id"])
        for event in truth.injections_of_kind("duplicate")
    }
    ordered = [s for s in ordered if s.sighting_id not in duplicate_ids]

    for earlier, later in itertools.pairwise(ordered):
        if earlier.camera_id == later.camera_id:
            continue
        elapsed = (
            truth.true_timestamps[later.sighting_id] - truth.true_timestamps[earlier.sighting_id]
        ).total_seconds()
        verdict = is_transition_plausible(topology, earlier.camera_id, later.camera_id, elapsed)
        assert verdict.plausible or verdict.reason.value == "no_link", (
            f"{name}: {earlier.camera_id} -> {later.camera_id}: {verdict}"
        )


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario__is_deterministic(name: str, topology: Topology) -> None:
    """A regression fixture that drifted would make every comparison meaningless."""
    assert _generate(name, topology).to_json_dict() == _generate(name, topology).to_json_dict()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Each scenario measurably matches its intent
# ---------------------------------------------------------------------------


def test_clean__has_no_corruptions_and_no_decoys(topology: Topology) -> None:
    """The floor case: anything that fails here is broken, not merely imprecise."""
    dataset = _generate("clean", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]

    assert truth.corruptions == []
    assert truth.read_failure_rate == 0.0
    assert [v.vehicle_id for v in truth.vehicles] == ["target"]

    target = truth.target
    assert target is not None
    for sighting_id in target.sighting_ids:
        sighting = next(s for s in dataset.sightings if s.sighting_id == sighting_id)  # type: ignore[attr-defined]
        assert sighting.plate_text_normalized == "ABC1234"


def test_realistic__has_moderate_noise_and_real_traffic(topology: Topology) -> None:
    """The everyday case thresholds are tuned against."""
    dataset = _generate("realistic", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]

    assert 0.15 < len(truth.corruptions) / len(dataset.sightings) < 0.6  # type: ignore[attr-defined]
    assert len([v for v in truth.vehicles if v.is_decoy]) >= 20
    assert truth.read_failure_rate < 0.20, "realistic must not be degraded in disguise"


def test_degraded__loses_a_large_share_of_plate_reads(topology: Topology) -> None:
    """Plate matching cannot carry this scenario; re-id has to.

    The threshold is stated rather than inferred so a drift in the noise model
    fails here instead of silently making the scenario easy.
    """
    truth = _generate("degraded", topology).ground_truth  # type: ignore[attr-defined]

    assert truth.read_failure_rate > 0.25, f"only {truth.read_failure_rate:.1%} unreadable"
    assert len(truth.corruptions_of_kind("substitution")) > 0
    assert len(truth.corruptions_of_kind("dropout")) > 0


def test_degraded__confidences_are_visibly_lower_than_clean(topology: Topology) -> None:
    """Low-quality reads must carry low confidence, as real OCR does."""
    import statistics

    degraded = _generate("degraded", topology)
    clean = _generate("clean", topology)

    def _mean_confidence(dataset: object) -> float:
        values = [
            s.plate_confidence
            for s in dataset.sightings  # type: ignore[attr-defined]
            if s.plate_confidence is not None
        ]
        return statistics.mean(values)

    assert _mean_confidence(degraded) < _mean_confidence(clean) - 0.1


def test_hard_negatives__contains_decoys_above_the_auto_accept_threshold(
    topology: Topology,
) -> None:
    """Demonstrably hard, not decorative.

    At least one decoy must clear the contract's 0.92 embedding auto-accept
    similarity, so appearance alone wrongly accepts it and topology has to be
    what rejects it.
    """
    dataset = _generate("hard_negatives", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]
    target = truth.target
    assert target is not None

    reference = next(
        s.embedding
        for s in dataset.sightings  # type: ignore[attr-defined]
        if s.sighting_id == target.sighting_ids[0]
    )
    assert reference is not None

    hard_ids = {
        sid
        for vehicle in truth.vehicles
        if vehicle.is_hard_negative
        for sid in vehicle.sighting_ids
    }
    scores = [
        cosine_similarity(reference, s.embedding)
        for s in dataset.sightings  # type: ignore[attr-defined]
        if s.sighting_id in hard_ids and s.embedding is not None
    ]

    assert scores, "the scenario produced no hard negatives at all"
    assert max(scores) > 0.92, f"hardest decoy only reaches {max(scores):.3f}"


def test_sparse_coverage__target_hops_are_multi_link(topology: Topology) -> None:
    """The target crosses uncovered ground, so single-hop reasoning misses the link."""
    from multicam_tracker.synth import leg_window

    dataset = _generate("sparse_coverage", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]
    target = truth.target
    assert target is not None

    by_id = {s.sighting_id: s for s in dataset.sightings}  # type: ignore[attr-defined]
    cameras = [by_id[sid].camera_id for sid in target.sighting_ids]

    for origin, destination in itertools.pairwise(cameras):
        assert topology.has_link(origin, destination) is False, "leg should not be direct"
        assert leg_window(topology, origin, destination).hops >= 2


def test_adversarial__contains_every_injected_failure_mode(topology: Topology) -> None:
    """All at once, because they interact: drift makes an outage look like a gap."""
    truth = _generate("adversarial", topology).ground_truth  # type: ignore[attr-defined]

    kinds = {event.kind for event in truth.injections}

    assert {
        "clock_drift",
        "outage",
        "duplicate",
        "plate_clone",
        "boundary_hop",
        "out_of_order",
    } <= kinds


def test_adversarial__records_arrive_out_of_chronological_order(topology: Topology) -> None:
    """Anything assuming a sorted input stream breaks here rather than in production."""
    dataset = _generate("adversarial", topology)

    times = [s.timestamp_utc for s in dataset.sightings]  # type: ignore[attr-defined]

    assert times != sorted(times)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_largest_scenario__generates_within_the_documented_budget(
    topology: Topology,
) -> None:
    """Guards against a change that made generation quadratic in vehicle count."""
    started = time.perf_counter()
    dataset = _generate("realistic", topology)
    elapsed = time.perf_counter() - started

    assert dataset.sightings  # type: ignore[attr-defined]
    assert elapsed < GENERATION_BUDGET_SEC, f"generation took {elapsed:.1f}s"
