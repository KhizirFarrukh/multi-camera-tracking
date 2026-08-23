"""Measured reconstruction against synthetic ground truth.

The hand-built cases in ``tests/unit/pathing/`` prove the algorithm does what it
claims on graphs a reader can verify. These prove it still does on data produced
by the real matchers, decoys and OCR failures included -- which is the only place
the difference between a good engine and a confident one shows up.

One number here is not a trade-off. **Teleport rate must be exactly zero on
every scenario.** A route can be incomplete, or uncertain, and still be useful.
A route containing a physically impossible hop is evidence of nothing while
looking like evidence of something.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from multicam_tracker.pathing import (
    ConfidenceStrategy,
    candidates_from_matches,
    evaluate_pathing,
    reconstruct_trajectory,
)
from multicam_tracker.synth import GroundTruth, generate_from_file
from multicam_tracker.topology import Topology, load_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_FILE = Path(__file__).parent / "baselines" / "pathing_metrics_baseline.json"

SCENARIOS = ("clean", "realistic", "degraded", "hard_negatives", "sparse_coverage", "adversarial")
DECOYED = ("hard_negatives", "adversarial")
"""Scenarios containing vehicles built to be mistaken for the target."""

TARGET_ID = "a0000000-0000-4000-8000-000000000001"
BASELINE_TOLERANCE = 0.02
PERFORMANCE_BUDGET_SEC = 60.0


@pytest.fixture(scope="module")
def topology() -> Topology:
    """Return the shipped topology.

    Returns:
        The graph the scenarios run over.
    """
    return load_topology(REPO_ROOT / "config" / "topology.yaml")


@pytest.fixture(scope="module")
def cameras(topology: Topology) -> dict[str, Any]:
    """Return the topology's cameras by id.

    Args:
        topology: The graph.

    Returns:
        Cameras keyed by id.
    """
    return dict(topology.cameras_by_id)


@pytest.fixture(scope="module")
def prepared(topology: Topology) -> dict[str, tuple[Any, list[Any]]]:
    """Generate every scenario and run matching over it once.

    Args:
        topology: The graph.

    Returns:
        Scenario name to ``(dataset, candidates)``.
    """
    built: dict[str, tuple[Any, list[Any]]] = {}
    for name in SCENARIOS:
        dataset = generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        built[name] = (
            dataset,
            candidates_from_matches(dataset, dataset.ground_truth, TARGET_ID, topology=topology),
        )
    return built


def _reconstruct(
    prepared: dict[str, tuple[Any, list[Any]]],
    name: str,
    topology: Topology,
    cameras: dict[str, Any],
    **overrides: Any,
) -> Any:
    """Reconstruct one scenario.

    Args:
        prepared: Datasets and candidates by name.
        name: Scenario to reconstruct.
        topology: The graph.
        cameras: Cameras by id.
        **overrides: Objective overrides.

    Returns:
        The reconstruction.
    """
    dataset, candidates = prepared[name]
    return reconstruct_trajectory(
        TARGET_ID,
        candidates,
        {sighting.sighting_id: sighting for sighting in dataset.sightings},
        topology,
        cameras=cameras,
        activity=list(dataset.sightings),
        **overrides,
    )


def _metrics(
    prepared: dict[str, tuple[Any, list[Any]]],
    name: str,
    topology: Topology,
    cameras: dict[str, Any],
    **overrides: Any,
) -> Any:
    """Score one scenario.

    Args:
        prepared: Datasets and candidates by name.
        name: Scenario to score.
        topology: The graph.
        cameras: Cameras by id.
        **overrides: Objective overrides.

    Returns:
        The metrics.
    """
    dataset, _ = prepared[name]
    return evaluate_pathing(
        dataset,
        dataset.ground_truth,
        topology,
        cameras=cameras,
        result=_reconstruct(prepared, name, topology, cameras, **overrides),
    )


def _baseline() -> dict[str, dict[str, Any]]:
    """Load the committed regression baseline.

    Returns:
        Scenario name to recorded metrics.
    """
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The property that is not a trade-off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIOS)
def test_no_scenario__produces_a_physically_impossible_hop(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """The exit criterion. Teleport rate is zero everywhere, without exception."""
    metrics = _metrics(prepared, name, topology, cameras)

    assert metrics.teleport_hops == 0, name
    assert metrics.teleport_rate == 0.0, name


def test_degraded__teleport_rate_is_zero_even_though_accuracy_falls(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Degrading data may cost recall. It may never buy an impossible route."""
    metrics = _metrics(prepared, "degraded", topology, cameras)

    assert metrics.teleport_hops == 0
    assert metrics.sighting_recall < 1.0, "if nothing is lost, the scenario is not degraded"
    assert metrics.sighting_precision == 1.0


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_clean__reconstructs_the_route_exactly(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Perfect reads and no decoys leave nothing to get wrong."""
    metrics = _metrics(prepared, "clean", topology, cameras)

    assert metrics.exact_path_match
    assert metrics.sighting_precision == 1.0
    assert metrics.sighting_recall == 1.0
    assert metrics.hop_accuracy == 1.0


def test_realistic__reconstructs_the_route_exactly(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """The documented target for ordinary conditions: an exact match."""
    metrics = _metrics(prepared, "realistic", topology, cameras)

    assert metrics.exact_path_match
    assert metrics.overall_confidence > 0.0


def test_sparse_coverage__spans_the_full_route_and_reports_the_gaps(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Thin coverage must produce one annotated route, not several fragments.

    The skip edges are what keep it connected, and the gap reports are what stop
    that connection reading as continuous observation.
    """
    metrics = _metrics(prepared, "sparse_coverage", topology, cameras)

    assert metrics.exact_path_match
    assert metrics.sighting_recall == 1.0
    assert metrics.gaps_reported > 0


@pytest.mark.parametrize("name", ["clean", "realistic", "degraded", "sparse_coverage"])
def test_scenarios_without_decoys__admit_no_other_vehicle_into_the_route(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """Precision is perfect wherever nothing was built to be mistaken for the target."""
    metrics = _metrics(prepared, name, topology, cameras)

    assert metrics.decoys_included == 0, name
    assert metrics.sighting_precision == 1.0, name


# ---------------------------------------------------------------------------
# Adversarial data: ambiguity instead of confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DECOYED)
def test_adversarial_scenarios__are_flagged_ambiguous_rather_than_answered_confidently(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """The honest answer where two routes explain the evidence equally well.

    Precision on these scenarios is genuinely poor, and no objective fixes that:
    the decoys are indistinguishable by construction. What must not happen is
    the engine presenting one of them as the answer.
    """
    result = _reconstruct(prepared, name, topology, cameras)

    assert result.is_ambiguous, name
    assert len(result.ambiguity.alternatives) > 1
    assert "does not distinguish" in result.ambiguity.describe()


def test_adversarial__the_alternatives_disagree_about_which_sightings_belong(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """An alternative that is the same route minus a hop is not an alternative."""
    result = _reconstruct(prepared, "adversarial", topology, cameras)

    best = set(result.ambiguity.alternatives[0].node_indices)
    other = set(result.ambiguity.alternatives[1].node_indices)

    assert best - other and other - best


@pytest.mark.parametrize("name", DECOYED)
def test_adversarial_scenarios__report_low_confidence_not_merely_a_flag(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """A route assembled through contested evidence must read as uncertain.

    The weakest-link aggregation is what makes the number say so; a flag alone
    would be easy to overlook in a list of trajectories.
    """
    metrics = _metrics(prepared, name, topology, cameras)

    assert metrics.overall_confidence < 0.5, name


def test_hard_negatives__the_target_sightings_are_still_found(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Poor precision here must not be poor recall in disguise."""
    metrics = _metrics(prepared, "hard_negatives", topology, cameras)

    assert metrics.sighting_recall >= 0.8


# ---------------------------------------------------------------------------
# The objective, as calibrated
# ---------------------------------------------------------------------------


def test_a_positive_inclusion_bonus__makes_the_adversarial_answer_confident(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Why the measured bonus is zero, asserted rather than asserted about.

    A per-node bonus pays a fixed amount for including a sighting whatever the
    evidence behind it. At 0.05 the engine chains decoys into the adversarial
    route until no competing account is left, and reports it as unambiguous --
    confidently wrong, which is the outcome the whole design exists to avoid.
    """
    calibrated = _reconstruct(prepared, "adversarial", topology, cameras)
    with_bonus = _reconstruct(
        prepared, "adversarial", topology, cameras, inclusion_bonus=0.05, gap_penalty=0.05
    )

    assert calibrated.is_ambiguous
    assert not with_bonus.is_ambiguous


def test_the_confidence_strategy__is_what_makes_a_contested_route_read_as_uncertain(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """The weakest link dominates, as configured, rather than being averaged away."""
    result = _reconstruct(
        prepared,
        "adversarial",
        topology,
        cameras,
        confidence_strategy=ConfidenceStrategy.MINIMUM,
    )

    assert result.trajectory is not None
    hops = [hop.hop_confidence for hop in result.trajectory.hops]
    assert result.trajectory.overall_confidence == pytest.approx(min(hops))


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_reconstruction__carries_a_complete_serializable_explanation(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """A trajectory a human cannot interrogate is not usable evidence."""
    result = _reconstruct(prepared, name, topology, cameras)
    explanation = result.explanation

    assert explanation is not None
    assert len(explanation.included) == len(result.trajectory.sightings)
    assert all(record.arrived_via for record in explanation.included)
    assert all(record.detail for record in explanation.excluded)
    json.dumps(explanation.to_json_dict())


def test_the_explanation__accounts_for_every_candidate_the_search_saw(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Included plus excluded is the whole candidate set. Nothing vanishes."""
    result = _reconstruct(prepared, "hard_negatives", topology, cameras)

    total = len(result.explanation.included) + result.explanation.excluded_total
    assert total == len(result.graph.nodes)


# ---------------------------------------------------------------------------
# Ground-truth provenance and regression guard
# ---------------------------------------------------------------------------


def test_metrics__are_computed_against_the_ground_truth_file_not_a_rederived_one(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The artifact later stages read is the artifact scored against."""
    dataset, _ = prepared["realistic"]
    path = dataset.ground_truth.write_json(tmp_path / "truth.json")
    result = _reconstruct(prepared, "realistic", topology, cameras)

    from_object = evaluate_pathing(
        dataset, dataset.ground_truth, topology, cameras=cameras, result=result
    )
    from_disk = evaluate_pathing(
        dataset, GroundTruth.read_json(path), topology, cameras=cameras, result=result
    )

    assert from_disk.to_json_dict() == from_object.to_json_dict()


@pytest.mark.parametrize("name", SCENARIOS)
def test_metrics__match_the_committed_baseline(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    name: str,
) -> None:
    """A quality regression must fail CI rather than be noticed later, or never."""
    recorded = _baseline()[name]
    current = _metrics(prepared, name, topology, cameras).to_json_dict()

    for key, expected in recorded.items():
        actual = current[key]
        if isinstance(expected, float):
            assert abs(actual - expected) <= BASELINE_TOLERANCE, f"{name}.{key}"
        else:
            assert actual == expected, f"{name}.{key}"


def test_baseline__covers_every_scenario_the_suite_evaluates() -> None:
    """A scenario missing from the baseline would be silently unguarded."""
    assert set(_baseline()) == set(SCENARIOS)


def test_reconstruction__is_deterministic_across_repeated_runs(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """An operator rerunning a search gets the same route, not a similar one."""
    routes = {
        tuple(
            sighting.sighting_id
            for sighting in _reconstruct(
                prepared, "adversarial", topology, cameras
            ).trajectory.sightings
        )
        for _ in range(3)
    }

    assert len(routes) == 1


def test_reconstruction__completes_within_the_documented_budget(topology: Topology) -> None:
    """Catches an accidental quadratic in the search, not slow hardware."""
    cameras = dict(topology.cameras_by_id)
    started = time.perf_counter()
    for name in SCENARIOS:
        dataset = generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        candidates = candidates_from_matches(
            dataset, dataset.ground_truth, TARGET_ID, topology=topology
        )
        reconstruct_trajectory(
            TARGET_ID,
            candidates,
            {sighting.sighting_id: sighting for sighting in dataset.sightings},
            topology,
            cameras=cameras,
            activity=list(dataset.sightings),
        )
    elapsed = time.perf_counter() - started

    assert elapsed < PERFORMANCE_BUDGET_SEC, f"took {elapsed:.1f}s"
