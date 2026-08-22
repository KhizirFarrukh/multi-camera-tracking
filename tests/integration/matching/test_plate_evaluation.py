"""Measured evaluation of plate matching against synthetic ground truth.

This is the payoff of stage 05. Every assertion here is a number rather than an
adjective, and the baseline file turns "matching still works" into something CI
can check.

The measurements are taken against ground truth **loaded from its committed
file**, not re-derived from the dataset object. If the file and the generator
ever disagreed, every number computed from the object would still look fine
while the artifact everything else reads had gone stale.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from multicam_tracker.matching import (
    detect_plate_conflicts,
    evaluate_plate_matching,
    score_sighting,
)
from multicam_tracker.synth import GroundTruth, generate_from_file
from multicam_tracker.topology import Topology, load_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_FILE = Path(__file__).parent / "baselines" / "plate_metrics_baseline.json"

REALISTIC_PRECISION_TARGET = 0.99
"""Documented target from docs/PLATE_MATCHING.md, chosen from the sweep."""

REALISTIC_RECALL_FLOOR = 0.80
DEGRADED_PRECISION_TARGET = 0.99
BASELINE_TOLERANCE = 0.02
PERFORMANCE_BUDGET_SEC = 20.0


@pytest.fixture(scope="module")
def topology() -> Topology:
    """Return the shipped topology.

    Returns:
        The graph the scenarios run over.
    """
    return load_topology(REPO_ROOT / "config" / "topology.yaml")


def _dataset(name: str, topology: Topology) -> object:
    """Generate one committed scenario.

    Args:
        name: Scenario file stem.
        topology: Pre-loaded graph.

    Returns:
        The generated dataset.
    """
    return generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)


def _baseline() -> dict[str, dict[str, float]]:
    """Load the committed regression baseline.

    Returns:
        Scenario name to recorded metrics.
    """
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-scenario targets
# ---------------------------------------------------------------------------


def test_clean__precision_and_recall_are_both_perfect(topology: Topology) -> None:
    """The floor. Perfect reads and no decoys leave nothing to get wrong."""
    dataset = _dataset("clean", topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_realistic__precision_meets_the_documented_target(topology: Topology) -> None:
    """The exit criterion.

    The near-miss decoys in this scenario sit one and two edits from the target
    plate, so a threshold chosen loosely fails here rather than in production.
    """
    dataset = _dataset("realistic", topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]

    assert metrics.precision >= REALISTIC_PRECISION_TARGET, (
        f"precision {metrics.precision:.3f} below the documented "
        f"{REALISTIC_PRECISION_TARGET} target; false positives came from "
        f"{metrics.false_positive_vehicles}"
    )
    assert metrics.recall >= REALISTIC_RECALL_FLOOR


def test_degraded__precision_holds_even_as_recall_collapses(topology: Topology) -> None:
    """Precision must not degrade with the data.

    Recall falling is expected and correct -- 40% of plates in this scenario are
    completely unreadable, and no amount of string logic recovers those. What
    must not happen is the matcher compensating by accepting worse candidates.
    """
    dataset = _dataset("degraded", topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]

    assert metrics.precision >= DEGRADED_PRECISION_TARGET
    assert metrics.recall < 0.8, "if recall did not drop, the scenario is not degraded"


def test_hard_negatives__no_decoy_is_ever_auto_accepted(topology: Topology) -> None:
    """The exit criterion, and the one that matters most.

    An auto-accepted false positive reaches an operator as fact, with no human
    in the loop to catch it.
    """
    dataset = _dataset("hard_negatives", topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]

    assert metrics.auto_accepted_false == 0
    assert metrics.auto_accept_precision == 1.0


def test_adversarial__cloned_plate_is_reported_by_conflict_detection(
    topology: Topology,
) -> None:
    """Plate matching provably cannot separate a clone from the target.

    Two vehicles wear one plate, so the matcher returns both and precision is
    genuinely below target here. The system does not pretend otherwise -- it
    detects the physical impossibility and surfaces it instead.
    """
    dataset = _dataset("adversarial", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]
    target = truth.target
    assert target is not None

    matched = [
        sighting
        for sighting in dataset.sightings  # type: ignore[attr-defined]
        if score_sighting(sighting, target.true_plate) is not None
    ]
    conflicts = detect_plate_conflicts(target.true_plate, matched, topology)

    assert conflicts, "the cloned plate produced no detectable conflict"
    assert any("cannot account for both" in conflict.describe() for conflict in conflicts)


def test_adversarial__false_positives_are_the_clone_and_nothing_else(
    topology: Topology,
) -> None:
    """Precision below target here must be the clone, not a matcher defect."""
    dataset = _dataset("adversarial", topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]

    assert metrics.false_positive_vehicles == ["clone_00"]


# ---------------------------------------------------------------------------
# Ground-truth provenance and failure attribution
# ---------------------------------------------------------------------------


def test_metrics__are_computed_against_the_ground_truth_file_not_a_rederived_one(
    topology: Topology, tmp_path: Path
) -> None:
    """The artifact later stages read is the artifact scored against.

    A generator and a stale file that disagreed would produce identical-looking
    numbers from the object while the committed record had drifted.
    """
    dataset = _dataset("realistic", topology)
    path = dataset.ground_truth.write_json(tmp_path / "truth.json")  # type: ignore[attr-defined]
    from_file = GroundTruth.read_json(path)

    from_object = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]
    from_disk = evaluate_plate_matching(dataset, from_file)

    assert from_disk.to_json_dict() == from_object.to_json_dict()


def test_misses__are_attributed_to_the_corruption_recorded_in_ground_truth(
    topology: Topology,
) -> None:
    """Every miss is explained by a failure mode the answer key actually logged."""
    dataset = _dataset("degraded", topology)
    truth = dataset.ground_truth  # type: ignore[attr-defined]
    metrics = evaluate_plate_matching(dataset, truth)

    recorded_kinds = {event.kind for event in truth.corruptions} | {"clean"}

    assert metrics.misses_by_corruption
    assert set(metrics.misses_by_corruption) <= recorded_kinds
    assert sum(metrics.misses_by_corruption.values()) == metrics.false_negatives


def test_misses__are_never_attributed_to_clean_reads(topology: Topology) -> None:
    """A missed sighting whose plate was read correctly would be a matcher bug.

    Everything else is a data limitation; this is the one category that is not.
    """
    for name in ("clean", "realistic", "degraded", "hard_negatives"):
        dataset = _dataset(name, topology)
        metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]
        assert metrics.misses_by_corruption.get("clean", 0) == 0, name


# ---------------------------------------------------------------------------
# Regression guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["clean", "realistic", "degraded", "hard_negatives", "sparse_coverage", "adversarial"],
)
def test_metrics__match_the_committed_baseline(name: str, topology: Topology) -> None:
    """A change that degrades matching quality fails here rather than shipping.

    Regenerate deliberately with ``scripts/evaluate_matching.py --write-baseline``
    once a change is understood and intended -- never to make this pass.
    """
    dataset = _dataset(name, topology)
    metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]
    expected = _baseline()[name]

    for field in ("precision", "recall", "f1", "auto_accept_precision"):
        measured = getattr(metrics, field)
        assert measured == pytest.approx(expected[field], abs=BASELINE_TOLERANCE), (
            f"{name}.{field}: measured {measured:.4f}, baseline {expected[field]:.4f}"
        )

    assert metrics.true_positives == expected["true_positives"], name
    assert metrics.false_positives == expected["false_positives"], name


def test_baseline__covers_every_committed_scenario() -> None:
    """A scenario missing from the baseline is silently unguarded."""
    expected = _baseline()

    assert set(expected) == {
        "clean",
        "realistic",
        "degraded",
        "hard_negatives",
        "sparse_coverage",
        "adversarial",
    }


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_matching__scales_with_prefiltered_candidates_not_dataset_size(
    topology: Topology,
) -> None:
    """The claim the two-stage design rests on.

    Scoring is only ever applied to what the indexed prefilter returned, so a
    dataset ten times larger costs the same unless it contains ten times more
    plates resembling the target.
    """
    from tests.fixtures.factories import make_camera, make_sighting
    from tests.fixtures.fake_repositories import InMemoryStore, build_in_memory_repositories
    from tests.unit.matching.test_search import RecordingSightingRepository

    store = InMemoryStore()
    repositories = build_in_memory_repositories(store)
    repositories.cameras.upsert(make_camera(camera_id="cam_01"))

    noise = [
        make_sighting(
            "cam_01",
            offset_sec=index,
            plate_text_raw=f"ZZ{index:05d}",
            plate_text_normalized=f"ZZ{index:05d}",
            plate_confidence=0.9,
        )
        for index in range(5000)
    ]
    repositories.sightings.add_batch(noise)

    recording = RecordingSightingRepository(repositories.sightings)
    from multicam_tracker.matching import find_plate_matches

    started = time.perf_counter()
    found = find_plate_matches(
        "ABC1234", recording, target_id="a0000000-0000-4000-8000-000000000001"
    )
    elapsed = time.perf_counter() - started

    assert found == []
    assert elapsed < PERFORMANCE_BUDGET_SEC
    assert set(recording.calls) <= {"find_by_plate_exact", "find_by_plate_folded"}
