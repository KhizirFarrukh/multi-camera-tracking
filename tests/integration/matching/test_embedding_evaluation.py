"""Measured evaluation of the embedding path against synthetic ground truth.

Stage 06 measured what a plate can do. This measures what appearance recovers on
top of it, and the number that justifies the whole stage is
``plate_failed_recovered``: sightings whose plate was unreadable that appearance
found anyway.

Two things here are deliberately not what the stage prompt anticipated, and both
are recorded rather than smoothed over:

* The false-positive *rate* target is stated per scenario class. On data with no
  visual decoys it is 0.000; on ``hard_negatives`` 21 decoys sit inside the
  true-match similarity band (0.9347-0.9423 against 0.9379-0.9423), so no
  threshold separates them and the meaningful target there is that **none is
  auto-accepted**.
* Combined evidence scores below plate-only evidence on exactly two sightings
  across every scenario, and both belong to the cloned-plate vehicle. That is
  the disagreement rule working: a strong plate the appearance contradicts is
  forced to human review, which necessarily means dropping the score.

Every number is asserted against the committed baseline as well, so a change
that quietly degrades matching fails CI rather than going unnoticed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from multicam_tracker.matching import (
    combine_evidence,
    evaluate_plate_matching,
    evaluate_reid_matching,
)
from multicam_tracker.matching.aggregation import aggregate_similarity
from multicam_tracker.matching.plate_match import PlateMatchMethod, classify_plate_match
from multicam_tracker.matching.scoring import score_plate_match
from multicam_tracker.models import ReviewStatus
from multicam_tracker.synth import GroundTruth, generate_from_file
from multicam_tracker.topology import Topology, load_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_FILE = Path(__file__).parent / "baselines" / "embedding_metrics_baseline.json"

SCENARIOS = ("clean", "realistic", "degraded", "hard_negatives", "sparse_coverage", "adversarial")

UNDECOYED = ("clean", "realistic", "degraded", "sparse_coverage")
"""Scenarios containing no vehicle built to resemble the target."""

DECOYED = ("hard_negatives", "adversarial")
"""Scenarios whose decoys are drawn deliberately close in embedding space."""

UNDECOYED_FP_RATE_CEILING = 0.01
"""Documented target from docs/REID_MATCHING.md. Measured 0.000."""

BASELINE_TOLERANCE = 0.02
PERFORMANCE_BUDGET_SEC = 60.0
"""Generous by design: this catches an accidental quadratic, not slow hardware."""


@pytest.fixture(scope="module")
def topology() -> Topology:
    """Return the shipped topology.

    Returns:
        The graph the scenarios run over.
    """
    return load_topology(REPO_ROOT / "config" / "topology.yaml")


@pytest.fixture(scope="module")
def datasets(topology: Topology) -> dict[str, Any]:
    """Generate every committed scenario once.

    Generation is deterministic but not free, and every test in this module
    wants the same data.

    Args:
        topology: Pre-loaded graph.

    Returns:
        Scenario name to generated dataset.
    """
    return {
        name: generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        for name in SCENARIOS
    }


def _metrics(datasets: dict[str, Any], name: str, topology: Topology | None, **kwargs: Any) -> Any:
    """Score one scenario.

    Args:
        datasets: The generated datasets.
        name: Scenario to score.
        topology: Graph to constrain with, or ``None`` for the unconstrained
            baseline.
        **kwargs: Threshold overrides.

    Returns:
        The metrics.
    """
    dataset = datasets[name]
    return evaluate_reid_matching(dataset, dataset.ground_truth, topology=topology, **kwargs)


def _baseline() -> dict[str, dict[str, Any]]:
    """Load the committed regression baseline.

    Returns:
        Scenario name to recorded metrics.
    """
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ranking quality
# ---------------------------------------------------------------------------


def test_clean__rank_one_accuracy_is_perfect(datasets: dict[str, Any], topology: Topology) -> None:
    """With undegraded images the true match must be the single best candidate."""
    metrics = _metrics(datasets, "clean", topology)

    assert metrics.cmc[1] == 1.0
    assert metrics.embedding_precision == 1.0


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scenario__the_true_match_appears_within_the_first_five(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """Ranking, not thresholding, is what a human review queue depends on.

    Even where precision is poor because decoys are interleaved, the operator
    sees a true match at the top of the page.
    """
    metrics = _metrics(datasets, name, topology)

    assert metrics.cmc[1] == 1.0, name
    assert metrics.cmc[5] == 1.0, name


# ---------------------------------------------------------------------------
# The recovery that justifies the stage
# ---------------------------------------------------------------------------


def test_degraded__combined_recall_exceeds_plate_only_recall(
    datasets: dict[str, Any], topology: Topology
) -> None:
    """The whole point, quantified.

    Plate matching cannot find a sighting whose plate was never readable.
    Appearance can, and here it recovers every one of them.
    """
    dataset = datasets["degraded"]
    plate_only = evaluate_plate_matching(dataset, dataset.ground_truth)
    combined = _metrics(datasets, "degraded", topology)

    assert combined.combined_recall > plate_only.recall
    assert combined.plate_failed_total == 3
    assert combined.plate_failed_recovered == 3
    assert combined.plate_failure_recovery_rate == 1.0


@pytest.mark.parametrize("name", ["degraded", "sparse_coverage", "hard_negatives", "adversarial"])
def test_every_plate_failure__is_recovered_by_appearance(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """If this number were zero, re-id would be complexity with no return."""
    metrics = _metrics(datasets, name, topology)

    assert metrics.plate_failed_total > 0, f"{name} exercises no plate failure"
    assert metrics.plate_failed_recovered == metrics.plate_failed_total


# ---------------------------------------------------------------------------
# Safety: what reaches an operator without a human in the loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIOS)
def test_no_scenario__auto_accepts_a_false_positive(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """The exit criterion. An auto-accepted false positive reaches an operator as fact."""
    assert _metrics(datasets, name, topology).embedding_auto_accepted_false == 0, name


@pytest.mark.parametrize("name", UNDECOYED)
def test_scenarios_without_decoys__meet_the_documented_false_positive_target(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """Where nothing resembles the target, nothing wrong may be returned at all."""
    metrics = _metrics(datasets, name, topology)

    assert metrics.embedding_false_positive_rate <= UNDECOYED_FP_RATE_CEILING, name
    assert metrics.embedding_recall == 1.0, name


def test_hard_negatives__decoys_are_returned_for_review_rather_than_accepted(
    datasets: dict[str, Any], topology: Topology
) -> None:
    """Documents the honest limit of appearance evidence.

    These decoys sit inside the true-match similarity band, so precision here is
    genuinely poor and no threshold fixes it. What the configuration guarantees
    is that every one of them is queued for a human instead of asserted.
    """
    metrics = _metrics(datasets, "hard_negatives", topology)

    assert metrics.embedding_false_positives > 0, "the scenario is not adversarial any more"
    assert metrics.embedding_auto_accepted_false == 0
    assert metrics.embedding_recall == 1.0


def test_hard_negatives__the_margin_rule_demotes_the_leader_when_a_decoy_is_close_behind(
    datasets: dict[str, Any], topology: Topology
) -> None:
    """The margin rule, exercised at a threshold where auto-accept actually fires.

    At the shipped 0.95 nothing on this scenario reaches auto-accept, so the rule
    is dormant. Dropped to 0.92 the leader qualifies -- and a decoy 0.003 behind
    it makes that a coin flip, which the rule refuses to call.
    """
    permissive = _metrics(
        datasets, "hard_negatives", topology, auto_accept_similarity=0.92, margin_min=0.0
    )
    guarded = _metrics(
        datasets, "hard_negatives", topology, auto_accept_similarity=0.92, margin_min=0.04
    )

    assert permissive.margin_downgrades == 0
    assert guarded.margin_downgrades == 1


def test_hard_negatives__the_shipped_threshold_is_what_removes_the_auto_accepted_decoys(
    datasets: dict[str, Any], topology: Topology
) -> None:
    """Cites the calibration. 0.92 auto-accepts 21 decoys; 0.95 auto-accepts none."""
    contract_prior = _metrics(datasets, "hard_negatives", topology, auto_accept_similarity=0.92)
    measured = _metrics(datasets, "hard_negatives", topology, auto_accept_similarity=0.95)

    assert contract_prior.embedding_auto_accepted_false == 21
    assert measured.embedding_auto_accepted_false == 0
    assert measured.embedding_recall == contract_prior.embedding_recall


# ---------------------------------------------------------------------------
# The topology constraint, measured against its absence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DECOYED)
def test_constrained_retrieval__beats_unconstrained_precision_where_decoys_exist(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """The design claim, as a measurement.

    Both runs use identical thresholds, so the only difference is the topology.
    A decoy that looks identical to the target is harmless if it was never
    anywhere the target could have been.
    """
    unconstrained = _metrics(datasets, name, None)
    constrained = _metrics(datasets, name, topology)

    assert constrained.embedding_precision > unconstrained.embedding_precision, name
    assert constrained.embedding_recall == unconstrained.embedding_recall, name


@pytest.mark.parametrize("name", ["sparse_coverage", "realistic", "degraded"])
def test_constrained_retrieval__searches_a_smaller_space_at_equal_recall(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """What the constraint buys where there are no lookalikes to remove.

    Precision is already perfect on these scenarios, so the constraint cannot
    improve it. What it does improve is how much the query has to examine --
    which is the difference between a query and an outage at production volume.
    """
    unconstrained = _metrics(datasets, name, None)
    constrained = _metrics(datasets, name, topology)

    assert constrained.candidates_considered < unconstrained.candidates_considered, name
    assert constrained.embedding_recall == unconstrained.embedding_recall, name


@pytest.mark.parametrize("name", SCENARIOS)
def test_constrained_retrieval__never_costs_recall(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """The constraint must remove opportunities to be wrong, not true matches."""
    assert (
        _metrics(datasets, name, topology).embedding_recall
        >= _metrics(datasets, name, None).embedding_recall
    ), name


# ---------------------------------------------------------------------------
# Combined evidence versus plate evidence alone
# ---------------------------------------------------------------------------


def _combined_versus_plate(dataset: Any, truth: GroundTruth) -> list[tuple[str, float, float, Any]]:
    """Score every sighting both ways.

    Args:
        dataset: The generated sightings.
        truth: The answer key, for the target plate and its reference sighting.

    Returns:
        ``(sighting_id, plate_only_score, combined_score, verdict)`` per sighting.
    """
    target = truth.target
    assert target is not None
    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    embedded = [by_id[sid] for sid in target.sighting_ids if sid in by_id and by_id[sid].embedding]
    references = [list(embedded[0].embedding)]

    rows: list[tuple[str, float, float, Any]] = []
    for sighting in dataset.sightings:
        if sighting.sighting_id == embedded[0].sighting_id:
            continue
        plate_result = (
            classify_plate_match(sighting.plate_text_normalized, target.true_plate)
            if sighting.plate_text_normalized
            else None
        )
        similarity = (
            aggregate_similarity(sighting.embedding, references)
            if sighting.embedding is not None
            else None
        )
        verdict = combine_evidence(plate_result, sighting.plate_confidence, similarity)
        plate_only = (
            score_plate_match(plate_result, sighting.plate_confidence)
            if plate_result is not None and plate_result.method is not PlateMatchMethod.NO_MATCH
            else 0.0
        )
        rows.append((sighting.sighting_id, plate_only, verdict.score, verdict))
    return rows


@pytest.mark.parametrize("name", SCENARIOS)
def test_combined_evidence__never_scores_below_plate_only_unless_the_signals_conflict(
    datasets: dict[str, Any], name: str
) -> None:
    """The fallback never hurts -- except where it is supposed to.

    The stage prompt states the property without exception. The disagreement
    rule from 7.5 is the exception, and it is the more important of the two: a
    strong plate that appearance contradicts must go to a human, and forcing
    review means dropping below the auto-accept threshold. Every sighting where
    combined scores lower is checked to be exactly that case.
    """
    dataset = datasets[name]
    for sighting_id, plate_only, combined, verdict in _combined_versus_plate(
        dataset, dataset.ground_truth
    ):
        if combined + 1e-12 < plate_only:
            assert verdict.conflicting, f"{name}/{sighting_id} lost score without a conflict"
            assert verdict.review_status is ReviewStatus.PENDING_REVIEW


def test_adversarial__appearance_catches_the_cloned_plate(datasets: dict[str, Any]) -> None:
    """Re-id earning its place a second way.

    A clone wears the target plate perfectly, so plate matching alone cannot
    separate them. Appearance disagrees, and the disagreement is what routes the
    clone to a human instead of an operator being told it is the target.
    """
    dataset = datasets["adversarial"]
    truth = dataset.ground_truth
    conflicting = [
        (sighting_id, verdict)
        for sighting_id, _plate, _combined, verdict in _combined_versus_plate(dataset, truth)
        if verdict.conflicting
    ]

    assert conflicting, "the clone produced no disagreement"
    owners = {truth.vehicle_for(sighting_id) for sighting_id, _ in conflicting}
    assert owners == {"clone_00"}
    assert all(verdict.review_status is ReviewStatus.PENDING_REVIEW for _, verdict in conflicting)
    assert all("does not say which" in verdict.disagreement for _, verdict in conflicting)


@pytest.mark.parametrize("name", UNDECOYED)
def test_scenarios_without_decoys__produce_no_disagreements_at_all(
    datasets: dict[str, Any], name: str
) -> None:
    """A conflict must mean something. Routine data must not manufacture them."""
    dataset = datasets[name]
    rows = _combined_versus_plate(dataset, dataset.ground_truth)

    assert not [row for row in rows if row[3].conflicting], name


# ---------------------------------------------------------------------------
# Provenance and regression guard
# ---------------------------------------------------------------------------


def test_metrics__are_computed_against_the_ground_truth_file_not_a_rederived_one(
    datasets: dict[str, Any], topology: Topology, tmp_path: Path
) -> None:
    """The artifact later stages read is the artifact scored against."""
    dataset = datasets["realistic"]
    path = dataset.ground_truth.write_json(tmp_path / "truth.json")

    from_object = evaluate_reid_matching(dataset, dataset.ground_truth, topology=topology)
    from_disk = evaluate_reid_matching(dataset, GroundTruth.read_json(path), topology=topology)

    assert from_disk.to_json_dict() == from_object.to_json_dict()


@pytest.mark.parametrize("name", SCENARIOS)
def test_metrics__match_the_committed_baseline(
    datasets: dict[str, Any], topology: Topology, name: str
) -> None:
    """A quality regression must fail CI rather than be noticed later, or never."""
    recorded = _baseline()[name]
    current = _metrics(datasets, name, topology).to_json_dict()

    for key, expected in recorded.items():
        actual = current[key]
        if isinstance(expected, float):
            assert abs(actual - expected) <= BASELINE_TOLERANCE, f"{name}.{key}"
        else:
            assert actual == expected, f"{name}.{key}"


def test_baseline__covers_every_scenario_the_suite_evaluates() -> None:
    """A scenario missing from the baseline would be silently unguarded."""
    assert set(_baseline()) == set(SCENARIOS)


def test_evaluation__completes_within_the_documented_budget(topology: Topology) -> None:
    """Catches an accidental quadratic in the search space, not slow hardware."""
    started = time.perf_counter()
    for name in SCENARIOS:
        dataset = generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        evaluate_reid_matching(dataset, dataset.ground_truth, topology=topology)
    elapsed = time.perf_counter() - started

    assert elapsed < PERFORMANCE_BUDGET_SEC, f"took {elapsed:.1f}s"
