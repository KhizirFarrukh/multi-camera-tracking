"""Measure matching against the committed synthetic scenarios.

Modes:

* default -- score plate matching on every scenario at the configured
  thresholds and print a table
* ``--sweep`` -- vary the fuzzy distance cutoff and the review floor, and print
  the precision/recall surface
* ``--reid`` -- the same for the stage 07 embedding path, reporting the
  plate-failure recovery that justifies it
* ``--compare-constraint`` -- constrained versus unconstrained retrieval at
  identical thresholds, which is the measurement behind the design claim

The sweep is how the thresholds in ``config/thresholds.yaml`` were chosen.
Picking them by intuition is exactly what stage 06 forbids, because the
intuitive value and the measured one differ substantially: an arbitrary
single-character substitution and a genuine OCR confusion are both "distance 1",
and only the weighted distance tells them apart.

Examples::

    python scripts/evaluate_matching.py
    python scripts/evaluate_matching.py --sweep
    python scripts/evaluate_matching.py --write-baseline
    python scripts/evaluate_matching.py --reid
    python scripts/evaluate_matching.py --reid --sweep
    python scripts/evaluate_matching.py --reid --write-baseline
    python scripts/evaluate_matching.py --compare-constraint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_tracker.logging_config import configure_logging  # noqa: E402
from multicam_tracker.matching import (  # noqa: E402
    evaluate_plate_matching,
    evaluate_reid_matching,
)
from multicam_tracker.synth import generate_from_file  # noqa: E402
from multicam_tracker.topology import Topology, load_topology  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_DIR = REPO_ROOT / "tests" / "integration" / "matching" / "baselines"
BASELINE_FILE = BASELINE_DIR / "plate_metrics_baseline.json"
REID_BASELINE_FILE = BASELINE_DIR / "embedding_metrics_baseline.json"
SCENARIOS = [
    "clean",
    "realistic",
    "degraded",
    "hard_negatives",
    "sparse_coverage",
    "adversarial",
]

DISTANCE_GRID = [0.4, 0.5, 0.6, 0.9, 1.0, 1.1, 1.5, 2.0]
REVIEW_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]

SCENARIO_COL = "scenario"
"""Shared column heading, so the three tables line up under one another."""

SIMILARITY_GRID = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97]
MARGIN_GRID = [0.0, 0.02, 0.04, 0.08]
REVIEW_SIMILARITY_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]


def _datasets(topology: Topology) -> dict[str, object]:
    """Generate every committed scenario once.

    Args:
        topology: Pre-loaded graph, so the file is parsed once.

    Returns:
        Scenario name to generated dataset.
    """
    return {
        name: generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        for name in SCENARIOS
    }


def report(datasets: dict[str, object]) -> dict[str, dict[str, object]]:
    """Score every scenario at the configured thresholds and print a table.

    Args:
        datasets: Generated datasets by scenario name.

    Returns:
        Scenario name to serialized metrics.
    """
    header = (
        f"{'scenario':<16} {'precision':>9} {'recall':>7} {'f1':>6} "
        f"{'auto-P':>7} {'TP':>4} {'FP':>4} {'FN':>4}  misses"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, object]] = {}
    for name, dataset in datasets.items():
        metrics = evaluate_plate_matching(dataset, dataset.ground_truth)  # type: ignore[attr-defined]
        results[name] = metrics.to_json_dict()
        misses = ", ".join(f"{k}={v}" for k, v in sorted(metrics.misses_by_corruption.items()))
        print(
            f"{name:<16} {metrics.precision:>9.3f} {metrics.recall:>7.3f} "
            f"{metrics.f1:>6.3f} {metrics.auto_accept_precision:>7.3f} "
            f"{metrics.true_positives:>4} {metrics.false_positives:>4} "
            f"{metrics.false_negatives:>4}  {misses}"
        )
    return results


def sweep(datasets: dict[str, object]) -> None:
    """Print the precision/recall surface over the threshold grid.

    Precision is reported for ``realistic`` because that is the scenario the
    documented target is stated against, and recall for ``degraded`` because
    that is where a too-strict cutoff shows up first.

    Args:
        datasets: Generated datasets by scenario name.
    """
    print(
        f"{'max_weighted':>12} {'review_min':>10} "
        f"{'realistic P':>12} {'realistic R':>12} "
        f"{'degraded P':>11} {'degraded R':>11} "
        f"{'hardneg autoFP':>15}"
    )
    print("-" * 88)

    for distance in DISTANCE_GRID:
        for review in REVIEW_GRID:
            realistic = evaluate_plate_matching(
                datasets["realistic"],  # type: ignore[arg-type]
                datasets["realistic"].ground_truth,  # type: ignore[attr-defined]
                review_min=review,
                max_weighted_distance=distance,
            )
            degraded = evaluate_plate_matching(
                datasets["degraded"],  # type: ignore[arg-type]
                datasets["degraded"].ground_truth,  # type: ignore[attr-defined]
                review_min=review,
                max_weighted_distance=distance,
            )
            hard = evaluate_plate_matching(
                datasets["hard_negatives"],  # type: ignore[arg-type]
                datasets["hard_negatives"].ground_truth,  # type: ignore[attr-defined]
                review_min=review,
                max_weighted_distance=distance,
            )
            print(
                f"{distance:>12.2f} {review:>10.2f} "
                f"{realistic.precision:>12.3f} {realistic.recall:>12.3f} "
                f"{degraded.precision:>11.3f} {degraded.recall:>11.3f} "
                f"{hard.auto_accepted_false:>15}"
            )


def reid_report(datasets: dict[str, object], topology: Topology) -> dict[str, dict[str, object]]:
    """Score the embedding path on every scenario and print a table.

    Args:
        datasets: Generated datasets by scenario name.
        topology: The graph, used to constrain each search.

    Returns:
        Scenario name to serialized metrics.
    """
    header = (
        f"{SCENARIO_COL:<16} {'embP':>6} {'FPrate':>7} {'autoFP':>7} "
        f"{'combP':>6} {'combR':>6} {'recovered':>10} {'cmc1':>5} {'cmc5':>5}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, object]] = {}
    for name, dataset in datasets.items():
        metrics = evaluate_reid_matching(
            dataset,  # type: ignore[arg-type]
            dataset.ground_truth,  # type: ignore[attr-defined]
            topology=topology,
        )
        results[name] = metrics.to_json_dict()
        recovered = f"{metrics.plate_failed_recovered}/{metrics.plate_failed_total}"
        print(
            f"{name:<16} {metrics.embedding_precision:>6.3f} "
            f"{metrics.embedding_false_positive_rate:>7.3f} "
            f"{metrics.embedding_auto_accepted_false:>7} "
            f"{metrics.combined_precision:>6.3f} {metrics.combined_recall:>6.3f} "
            f"{recovered:>10} {metrics.cmc.get(1, 0.0):>5.2f} {metrics.cmc.get(5, 0.0):>5.2f}"
        )
    return results


def reid_sweep(datasets: dict[str, object], topology: Topology) -> None:
    """Print the similarity/margin surface on the two decisive scenarios.

    ``degraded`` is where a too-strict threshold destroys the recovery this
    stage exists for, and ``hard_negatives`` is where a too-loose one
    auto-accepts a decoy. A threshold has to be read off both at once.

    Args:
        datasets: Generated datasets by scenario name.
        topology: The graph, used to constrain each search.
    """
    header = (
        f"{'accept':>7} {'margin':>7} "
        f"{'degradedR':>10} {'recov':>6} "
        f"{'hardnegP':>9} {'hardnegFP':>10} {'hardnegAutoFP':>14} {'hardnegDown':>12} "
        f"{'cleanDown':>10}"
    )
    print(header)
    print("-" * len(header))

    for accept in SIMILARITY_GRID:
        for margin in MARGIN_GRID:
            degraded = evaluate_reid_matching(
                datasets["degraded"],  # type: ignore[arg-type]
                datasets["degraded"].ground_truth,  # type: ignore[attr-defined]
                auto_accept_similarity=accept,
                margin_min=margin,
                topology=topology,
            )
            hard = evaluate_reid_matching(
                datasets["hard_negatives"],  # type: ignore[arg-type]
                datasets["hard_negatives"].ground_truth,  # type: ignore[attr-defined]
                auto_accept_similarity=accept,
                margin_min=margin,
                topology=topology,
            )
            # ``clean`` is the only scenario whose similarities reach the
            # auto-accept band at every setting, so it is where the margin rule
            # can be seen firing at all.
            clean = evaluate_reid_matching(
                datasets["clean"],  # type: ignore[arg-type]
                datasets["clean"].ground_truth,  # type: ignore[attr-defined]
                auto_accept_similarity=accept,
                margin_min=margin,
                topology=topology,
            )
            recovered = f"{degraded.plate_failed_recovered}/{degraded.plate_failed_total}"
            print(
                f"{accept:>7.2f} {margin:>7.2f} "
                f"{degraded.embedding_recall:>10.3f} {recovered:>6} "
                f"{hard.embedding_precision:>9.3f} "
                f"{hard.embedding_false_positive_rate:>10.3f} "
                f"{hard.embedding_auto_accepted_false:>14} "
                f"{hard.margin_downgrades:>12} {clean.margin_downgrades:>10}"
            )


def reid_review_floor_sweep(datasets: dict[str, object], topology: Topology) -> None:
    """Print what the review floor costs and buys.

    The floor decides what a human ever sees. Too low and the queue fills with
    noise; too high and the plate-failure recovery this stage exists for
    disappears before anyone can look at it.

    Args:
        datasets: Generated datasets by scenario name.
        topology: The graph, used to constrain each search.
    """
    header = (
        f"{'review':>7} {'degradedR':>10} {'recov':>6} "
        f"{'hardnegP':>9} {'hardnegR':>9} {'hardnegAutoFP':>14}"
    )
    print(header)
    print("-" * len(header))

    for review in REVIEW_SIMILARITY_GRID:
        degraded = evaluate_reid_matching(
            datasets["degraded"],  # type: ignore[arg-type]
            datasets["degraded"].ground_truth,  # type: ignore[attr-defined]
            review_similarity=review,
            topology=topology,
        )
        hard = evaluate_reid_matching(
            datasets["hard_negatives"],  # type: ignore[arg-type]
            datasets["hard_negatives"].ground_truth,  # type: ignore[attr-defined]
            review_similarity=review,
            topology=topology,
        )
        recovered = f"{degraded.plate_failed_recovered}/{degraded.plate_failed_total}"
        print(
            f"{review:>7.2f} {degraded.embedding_recall:>10.3f} {recovered:>6} "
            f"{hard.embedding_precision:>9.3f} {hard.embedding_recall:>9.3f} "
            f"{hard.embedding_auto_accepted_false:>14}"
        )


def constraint_comparison(datasets: dict[str, object], topology: Topology) -> None:
    """Print constrained versus unconstrained retrieval, scenario by scenario.

    Both columns run at identical thresholds, so the only difference is the
    topology. This is what turns "restricting the search space beats raising the
    threshold" from an assertion into a measurement.

    Args:
        datasets: Generated datasets by scenario name.
        topology: The graph.
    """
    header = (
        f"{SCENARIO_COL:<16} {'unconP':>7} {'conP':>6} "
        f"{'uncon autoFP':>13} {'con autoFP':>11} {'unconR':>7} {'conR':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, dataset in datasets.items():
        unconstrained = evaluate_reid_matching(
            dataset,  # type: ignore[arg-type]
            dataset.ground_truth,  # type: ignore[attr-defined]
            topology=None,
        )
        constrained = evaluate_reid_matching(
            dataset,  # type: ignore[arg-type]
            dataset.ground_truth,  # type: ignore[attr-defined]
            topology=topology,
        )
        print(
            f"{name:<16} {unconstrained.embedding_precision:>7.3f} "
            f"{constrained.embedding_precision:>6.3f} "
            f"{unconstrained.embedding_auto_accepted_false:>13} "
            f"{constrained.embedding_auto_accepted_false:>11} "
            f"{unconstrained.embedding_recall:>7.3f} {constrained.embedding_recall:>6.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation.

    Args:
        argv: Command-line arguments.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="Print the threshold sweep")
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Evaluate the stage 07 embedding path instead of plate matching",
    )
    parser.add_argument(
        "--compare-constraint",
        action="store_true",
        help="Print constrained versus unconstrained embedding retrieval",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite the committed regression baseline with the current numbers",
    )
    args = parser.parse_args(argv)

    configure_logging(json_output=False, level="ERROR")
    topology = load_topology(REPO_ROOT / "config" / "topology.yaml")
    datasets = _datasets(topology)

    if args.compare_constraint:
        constraint_comparison(datasets, topology)
        return 0

    if args.reid:
        if args.sweep:
            reid_sweep(datasets, topology)
            print()
            reid_review_floor_sweep(datasets, topology)
            return 0
        results = reid_report(datasets, topology)
        target_file = REID_BASELINE_FILE
    else:
        if args.sweep:
            sweep(datasets)
            return 0
        results = report(datasets)
        target_file = BASELINE_FILE

    if args.write_baseline:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nbaseline -> {target_file}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
