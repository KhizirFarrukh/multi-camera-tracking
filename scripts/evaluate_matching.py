"""Measure plate matching against the committed synthetic scenarios.

Two modes:

* default -- score every scenario at the configured thresholds and print a table
* ``--sweep`` -- vary the fuzzy distance cutoff and the review floor, and print
  the precision/recall surface

The sweep is how the thresholds in ``config/thresholds.yaml`` were chosen.
Picking them by intuition is exactly what stage 06 forbids, because the
intuitive value and the measured one differ substantially: an arbitrary
single-character substitution and a genuine OCR confusion are both "distance 1",
and only the weighted distance tells them apart.

Examples::

    python scripts/evaluate_matching.py
    python scripts/evaluate_matching.py --sweep
    python scripts/evaluate_matching.py --write-baseline
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
from multicam_tracker.matching import evaluate_plate_matching  # noqa: E402
from multicam_tracker.synth import generate_from_file  # noqa: E402
from multicam_tracker.topology import Topology, load_topology  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_FILE = (
    REPO_ROOT / "tests" / "integration" / "matching" / "baselines" / "plate_metrics_baseline.json"
)
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
        "--write-baseline",
        action="store_true",
        help="Overwrite the committed regression baseline with the current numbers",
    )
    args = parser.parse_args(argv)

    configure_logging(json_output=False, level="ERROR")
    datasets = _datasets(load_topology(REPO_ROOT / "config" / "topology.yaml"))

    if args.sweep:
        sweep(datasets)
        return 0

    results = report(datasets)

    if args.write_baseline:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nbaseline -> {BASELINE_FILE}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
