"""Measure trajectory reconstruction against the committed synthetic scenarios.

Modes:

* default -- reconstruct every scenario at the configured objective and print a
  table
* ``--sweep`` -- vary the per-node inclusion bonus and the gap-edge penalty, and
  print what each setting does to precision, recall, and the ambiguity flags
* ``--explain SCENARIO`` -- print one reconstruction in full: the route, the
  hops, the gaps, and why the top rejected candidates were rejected

The sweep is how the values in ``config/thresholds.yaml`` were chosen, and it is
worth re-reading before changing them: the intuitive setting for the inclusion
bonus turns out to make the engine confidently wrong on adversarial data.

Examples::

    python scripts/evaluate_pathing.py
    python scripts/evaluate_pathing.py --sweep
    python scripts/evaluate_pathing.py --explain adversarial
    python scripts/evaluate_pathing.py --write-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_tracker.logging_config import configure_logging  # noqa: E402
from multicam_tracker.pathing import (  # noqa: E402
    candidates_from_matches,
    evaluate_pathing,
    reconstruct_trajectory,
)
from multicam_tracker.synth import generate_from_file  # noqa: E402
from multicam_tracker.topology import Topology, load_topology  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
BASELINE_FILE = (
    REPO_ROOT / "tests" / "integration" / "pathing" / "baselines" / "pathing_metrics_baseline.json"
)
SCENARIOS = [
    "clean",
    "realistic",
    "degraded",
    "hard_negatives",
    "sparse_coverage",
    "adversarial",
]

TARGET_ID = "a0000000-0000-4000-8000-000000000001"

BONUS_GRID = [0.0, 0.02, 0.05, 0.15, 0.25]
GAP_PENALTY_GRID = [0.02, 0.05, 0.08, 0.15, 0.35, 0.50]


def _prepared(topology: Topology) -> dict[str, tuple[Any, list[Any]]]:
    """Generate every scenario and run matching over it once.

    Matching is the expensive half and does not depend on the objective, so the
    sweep runs it once rather than once per grid point.

    Args:
        topology: Pre-loaded graph.

    Returns:
        Scenario name to ``(dataset, candidates)``.
    """
    prepared: dict[str, tuple[Any, list[Any]]] = {}
    for name in SCENARIOS:
        dataset = generate_from_file(SCENARIO_DIR / f"{name}.yaml", topology=topology)
        prepared[name] = (
            dataset,
            candidates_from_matches(dataset, dataset.ground_truth, TARGET_ID, topology=topology),
        )
    return prepared


def _reconstruct(
    dataset: Any,
    candidates: list[Any],
    topology: Topology,
    cameras: dict[str, Any],
    **overrides: Any,
) -> Any:
    """Reconstruct one scenario.

    Args:
        dataset: The generated sightings.
        candidates: Match candidates for the target.
        topology: The camera graph.
        cameras: Cameras by id.
        **overrides: Objective overrides.

    Returns:
        The reconstruction.
    """
    return reconstruct_trajectory(
        TARGET_ID,
        candidates,
        {sighting.sighting_id: sighting for sighting in dataset.sightings},
        topology,
        cameras=cameras,
        activity=list(dataset.sightings),
        **overrides,
    )


def report(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Score every scenario at the configured objective and print a table.

    Args:
        prepared: Datasets and candidates by scenario name.
        topology: The camera graph.
        cameras: Cameras by id.

    Returns:
        Scenario name to serialized metrics.
    """
    header = (
        f"{'scenario':<16} {'exact':>6} {'P':>6} {'R':>6} {'hopAcc':>7} "
        f"{'teleport':>9} {'gaps':>5} {'ambig':>6} {'conf':>6} {'decoys':>7}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, Any]] = {}
    for name, (dataset, candidates) in prepared.items():
        result = _reconstruct(dataset, candidates, topology, cameras)
        metrics = evaluate_pathing(
            dataset, dataset.ground_truth, topology, cameras=cameras, result=result
        )
        results[name] = metrics.to_json_dict()
        print(
            f"{name:<16} {metrics.exact_path_match!s:>6} "
            f"{metrics.sighting_precision:>6.3f} {metrics.sighting_recall:>6.3f} "
            f"{metrics.hop_accuracy:>7.3f} {metrics.teleport_hops:>9} "
            f"{metrics.gaps_reported:>5} {metrics.is_ambiguous!s:>6} "
            f"{metrics.overall_confidence:>6.3f} {metrics.decoys_included:>7}"
        )
    return results


def sweep(
    prepared: dict[str, tuple[Any, list[Any]]], topology: Topology, cameras: dict[str, Any]
) -> None:
    """Print what each objective setting does across every scenario.

    The two columns to read first are ``ambigHN`` and ``ambigADV``. A setting
    that turns either to ``False`` has made the engine confident about a case
    where two routes explain the evidence equally well, and no amount of
    precision elsewhere compensates for that.

    Args:
        prepared: Datasets and candidates by scenario name.
        topology: The camera graph.
        cameras: Cameras by id.
    """
    header = (
        f"{'bonus':>6} {'gapPen':>7} {'exact':>6} {'meanP':>6} {'meanR':>6} "
        f"{'decoys':>7} {'teleport':>9} {'ambigHN':>8} {'ambigADV':>9}"
    )
    print(header)
    print("-" * len(header))

    for bonus in BONUS_GRID:
        for penalty in GAP_PENALTY_GRID:
            exact = 0
            precisions: list[float] = []
            recalls: list[float] = []
            decoys = 0
            teleports = 0
            ambiguous: dict[str, bool] = {}

            for name, (dataset, candidates) in prepared.items():
                result = _reconstruct(
                    dataset,
                    candidates,
                    topology,
                    cameras,
                    inclusion_bonus=bonus,
                    gap_penalty=penalty,
                )
                metrics = evaluate_pathing(
                    dataset, dataset.ground_truth, topology, cameras=cameras, result=result
                )
                exact += int(metrics.exact_path_match)
                precisions.append(metrics.sighting_precision)
                recalls.append(metrics.sighting_recall)
                decoys += metrics.decoys_included
                teleports += metrics.teleport_hops
                ambiguous[name] = metrics.is_ambiguous

            print(
                f"{bonus:>6.2f} {penalty:>7.2f} {exact:>6} "
                f"{sum(precisions) / len(precisions):>6.3f} "
                f"{sum(recalls) / len(recalls):>6.3f} {decoys:>7} {teleports:>9} "
                f"{ambiguous['hard_negatives']!s:>8} {ambiguous['adversarial']!s:>9}"
            )


def explain(
    prepared: dict[str, tuple[Any, list[Any]]],
    topology: Topology,
    cameras: dict[str, Any],
    scenario: str,
) -> int:
    """Print one reconstruction in full.

    Args:
        prepared: Datasets and candidates by scenario name.
        topology: The camera graph.
        cameras: Cameras by id.
        scenario: Which scenario to explain.

    Returns:
        ``0`` on success, ``2`` when the scenario is unknown.
    """
    if scenario not in prepared:
        print(f"unknown scenario {scenario!r}; expected one of {', '.join(SCENARIOS)}")
        return 2

    dataset, candidates = prepared[scenario]
    result = _reconstruct(dataset, candidates, topology, cameras)
    if result.trajectory is None:
        print(f"{scenario}: no candidate survived preparation; there is no route to explain")
        return 0

    truth = set(dataset.ground_truth.target.sighting_ids)
    print(f"=== {scenario} ===")
    print(f"score {result.path.score:.3f}, confidence {result.trajectory.overall_confidence:.3f}")
    print(f"objective: {result.explanation.objective}")
    print(f"ambiguity: {result.ambiguity.describe()}")
    print("\nroute:")
    for record in result.explanation.included:
        mark = "true " if record.sighting_id in truth else "DECOY"
        hop = "-" if record.hop_confidence is None else f"{record.hop_confidence:.3f}"
        print(f"  [{mark}] {record.camera_id:<8} {record.timestamp_utc}  hop={hop}")
        print(f"          via {record.arrived_via}")

    if result.gaps:
        print("\ngaps:")
        for gap in result.gaps:
            print(f"  [{gap.kind.value}] {gap.describe()}")

    if result.explanation.excluded:
        print(f"\nexcluded ({result.explanation.excluded_total} in total, top by confidence):")
        for record in result.explanation.excluded:
            print(
                f"  {record.camera_id:<8} {record.timestamp_utc} "
                f"conf={record.match_confidence:.2f} [{record.reason.value}]"
            )
            print(f"          {record.detail}")

    if result.movement.hops:
        print("\nmovement:")
        for position, hop in enumerate(result.movement.hops):
            flag = "  IMPLAUSIBLE" if hop.implausible_speed else ""
            print(
                f"  hop {position}: {hop.distance_meters:>7.0f}m in {hop.elapsed_sec:>6.0f}s "
                f"= {hop.speed_kph:>6.1f} km/h heading {hop.heading}{flag}"
            )
        print(
            f"  total {result.movement.total_distance_meters:.0f}m, "
            f"average {result.movement.average_speed_kph:.1f} km/h, "
            f"net direction {result.movement.dominant_direction}, "
            f"reversals at {result.movement.reversals or 'none'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation.

    Args:
        argv: Command-line arguments.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="Print the objective sweep")
    parser.add_argument("--explain", metavar="SCENARIO", help="Explain one reconstruction in full")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite the committed regression baseline with the current numbers",
    )
    args = parser.parse_args(argv)

    configure_logging(json_output=False, level="ERROR")
    topology = load_topology(REPO_ROOT / "config" / "topology.yaml")
    cameras = dict(topology.cameras_by_id)
    prepared = _prepared(topology)

    if args.explain:
        return explain(prepared, topology, cameras, args.explain)

    if args.sweep:
        sweep(prepared, topology, cameras)
        return 0

    results = report(prepared, topology, cameras)

    if args.write_baseline:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nbaseline -> {BASELINE_FILE}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
