"""Estimate each camera's clock offset from reference vehicle passes.

The operator procedure this supports: drive a vehicle with a known plate along a
known route past several cameras, a few times, then run this. It reports what
each camera's clock appears to be doing and how much to trust that estimate.

It **reports**; it does not apply anything. Correcting a clock rewrites every
timestamp that camera ever recorded and invalidates the routes built from them,
which is a decision an operator makes deliberately after reading these numbers
-- not a side effect of running a diagnostic.

Modes:

* default -- estimate from a scenario's traffic and print a table
* ``--drift`` -- report the drift analysis per camera, which distinguishes a
  constant offset (correct it once) from a clock that is still drifting (fix the
  camera's time client)

Examples::

    python scripts/estimate_camera_offsets.py
    python scripts/estimate_camera_offsets.py --scenario adversarial --drift
    python scripts/estimate_camera_offsets.py --reference-camera cam_01
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_tracker.exceptions import ValidationError  # noqa: E402
from multicam_tracker.logging_config import configure_logging  # noqa: E402
from multicam_tracker.synth import generate_from_file  # noqa: E402
from multicam_tracker.timesync import ReferencePass, detect_drift, estimate_offsets  # noqa: E402
from multicam_tracker.topology import Topology, load_topology  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"


def reference_passes(dataset: Any, topology: Topology) -> list[ReferencePass]:
    """Build reference passes from vehicles whose route is known.

    Route order comes from what the vehicle actually did, not from the reported
    timestamps. Re-sorting by reported time would let a drifting camera reorder
    the very legs being measured, and the drift would hide inside its own
    symptom.

    Args:
        dataset: The generated dataset, standing in for a real reference drive.
        topology: The camera graph, for the expected transit.

    Returns:
        One pass per consecutive camera pair per vehicle.
    """
    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}

    passes: list[ReferencePass] = []
    for vehicle in dataset.ground_truth.vehicles:
        route = [by_id[sid] for sid in vehicle.sighting_ids if sid in by_id]
        for origin, destination in pairwise(route):
            link = topology.get_link(origin.camera_id, destination.camera_id)
            if link is None:
                continue
            passes.append(
                ReferencePass(
                    from_camera_id=origin.camera_id,
                    to_camera_id=destination.camera_id,
                    departure_utc=origin.timestamp_utc,
                    arrival_utc=destination.timestamp_utc,
                    expected_sec=(link.min_travel_time_sec + link.max_travel_time_sec) / 2,
                    vehicle_id=vehicle.vehicle_id,
                )
            )
    return passes


def report_offsets(
    passes: list[ReferencePass], topology: Topology, reference_camera: str | None
) -> int:
    """Estimate and print each camera's offset.

    Args:
        passes: The reference passes.
        topology: The camera graph.
        reference_camera: Camera to pin at zero, or ``None`` for the
            best-constrained one.

    Returns:
        ``0`` on success, ``2`` when the system is too underdetermined to answer.
    """
    try:
        estimation = estimate_offsets(
            passes, topology, reference_camera_id=reference_camera, min_reference_passes=2
        )
    except ValidationError as exc:
        print(f"cannot estimate: {exc}")
        return 2

    header = f"{'camera':<10} {'correction (ms)':>16} {'95% interval':>26} {'passes':>7}"
    print(header)
    print("-" * len(header))

    for camera_id in sorted(estimation.estimates):
        estimate = estimation.estimates[camera_id]
        low, high = estimate.confidence_interval_ms
        marker = "  (reference)" if estimate.is_reference else ""
        print(
            f"{camera_id:<10} {estimate.offset_ms:>16.0f} "
            f"{f'{low:+.0f} to {high:+.0f}':>26} {estimate.pass_count:>7}{marker}"
        )

    if estimation.excluded_camera_ids:
        print(
            f"\nnot estimated (too few reference passes): "
            f"{', '.join(sorted(estimation.excluded_camera_ids))}"
        )
    if estimation.outlier_pass_count:
        print(
            f"{estimation.outlier_pass_count} pass(es) disagreed with their link's median "
            f"and did not affect the estimate"
        )

    print(
        "\nNothing has been changed. Apply a correction with the offset column, "
        "which rewrites that camera's stored timestamps and flags the routes built "
        "from them."
    )
    return 0


def report_drift(passes: list[ReferencePass], topology: Topology) -> int:
    """Print the drift analysis per camera.

    Args:
        passes: The reference passes.
        topology: The camera graph, for the camera list.

    Returns:
        ``0`` on success.
    """
    header = f"{'camera':<10} {'mean (ms)':>11} {'rate (ms/h)':>12} {'samples':>8}  alert"
    print(header)
    print("-" * (len(header) + 30))

    for camera_id in sorted(topology.camera_ids):
        analysis = detect_drift(camera_id, passes)
        if analysis.sample_count == 0:
            continue
        verdict = analysis.alert.kind if analysis.alert else "-"
        print(
            f"{camera_id:<10} {analysis.mean_offset_ms:>11.0f} "
            f"{analysis.rate_ms_per_hour:>12.0f} {analysis.sample_count:>8}  {verdict}"
        )
        if analysis.alert is not None:
            print(f"           {analysis.alert.describe()}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the estimator.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="adversarial",
        help="Committed scenario to read reference passes from",
    )
    parser.add_argument(
        "--reference-camera",
        default=None,
        help="Camera pinned to zero. Defaults to the best-constrained one.",
    )
    parser.add_argument(
        "--drift", action="store_true", help="Report the drift analysis instead of the offsets"
    )
    args = parser.parse_args(argv)

    configure_logging(json_output=False, level="ERROR")
    topology = load_topology(REPO_ROOT / "config" / "topology.yaml")

    scenario_file = SCENARIO_DIR / f"{args.scenario}.yaml"
    if not scenario_file.exists():
        print(f"unknown scenario {args.scenario!r}: {scenario_file} does not exist")
        return 2

    dataset = generate_from_file(scenario_file, topology=topology)
    passes = reference_passes(dataset, topology)
    print(f"{len(passes)} reference passes from scenario {args.scenario!r}\n")

    if args.drift:
        return report_drift(passes, topology)
    return report_offsets(passes, topology, args.reference_camera)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
