"""Generate a synthetic dataset from a scenario file.

Examples::

    # Write JSON next to the scenario
    python scripts/generate_dataset.py tests/fixtures/scenarios/realistic.yaml \\
        --out build/realistic.json

    # Override the seed and load straight into the configured database
    python scripts/generate_dataset.py tests/fixtures/scenarios/realistic.yaml \\
        --seed 42 --to-database

Loading into the database syncs the topology first, because a sighting carries a
foreign key to its camera and the cameras have to exist before the rows that
reference them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_tracker.exceptions import MulticamTrackerError  # noqa: E402
from multicam_tracker.logging_config import configure_logging, get_logger  # noqa: E402
from multicam_tracker.synth import generate_from_file, load_scenario  # noqa: E402
from multicam_tracker.topology import load_topology, sync_topology_to_db  # noqa: E402

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic sighting dataset with ground truth.",
    )
    parser.add_argument("scenario", type=Path, help="Path to a scenario YAML file")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the scenario's seed. The seed used is recorded in the ground truth.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the dataset here as JSON. Ground truth goes to <out>.truth.json.",
    )
    parser.add_argument(
        "--to-database",
        action="store_true",
        help="Sync the topology and insert the sightings into the configured database",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the human-readable summary")
    return parser


def _write_files(dataset: object, out: Path) -> tuple[Path, Path]:
    """Write the dataset and its ground truth to disk.

    Args:
        dataset: The generated dataset.
        out: Destination for the dataset JSON.

    Returns:
        ``(dataset_path, ground_truth_path)``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dataset.to_json_dict(), indent=2, sort_keys=True),  # type: ignore[attr-defined]
        encoding="utf-8",
    )
    truth_path = out.with_suffix(".truth.json")
    dataset.ground_truth.write_json(truth_path)  # type: ignore[attr-defined]
    return out, truth_path


def _load_into_database(dataset: object, scenario_topology: Path) -> int:
    """Sync the topology and insert the dataset's sightings.

    Args:
        dataset: The generated dataset.
        scenario_topology: Topology file the scenario referenced.

    Returns:
        The number of sightings inserted.
    """
    from multicam_tracker.db.repositories import (
        PostgresCameraLinkRepository,
        PostgresCameraRepository,
        PostgresSightingRepository,
    )
    from multicam_tracker.db.session import session_scope
    from multicam_tracker.synth import load_dataset_into

    with session_scope() as session:
        sync_topology_to_db(
            load_topology(scenario_topology),
            PostgresCameraRepository(session),
            PostgresCameraLinkRepository(session),
        )
        return load_dataset_into(dataset, PostgresSightingRepository(session))  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    """Run the generator.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled project error.
    """
    args = build_parser().parse_args(argv)
    configure_logging(json_output=False, level="INFO")

    try:
        scenario = load_scenario(args.scenario)
        dataset = generate_from_file(args.scenario, seed=args.seed)
    except MulticamTrackerError as exc:
        logger.error("generation_failed", reason=str(exc))
        return 1

    written: tuple[Path, Path] | None = None
    if args.out is not None:
        written = _write_files(dataset, args.out)

    inserted = None
    if args.to_database:
        try:
            inserted = _load_into_database(dataset, scenario.topology_file)
        except MulticamTrackerError as exc:
            logger.error("database_load_failed", reason=str(exc))
            return 1

    if not args.quiet:
        truth = dataset.ground_truth
        print(f"scenario     : {dataset.scenario_name}")
        print(f"seed         : {dataset.seed}")
        print(f"vehicles     : {len(truth.vehicles)}")
        print(f"sightings    : {len(dataset.sightings)}")
        print(f"corruptions  : {len(truth.corruptions)}")
        print(f"read failures: {truth.read_failure_rate:.1%}")
        print(f"injections   : {len(truth.injections)}")
        if written is not None:
            print(f"dataset      -> {written[0]}")
            print(f"ground truth -> {written[1]}")
        if inserted is not None:
            print(f"inserted     : {inserted} sighting(s)")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
