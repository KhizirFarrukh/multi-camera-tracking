"""Shared fixtures for the synthetic-data tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from multicam_tracker.synth import Scenario, VehicleSpec, load_scenario
from multicam_tracker.topology import Topology, load_topology

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_FILE = REPO_ROOT / "config" / "topology.yaml"
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
DEPARTURE = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
TARGET_ROUTE = ["cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]


@pytest.fixture(scope="session")
def topology() -> Topology:
    """Return the shipped example topology.

    Session-scoped: it is read-only and parsing it per test is wasted work.

    Returns:
        The graph the scenarios run over.
    """
    return load_topology(TOPOLOGY_FILE)


@pytest.fixture
def base_scenario() -> Scenario:
    """Return a minimal single-vehicle scenario with no noise or traffic.

    A test that wants noise or decoys turns exactly one thing on, so a failure
    points at that one thing.

    Returns:
        The scenario.
    """
    return Scenario(
        name="unit",
        topology_file=TOPOLOGY_FILE,
        seed=7,
        vehicles=[
            VehicleSpec(
                vehicle_id="target",
                plate="ABC1234",
                route=TARGET_ROUTE,
                departure_utc=DEPARTURE,
                is_target=True,
            )
        ],
    )


def committed_scenario(name: str) -> Scenario:
    """Load one of the six committed regression scenarios.

    Args:
        name: Scenario file stem.

    Returns:
        The loaded scenario.
    """
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")
