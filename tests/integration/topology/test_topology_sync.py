"""Integration tests for topology persistence against a real database.

The same cases run against the in-memory fakes in ``tests/unit/topology/``. They
are repeated here because two of them depend on behaviour only Postgres can
supply: the ``ON DELETE RESTRICT`` foreign key that refuses to remove a camera
holding sightings, and the ``ON CONFLICT DO UPDATE`` that makes a re-sync an
update rather than a duplicate-key error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.topology import (
    Topology,
    load_topology,
    load_topology_from_db,
    sync_topology_to_db,
)
from tests.fixtures.factories import make_camera, make_sighting

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REPO_CONFIG = Path(__file__).resolve().parents[3] / "config" / "topology.yaml"


@pytest.fixture
def topology() -> Topology:
    """Return the shared valid topology fixture.

    Returns:
        The graph.
    """
    return load_topology(FIXTURES / "topology_valid.yaml")


def test_sync__inserts_all_cameras_and_links(
    postgres_repositories: Any, topology: Topology
) -> None:
    """The happy path against a real schema, foreign keys and all."""
    report = sync_topology_to_db(
        topology, postgres_repositories.cameras, postgres_repositories.links
    )

    assert report.cameras_written == 4
    assert len(postgres_repositories.cameras.list_enabled()) == 4
    assert len(postgres_repositories.links.list_all()) == 2


def test_sync__run_twice__is_idempotent(postgres_repositories: Any, topology: Topology) -> None:
    """The second run must be an upsert, not a duplicate-key violation."""
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)

    assert len(postgres_repositories.cameras.list_enabled()) == 4
    assert len(postgres_repositories.links.list_all()) == 2


def test_sync__modified_link__updates_the_existing_row(
    postgres_repositories: Any, topology: Topology, tmp_path: Path
) -> None:
    """Editing a window in YAML revises the row rather than adding a second."""
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)

    revised_file = tmp_path / "topology.yaml"
    revised_file.write_text(
        (FIXTURES / "topology_valid.yaml")
        .read_text(encoding="utf-8")
        .replace("max_travel_time_sec: 300.0", "max_travel_time_sec: 999.0"),
        encoding="utf-8",
    )
    sync_topology_to_db(
        load_topology(revised_file), postgres_repositories.cameras, postgres_repositories.links
    )

    stored = postgres_repositories.links.get_link("cam_a", "cam_b")
    assert stored is not None
    assert stored.max_travel_time_sec == 999.0
    assert len(postgres_repositories.links.list_all()) == 2


def test_sync__pruning_a_camera_with_no_sightings__removes_it(
    postgres_repositories: Any, topology: Topology
) -> None:
    """An opted-in prune of an evidence-free camera succeeds."""
    postgres_repositories.cameras.upsert(make_camera(camera_id="cam_retired"))

    report = sync_topology_to_db(
        topology,
        postgres_repositories.cameras,
        postgres_repositories.links,
        sighting_repo=postgres_repositories.sightings,
        prune_missing=True,
    )

    assert report.cameras_deleted == ["cam_retired"]
    assert postgres_repositories.cameras.get("cam_retired") is None


def test_sync__pruning_a_camera_with_sightings__raises_listing_it(
    postgres_repositories: Any, topology: Topology
) -> None:
    """The check runs before any DELETE, so the FK never has to fire.

    Attempting the delete and catching the violation would abort the whole
    transaction in Postgres, taking the rest of the sync down with it.
    """
    postgres_repositories.cameras.upsert(make_camera(camera_id="cam_retired"))
    postgres_repositories.sightings.add(make_sighting("cam_retired"))

    with pytest.raises(TopologyError) as excinfo:
        sync_topology_to_db(
            topology,
            postgres_repositories.cameras,
            postgres_repositories.links,
            sighting_repo=postgres_repositories.sightings,
            prune_missing=True,
        )

    assert excinfo.value.context["camera_ids"] == ["cam_retired"]
    assert postgres_repositories.cameras.get("cam_retired") is not None


def test_load_from_db__reproduces_the_file_defined_topology(
    postgres_repositories: Any, topology: Topology
) -> None:
    """The database is the runtime source of truth and must agree with the file."""
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)

    from_db = load_topology_from_db(postgres_repositories.cameras, postgres_repositories.links)

    assert from_db.camera_ids == topology.camera_ids
    assert [edge.key for edge in from_db.list_edges()] == [
        edge.key for edge in topology.list_edges()
    ]


def test_load_from_db__preserves_windows_and_direction(
    postgres_repositories: Any, topology: Topology
) -> None:
    """The constraint that does the pruning must survive the round trip intact."""
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)

    from_db = load_topology_from_db(postgres_repositories.cameras, postgres_repositories.links)
    link = from_db.get_link("cam_a", "cam_b")

    assert link is not None
    assert link.min_travel_time_sec == 60.0
    assert link.max_travel_time_sec == 300.0
    assert from_db.has_link("cam_c", "cam_b") is False, "one-way link stays one-way"


def test_sync__shipped_example_config__round_trips_through_the_database(
    postgres_repositories: Any,
) -> None:
    """The documented example is the one an operator will actually sync first."""
    from_file = load_topology(REPO_CONFIG)

    sync_topology_to_db(from_file, postgres_repositories.cameras, postgres_repositories.links)
    from_db = load_topology_from_db(postgres_repositories.cameras, postgres_repositories.links)

    assert from_db.camera_ids == from_file.camera_ids
    assert from_db.has_link("cam_04", "cam_05") is True
    assert from_db.has_link("cam_05", "cam_04") is False
