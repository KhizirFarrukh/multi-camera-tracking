"""Unit tests for :mod:`multicam_tracker.topology.sync`, against the fakes.

These run without a database because stage 03's in-memory repositories honour
the same semantics as Postgres -- including the foreign key that stops a camera
with sightings being deleted. The conformance suite is what makes that
substitution sound; the same cases run against a real database in
``tests/integration/topology/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.topology import load_topology, load_topology_from_db, sync_topology_to_db
from tests.fixtures.factories import make_camera, make_sighting
from tests.fixtures.fake_repositories import RepositorySet, build_in_memory_repositories

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def repositories() -> RepositorySet:
    """Return a fresh set of in-memory repositories.

    Returns:
        Repositories sharing one store.
    """
    return build_in_memory_repositories()


def _load() -> object:
    """Load the shared valid topology fixture.

    Returns:
        The graph.
    """
    return load_topology(FIXTURES / "topology_valid.yaml")


def test_sync__writes_every_camera_and_declared_link(repositories: RepositorySet) -> None:
    """The happy path: file contents land in the database."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    report = sync_topology_to_db(topology, repositories.cameras, repositories.links)

    assert report.cameras_written == 4
    assert len(repositories.cameras.list_enabled()) == 4
    assert repositories.cameras.get("cam_iso") is not None


def test_sync__stores_each_link_once_in_its_declared_direction(
    repositories: RepositorySet,
) -> None:
    """The database holds what the author wrote, not a derived duplicate.

    Storing the synthesized reverse too would create a second row that silently
    drifts if the original is edited.
    """
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    sync_topology_to_db(topology, repositories.cameras, repositories.links)

    pairs = [(link.from_camera_id, link.to_camera_id) for link in repositories.links.list_all()]
    assert pairs == [("cam_a", "cam_b"), ("cam_b", "cam_c")]


def test_sync__run_twice__is_idempotent(repositories: RepositorySet) -> None:
    """Re-syncing an unchanged file must change nothing."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    sync_topology_to_db(topology, repositories.cameras, repositories.links)
    sync_topology_to_db(topology, repositories.cameras, repositories.links)

    assert len(repositories.cameras.list_enabled()) == 4
    assert len(repositories.links.list_all()) == 2


def test_sync__changed_link__updates_the_existing_row(
    repositories: RepositorySet, tmp_path: Path
) -> None:
    """Editing a window in YAML revises the row rather than adding a second."""
    original = load_topology(FIXTURES / "topology_valid.yaml")
    sync_topology_to_db(original, repositories.cameras, repositories.links)

    revised_file = tmp_path / "topology.yaml"
    revised_file.write_text(
        (FIXTURES / "topology_valid.yaml")
        .read_text(encoding="utf-8")
        .replace("max_travel_time_sec: 300.0", "max_travel_time_sec: 999.0"),
        encoding="utf-8",
    )
    sync_topology_to_db(load_topology(revised_file), repositories.cameras, repositories.links)

    stored = repositories.links.get_link("cam_a", "cam_b")
    assert stored is not None
    assert stored.max_travel_time_sec == 999.0
    assert len(repositories.links.list_all()) == 2


def test_sync__camera_missing_from_the_file__is_retained_by_default(
    repositories: RepositorySet,
) -> None:
    """Deleting infrastructure must be a decision, not a side effect of a sync."""
    repositories.cameras.upsert(make_camera(camera_id="cam_retired"))

    report = sync_topology_to_db(_load(), repositories.cameras, repositories.links)  # type: ignore[arg-type]

    assert report.cameras_retained == ["cam_retired"]
    assert repositories.cameras.get("cam_retired") is not None


def test_sync__pruning_a_camera_with_no_sightings__removes_it(
    repositories: RepositorySet,
) -> None:
    """An opted-in prune of an evidence-free camera succeeds."""
    repositories.cameras.upsert(make_camera(camera_id="cam_retired"))

    report = sync_topology_to_db(
        _load(),  # type: ignore[arg-type]
        repositories.cameras,
        repositories.links,
        sighting_repo=repositories.sightings,
        prune_missing=True,
    )

    assert report.cameras_deleted == ["cam_retired"]
    assert repositories.cameras.get("cam_retired") is None


def test_sync__pruning_a_camera_with_sightings__raises_listing_it(
    repositories: RepositorySet,
) -> None:
    """Evidence outlives the hardware that recorded it."""
    repositories.cameras.upsert(make_camera(camera_id="cam_retired"))
    repositories.sightings.add(make_sighting("cam_retired"))

    with pytest.raises(TopologyError) as excinfo:
        sync_topology_to_db(
            _load(),  # type: ignore[arg-type]
            repositories.cameras,
            repositories.links,
            sighting_repo=repositories.sightings,
            prune_missing=True,
        )

    assert excinfo.value.context["camera_ids"] == ["cam_retired"]
    assert repositories.cameras.get("cam_retired") is not None


def test_sync__blocked_prune__deletes_nothing_at_all(repositories: RepositorySet) -> None:
    """All-or-nothing: a partial prune would leave the operator guessing."""
    repositories.cameras.upsert(make_camera(camera_id="cam_empty"))
    repositories.cameras.upsert(make_camera(camera_id="cam_busy"))
    repositories.sightings.add(make_sighting("cam_busy"))

    with pytest.raises(TopologyError):
        sync_topology_to_db(
            _load(),  # type: ignore[arg-type]
            repositories.cameras,
            repositories.links,
            sighting_repo=repositories.sightings,
            prune_missing=True,
        )

    assert repositories.cameras.get("cam_empty") is not None, "innocent camera survived"


def test_sync__pruning_without_a_sighting_repository__is_refused(
    repositories: RepositorySet,
) -> None:
    """Without it there is no way to know which cameras hold evidence."""
    with pytest.raises(TopologyError, match="Pruning requires"):
        sync_topology_to_db(
            _load(),  # type: ignore[arg-type]
            repositories.cameras,
            repositories.links,
            prune_missing=True,
        )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_load_from_db__reproduces_the_file_defined_topology(
    repositories: RepositorySet,
) -> None:
    """The database is the runtime source of truth and must agree with the file."""
    from_file = load_topology(FIXTURES / "topology_valid.yaml")
    sync_topology_to_db(from_file, repositories.cameras, repositories.links)

    from_db = load_topology_from_db(repositories.cameras, repositories.links)

    assert from_db.camera_ids == from_file.camera_ids
    assert [edge.key for edge in from_db.list_edges()] == [
        edge.key for edge in from_file.list_edges()
    ]


def test_load_from_db__reexpands_bidirectional_links(repositories: RepositorySet) -> None:
    """One stored row becomes two directed edges again, exactly as on load."""
    sync_topology_to_db(_load(), repositories.cameras, repositories.links)  # type: ignore[arg-type]

    from_db = load_topology_from_db(repositories.cameras, repositories.links)

    assert from_db.has_link("cam_a", "cam_b") is True
    assert from_db.has_link("cam_b", "cam_a") is True
    assert from_db.has_link("cam_c", "cam_b") is False, "one-way link stays one-way"


def test_load_from_db__preserves_travel_windows(repositories: RepositorySet) -> None:
    """The constraint that does the pruning must survive the round trip intact."""
    sync_topology_to_db(_load(), repositories.cameras, repositories.links)  # type: ignore[arg-type]

    link = load_topology_from_db(repositories.cameras, repositories.links).get_link(
        "cam_a", "cam_b"
    )

    assert link is not None
    assert link.min_travel_time_sec == 60.0
    assert link.max_travel_time_sec == 300.0


def test_load_from_db__empty_database__yields_an_empty_topology(
    repositories: RepositorySet,
) -> None:
    """Boundary: nothing synced yet is a valid, empty graph."""
    topology = load_topology_from_db(repositories.cameras, repositories.links)

    assert len(topology) == 0
