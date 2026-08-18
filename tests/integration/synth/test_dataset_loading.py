"""Integration tests loading a generated dataset into a real database.

Generation and persistence are developed separately, so this is the first place
they meet. Two things can only be checked here: that the schema's constraints
accept everything the generator produces, and that a 512-dimensional vector
survives the pgvector round trip still L2-normalized within float32 tolerance --
the model rejects anything that does not.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from multicam_tracker.synth import generate, load_dataset_into, load_scenario
from multicam_tracker.topology import load_topology, sync_topology_to_db

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "scenarios" / "realistic.yaml"
TOPOLOGY_FILE = REPO_ROOT / "config" / "topology.yaml"

# pgvector stores float32; a float64 round trip therefore loses precision, and
# the tolerance has to allow for it while still catching a genuinely broken
# vector.
FLOAT32_TOLERANCE = 1e-4


@pytest.fixture
def seeded(postgres_repositories: Any) -> Any:
    """Sync the topology and generate a dataset ready to load.

    Cameras must exist before sightings that reference them, so the topology
    sync is part of the fixture rather than something each test remembers.

    Returns:
        ``(repositories, dataset)``.
    """
    topology = load_topology(TOPOLOGY_FILE)
    sync_topology_to_db(topology, postgres_repositories.cameras, postgres_repositories.links)
    dataset = generate(load_scenario(SCENARIO), topology=topology)
    return postgres_repositories, dataset


def test_dataset__loads_without_constraint_violations(seeded: Any) -> None:
    """Every CHECK and every foreign key must accept what the generator emits."""
    repositories, dataset = seeded

    inserted = load_dataset_into(dataset, repositories.sightings)

    assert inserted == len(dataset.sightings)


def test_dataset__all_referenced_cameras_exist_after_topology_sync(seeded: Any) -> None:
    """A sighting on an unknown camera would fail the foreign key, not silently drop."""
    repositories, dataset = seeded
    known = {camera.camera_id for camera in repositories.cameras.list_enabled()}

    assert {sighting.camera_id for sighting in dataset.sightings} <= known


def test_dataset__loading_twice_with_conflict_ignore__produces_no_duplicates(
    seeded: Any,
) -> None:
    """Re-running a demo script must be idempotent, not a duplicate-key crash."""
    repositories, dataset = seeded

    first = load_dataset_into(dataset, repositories.sightings, ignore_conflicts=True)
    second = load_dataset_into(dataset, repositories.sightings, ignore_conflicts=True)

    assert first == len(dataset.sightings)
    assert second == 0


def test_dataset__sightings_read_back__equal_the_generated_ones(seeded: Any) -> None:
    """A field lost in the mapping layer would silently change what is measured."""
    repositories, dataset = seeded
    load_dataset_into(dataset, repositories.sightings)

    sample = dataset.sightings[:20]
    for generated in sample:
        stored = repositories.sightings.get(generated.sighting_id)
        assert stored is not None, generated.sighting_id
        assert stored.camera_id == generated.camera_id
        assert stored.timestamp_utc == generated.timestamp_utc
        assert stored.plate_text_normalized == generated.plate_text_normalized
        assert stored.object_class == generated.object_class


def test_dataset__embeddings__survive_the_round_trip_still_normalized(
    seeded: Any,
) -> None:
    """pgvector stores float32, and the Sighting model rejects a denormalized vector.

    If the round trip pushed the norm outside the 1e-3 tolerance, every sighting
    read back from the database would fail validation.
    """
    repositories, dataset = seeded
    load_dataset_into(dataset, repositories.sightings)

    for generated in dataset.sightings[:20]:
        stored = repositories.sightings.get(generated.sighting_id)
        assert stored is not None
        assert stored.embedding is not None
        assert generated.embedding is not None

        norm = math.sqrt(sum(component**2 for component in stored.embedding))
        assert norm == pytest.approx(1.0, abs=1e-3)
        assert stored.embedding == pytest.approx(generated.embedding, abs=FLOAT32_TOLERANCE)


def test_dataset__plate_folded_column__is_populated_for_readable_plates(
    seeded: Any,
) -> None:
    """The fuzzy prefilter index depends on the generated column being maintained."""
    repositories, dataset = seeded
    load_dataset_into(dataset, repositories.sightings)

    with_plate = next(s for s in dataset.sightings if s.plate_text_normalized is not None)
    found = repositories.sightings.find_by_plate_folded(with_plate.plate_text_normalized)

    assert any(s.sighting_id == with_plate.sighting_id for s in found)


def test_dataset__target_sightings__are_retrievable_by_its_true_plate(
    seeded: Any,
) -> None:
    """The end-to-end point of the dataset: the target can be found by plate.

    Only the uncorrupted reads are expected back -- that is precisely the recall
    gap stage 06 exists to close.
    """
    repositories, dataset = seeded
    load_dataset_into(dataset, repositories.sightings)

    truth = dataset.ground_truth
    target = truth.target
    assert target is not None

    found = repositories.sightings.find_by_plate_exact(target.true_plate)
    found_ids = {s.sighting_id for s in found}

    corrupted = {
        event.sighting_id
        for event in truth.corruptions
        if event.observed_plate != target.true_plate
    }
    expected = set(target.sighting_ids) - corrupted

    assert expected <= found_ids
