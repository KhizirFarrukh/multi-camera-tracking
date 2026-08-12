"""Fixtures that supply one repository set per backend.

The ``repositories`` fixture is parametrized, so every test in this package runs
twice: once against the in-memory fakes as a fast unit test, and once against a
real Postgres in a container as an integration test. That is the whole point of
the package -- it is what makes a unit test written against a fake a truthful
prediction about production.

When Docker is unavailable the Postgres parameter skips and the in-memory
parameter still runs, so the suite stays useful on a machine without it.

The container, migration, and Postgres-repository fixtures live in
``tests/conftest.py`` because ``tests/integration/`` needs them too.
"""

from __future__ import annotations

import pytest

from tests.fixtures.factories import make_camera
from tests.fixtures.fake_repositories import RepositorySet, build_in_memory_repositories


@pytest.fixture(
    params=[
        pytest.param("memory", marks=pytest.mark.unit),
        pytest.param("postgres", marks=[pytest.mark.integration, pytest.mark.slow]),
    ]
)
def repositories(request: pytest.FixtureRequest) -> RepositorySet:
    """Return a repository set for the current backend parameter.

    Args:
        request: Pytest request carrying the backend name.

    Returns:
        The repository set under test.
    """
    if request.param == "memory":
        return build_in_memory_repositories()
    return request.getfixturevalue("postgres_repositories")  # type: ignore[no-any-return]


@pytest.fixture
def seeded_cameras(repositories: RepositorySet) -> list[str]:
    """Insert the cameras every sighting test needs and return their ids.

    Sightings carry a foreign key to ``cameras``, so a test that inserts one
    without a camera would be testing the foreign key rather than the query it
    meant to test.

    Args:
        repositories: The backend under test.

    Returns:
        The seeded camera ids.
    """
    camera_ids = ["cam_01", "cam_02", "cam_03"]
    for camera_id in camera_ids:
        repositories.cameras.upsert(make_camera(camera_id=camera_id))
    return camera_ids
