"""Dual-backend repositories for the clock-correction tests.

Mirrors the fixture in ``tests/conformance/`` deliberately rather than importing
it: the claim under test here is that a clock correction is *transactional*, and
that is a claim about a database. The in-memory run keeps the tests useful on a
machine without Docker; the Postgres run is what actually substantiates them.
"""

from __future__ import annotations

import pytest

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
