"""The embedding search, run against a real pgvector index.

Every unit test of :func:`find_embedding_matches` runs against the in-memory
fake, which computes cosine similarity exactly in Python. Production runs
against an approximate HNSW index. If those two disagreed, every fast test in
the suite would be a confident prediction about a system that does not exist --
so the same query is run both ways here on identical data and the rankings are
compared.

These tests need Docker. Without it they skip rather than fail, which is why the
parity claim has to be re-checked in CI, where the container does run.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from multicam_tracker.db.repositories import PostgresSightingRepository
from multicam_tracker.matching import find_embedding_matches
from multicam_tracker.models import Sighting, Target
from tests.fixtures.factories import (
    BASE_INSTANT,
    DEFAULT_EMBEDDING_DIM,
    make_camera,
    make_sighting,
    unit_vector,
)
from tests.fixtures.fake_repositories import InMemoryStore, build_in_memory_repositories
from tests.fixtures.topologies import make_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODEL_VERSION = "reid_v1"
CAMERAS = ("cam_01", "cam_02", "cam_03")

LARGE_CORPUS = 100_000
SEARCH_BUDGET_SEC = 2.0
"""Documented budget for one constrained search over the large corpus.

Only the search is timed. Building the corpus is setup, and at 512 dimensions it
moves roughly 200 MB into the database.
"""


def _topology() -> Any:
    """Return a three-camera chain.

    Returns:
        ``cam_01 -> cam_02 -> cam_03``.
    """
    return make_topology(
        list(CAMERAS),
        [("cam_01", "cam_02", 60.0, 900.0), ("cam_02", "cam_03", 60.0, 900.0)],
    )


def _corpus() -> list[Sighting]:
    """Build the shared candidate set.

    Returns:
        Sightings spread over the reachable cameras and window, each carrying a
        distinct embedding.
    """
    return [
        make_sighting(
            CAMERAS[index % 3],
            offset_sec=120.0 + index * 3.0,
            embedding=unit_vector(seed=index + 10),
            embedding_model_version=MODEL_VERSION,
        )
        for index in range(60)
    ]


@pytest.fixture
def postgres_sightings(migrated_engine: Engine) -> Iterator[PostgresSightingRepository]:
    """Yield a Postgres sighting repository in a rolled-back transaction.

    Yields:
        A repository with the test cameras already seeded.
    """
    from multicam_tracker.db.repositories import PostgresCameraRepository

    connection = migrated_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )
    cameras = PostgresCameraRepository(session)
    for camera_id in CAMERAS:
        cameras.upsert(make_camera(camera_id=camera_id))

    try:
        yield PostgresSightingRepository(session)
    finally:
        session.close()
        transaction.rollback()
        connection.close()


def _in_memory(sightings: list[Sighting]) -> Any:
    """Build an in-memory repository holding the same data.

    Args:
        sightings: The corpus to load.

    Returns:
        The in-memory sighting repository.
    """
    repositories = build_in_memory_repositories(InMemoryStore())
    for camera_id in CAMERAS:
        repositories.cameras.upsert(make_camera(camera_id=camera_id))
    for sighting in sightings:
        repositories.sightings.add(sighting)
    return repositories.sightings


def _anchor() -> Sighting:
    """Return the confirmed sighting every search starts from.

    Returns:
        A sighting at the first camera carrying the reference embedding.
    """
    return make_sighting(
        "cam_01",
        offset_sec=0.0,
        embedding=unit_vector(seed=1),
        embedding_model_version=MODEL_VERSION,
    )


def _target() -> Target:
    """Return a target whose reference is the anchor embedding.

    Returns:
        The target being searched for.
    """
    return Target(
        label="parity target",
        plate_query="ABC1234",
        reference_embeddings=[unit_vector(seed=1)],
        created_at=BASE_INSTANT,
    )


def test_ranking__postgres_and_in_memory__agree_on_identical_data(
    postgres_sightings: PostgresSightingRepository,
) -> None:
    """The fake must be a truthful prediction about production, not a stand-in.

    Same corpus, same anchor, same thresholds. Any divergence here means every
    unit test written against the fake is measuring something else.
    """
    corpus = _corpus()
    anchor = _anchor()
    postgres_sightings.add_batch([anchor, *corpus])
    memory = _in_memory([anchor, *corpus])

    kwargs: dict[str, Any] = {
        "anchor_sighting": anchor,
        "model_version": MODEL_VERSION,
        "k": 50,
        "max_horizon_sec": 1800.0,
    }
    from_postgres = find_embedding_matches(_target(), postgres_sightings, _topology(), **kwargs)
    from_memory = find_embedding_matches(_target(), memory, _topology(), **kwargs)

    assert [result.sighting.sighting_id for result in from_postgres] == [
        result.sighting.sighting_id for result in from_memory
    ]
    for postgres_result, memory_result in zip(from_postgres, from_memory, strict=True):
        assert postgres_result.similarity == pytest.approx(memory_result.similarity, abs=1e-5)
        assert postgres_result.review_status is memory_result.review_status


def test_ranking__the_approximate_index_and_an_exact_scan__agree_on_the_top_hit(
    postgres_sightings: PostgresSightingRepository,
) -> None:
    """HNSW is approximate by design, and a wrong top-1 would be invisible.

    The result still looks like a ranking, so nothing downstream can detect it.
    """
    corpus = _corpus()
    anchor = _anchor()
    postgres_sightings.add_batch([anchor, *corpus])

    kwargs: dict[str, Any] = {
        "anchor_sighting": anchor,
        "model_version": MODEL_VERSION,
        "k": 20,
        "max_horizon_sec": 1800.0,
    }
    approximate = find_embedding_matches(_target(), postgres_sightings, _topology(), **kwargs)

    postgres_sightings.session.execute(text("SET LOCAL enable_indexscan = off"))
    postgres_sightings.session.execute(text("SET LOCAL enable_bitmapscan = off"))
    exact = find_embedding_matches(_target(), postgres_sightings, _topology(), **kwargs)
    postgres_sightings.session.execute(text("SET LOCAL enable_indexscan = on"))
    postgres_sightings.session.execute(text("SET LOCAL enable_bitmapscan = on"))

    assert approximate and exact
    assert approximate[0].sighting.sighting_id == exact[0].sighting.sighting_id


def test_model_version__is_filtered_in_the_database__not_after_the_read(
    postgres_sightings: PostgresSightingRepository,
) -> None:
    """A migrating table holds both versions, and the search must still work.

    Filtering afterwards would return k rows of which most were unusable; the
    restriction has to be part of the query.
    """
    anchor = _anchor()
    superseded = [
        make_sighting(
            "cam_02",
            offset_sec=200.0 + index,
            embedding=unit_vector(seed=1),
            embedding_model_version="reid_v2",
        )
        for index in range(10)
    ]
    postgres_sightings.add_batch([anchor, *_corpus(), *superseded])

    results = find_embedding_matches(
        _target(),
        postgres_sightings,
        _topology(),
        anchor_sighting=anchor,
        model_version=MODEL_VERSION,
        k=50,
        max_horizon_sec=1800.0,
    )

    returned = {result.sighting.sighting_id for result in results}
    assert results
    assert not returned & {sighting.sighting_id for sighting in superseded}


def test_search__over_one_hundred_thousand_embeddings__completes_within_the_budget(
    postgres_sightings: PostgresSightingRepository,
) -> None:
    """The scale the index exists for.

    Vectors are generated with numpy rather than the pure-Python helper: 51.2
    million Gaussian samples through ``random`` would dominate the test.
    """
    generator = np.random.default_rng(seed=7)
    raw = generator.standard_normal((LARGE_CORPUS, DEFAULT_EMBEDDING_DIM))
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    anchor = _anchor()
    postgres_sightings.add(anchor)
    for start in range(0, LARGE_CORPUS, 10_000):
        postgres_sightings.add_batch(
            [
                make_sighting(
                    CAMERAS[index % 3],
                    offset_sec=120.0 + (index % 600),
                    embedding=normalized[index].tolist(),
                    embedding_model_version=MODEL_VERSION,
                ).model_copy(update={"sighting_id": str(uuid.uuid4())})
                for index in range(start, min(start + 10_000, LARGE_CORPUS))
            ]
        )

    started = time.perf_counter()
    results = find_embedding_matches(
        _target(),
        postgres_sightings,
        _topology(),
        anchor_sighting=anchor,
        model_version=MODEL_VERSION,
        k=50,
        max_horizon_sec=1800.0,
    )
    elapsed = time.perf_counter() - started

    assert isinstance(results, list)
    assert elapsed < SEARCH_BUDGET_SEC, f"search took {elapsed:.2f}s over {LARGE_CORPUS} rows"
