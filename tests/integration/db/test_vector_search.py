"""Integration tests for pgvector search and batch-insert throughput.

Two things are worth proving against a real database rather than a fake. First,
that the HNSW index is actually *used* -- an unused index is a silent
performance cliff that only appears at production data volumes. Second, that the
approximate index agrees with an exact scan on the same data, because an index
that returns plausible-but-wrong neighbours would degrade re-id matching in a
way no unit test could detect.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from multicam_tracker.db.repositories import PostgresSightingRepository
from tests.fixtures.factories import make_camera, make_sighting, unit_vector

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BATCH_SIZE = 10_000
BATCH_BUDGET_SEC = 60.0
"""Deliberately loose. The assertion exists to catch a regression to per-row
inserts -- which would take minutes -- not to benchmark CI hardware."""


@pytest.fixture
def repository(migrated_engine: Engine) -> Iterator[PostgresSightingRepository]:
    """Yield a sighting repository in a rolled-back transaction, with cameras seeded.

    Yields:
        A repository bound to an isolated transaction.
    """
    from multicam_tracker.db.repositories import PostgresCameraRepository

    connection = migrated_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )
    cameras = PostgresCameraRepository(session)
    for camera_id in ("cam_01", "cam_02"):
        cameras.upsert(make_camera(camera_id=camera_id))

    try:
        yield PostgresSightingRepository(session)
    finally:
        session.close()
        transaction.rollback()
        connection.close()


def test_vector_search__known_nearest_vector__is_returned_first(
    repository: PostgresSightingRepository,
) -> None:
    """With a seeded set, the exact match ranks first and scores ~1.0."""
    query = unit_vector(seed=1)
    expected = make_sighting("cam_01", offset_sec=0, embedding=query)
    repository.add_batch(
        [
            expected,
            *(
                make_sighting("cam_01", offset_sec=index, embedding=unit_vector(seed=100 + index))
                for index in range(1, 20)
            ),
        ]
    )

    found = repository.find_nearest_by_embedding(query, k=5)

    assert found[0].sighting.sighting_id == expected.sighting_id
    assert found[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_vector_search__results_are_ordered_by_descending_similarity(
    repository: PostgresSightingRepository,
) -> None:
    """Ordering is a contract the matching engine relies on."""
    repository.add_batch(
        [
            make_sighting("cam_01", offset_sec=index, embedding=unit_vector(seed=index))
            for index in range(10)
        ]
    )

    found = repository.find_nearest_by_embedding(unit_vector(seed=0), k=10)

    similarities = [match.similarity for match in found]
    assert similarities == sorted(similarities, reverse=True)


def test_vector_search__index_scan_and_sequential_scan__agree(
    repository: PostgresSightingRepository, migrated_engine: Engine
) -> None:
    """The approximate index must not disagree with the exact answer at this scale.

    HNSW is approximate by design. On a small, well-separated set it should be
    exact; if it is not, the recall parameters need revisiting before stage 07
    calibrates thresholds against it.
    """
    query = unit_vector(seed=1)
    repository.add_batch(
        [
            make_sighting("cam_01", offset_sec=index, embedding=unit_vector(seed=index))
            for index in range(25)
        ]
    )

    with_index = repository.find_nearest_by_embedding(query, k=5)

    repository.session.execute(text("SET LOCAL enable_indexscan = off"))
    repository.session.execute(text("SET LOCAL enable_bitmapscan = off"))
    without_index = repository.find_nearest_by_embedding(query, k=5)
    repository.session.execute(text("SET LOCAL enable_indexscan = on"))
    repository.session.execute(text("SET LOCAL enable_bitmapscan = on"))

    assert [m.sighting.sighting_id for m in with_index] == [
        m.sighting.sighting_id for m in without_index
    ]


def test_vector_search__hnsw_index__is_used_by_the_planner(migrated_engine: Engine) -> None:
    """An index the planner ignores is a silent performance cliff at scale.

    Uses its own committed data because the planner will not choose an index
    scan over a handful of rows; the table needs enough content for the index to
    win. The rows are removed afterwards.
    """
    query = "[" + ",".join(str(value) for value in unit_vector(seed=1)) + "]"

    with migrated_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO cameras (camera_id, name, lat, lon) VALUES ('cam_hnsw', 'x', 0, 0)")
        )
        connection.execute(
            text(
                """
                INSERT INTO sightings (
                    sighting_id, camera_id, timestamp_utc, raw_timestamp, object_class,
                    detection_confidence, bbox, frame_index, embedding, source_id, created_at
                )
                SELECT
                    gen_random_uuid(), 'cam_hnsw', now(), now(), 'car',
                    0.5, ARRAY[0,0,10,10], 0,
                    (SELECT array_agg(random())::vector FROM generate_series(1, 512)),
                    'seed', now()
                FROM generate_series(1, 2000)
                """
            )
        )
        connection.execute(text("ANALYZE sightings"))

    try:
        with migrated_engine.connect() as connection:
            plan = "\n".join(
                row[0]
                for row in connection.execute(
                    text(
                        "EXPLAIN SELECT sighting_id FROM sightings "
                        "WHERE embedding IS NOT NULL "
                        f"ORDER BY embedding <=> '{query}'::vector LIMIT 5"
                    )
                )
            )

        assert "ix_sightings_embedding_hnsw" in plan, plan
    finally:
        with migrated_engine.begin() as connection:
            connection.execute(text("DELETE FROM sightings WHERE camera_id = 'cam_hnsw'"))
            connection.execute(text("DELETE FROM cameras WHERE camera_id = 'cam_hnsw'"))


def test_batch_insert__ten_thousand_sightings__completes_within_the_budget(
    repository: PostgresSightingRepository,
) -> None:
    """Guards the bulk path against regressing to per-row inserts.

    Embeddings are omitted: at 512 dimensions they dominate the payload, and the
    property under test is statement batching, not vector throughput.
    """
    batch = [
        make_sighting("cam_01", offset_sec=index).model_copy(
            update={"sighting_id": str(uuid.uuid4())}
        )
        for index in range(BATCH_SIZE)
    ]

    started = time.perf_counter()
    inserted = repository.add_batch(batch)
    elapsed = time.perf_counter() - started

    assert inserted == BATCH_SIZE
    assert elapsed < BATCH_BUDGET_SEC, f"batch insert took {elapsed:.1f}s"
