"""Integration tests for database-level constraint enforcement.

These deliberately bypass the Pydantic models and write raw SQL. The models
guard the application path; these constraints guard everything else -- a
migration, a manual ``psql`` session, a future service in another language. A
constraint that exists only in Python is not a constraint on the data.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.slow]

INSERT_CAMERA = text(
    "INSERT INTO cameras (camera_id, name, lat, lon) VALUES (:cid, :name, :lat, :lon)"
)

INSERT_SIGHTING = text(
    """
    INSERT INTO sightings (
        sighting_id, camera_id, timestamp_utc, raw_timestamp, clock_offset_applied_ms,
        object_class, detection_confidence, bbox, frame_index, source_id, created_at
    ) VALUES (
        :sid, :cid, :ts, :ts, 0, 'car', :confidence, :bbox, 0, 'src', :ts
    )
    """
)


@pytest.fixture
def connection(migrated_engine: Engine) -> Iterator[Connection]:
    """Yield a connection whose transaction is rolled back afterwards.

    Yields:
        An open connection inside an uncommitted transaction.
    """
    with migrated_engine.connect() as conn:
        transaction = conn.begin()
        try:
            yield conn
        finally:
            transaction.rollback()


def _add_camera(connection: Connection, camera_id: str = "cam_01") -> str:
    """Insert a camera and return its id.

    Args:
        connection: The open connection.
        camera_id: Identifier to insert.

    Returns:
        The inserted camera id.
    """
    connection.execute(INSERT_CAMERA, {"cid": camera_id, "name": "Test", "lat": 0.0, "lon": 0.0})
    return camera_id


@pytest.mark.parametrize("confidence", [-0.01, 1.01], ids=["below", "above"])
def test_constraints__detection_confidence_out_of_range__is_rejected(
    connection: Connection, confidence: float
) -> None:
    """A confidence outside [0, 1] cannot be compared against a threshold."""
    _add_camera(connection)

    with pytest.raises(IntegrityError, match="ck_sightings_detection_confidence"):
        connection.execute(
            INSERT_SIGHTING,
            {
                "sid": str(uuid.uuid4()),
                "cid": "cam_01",
                "ts": "2026-08-10T14:22:11.500+00",
                "confidence": confidence,
                "bbox": [0, 0, 10, 10],
            },
        )


def test_constraints__bbox_of_the_wrong_length__is_rejected(connection: Connection) -> None:
    """A bbox is exactly four coordinates."""
    _add_camera(connection)

    with pytest.raises(IntegrityError, match="ck_sightings_bbox"):
        connection.execute(
            INSERT_SIGHTING,
            {
                "sid": str(uuid.uuid4()),
                "cid": "cam_01",
                "ts": "2026-08-10T14:22:11.500+00",
                "confidence": 0.5,
                "bbox": [0, 0, 10],
            },
        )


def test_constraints__inverted_bbox__is_rejected(connection: Connection) -> None:
    """A zero- or negative-area box crops to nothing."""
    _add_camera(connection)

    with pytest.raises(IntegrityError, match="ck_sightings_bbox_area"):
        connection.execute(
            INSERT_SIGHTING,
            {
                "sid": str(uuid.uuid4()),
                "cid": "cam_01",
                "ts": "2026-08-10T14:22:11.500+00",
                "confidence": 0.5,
                "bbox": [100, 100, 50, 50],
            },
        )


def test_constraints__duplicate_camera_link_pair__is_rejected(connection: Connection) -> None:
    """One declared travel window per ordered pair."""
    _add_camera(connection, "cam_01")
    _add_camera(connection, "cam_02")
    insert = text(
        "INSERT INTO camera_links (from_camera_id, to_camera_id, "
        "min_travel_time_sec, max_travel_time_sec) VALUES ('cam_01', 'cam_02', 10, 20)"
    )
    connection.execute(insert)

    with pytest.raises(IntegrityError):
        connection.execute(insert)


def test_constraints__inverted_travel_window__is_rejected(connection: Connection) -> None:
    """A window whose max is below its min accepts nothing."""
    _add_camera(connection, "cam_01")
    _add_camera(connection, "cam_02")

    with pytest.raises(IntegrityError, match="ck_camera_links_window_ordered"):
        connection.execute(
            text(
                "INSERT INTO camera_links (from_camera_id, to_camera_id, "
                "min_travel_time_sec, max_travel_time_sec) VALUES ('cam_01', 'cam_02', 300, 45)"
            )
        )


def test_constraints__deleting_a_camera_with_sightings__is_rejected(
    connection: Connection,
) -> None:
    """ON DELETE RESTRICT: evidence is never orphaned by removing a camera."""
    _add_camera(connection)
    connection.execute(
        INSERT_SIGHTING,
        {
            "sid": str(uuid.uuid4()),
            "cid": "cam_01",
            "ts": "2026-08-10T14:22:11.500+00",
            "confidence": 0.5,
            "bbox": [0, 0, 10, 10],
        },
    )

    with pytest.raises(IntegrityError, match="fk_sightings_camera"):
        connection.execute(text("DELETE FROM cameras WHERE camera_id = 'cam_01'"))


def test_constraints__deleting_a_target__cascades_to_its_match_candidates(
    connection: Connection,
) -> None:
    """Candidates have no meaning without the target they were scored against."""
    _add_camera(connection)
    sighting_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())

    connection.execute(
        INSERT_SIGHTING,
        {
            "sid": sighting_id,
            "cid": "cam_01",
            "ts": "2026-08-10T14:22:11.500+00",
            "confidence": 0.5,
            "bbox": [0, 0, 10, 10],
        },
    )
    connection.execute(
        text(
            "INSERT INTO targets (target_id, label, plate_query, created_at) "
            "VALUES (:tid, 'Test', 'ABC1234', '2026-08-10T14:00:00+00')"
        ),
        {"tid": target_id},
    )
    connection.execute(
        text(
            "INSERT INTO match_candidates (target_id, sighting_id, match_method, "
            "match_score, plate_edit_distance, review_status) "
            "VALUES (:tid, :sid, 'plate_exact', 0.9, 0, 'auto_accepted')"
        ),
        {"tid": target_id, "sid": sighting_id},
    )

    connection.execute(text("DELETE FROM targets WHERE target_id = :tid"), {"tid": target_id})

    remaining = connection.execute(
        text("SELECT count(*) FROM match_candidates WHERE target_id = :tid"), {"tid": target_id}
    ).scalar_one()
    assert remaining == 0


def test_constraints__plate_without_confidence__is_rejected(connection: Connection) -> None:
    """Matching weights every read by its confidence; an unweighted read has none."""
    _add_camera(connection)

    with pytest.raises(IntegrityError, match="ck_sightings_plate_needs_confidence"):
        connection.execute(
            text(
                """
                INSERT INTO sightings (
                    sighting_id, camera_id, timestamp_utc, raw_timestamp,
                    object_class, detection_confidence, bbox, frame_index,
                    plate_text_normalized, source_id, created_at
                ) VALUES (
                    :sid, 'cam_01', :ts, :ts, 'car', 0.5, :bbox, 0, 'ABC1234', 'src', :ts
                )
                """
            ),
            {
                "sid": str(uuid.uuid4()),
                "ts": "2026-08-10T14:22:11.500+00",
                "bbox": [0, 0, 10, 10],
            },
        )


def test_constraints__unsearchable_target__is_rejected(connection: Connection) -> None:
    """A target with neither a plate nor embeddings matches nothing by construction."""
    with pytest.raises(IntegrityError, match="ck_targets_searchable"):
        connection.execute(
            text(
                "INSERT INTO targets (target_id, label, created_at) "
                "VALUES (:tid, 'Nothing', '2026-08-10T14:00:00+00')"
            ),
            {"tid": str(uuid.uuid4())},
        )


def test_constraints__plate_folded__is_computed_by_the_database(
    connection: Connection,
) -> None:
    """The generated column folds on write, without the application supplying it."""
    _add_camera(connection)
    sighting_id = str(uuid.uuid4())
    connection.execute(
        text(
            """
            INSERT INTO sightings (
                sighting_id, camera_id, timestamp_utc, raw_timestamp,
                object_class, detection_confidence, bbox, frame_index,
                plate_text_normalized, plate_confidence, source_id, created_at
            ) VALUES (
                :sid, 'cam_01', :ts, :ts, 'car', 0.5, :bbox, 0, 'ABC1234', 0.9, 'src', :ts
            )
            """
        ),
        {"sid": sighting_id, "ts": "2026-08-10T14:22:11.500+00", "bbox": [0, 0, 10, 10]},
    )

    folded = connection.execute(
        text("SELECT plate_folded FROM sightings WHERE sighting_id = :sid"), {"sid": sighting_id}
    ).scalar_one()

    # O -> 0, B -> 8: ABC1234 folds to A8C1234.
    assert folded == "A8C1234"
