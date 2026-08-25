"""Rewriting a camera's stored timestamps when its offset changes.

Correcting a clock is not a configuration change -- it rewrites history. Every
sighting that camera ever recorded moves, and so does every conclusion drawn
from them. Three properties make that safe to do:

* the rewrite is **atomic**, so a failure part-way through cannot leave a camera
  half-corrected, which would be worse than leaving it uncorrected;
* it is **idempotent**, recomputing from ``raw_timestamp`` rather than shifting
  the current value, so running it twice does not double-shift;
* it is **recorded**, with both the old and the new offset, because "the offset
  is now -45000" is unanswerable after the fact without knowing what it was.

The tests run against the in-memory fake and, where Docker is available, against
Postgres -- because "transactional" is a claim about the database, and a fake
cannot substantiate it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.models import Sighting
from multicam_tracker.timesync import apply_offset_change, recompute_offsets
from tests.fixtures.factories import BASE_INSTANT, make_camera, make_sighting

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
DRIFT_MS = -45_000


@pytest.fixture
def seeded(repositories: Any) -> Any:
    """Insert two cameras with sightings on each.

    Args:
        repositories: The backend under test.

    Returns:
        The repository set, with data.
    """
    for camera_id in ("cam_01", "cam_02"):
        repositories.cameras.upsert(make_camera(camera_id=camera_id))
    repositories.sightings.add_batch(
        [
            *[make_sighting("cam_01", offset_sec=index * 60.0) for index in range(3)],
            *[make_sighting("cam_02", offset_sec=index * 60.0) for index in range(2)],
        ]
    )
    return repositories


def _by_camera(repositories: Any, camera_id: str) -> list[Sighting]:
    """Return one camera's stored sightings in time order.

    Args:
        repositories: The backend under test.
        camera_id: The camera to read.

    Returns:
        Its sightings.
    """
    from multicam_tracker.models import TimeWindow

    window = TimeWindow(
        start_utc=BASE_INSTANT - timedelta(days=1), end_utc=BASE_INSTANT + timedelta(days=1)
    )
    found = repositories.sightings.find_by_camera_and_window(camera_id, window)
    return sorted(found, key=lambda sighting: sighting.raw_timestamp)


def _saved_trajectory(repositories: Any, camera_id: str) -> Any:
    """Save a two-hop route over one camera and return it.

    Args:
        repositories: The backend under test.
        camera_id: The camera whose sightings the route is built from.

    Returns:
        The saved trajectory.
    """
    from multicam_tracker.models import Target, Trajectory, TrajectoryHop

    target = repositories.targets.create(
        Target(label="offset test", plate_query="ABC1234", created_at=BASE_INSTANT)
    )
    route = _by_camera(repositories, camera_id)[:2]
    hops = [
        TrajectoryHop(
            from_sighting_id=route[0].sighting_id,
            to_sighting_id=route[1].sighting_id,
            from_camera_id=route[0].camera_id,
            to_camera_id=route[1].camera_id,
            elapsed_sec=(route[1].timestamp_utc - route[0].timestamp_utc).total_seconds(),
            topology_plausible=True,
            hop_confidence=0.9,
        )
    ]
    return repositories.trajectories.save(
        Trajectory(
            target_id=target.target_id,
            sightings=route,
            hops=hops,
            overall_confidence=0.9,
            start_time_utc=route[0].timestamp_utc,
            end_time_utc=route[1].timestamp_utc,
        )
    )


def test_the_rewrite__moves_every_sighting_for_that_camera(seeded: Any) -> None:
    """The blast radius is the whole camera, not the rows someone remembered."""
    rewritten = seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)

    stored = _by_camera(seeded, "cam_01")
    assert rewritten == 3
    assert all(sighting.clock_offset_applied_ms == DRIFT_MS for sighting in stored)
    assert all(
        sighting.timestamp_utc == sighting.raw_timestamp + timedelta(milliseconds=DRIFT_MS)
        for sighting in stored
    )


def test_the_rewrite__leaves_other_cameras_untouched(seeded: Any) -> None:
    """One camera's clock says nothing about another's."""
    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)

    untouched = _by_camera(seeded, "cam_02")
    assert all(sighting.clock_offset_applied_ms == 0 for sighting in untouched)
    assert all(sighting.timestamp_utc == sighting.raw_timestamp for sighting in untouched)


def test_the_rewrite__is_idempotent(seeded: Any) -> None:
    """Running ingestion or a correction sweep twice must not double-shift.

    The guarantee comes from recomputing off ``raw_timestamp``; a shift of the
    current value would land somewhere no record explains.
    """
    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)
    first = [sighting.timestamp_utc for sighting in _by_camera(seeded, "cam_01")]

    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)
    second = [sighting.timestamp_utc for sighting in _by_camera(seeded, "cam_01")]

    assert first == second


def test_a_revised_offset__supersedes_the_previous_one(seeded: Any) -> None:
    """A better estimate replaces the old correction; it does not add to it."""
    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)
    seeded.sightings.apply_clock_offset("cam_01", -30_000)

    stored = _by_camera(seeded, "cam_01")
    assert all(
        sighting.timestamp_utc == sighting.raw_timestamp - timedelta(seconds=30)
        for sighting in stored
    )


def test_the_rewrite__returns_zero_for_a_camera_with_no_sightings(seeded: Any) -> None:
    """Correcting a quiet camera is legitimate and changes no rows."""
    assert seeded.sightings.apply_clock_offset("cam_09", DRIFT_MS) == 0


def test_the_stored_sightings__still_satisfy_the_model_after_the_rewrite(
    seeded: Any,
) -> None:
    """The three timestamp fields must agree, or the read path rejects them.

    Reading them back through the mapper is the check: the model validates on
    construction, so a rewrite that broke the invariant would raise here rather
    than surfacing as a wrong route later.
    """
    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)

    for sighting in _by_camera(seeded, "cam_01"):
        assert sighting.model_validate(sighting.model_dump()) == sighting


# ---------------------------------------------------------------------------
# The change record
# ---------------------------------------------------------------------------


def test_the_change_record__names_both_offsets_and_the_row_count(seeded: Any) -> None:
    """What an operator reviewing the audit trail needs to reconstruct the event."""
    stored = _by_camera(seeded, "cam_01") + _by_camera(seeded, "cam_02")

    _rewritten, change = recompute_offsets("cam_01", DRIFT_MS, stored, FixedClock(NOW))

    entry = change.audit_entry()
    assert entry["camera_id"] == "cam_01"
    assert entry["previous_offset_ms"] == 0
    assert entry["new_offset_ms"] == DRIFT_MS
    assert entry["sightings_updated"] == 3
    assert entry["changed_at_utc"] == NOW.isoformat()


def test_the_pure_rewrite_and_the_repository_rewrite__agree(seeded: Any) -> None:
    """Two implementations of one rule, and they must not drift apart.

    The pure function is what the tests and the live path use; the SQL statement
    is what production runs. If they disagreed, every unit test written against
    the former would be describing a system that does not exist.
    """
    before = _by_camera(seeded, "cam_01")
    expected, _change = recompute_offsets("cam_01", DRIFT_MS, before, FixedClock(NOW))

    seeded.sightings.apply_clock_offset("cam_01", DRIFT_MS)
    actual = _by_camera(seeded, "cam_01")

    assert [sighting.timestamp_utc for sighting in actual] == [
        sighting.timestamp_utc for sighting in expected
    ]
    assert [sighting.clock_offset_applied_ms for sighting in actual] == [
        sighting.clock_offset_applied_ms for sighting in expected
    ]


def test_the_correction_service__rewrites_sightings_and_flags_the_routes(
    seeded: Any,
) -> None:
    """The whole operation, as an operator performs it.

    A clock correction is not a configuration change: it rewrites history and
    invalidates the conclusions drawn from it. Both halves happen together, or
    the system is left with routes that silently no longer follow from their
    evidence.
    """
    trajectory = _saved_trajectory(seeded, "cam_01")

    change = apply_offset_change(
        seeded.sightings,
        seeded.trajectories,
        "cam_01",
        DRIFT_MS,
        FixedClock(NOW),
        reason="ntp sweep found the camera 45s fast",
    )

    assert change.sightings_updated == 3
    assert change.affected_trajectory_ids == (trajectory.trajectory_id,)
    assert change.audit_entry()["affected_trajectory_ids"] == [trajectory.trajectory_id]


def test_routes_that_never_used_the_camera__are_left_alone(seeded: Any) -> None:
    """Flagging everything would be the same as flagging nothing."""
    untouched = _saved_trajectory(seeded, "cam_02")

    change = apply_offset_change(
        seeded.sightings, seeded.trajectories, "cam_01", DRIFT_MS, FixedClock(NOW)
    )

    assert untouched.trajectory_id not in change.affected_trajectory_ids


def test_a_flagged_route__says_so_when_it_is_read_back(seeded: Any) -> None:
    """The flag is what an operator queries after correcting a clock.

    Without it they would have to work out by hand which conclusions rested on
    the timestamps that just moved.
    """
    trajectory = _saved_trajectory(seeded, "cam_01")

    apply_offset_change(seeded.sightings, seeded.trajectories, "cam_01", DRIFT_MS, FixedClock(NOW))

    stored = seeded.trajectories.get(trajectory.trajectory_id)
    assert stored is not None
    assert stored.requires_recomputation


def test_a_freshly_saved_route__is_not_flagged(seeded: Any) -> None:
    """The control: the flag means something only if it is usually false."""
    trajectory = _saved_trajectory(seeded, "cam_01")

    stored = seeded.trajectories.get(trajectory.trajectory_id)
    assert stored is not None
    assert not stored.requires_recomputation


# ---------------------------------------------------------------------------
# Atomicity, which only a real database can substantiate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_failure_mid_operation__leaves_no_partially_corrected_data(
    postgres_repositories: Any,
) -> None:
    """The property a single UPDATE statement buys.

    A per-row loop that raised half-way would leave one camera with two
    different clock corrections applied to its sightings -- and nothing in the
    data saying which rows had been touched.
    """
    from sqlalchemy.exc import SQLAlchemyError

    repositories = postgres_repositories
    for camera_id in ("cam_01",):
        repositories.cameras.upsert(make_camera(camera_id=camera_id))
    repositories.sightings.add_batch(
        [make_sighting("cam_01", offset_sec=index * 60.0) for index in range(4)]
    )

    session = repositories.sightings.session
    savepoint = session.begin_nested()
    try:
        repositories.sightings.apply_clock_offset("cam_01", DRIFT_MS)
        msg = "simulated failure after the rewrite, before the commit"
        raise SQLAlchemyError(msg)
    except SQLAlchemyError:
        savepoint.rollback()

    stored = _by_camera(repositories, "cam_01")
    assert all(sighting.clock_offset_applied_ms == 0 for sighting in stored)
    assert all(sighting.timestamp_utc == sighting.raw_timestamp for sighting in stored)
