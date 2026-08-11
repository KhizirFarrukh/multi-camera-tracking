"""One behavioural contract, run against every repository implementation.

Each test here executes twice -- against the in-memory fakes and against a real
Postgres -- from the same body. If a fake ever drifts from the database, the
divergence surfaces here rather than three stages later as a unit test that
passes while production misbehaves.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import (
    MatchMethod,
    ReviewStatus,
    Sighting,
    TimeWindow,
)
from tests.fixtures.factories import (
    BASE_INSTANT,
    make_camera,
    make_link,
    make_match_candidate,
    make_sighting,
    make_target,
    make_trajectory,
    unit_vector,
)
from tests.fixtures.fake_repositories import RepositorySet


def _window(start_min: float, end_min: float) -> TimeWindow:
    """Build a window at minute offsets from the base instant.

    Args:
        start_min: Minutes after the base instant for the start.
        end_min: Minutes after the base instant for the end.

    Returns:
        A half-open window.
    """
    return TimeWindow(
        start_utc=BASE_INSTANT + timedelta(minutes=start_min),
        end_utc=BASE_INSTANT + timedelta(minutes=end_min),
    )


def _plated(camera_id: str, offset_sec: float, plate: str, **overrides: object) -> Sighting:
    """Build a sighting carrying a normalized plate.

    Args:
        camera_id: The observing camera.
        offset_sec: Seconds after the base instant.
        plate: The normalized plate text.
        **overrides: Extra field overrides.

    Returns:
        A validated sighting.
    """
    return make_sighting(
        camera_id,
        offset_sec=offset_sec,
        plate_text_raw=plate,
        plate_text_normalized=plate,
        plate_confidence=0.9,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Cameras and links
# ---------------------------------------------------------------------------


def test_camera__add_then_get__returns_an_equal_camera(repositories: RepositorySet) -> None:
    """The most basic round trip through the store."""
    camera = make_camera(camera_id="cam_07", name="Test Corner", heading_degrees=91.5)
    repositories.cameras.upsert(camera)

    assert repositories.cameras.get("cam_07") == camera


def test_camera__get_unknown_id__returns_none(repositories: RepositorySet) -> None:
    """Absence is reported as None, not as an error."""
    assert repositories.cameras.get("cam_missing") is None


def test_camera__upsert_twice__replaces_rather_than_duplicates(
    repositories: RepositorySet,
) -> None:
    """Reloading a topology file must not create a second row."""
    repositories.cameras.upsert(make_camera(camera_id="cam_07", name="First"))
    repositories.cameras.upsert(make_camera(camera_id="cam_07", name="Second"))

    stored = repositories.cameras.get("cam_07")
    assert stored is not None
    assert stored.name == "Second"
    assert len(repositories.cameras.list_enabled()) == 1


def test_camera__list_enabled__excludes_disabled_and_orders_by_id(
    repositories: RepositorySet,
) -> None:
    """Disabled cameras stay in the table but out of the working set."""
    repositories.cameras.upsert(make_camera(camera_id="cam_02"))
    repositories.cameras.upsert(make_camera(camera_id="cam_01"))
    repositories.cameras.upsert(make_camera(camera_id="cam_03", enabled=False))

    assert [c.camera_id for c in repositories.cameras.list_enabled()] == ["cam_01", "cam_02"]


def test_camera__delete_unused__removes_the_row(repositories: RepositorySet) -> None:
    """A camera with no sightings can be deleted outright."""
    repositories.cameras.upsert(make_camera(camera_id="cam_09"))

    assert repositories.cameras.delete("cam_09") is True
    assert repositories.cameras.get("cam_09") is None


def test_camera__delete_unknown__returns_false(repositories: RepositorySet) -> None:
    """Deleting nothing is not an error."""
    assert repositories.cameras.delete("cam_missing") is False


def test_camera__delete_with_sightings__raises_storage_error(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """ON DELETE RESTRICT: evidence is never orphaned by removing a camera."""
    repositories.sightings.add(make_sighting("cam_01"))

    with pytest.raises(StorageError):
        repositories.cameras.delete("cam_01")


def test_link__upsert_then_get__returns_an_equal_link(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Links round-trip through the store."""
    link = make_link("cam_01", "cam_02", min_travel_time_sec=30.0, max_travel_time_sec=200.0)
    repositories.links.upsert(link)

    assert repositories.links.get_link("cam_01", "cam_02") == link


def test_link__get_links_from__returns_only_declared_origins_ordered(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Bidirectional expansion is stage 04's job, not the repository's."""
    repositories.links.upsert(make_link("cam_01", "cam_03"))
    repositories.links.upsert(make_link("cam_01", "cam_02"))
    repositories.links.upsert(make_link("cam_02", "cam_03"))

    from_one = repositories.links.get_links_from("cam_01")

    assert [link.to_camera_id for link in from_one] == ["cam_02", "cam_03"]


def test_link__list_all__orders_by_origin_then_destination(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A stable order keeps topology dumps diffable."""
    repositories.links.upsert(make_link("cam_02", "cam_03"))
    repositories.links.upsert(make_link("cam_01", "cam_03"))
    repositories.links.upsert(make_link("cam_01", "cam_02"))

    pairs = [(link.from_camera_id, link.to_camera_id) for link in repositories.links.list_all()]
    assert pairs == [("cam_01", "cam_02"), ("cam_01", "cam_03"), ("cam_02", "cam_03")]


def test_link__unknown_camera__raises_storage_error(repositories: RepositorySet) -> None:
    """A link to a camera that does not exist is rejected by the foreign key."""
    repositories.cameras.upsert(make_camera(camera_id="cam_01"))

    with pytest.raises(StorageError):
        repositories.links.upsert(make_link("cam_01", "cam_nope"))


# ---------------------------------------------------------------------------
# Sightings: writes
# ---------------------------------------------------------------------------


def test_sighting__add_then_get__returns_an_equal_sighting(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Full fidelity through the store, embedding and plate included."""
    sighting = _plated("cam_01", 0, "ABC1234", embedding=unit_vector(seed=1))
    repositories.sightings.add(sighting)

    assert repositories.sightings.get(sighting.sighting_id) == sighting


def test_sighting__get_unknown_id__returns_none(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Absence is reported as None."""
    assert repositories.sightings.get("11111111-1111-4111-8111-111111111111") is None


def test_sighting__add_batch__inserts_all_and_returns_the_count(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The bulk path must not silently drop rows."""
    batch = [make_sighting("cam_01", offset_sec=index) for index in range(5)]

    assert repositories.sightings.add_batch(batch) == 5
    assert len(repositories.sightings.find_by_window(_window(-1, 60))) == 5


def test_sighting__add_batch_empty__returns_zero(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Boundary: an empty batch is a no-op, not an error."""
    assert repositories.sightings.add_batch([]) == 0


def test_sighting__add_batch_with_conflicts_ignored__inserts_only_the_new_rows(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Re-processing a video must be idempotent, not a duplicate-key crash."""
    original = [make_sighting("cam_01", offset_sec=index) for index in range(3)]
    repositories.sightings.add_batch(original)

    extra = make_sighting("cam_01", offset_sec=99)
    inserted = repositories.sightings.add_batch([*original, extra], ignore_conflicts=True)

    assert inserted == 1
    assert len(repositories.sightings.find_by_window(_window(-1, 60))) == 4


def test_sighting__add_batch_with_conflicts_not_ignored__raises(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Without the flag, a duplicate id is a genuine error."""
    batch = [make_sighting("cam_01", offset_sec=0)]
    repositories.sightings.add_batch(batch)

    with pytest.raises(StorageError):
        repositories.sightings.add_batch(batch)


def test_sighting__referencing_an_unknown_camera__raises_storage_error(
    repositories: RepositorySet,
) -> None:
    """The foreign key surfaces as StorageError, never as a raw driver error."""
    with pytest.raises(StorageError) as excinfo:
        repositories.sightings.add(make_sighting("cam_ghost"))

    assert isinstance(excinfo.value, StorageError)


# ---------------------------------------------------------------------------
# Sightings: plate queries
# ---------------------------------------------------------------------------


def test_find_by_plate_exact__returns_only_exact_matches(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A near-miss is not an exact match, however plausible it looks."""
    wanted = _plated("cam_01", 0, "ABC1234")
    repositories.sightings.add_batch(
        [wanted, _plated("cam_01", 10, "ABC1235"), _plated("cam_01", 20, "A8C1234")]
    )

    found = repositories.sightings.find_by_plate_exact("ABC1234")

    assert [s.sighting_id for s in found] == [wanted.sighting_id]


def test_find_by_plate_exact__ignores_sightings_without_a_plate(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Most sightings have no plate; none of them can match one."""
    repositories.sightings.add_batch([make_sighting("cam_01", offset_sec=0)])

    assert repositories.sightings.find_by_plate_exact("ABC1234") == []


def test_find_by_plate_exact__window_boundaries_are_half_open(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Start is inclusive, end is exclusive -- the documented convention."""
    at_start = _plated("cam_01", 0, "ABC1234")
    inside = _plated("cam_01", 60, "ABC1234")
    at_end = _plated("cam_01", 120, "ABC1234")
    repositories.sightings.add_batch([at_start, inside, at_end])

    found = repositories.sightings.find_by_plate_exact("ABC1234", _window(0, 2))

    assert [s.sighting_id for s in found] == [at_start.sighting_id, inside.sighting_id]


def test_find_by_plate_folded__matches_ambiguous_substitutions(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The whole point of the folded column: 0/O, 1/I, 8/B collapse together."""
    exact = _plated("cam_01", 0, "ABC1234")
    confused = _plated("cam_02", 30, "A8C1Z34")
    unrelated = _plated("cam_03", 60, "XYZ9999")
    repositories.sightings.add_batch([exact, confused, unrelated])

    found = repositories.sightings.find_by_plate_folded("ABC1234")

    assert {s.sighting_id for s in found} == {exact.sighting_id, confused.sighting_id}


def test_find_by_plate_folded__includes_the_exact_match(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """An exact match folds identically, so the prefilter must not exclude it."""
    exact = _plated("cam_01", 0, "ABC1234")
    repositories.sightings.add(exact)

    assert [s.sighting_id for s in repositories.sightings.find_by_plate_folded("ABC1234")] == [
        exact.sighting_id
    ]


# ---------------------------------------------------------------------------
# Sightings: window queries
# ---------------------------------------------------------------------------


def test_find_by_camera_and_window__orders_strictly_by_ascending_timestamp(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Trajectory assembly depends on this order; insertion order is not it."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=120),
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_01", offset_sec=60),
        ]
    )

    found = repositories.sightings.find_by_camera_and_window("cam_01", _window(-1, 10))

    timestamps = [s.timestamp_utc for s in found]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == 3


def test_find_by_camera_and_window__excludes_other_cameras(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A per-camera query must not leak a neighbour's sightings."""
    repositories.sightings.add_batch(
        [make_sighting("cam_01", offset_sec=0), make_sighting("cam_02", offset_sec=10)]
    )

    found = repositories.sightings.find_by_camera_and_window("cam_01", _window(-1, 10))

    assert [s.camera_id for s in found] == ["cam_01"]


def test_find_by_camera_and_window__camera_with_no_sightings__returns_empty_list(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Empty list, never None -- callers iterate the result unconditionally."""
    result = repositories.sightings.find_by_camera_and_window("cam_03", _window(-1, 10))

    assert result == []


def test_find_by_window__spans_all_cameras_in_global_time_order(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Cross-camera assembly needs one globally ordered stream."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_02", offset_sec=30),
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_03", offset_sec=60),
        ]
    )

    found = repositories.sightings.find_by_window(_window(-1, 10))

    assert [s.camera_id for s in found] == ["cam_01", "cam_02", "cam_03"]


def test_find_by_window__boundaries_are_half_open(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Same convention everywhere a window appears."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_01", offset_sec=120),
        ]
    )

    found = repositories.sightings.find_by_window(_window(0, 2))

    assert len(found) == 1


# ---------------------------------------------------------------------------
# Sightings: vector search
# ---------------------------------------------------------------------------


def test_find_nearest_by_embedding__orders_by_descending_similarity(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The known-nearest vector comes back first."""
    query = unit_vector(seed=1)
    identical = make_sighting("cam_01", offset_sec=0, embedding=query)
    different = make_sighting("cam_01", offset_sec=10, embedding=unit_vector(seed=99))
    repositories.sightings.add_batch([different, identical])

    found = repositories.sightings.find_nearest_by_embedding(query, k=2)

    assert found[0].sighting.sighting_id == identical.sighting_id
    assert found[0].similarity > found[1].similarity
    assert found[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_find_nearest_by_embedding__respects_k_exactly(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Boundary: K is a hard limit, not a hint."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=index, embedding=unit_vector(seed=index))
            for index in range(5)
        ]
    )

    assert len(repositories.sightings.find_nearest_by_embedding(unit_vector(seed=0), k=2)) == 2


def test_find_nearest_by_embedding__camera_filter__excludes_other_cameras(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Stage 07 narrows the search to topologically reachable cameras."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=0, embedding=unit_vector(seed=1)),
            make_sighting("cam_02", offset_sec=10, embedding=unit_vector(seed=1)),
        ]
    )

    found = repositories.sightings.find_nearest_by_embedding(
        unit_vector(seed=1), k=10, camera_ids=["cam_02"]
    )

    assert [match.sighting.camera_id for match in found] == ["cam_02"]


def test_find_nearest_by_embedding__ignores_sightings_without_an_embedding(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A NULL embedding cannot be compared; counting it as 0 would displace a real hit."""
    with_vector = make_sighting("cam_01", offset_sec=0, embedding=unit_vector(seed=1))
    repositories.sightings.add_batch([with_vector, make_sighting("cam_01", offset_sec=10)])

    found = repositories.sightings.find_nearest_by_embedding(unit_vector(seed=1), k=10)

    assert [match.sighting.sighting_id for match in found] == [with_vector.sighting_id]


def test_find_nearest_by_embedding__window_filter__applies(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Vector search is combined with a time constraint, not applied blindly."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=0, embedding=unit_vector(seed=1)),
            make_sighting("cam_01", offset_sec=600, embedding=unit_vector(seed=1)),
        ]
    )

    found = repositories.sightings.find_nearest_by_embedding(
        unit_vector(seed=1), k=10, window=_window(0, 5)
    )

    assert len(found) == 1


def test_find_nearest_by_embedding__k_of_zero__returns_empty(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Boundary: asking for nothing returns nothing rather than everything."""
    repositories.sightings.add(make_sighting("cam_01", embedding=unit_vector(seed=1)))

    assert repositories.sightings.find_nearest_by_embedding(unit_vector(seed=1), k=0) == []


def test_find_nearest_by_embedding__wrong_dimension__raises_storage_error(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The error names the expected width, which the driver's would not."""
    with pytest.raises(StorageError) as excinfo:
        repositories.sightings.find_nearest_by_embedding([0.1, 0.2], k=1)

    assert excinfo.value.context["expected"] == 512


# ---------------------------------------------------------------------------
# Sightings: observability and retention
# ---------------------------------------------------------------------------


def test_count_by_camera_hour__buckets_across_an_hour_boundary(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Two sightings either side of the hour mark land in different buckets."""
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_01", offset_sec=60),
            make_sighting("cam_01", offset_sec=3600),
        ]
    )

    counts = repositories.sightings.count_by_camera_hour(_window(-1, 180))

    assert [entry.sighting_count for entry in counts] == [2, 1]
    assert counts[0].hour_start_utc.hour == BASE_INSTANT.hour
    assert counts[1].hour_start_utc.hour == BASE_INSTANT.hour + 1


def test_count_by_camera_hour__buckets_across_a_utc_day_boundary(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Day rollover must not merge or drop a bucket."""
    seconds_to_midnight = (
        24 * 3600 - BASE_INSTANT.hour * 3600 - BASE_INSTANT.minute * 60 - BASE_INSTANT.second
    )
    repositories.sightings.add_batch(
        [
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_01", offset_sec=seconds_to_midnight + 60),
        ]
    )

    counts = repositories.sightings.count_by_camera_hour(_window(-1, 24 * 60 + 60))

    assert len(counts) == 2
    assert counts[0].hour_start_utc.day != counts[1].hour_start_utc.day


def test_count_by_camera_hour__separates_cameras(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Buckets are per camera, ordered by camera then hour."""
    repositories.sightings.add_batch(
        [make_sighting("cam_02", offset_sec=0), make_sighting("cam_01", offset_sec=0)]
    )

    counts = repositories.sightings.count_by_camera_hour(_window(-1, 60))

    assert [entry.camera_id for entry in counts] == ["cam_01", "cam_02"]


def test_delete_older_than__removes_exactly_the_rows_before_the_cutoff(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Boundary: a row exactly at the cutoff survives."""
    old = make_sighting(
        "cam_01",
        offset_sec=0,
        created_at=BASE_INSTANT - timedelta(days=2),
        thumbnail_path="storage/thumbnails/old.jpg",
    )
    at_cutoff = make_sighting("cam_01", offset_sec=10, created_at=BASE_INSTANT)
    recent = make_sighting("cam_01", offset_sec=20, created_at=BASE_INSTANT + timedelta(days=1))
    repositories.sightings.add_batch([old, at_cutoff, recent])

    result = repositories.sightings.delete_older_than(BASE_INSTANT)

    assert result.deleted_count == 1
    assert result.thumbnail_paths == ["storage/thumbnails/old.jpg"]
    assert repositories.sightings.get(old.sighting_id) is None
    assert repositories.sightings.get(at_cutoff.sighting_id) is not None


def test_delete_older_than__cutoff_before_all_data__deletes_nothing(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Boundary: a purge with nothing to purge reports zero, not an error."""
    repositories.sightings.add(make_sighting("cam_01"))

    result = repositories.sightings.delete_older_than(BASE_INSTANT - timedelta(days=30))

    assert result.deleted_count == 0
    assert result.thumbnail_paths == []


def test_delete_older_than__cascades_to_match_candidates(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A match to a purged sighting cannot survive it."""
    sighting = make_sighting("cam_01", created_at=BASE_INSTANT - timedelta(days=2))
    repositories.sightings.add(sighting)
    target = repositories.targets.create(make_target())
    repositories.matches.upsert_candidate(
        make_match_candidate(target_id=target.target_id, sighting_id=sighting.sighting_id)
    )

    repositories.sightings.delete_older_than(BASE_INSTANT)

    assert repositories.matches.list_for_target(target.target_id) == []


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_target__create_then_get__returns_an_equal_target(
    repositories: RepositorySet,
) -> None:
    """Targets round-trip, reference embeddings included."""
    target = make_target(reference_embeddings=[unit_vector(seed=3)])
    repositories.targets.create(target)

    assert repositories.targets.get(target.target_id) == target


def test_target__create_duplicate_id__raises_storage_error(
    repositories: RepositorySet,
) -> None:
    """Two searches cannot share one identity."""
    target = repositories.targets.create(make_target())

    with pytest.raises(StorageError):
        repositories.targets.create(target)


def test_target__deactivate__removes_it_from_the_active_list_but_keeps_the_row(
    repositories: RepositorySet,
) -> None:
    """The row survives for the audit trail; only the flag changes."""
    target = repositories.targets.create(make_target())

    assert repositories.targets.deactivate(target.target_id) is True
    assert repositories.targets.list_active() == []

    stored = repositories.targets.get(target.target_id)
    assert stored is not None
    assert stored.active is False


def test_target__deactivate_unknown__returns_false(repositories: RepositorySet) -> None:
    """Deactivating nothing is not an error."""
    assert repositories.targets.deactivate("11111111-1111-4111-8111-111111111111") is False


def test_target__list_active__is_newest_first(repositories: RepositorySet) -> None:
    """The review UI shows the most recent search at the top."""
    older = repositories.targets.create(
        make_target(label="Older", created_at=BASE_INSTANT - timedelta(hours=1))
    )
    newer = repositories.targets.create(make_target(label="Newer", created_at=BASE_INSTANT))

    assert [t.target_id for t in repositories.targets.list_active()] == [
        newer.target_id,
        older.target_id,
    ]


# ---------------------------------------------------------------------------
# Match candidates
# ---------------------------------------------------------------------------


def test_match__upsert_twice__updates_rather_than_duplicates(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Re-running the matcher revises its verdict; it does not stack rows."""
    sighting = repositories.sightings.add(make_sighting("cam_01"))
    target = repositories.targets.create(make_target())

    base = make_match_candidate(
        target_id=target.target_id, sighting_id=sighting.sighting_id, match_score=0.5
    )
    repositories.matches.upsert_candidate(base)
    repositories.matches.upsert_candidate(base.model_copy(update={"match_score": 0.95}))

    stored = repositories.matches.list_for_target(target.target_id)
    assert len(stored) == 1
    assert stored[0].match_score == pytest.approx(0.95)


def test_match__list_for_target_filtered_by_status__returns_only_that_status(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The review queue asks for exactly one status at a time."""
    target = repositories.targets.create(make_target())
    pending = repositories.sightings.add(make_sighting("cam_01", offset_sec=0))
    accepted = repositories.sightings.add(make_sighting("cam_01", offset_sec=10))

    repositories.matches.bulk_upsert(
        [
            make_match_candidate(
                target_id=target.target_id,
                sighting_id=pending.sighting_id,
                match_method=MatchMethod.PLATE_FUZZY,
                plate_edit_distance=2,
                review_status=ReviewStatus.PENDING_REVIEW,
            ),
            make_match_candidate(
                target_id=target.target_id,
                sighting_id=accepted.sighting_id,
                review_status=ReviewStatus.AUTO_ACCEPTED,
            ),
        ]
    )

    found = repositories.matches.list_for_target(target.target_id, ReviewStatus.PENDING_REVIEW)

    assert [c.sighting_id for c in found] == [pending.sighting_id]


def test_match__list_for_target__orders_by_descending_score(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Strongest evidence is reviewed first."""
    target = repositories.targets.create(make_target())
    weak = repositories.sightings.add(make_sighting("cam_01", offset_sec=0))
    strong = repositories.sightings.add(make_sighting("cam_01", offset_sec=10))

    repositories.matches.bulk_upsert(
        [
            make_match_candidate(
                target_id=target.target_id, sighting_id=weak.sighting_id, match_score=0.4
            ),
            make_match_candidate(
                target_id=target.target_id, sighting_id=strong.sighting_id, match_score=0.99
            ),
        ]
    )

    found = repositories.matches.list_for_target(target.target_id)

    assert [c.sighting_id for c in found] == [strong.sighting_id, weak.sighting_id]


def test_match__bulk_upsert_empty__returns_zero(repositories: RepositorySet) -> None:
    """Boundary: nothing to write is not an error."""
    assert repositories.matches.bulk_upsert([]) == 0


def test_match__update_review_status__changes_only_the_targeted_record(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """A human confirming one match must not disturb its neighbours."""
    target = repositories.targets.create(make_target())
    first = repositories.sightings.add(make_sighting("cam_01", offset_sec=0))
    second = repositories.sightings.add(make_sighting("cam_01", offset_sec=10))
    repositories.matches.bulk_upsert(
        [
            make_match_candidate(
                target_id=target.target_id,
                sighting_id=first.sighting_id,
                review_status=ReviewStatus.PENDING_REVIEW,
                match_score=0.9,
            ),
            make_match_candidate(
                target_id=target.target_id,
                sighting_id=second.sighting_id,
                review_status=ReviewStatus.PENDING_REVIEW,
                match_score=0.8,
            ),
        ]
    )

    changed = repositories.matches.update_review_status(
        target.target_id, first.sighting_id, ReviewStatus.CONFIRMED
    )

    assert changed is True
    statuses = {
        c.sighting_id: c.review_status
        for c in repositories.matches.list_for_target(target.target_id)
    }
    assert statuses[first.sighting_id] is ReviewStatus.CONFIRMED
    assert statuses[second.sighting_id] is ReviewStatus.PENDING_REVIEW


def test_match__update_review_status_unknown__returns_false(
    repositories: RepositorySet,
) -> None:
    """Updating nothing is not an error."""
    assert (
        repositories.matches.update_review_status(
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            ReviewStatus.CONFIRMED,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


def test_trajectory__save_then_get__returns_an_equal_trajectory(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Trajectories round-trip with their hops in order."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]
    repositories.sightings.add_batch(sightings)
    target = repositories.targets.create(make_target())
    trajectory = make_trajectory(sightings=sightings, target_id=target.target_id)

    repositories.trajectories.save(trajectory)

    assert repositories.trajectories.get(trajectory.trajectory_id) == trajectory


def test_trajectory__get_unknown_id__returns_none(repositories: RepositorySet) -> None:
    """Absence is reported as None."""
    assert repositories.trajectories.get("11111111-1111-4111-8111-111111111111") is None


def test_trajectory__list_for_target__is_newest_first(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Most recent reconstruction first."""
    target = repositories.targets.create(make_target())

    early = [make_sighting("cam_01", offset_sec=0), make_sighting("cam_02", offset_sec=60)]
    late = [make_sighting("cam_01", offset_sec=600), make_sighting("cam_02", offset_sec=660)]
    repositories.sightings.add_batch([*early, *late])

    first = repositories.trajectories.save(
        make_trajectory(sightings=early, target_id=target.target_id)
    )
    second = repositories.trajectories.save(
        make_trajectory(sightings=late, target_id=target.target_id)
    )

    listed = repositories.trajectories.list_for_target(target.target_id)
    assert [t.trajectory_id for t in listed] == [second.trajectory_id, first.trajectory_id]


def test_trajectory__save_twice__replaces_rather_than_duplicates(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """Re-saving a revised trajectory must not leave the old hops behind."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]
    repositories.sightings.add_batch(sightings)
    target = repositories.targets.create(make_target())
    trajectory = make_trajectory(sightings=sightings, target_id=target.target_id)

    repositories.trajectories.save(trajectory)
    revised = trajectory.model_copy(update={"overall_confidence": 0.42})
    repositories.trajectories.save(revised)

    stored = repositories.trajectories.get(trajectory.trajectory_id)
    assert stored is not None
    assert stored.overall_confidence == pytest.approx(0.42)
    assert len(stored.hops) == 1


def test_trajectory__after_its_sightings_are_purged__reports_them_missing(
    repositories: RepositorySet, seeded_cameras: list[str]
) -> None:
    """The documented retention behaviour.

    A trajectory is a historical conclusion. Retention may remove the sightings
    underneath it, and the repository says so rather than returning a silently
    shortened route -- which would look like a confident, different answer.
    """
    sightings = [
        make_sighting("cam_01", offset_sec=0, created_at=BASE_INSTANT - timedelta(days=2)),
        make_sighting("cam_02", offset_sec=90, created_at=BASE_INSTANT - timedelta(days=2)),
    ]
    repositories.sightings.add_batch(sightings)
    target = repositories.targets.create(make_target())
    trajectory = repositories.trajectories.save(
        make_trajectory(sightings=sightings, target_id=target.target_id)
    )

    repositories.sightings.delete_older_than(BASE_INSTANT)

    with pytest.raises(StorageError) as excinfo:
        repositories.trajectories.get(trajectory.trajectory_id)

    assert len(excinfo.value.context["missing_sighting_ids"]) == 2
