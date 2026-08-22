"""Unit tests for candidate search and plate-conflict detection.

The call-log test is the important one. "Never scan all sightings" is a
performance claim that cannot be verified by looking at results -- a full scan
returns the same answers, just slower, and at production data volumes that is
the difference between a query and an outage. So the repository records which
methods were called and the test asserts on that.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from multicam_tracker.matching import detect_plate_conflicts, find_plate_matches
from multicam_tracker.models import ReviewStatus, Sighting, TimeWindow
from multicam_tracker.topology import Topology
from tests.fixtures.factories import BASE_INSTANT, make_camera, make_sighting
from tests.fixtures.fake_repositories import InMemoryStore, build_in_memory_repositories
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

TARGET_PLATE = "ABC1234"
TARGET_ID = "a0000000-0000-4000-8000-000000000001"

FULL_SCAN_METHODS = {"find_by_window", "find_by_camera_and_window", "find_nearest_by_embedding"}
"""Methods that would read more than the indexed prefilter. None may be called."""


class RecordingSightingRepository:
    """Wraps a sighting repository and records which methods were called.

    Args:
        inner: The repository to delegate to.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> object:
        """Record the call and delegate.

        Args:
            name: Attribute being accessed.

        Returns:
            A wrapper that records the call before delegating.
        """
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def recorded(*args: object, **kwargs: object) -> object:
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return recorded


def _repositories() -> tuple[object, RecordingSightingRepository]:
    """Build in-memory repositories with a recording sighting repository.

    Returns:
        ``(repository_set, recording_sightings)``.
    """
    store = InMemoryStore()
    repositories = build_in_memory_repositories(store)
    for camera_id in ("cam_01", "cam_02", "cam_03"):
        repositories.cameras.upsert(make_camera(camera_id=camera_id))
    return repositories, RecordingSightingRepository(repositories.sightings)


def _plated(camera_id: str, offset_sec: float, plate: str, confidence: float = 0.95) -> Sighting:
    """Build a sighting carrying a plate.

    Args:
        camera_id: The observing camera.
        offset_sec: Seconds after the base instant.
        plate: The normalized plate.
        confidence: OCR confidence.

    Returns:
        A validated sighting.
    """
    return make_sighting(
        camera_id,
        offset_sec=offset_sec,
        plate_text_raw=plate,
        plate_text_normalized=plate,
        plate_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# find_plate_matches
# ---------------------------------------------------------------------------


def test_find__ranks_by_descending_score() -> None:
    """The strongest evidence is reviewed first."""
    repositories, recording = _repositories()
    weak = _plated("cam_01", 0, "A8C1234", confidence=0.6)
    strong = _plated("cam_02", 10, TARGET_PLATE, confidence=0.99)
    repositories.sightings.add_batch([weak, strong])

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert [c.sighting_id for c in found] == [strong.sighting_id, weak.sighting_id]
    assert found[0].match_score >= found[1].match_score


def test_find__assigns_auto_accepted_above_the_threshold() -> None:
    """A confident exact read needs no human."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, TARGET_PLATE, confidence=0.99))

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert found[0].review_status is ReviewStatus.AUTO_ACCEPTED


def test_find__assigns_pending_review_inside_the_review_band() -> None:
    """The uncertain middle goes to a human rather than being accepted."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, "A8C1234", confidence=0.75))

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert len(found) == 1
    assert found[0].review_status is ReviewStatus.PENDING_REVIEW


def test_find__excludes_candidates_below_the_review_floor() -> None:
    """Discarded entirely, not queued: a review queue full of noise is ignored."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, "A8C1234", confidence=0.05))

    assert find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID) == []


def test_find__time_window__excludes_out_of_window_sightings() -> None:
    """Bounding the search is what keeps it cheap and relevant."""
    repositories, recording = _repositories()
    inside = _plated("cam_01", 0, TARGET_PLATE)
    outside = _plated("cam_01", 7200, TARGET_PLATE)
    repositories.sightings.add_batch([inside, outside])

    window = TimeWindow(start_utc=BASE_INSTANT, end_utc=BASE_INSTANT + timedelta(minutes=30))
    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID, time_window=window)

    assert [c.sighting_id for c in found] == [inside.sighting_id]


def test_find__camera_filter__excludes_other_cameras() -> None:
    """Stage 07 narrows to topologically reachable cameras this way."""
    repositories, recording = _repositories()
    wanted = _plated("cam_02", 0, TARGET_PLATE)
    repositories.sightings.add_batch([wanted, _plated("cam_01", 10, TARGET_PLATE)])

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID, camera_ids=["cam_02"])

    assert [c.sighting_id for c in found] == [wanted.sighting_id]


def test_find__no_matches__returns_an_empty_list_not_none() -> None:
    """Callers iterate the result unconditionally."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, "XYZ9999"))

    assert find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID) == []


def test_find__uses_only_the_indexed_prefilter_paths() -> None:
    """The exit criterion, asserted on the call log rather than on results.

    A full scan would return the same answers, just slower -- which is exactly
    why it cannot be caught by checking output.
    """
    repositories, recording = _repositories()
    repositories.sightings.add_batch(
        [_plated("cam_01", index * 10, TARGET_PLATE) for index in range(5)]
    )

    find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert set(recording.calls) <= {"find_by_plate_exact", "find_by_plate_folded"}
    assert FULL_SCAN_METHODS.isdisjoint(recording.calls)


def test_find__deduplicates_across_the_two_prefilters() -> None:
    """An exact match folds identically, so both lookups return it."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, TARGET_PLATE))

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert len(found) == 1


def test_find__ignores_sightings_with_no_plate() -> None:
    """Those are stage 07's problem, not this one."""
    repositories, recording = _repositories()
    repositories.sightings.add(make_sighting("cam_01"))

    assert find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID) == []


def test_find__candidates_carry_the_edit_distance() -> None:
    """The stored record has to explain itself later."""
    repositories, recording = _repositories()
    repositories.sightings.add(_plated("cam_01", 0, "A8C1234"))

    found = find_plate_matches(TARGET_PLATE, recording, target_id=TARGET_ID)

    assert found[0].plate_edit_distance == 1
    assert found[0].match_method.value == "plate_fuzzy"


# ---------------------------------------------------------------------------
# detect_plate_conflicts
# ---------------------------------------------------------------------------


def _topology() -> Topology:
    """Return a two-camera graph with a 300-second minimum transit.

    Returns:
        The graph.
    """
    return make_topology(
        ["cam_01", "cam_02", "cam_03"],
        [("cam_01", "cam_02", 300.0, 900.0), ("cam_02", "cam_03", 300.0, 900.0)],
    )


def test_conflicts__impossible_interval__is_flagged() -> None:
    """One vehicle cannot be in two distant places at once."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 30, TARGET_PLATE)]

    conflicts = detect_plate_conflicts(TARGET_PLATE, sightings, topology)

    assert len(conflicts) == 1
    assert conflicts[0].minimum_transit_sec == pytest.approx(300.0)
    assert conflicts[0].deficit_sec == pytest.approx(270.0)


def test_conflicts__plausible_interval__is_not_flagged() -> None:
    """The same pair, given enough time, is an ordinary drive."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 600, TARGET_PLATE)]

    assert detect_plate_conflicts(TARGET_PLATE, sightings, topology) == []


def test_conflicts__boundary_interval__is_not_flagged() -> None:
    """Boundary: exactly the minimum transit is possible, not impossible."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 300, TARGET_PLATE)]

    assert detect_plate_conflicts(TARGET_PLATE, sightings, topology) == []


def test_conflicts__same_camera__is_never_flagged() -> None:
    """A vehicle lingering in frame or circling back is ordinary."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_01", 1, TARGET_PLATE)]

    assert detect_plate_conflicts(TARGET_PLATE, sightings, topology) == []


def test_conflicts__three_sightings_one_impossible_pair__flags_exactly_that_pair() -> None:
    """Only the genuinely impossible pair is reported.

    Timings are chosen so cam_01 to cam_03 (600s minimum over two legs) is
    comfortably satisfied while cam_02 to cam_03 is not -- otherwise the wider
    pair would be impossible too, and the test would not be isolating anything.
    """
    topology = _topology()
    first = _plated("cam_01", 0, TARGET_PLATE)
    second = _plated("cam_02", 700, TARGET_PLATE)
    third = _plated("cam_03", 800, TARGET_PLATE)

    conflicts = detect_plate_conflicts(TARGET_PLATE, [first, second, third], topology)

    assert len(conflicts) == 1
    assert conflicts[0].from_camera_id == "cam_02"
    assert conflicts[0].to_camera_id == "cam_03"


def test_conflicts__every_pair_is_examined_not_just_consecutive_ones() -> None:
    """A clone on a parallel route can leave each neighbouring pair looking fine
    while a wider pair is impossible."""
    topology = _topology()
    first = _plated("cam_01", 0, TARGET_PLATE)
    second = _plated("cam_02", 400, TARGET_PLATE)
    third = _plated("cam_03", 500, TARGET_PLATE)

    conflicts = detect_plate_conflicts(TARGET_PLATE, [first, second, third], topology)
    pairs = {(c.from_camera_id, c.to_camera_id) for c in conflicts}

    assert ("cam_01", "cam_03") in pairs, "the non-consecutive pair is impossible too"
    assert ("cam_02", "cam_03") in pairs


def test_conflicts__unlinked_pair__is_ignored_by_default() -> None:
    """An undeclared route is far more often a surveying gap than a clone.

    Reporting every such pair would bury the real conflicts.
    """
    topology = make_topology(["cam_01", "cam_02"], [])
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 5, TARGET_PLATE)]

    assert detect_plate_conflicts(TARGET_PLATE, sightings, topology) == []


def test_conflicts__unlinked_pair__is_reported_when_asked_for() -> None:
    """The stricter mode exists for an operator investigating a specific plate."""
    topology = make_topology(["cam_01", "cam_02"], [])
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 5, TARGET_PLATE)]

    conflicts = detect_plate_conflicts(TARGET_PLATE, sightings, topology, include_unlinked=True)

    assert len(conflicts) == 1
    assert conflicts[0].minimum_transit_sec == float("inf")
    assert "no route" in conflicts[0].describe()


def test_conflicts__describe__names_both_cameras_and_the_deficit() -> None:
    """An operator has to be able to act on this without opening a debugger."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_02", 30, TARGET_PLATE)]

    message = detect_plate_conflicts(TARGET_PLATE, sightings, topology)[0].describe()

    assert "cam_01" in message
    assert "cam_02" in message
    assert "270s faster" in message


def test_conflicts__single_sighting__has_nothing_to_conflict_with() -> None:
    """Boundary."""
    assert (
        detect_plate_conflicts(TARGET_PLATE, [_plated("cam_01", 0, TARGET_PLATE)], _topology())
        == []
    )


def test_conflicts__unknown_camera__is_skipped_rather_than_raising() -> None:
    """A sighting from a camera the topology has not surveyed cannot be judged."""
    topology = _topology()
    sightings = [_plated("cam_01", 0, TARGET_PLATE), _plated("cam_99", 5, TARGET_PLATE)]

    assert detect_plate_conflicts(TARGET_PLATE, sightings, topology) == []
