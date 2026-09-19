"""Recording and replaying real detections without a model.

The case worth the most attention is an unrecorded source. Returning nothing for
it would be indistinguishable from an empty road, and a test asserting "no false
positives" would pass triumphantly against nothing at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from multicam_tracker.exceptions import VisionError
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision import (
    FIXTURE_FORMAT_VERSION,
    FakeDetector,
    FixtureDetector,
    build_fixture_payload,
    write_fixture,
)
from tests.fixtures.vision import blank_frame, linear_script, make_detection

pytestmark = pytest.mark.unit

RECORDED_AT = datetime(2026, 8, 10, 16, 0, 0, tzinfo=UTC)


def recording(frames: int = 6, source_id: str = "cam_01_test") -> dict[str, object]:
    """Return a payload recorded from a scripted detector.

    Args:
        frames: How many frames the object appears in.
        source_id: Which source the frames claim to come from.

    Returns:
        The payload.
    """
    detector = FakeDetector(linear_script(frames=frames), model_id="recorder", model_version="2")
    sequence = [blank_frame(index, source_id=source_id) for index in range(frames)]
    return build_fixture_payload(
        detector,
        {source_id: sequence},
        recorded_at=RECORDED_AT,
        note="recorded from a scripted detector in a unit test",
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_build_fixture_payload__records_the_detectors_own_provenance() -> None:
    """A replay cannot silently claim to be something it is not."""
    payload = recording()

    assert payload["model_id"] == "recorder"
    assert payload["model_version"] == "2"
    assert payload["format_version"] == FIXTURE_FORMAT_VERSION
    assert "scripted detector" in str(payload["note"])


def test_build_fixture_payload__omits_frames_with_no_detections() -> None:
    """Most frames of most clips are empty; writing them all would dwarf the signal."""
    detector = FakeDetector({2: [make_detection((10, 10, 40, 40))]})
    frames = [blank_frame(index) for index in range(5)]

    payload = build_fixture_payload(detector, {"cam_01_test": frames}, recorded_at=RECORDED_AT)
    per_frame = payload["sources"]["cam_01_test"]["frames"]  # type: ignore[index]

    assert set(per_frame) == {"2"}
    assert payload["sources"]["cam_01_test"]["frame_count"] == 5  # type: ignore[index]


def test_write_fixture__is_byte_stable_for_an_unchanged_recording(tmp_path: Path) -> None:
    """A diff in a fixture file must mean the detections really changed."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    write_fixture(first, recording())
    write_fixture(second, recording())

    assert first.read_bytes() == second.read_bytes()


def test_write_fixture__creates_missing_directories(tmp_path: Path) -> None:
    """Recording into a fresh checkout must not require a manual mkdir."""
    destination = tmp_path / "nested" / "detections.json"

    write_fixture(destination, recording())

    assert destination.is_file()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_detect__replays_exactly_what_was_recorded() -> None:
    """The whole purpose: real detection patterns, no weights, no variance."""
    detector = FixtureDetector(recording())

    replayed = detector.detect(blank_frame(2, source_id="cam_01_test"))

    assert len(replayed) == 1
    assert replayed[0].object_class is ObjectClass.CAR


def test_detect__a_recorded_frame_with_nothing_in_it__returns_an_empty_list() -> None:
    """A covered source with a quiet frame is not an error; it is a quiet frame."""
    detector = FixtureDetector(recording(frames=3))

    assert detector.detect(blank_frame(99, source_id="cam_01_test")) == []


def test_detect__an_unrecorded_source__raises_rather_than_returning_nothing() -> None:
    """Silently returning nothing is indistinguishable from an empty road.

    A test asserting no false positives would then pass against nothing at all.
    """
    detector = FixtureDetector(recording())

    with pytest.raises(VisionError, match="does not cover that source"):
        detector.detect(blank_frame(0, source_id="some_other_clip"))


def test_replay__carries_the_recordings_provenance_onto_every_detection() -> None:
    """Which model put this box here has to be answerable from the record alone."""
    detector = FixtureDetector(recording())

    detection = detector.detect(blank_frame(0, source_id="cam_01_test"))[0]

    assert detection.model_id == "recorder"
    assert detection.model_version == "2"


def test_replay__reproduces_the_recorded_run_exactly() -> None:
    """The obligation that makes a fixture worth committing."""
    source_id = "cam_01_test"
    detector = FakeDetector(linear_script(frames=6), model_id="recorder", model_version="2")
    frames = [blank_frame(index, source_id=source_id) for index in range(6)]

    replay = FixtureDetector(recording())

    for frame in frames:
        live = detector.detect(frame)
        recorded = replay.detect(frame)
        assert [entry.bbox for entry in live] == [entry.bbox for entry in recorded]
        assert [round(entry.confidence, 6) for entry in live] == [
            entry.confidence for entry in recorded
        ]


def test_covered_sources__reports_what_the_recording_spans() -> None:
    """The error message for an unrecorded source is only useful with this list."""
    assert FixtureDetector(recording()).covered_sources == ("cam_01_test",)


# ---------------------------------------------------------------------------
# Malformed and stale files
# ---------------------------------------------------------------------------


def test_from_file__round_trips_through_disk(tmp_path: Path) -> None:
    """The committed path: write it, load it, replay it."""
    path = tmp_path / "detections.json"
    write_fixture(path, recording())

    detector = FixtureDetector.from_file(path)

    assert detector.path == path
    assert len(detector.detect(blank_frame(0, source_id="cam_01_test"))) == 1


def test_from_file__a_missing_file__names_the_script_that_records_one(tmp_path: Path) -> None:
    """The operator's next action is to run something; the error has to say what."""
    with pytest.raises(VisionError, match="record_detection_fixtures"):
        FixtureDetector.from_file(tmp_path / "absent.json")


def test_from_file__invalid_json__is_reported_as_unreadable(tmp_path: Path) -> None:
    """A truncated fixture must fail at load, not as mysterious empty detections."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(VisionError, match="could not be read"):
        FixtureDetector.from_file(path)


def test_constructor__an_older_format_version__is_refused(tmp_path: Path) -> None:
    """A stale recording is identifiable rather than merely suspected."""
    payload = dict(recording())
    payload["format_version"] = FIXTURE_FORMAT_VERSION - 1

    with pytest.raises(VisionError, match="not supported"):
        FixtureDetector(payload)


def test_constructor__a_recording_with_no_sources__is_refused() -> None:
    """Boundary: an empty recording would make every clip look like an empty road."""
    with pytest.raises(VisionError, match="no sources"):
        FixtureDetector({"format_version": FIXTURE_FORMAT_VERSION, "sources": {}})


def test_constructor__a_malformed_detection__names_the_file(tmp_path: Path) -> None:
    """A hand-edited fixture is the likely cause, and the path is what locates it."""
    path = tmp_path / "detections.json"
    payload = {
        "format_version": FIXTURE_FORMAT_VERSION,
        "model_id": "m",
        "model_version": "1",
        "sources": {"cam_01_test": {"frames": {"0": [{"confidence": 0.9}]}}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VisionError, match="no usable bbox"):
        FixtureDetector.from_file(path)
