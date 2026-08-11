"""Integration tests loading the shared JSON fixtures into domain models.

These files are the shared sample data later stages reuse -- stage 03 seeds the
database from them, stage 04 builds its topology from them, stage 05 calibrates
its generator against them. Loading them here proves the fixtures and the models
agree before anything else depends on that.

Marked ``integration`` because they read real files from disk rather than
constructed objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from multicam_tracker.models import Camera, CameraLink, ObjectClass, Sighting

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    """Read a fixture file and strip human-facing annotations.

    Keys beginning with an underscore document *why* a malformed record is
    malformed. They are stripped before validation because ``extra="forbid"``
    would otherwise report the annotation itself as the defect and mask the real
    one.

    Args:
        name: File name inside ``tests/fixtures/``.

    Returns:
        The decoded records with annotation keys removed.
    """
    records = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]


# ---------------------------------------------------------------------------
# Well-formed fixtures
# ---------------------------------------------------------------------------


def test_sample_cameras__load_into_camera_models() -> None:
    """Every camera in the shared fixture is valid."""
    cameras = [Camera.from_json_dict(record) for record in _load("sample_cameras.json")]

    assert len(cameras) == 4
    assert [camera.camera_id for camera in cameras] == ["cam_01", "cam_02", "cam_03", "cam_04"]


def test_sample_cameras__exercise_the_optional_fields() -> None:
    """The fixture is worth keeping only if it covers the interesting cases."""
    cameras = {
        camera.camera_id: camera
        for camera in (Camera.from_json_dict(record) for record in _load("sample_cameras.json"))
    }

    assert cameras["cam_03"].clock_offset_ms == 2000, "clock drift case"
    assert cameras["cam_04"].enabled is False, "disabled-camera case"
    assert cameras["cam_04"].heading_degrees == pytest.approx(359.9), "upper heading boundary"
    assert cameras["cam_02"].notes is None, "null optional case"


def test_sample_links__load_into_camera_link_models() -> None:
    """Every link in the shared fixture is valid."""
    links = [CameraLink.from_json_dict(record) for record in _load("sample_links.json")]

    assert len(links) == 4
    assert all(link.max_travel_time_sec > link.min_travel_time_sec for link in links)


def test_sample_links__reference_only_known_cameras() -> None:
    """A link to a camera that does not exist would break stage 04's graph build."""
    known = {Camera.from_json_dict(record).camera_id for record in _load("sample_cameras.json")}
    links = [CameraLink.from_json_dict(record) for record in _load("sample_links.json")]

    referenced = {link.from_camera_id for link in links} | {link.to_camera_id for link in links}
    assert referenced <= known


def test_sample_sightings__load_into_sighting_models() -> None:
    """Every sighting in the shared fixture is valid."""
    sightings = [Sighting.from_json_dict(record) for record in _load("sample_sightings.json")]

    assert len(sightings) == 4
    assert [sighting.camera_id for sighting in sightings] == [
        "cam_01",
        "cam_02",
        "cam_03",
        "cam_04",
    ]


def test_sample_sightings__are_ordered_and_exercise_clock_correction() -> None:
    """The fixture models a single vehicle's route, including a drifting camera."""
    sightings = [Sighting.from_json_dict(record) for record in _load("sample_sightings.json")]

    timestamps = [sighting.timestamp_utc for sighting in sightings]
    assert timestamps == sorted(timestamps), "sightings must form an ascending route"

    drifting = next(s for s in sightings if s.camera_id == "cam_03")
    assert drifting.clock_offset_applied_ms == 2000
    assert drifting.timestamp_utc > drifting.raw_timestamp

    assert sightings[0].has_plate is True, "a clean plate read"
    assert sightings[2].has_plate is False, "an unreadable plate"
    assert sightings[3].object_class is ObjectClass.TRUCK, "a non-car class"


def test_sample_sightings__round_trip_back_to_the_fixture_payload() -> None:
    """Serializing a loaded fixture reproduces the file's content exactly.

    This is what guarantees the on-disk format and the model stay in step: if a
    field were renamed or a datetime rendered differently, this fails.
    """
    records = _load("sample_sightings.json")

    for record in records:
        assert Sighting.from_json_dict(record).to_json_dict() == record


# ---------------------------------------------------------------------------
# Malformed fixture
# ---------------------------------------------------------------------------


def test_malformed_sightings__first_record__is_the_valid_control() -> None:
    """The file's first record is deliberately valid, isolating the later defects."""
    assert Sighting.from_json_dict(_load("malformed_sightings.json")[0])


@pytest.mark.parametrize(
    ("index", "expected_field", "expected_message"),
    [
        (1, "bbox", "x2 > x1"),
        (2, "timestamp_utc", "does not equal"),
        (3, "plate_confidence", "plate_confidence is required"),
    ],
    ids=["inverted-bbox", "timestamp-inconsistent", "plate-without-confidence"],
)
def test_malformed_sightings__each_bad_record__names_its_specific_defect(
    index: int, expected_field: str, expected_message: str
) -> None:
    """A validation failure must identify the record and the reason, not just fail.

    Stage 03 will load thousands of rows; an error saying only "invalid input"
    would be useless for finding the one bad record among them.
    """
    records = json.loads((FIXTURE_DIR / "malformed_sightings.json").read_text(encoding="utf-8"))
    record = records[index]
    payload = {key: value for key, value in record.items() if not key.startswith("_")}

    with pytest.raises(ValidationError) as excinfo:
        Sighting.from_json_dict(payload)

    message = str(excinfo.value)
    assert expected_message in message
    assert expected_field in message or expected_field in record["_defect"]

    # The fixture's own annotation must describe the defect the model reports,
    # so the file stays a trustworthy explanation of what it is testing.
    assert record["_defect"]


def test_malformed_sightings__every_annotated_record__actually_fails() -> None:
    """No record may carry a _defect note while quietly validating."""
    records = json.loads((FIXTURE_DIR / "malformed_sightings.json").read_text(encoding="utf-8"))

    for index, record in enumerate(records):
        payload = {key: value for key, value in record.items() if not key.startswith("_")}
        if "_defect" not in record:
            continue
        with pytest.raises(ValidationError):
            Sighting.from_json_dict(payload)
            pytest.fail(f"records[{index}] is annotated as malformed but validated cleanly")
