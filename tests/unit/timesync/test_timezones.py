"""Unit tests for local-time resolution.

Two hours a year are genuinely ambiguous in every zone that observes daylight
saving, and a library that picks one silently is wrong half the time. Half the
time is far too often when the consequence is putting a vehicle at a camera an
hour from where it was.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.timesync import resolve_local_time, zone_for

pytestmark = pytest.mark.unit

LONDON = "Europe/London"
KATHMANDU = "Asia/Kathmandu"
"""UTC+05:45 -- a non-integer offset, which is where naive hour arithmetic
breaks."""


def test_an_ordinary_local_time__converts_to_the_expected_utc_instant() -> None:
    """The base case: British Summer Time is UTC+1."""
    resolved = resolve_local_time(datetime(2026, 8, 10, 15, 22, 11), LONDON)

    assert resolved == datetime(2026, 8, 10, 14, 22, 11, tzinfo=UTC)


def test_a_winter_local_time__converts_with_no_offset() -> None:
    """The same zone, outside daylight saving."""
    resolved = resolve_local_time(datetime(2026, 1, 10, 14, 22, 11), LONDON)

    assert resolved == datetime(2026, 1, 10, 14, 22, 11, tzinfo=UTC)


def test_an_ambiguous_local_time__raises_naming_both_candidates() -> None:
    """The repeated hour when the clocks go back.

    01:30 happens twice on that morning, and the two readings are an hour apart
    in UTC. Guessing would be wrong half the time.
    """
    with pytest.raises(ValidationError, match="occurs twice") as excinfo:
        resolve_local_time(datetime(2026, 10, 25, 1, 30, 0), LONDON, camera_id="cam_03")

    context = excinfo.value.context
    assert context["camera_id"] == "cam_03"
    assert len(context["candidates"]) == 2
    assert "utc_offset" in str(excinfo.value)


def test_a_non_existent_local_time__raises_naming_the_instant_and_camera() -> None:
    """The skipped hour when the clocks go forward.

    01:30 does not exist that morning, so a camera reporting it has a wrong
    clock or a wrong configured zone -- both worth an operator's attention.
    """
    with pytest.raises(ValidationError, match="does not exist") as excinfo:
        resolve_local_time(datetime(2026, 3, 29, 1, 30, 0), LONDON, camera_id="cam_07")

    assert excinfo.value.context["camera_id"] == "cam_07"
    assert "2026-03-29T01:30:00" in str(excinfo.value)


def test_an_ambiguous_time_with_an_explicit_offset__resolves_to_that_reading() -> None:
    """How an operator answers the question the error asks."""
    first = resolve_local_time(
        datetime(2026, 10, 25, 1, 30, 0), LONDON, utc_offset=timedelta(hours=1)
    )
    second = resolve_local_time(datetime(2026, 10, 25, 1, 30, 0), LONDON, utc_offset=timedelta(0))

    assert first == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    assert second == datetime(2026, 10, 25, 1, 30, tzinfo=UTC)


def test_an_offset_the_zone_never_uses__is_rejected() -> None:
    """An arbitrary offset would be a second guess dressed as a decision."""
    with pytest.raises(ValidationError, match="is not one"):
        resolve_local_time(datetime(2026, 10, 25, 1, 30, 0), LONDON, utc_offset=timedelta(hours=7))


def test_a_non_integer_offset_zone__converts_correctly() -> None:
    """UTC+05:45. Hour-based arithmetic gets this wrong by 45 minutes."""
    resolved = resolve_local_time(datetime(2026, 8, 10, 20, 7, 11), KATHMANDU)

    assert resolved == datetime(2026, 8, 10, 14, 22, 11, tzinfo=UTC)


def test_two_cameras_in_different_zones__agree_on_one_real_instant() -> None:
    """The property the whole system depends on: comparability."""
    london = resolve_local_time(datetime(2026, 8, 10, 15, 22, 11), LONDON, camera_id="cam_01")
    kathmandu = resolve_local_time(datetime(2026, 8, 10, 20, 7, 11), KATHMANDU, camera_id="cam_02")

    assert london == kathmandu


def test_an_already_aware_timestamp__is_converted_without_ambiguity() -> None:
    """It carries its own offset, so no zone rule has to be guessed at."""
    aware = datetime(2026, 10, 25, 1, 30, tzinfo=ZoneInfo(LONDON), fold=1)

    assert resolve_local_time(aware, LONDON) == aware.astimezone(UTC)


def test_an_unknown_timezone__is_rejected_rather_than_defaulting_to_utc() -> None:
    """Defaulting would silently shift every timestamp from that camera."""
    with pytest.raises(ValidationError, match="Unknown timezone"):
        zone_for("Mars/Olympus_Mons")
