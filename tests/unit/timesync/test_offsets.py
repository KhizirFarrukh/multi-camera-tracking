"""Unit tests for clock offset application.

Idempotency is the property worth the most here. Correction happens once, at
ingestion, and a value corrected twice is not obviously wrong -- it is merely
somewhere else, and every downstream conclusion follows from the wrong place.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multicam_tracker.clock import FixedClock
from multicam_tracker.timesync import apply_offset, correct_sighting, recompute_offsets
from tests.fixtures.factories import BASE_INSTANT, make_sighting

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 23, 9, 0, 0, tzinfo=UTC)
_HYPOTHESIS = settings(max_examples=200, deadline=None)


def test_a_positive_offset__shifts_the_corrected_time_later() -> None:
    """A camera running slow needs time added."""
    corrected = apply_offset(BASE_INSTANT, 45_000)

    assert corrected == BASE_INSTANT + timedelta(seconds=45)


def test_a_negative_offset__shifts_the_corrected_time_earlier() -> None:
    """A camera running fast needs time subtracted."""
    corrected = apply_offset(BASE_INSTANT, -45_000)

    assert corrected == BASE_INSTANT - timedelta(seconds=45)


def test_a_zero_offset__leaves_the_timestamp_untouched() -> None:
    """The ordinary case for a healthy camera."""
    assert apply_offset(BASE_INSTANT, 0) == BASE_INSTANT


def test_correcting_a_sighting__records_the_offset_it_applied() -> None:
    """The Sighting contract keeps all three values so the correction is auditable."""
    corrected = correct_sighting(make_sighting("cam_01"), -45_000)

    assert corrected.clock_offset_applied_ms == -45_000
    assert corrected.timestamp_utc == corrected.raw_timestamp - timedelta(seconds=45)


def test_a_corrected_sighting__satisfies_the_model_cross_field_validator() -> None:
    """Constructing it is the check: the model rejects inconsistent triples."""
    corrected = correct_sighting(make_sighting("cam_01"), 1_500)

    assert corrected.model_validate(corrected.model_dump()) == corrected


def test_correcting_twice__does_not_double_shift() -> None:
    """Idempotency at the boundary.

    Recomputing from ``raw_timestamp`` rather than adjusting ``timestamp_utc``
    is what makes running ingestion twice over the same record harmless.
    """
    once = correct_sighting(make_sighting("cam_01"), -45_000)
    twice = correct_sighting(once, -45_000)

    assert twice.timestamp_utc == once.timestamp_utc


def test_correcting_with_a_different_offset__replaces_rather_than_accumulates() -> None:
    """A revised estimate supersedes the old one; it does not add to it."""
    first = correct_sighting(make_sighting("cam_01"), -45_000)
    revised = correct_sighting(first, -30_000)

    assert revised.timestamp_utc == revised.raw_timestamp - timedelta(seconds=30)


# ---------------------------------------------------------------------------
# Recomputing a camera's stored sightings
# ---------------------------------------------------------------------------


def test_recompute__rewrites_only_the_named_camera_s_sightings() -> None:
    """A mixed batch is safe to pass; other cameras come back untouched."""
    sightings = [
        make_sighting("cam_01", offset_sec=0.0),
        make_sighting("cam_02", offset_sec=60.0),
        make_sighting("cam_01", offset_sec=120.0),
    ]

    rewritten, change = recompute_offsets("cam_01", -45_000, sightings, FixedClock(NOW))

    assert change.sightings_updated == 2
    assert [s.clock_offset_applied_ms for s in rewritten] == [-45_000, 0, -45_000]


def test_recompute__records_both_the_old_and_the_new_offset() -> None:
    """ "The offset is now -45000" is unanswerable without knowing what it was."""
    sightings = [correct_sighting(make_sighting("cam_01"), -10_000)]

    _rewritten, change = recompute_offsets("cam_01", -45_000, sightings, FixedClock(NOW))

    assert change.previous_offset_ms == -10_000
    assert change.new_offset_ms == -45_000
    assert change.shift_ms == -35_000


def test_recompute__produces_an_audit_entry_naming_the_blast_radius() -> None:
    """An operator reviewing the change needs to know how much data moved."""
    sightings = [make_sighting("cam_01", offset_sec=index * 60.0) for index in range(4)]

    _rewritten, change = recompute_offsets(
        "cam_01", 2_000, sightings, FixedClock(NOW), reason="ntp sweep"
    )
    entry = change.audit_entry()

    assert entry["event"] == "camera_clock_offset_changed"
    assert entry["sightings_updated"] == 4
    assert entry["reason"] == "ntp sweep"
    assert entry["changed_at_utc"] == NOW.isoformat()


def test_recompute__leaves_the_input_sightings_unmodified() -> None:
    """The caller owns the transaction, so nothing may be mutated in place.

    Returning new objects is what lets a repository write all of them or none.
    """
    original = make_sighting("cam_01")
    sightings = [original]

    recompute_offsets("cam_01", -45_000, sightings, FixedClock(NOW))

    assert original.clock_offset_applied_ms == 0
    assert sightings[0] is original


def test_recompute__on_a_camera_with_no_sightings__is_a_no_op_that_still_records() -> None:
    """Changing a quiet camera's offset is a real change, worth an audit line."""
    rewritten, change = recompute_offsets(
        "cam_09", -1_000, [make_sighting("cam_01")], FixedClock(NOW)
    )

    assert change.sightings_updated == 0
    assert rewritten[0].clock_offset_applied_ms == 0


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


@_HYPOTHESIS
@given(
    offset_ms=st.integers(min_value=-86_400_000, max_value=86_400_000),
    shift_sec=st.integers(min_value=0, max_value=100_000),
)
def test_the_three_timestamp_fields__always_agree(offset_ms: int, shift_sec: int) -> None:
    """Property: ``timestamp_utc - raw_timestamp`` is exactly the recorded offset.

    The invariant every elapsed-time calculation in the system depends on.
    """
    corrected = correct_sighting(make_sighting("cam_01", offset_sec=shift_sec), offset_ms)

    delta_ms = (corrected.timestamp_utc - corrected.raw_timestamp).total_seconds() * 1000

    assert round(delta_ms) == corrected.clock_offset_applied_ms == offset_ms
