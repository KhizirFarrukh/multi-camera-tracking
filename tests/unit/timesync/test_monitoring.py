"""Unit tests for the clock-probe monitoring hook.

Direct measurement beats inference wherever a device exposes its clock, which is
why this is the preferred mechanism and reference-event estimation is the
fallback. The failure path matters as much as the happy one: one unreachable
camera must not abort a sweep of forty.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.timesync import ClockSample, ClockSampleSeries, probe_cameras

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def test_a_camera_ahead_of_the_server__reports_a_positive_offset() -> None:
    """The sign convention, stated once and tested."""
    sample = ClockSample(
        camera_id="cam_03", observed_at_utc=NOW, camera_time_utc=NOW + timedelta(seconds=45)
    )

    assert sample.offset_ms == pytest.approx(45_000)


def test_the_round_trip__is_halved_out_of_the_measurement() -> None:
    """The usual assumption: request and response took equally long."""
    sample = ClockSample(
        camera_id="cam_03",
        observed_at_utc=NOW,
        camera_time_utc=NOW + timedelta(milliseconds=1000),
        round_trip_ms=200.0,
    )

    assert sample.offset_ms == pytest.approx(900.0)


def test_the_suggested_correction__is_the_negation_of_the_measured_offset() -> None:
    """A camera running 800 ms fast needs 800 ms subtracted."""
    series = ClockSampleSeries(camera_id="cam_03")
    series.record(
        ClockSample(
            camera_id="cam_03",
            observed_at_utc=NOW,
            camera_time_utc=NOW + timedelta(milliseconds=800),
        )
    )

    assert series.suggested_offset_ms() == -800


def test_an_empty_series__suggests_nothing() -> None:
    """No measurement is not a measurement of zero."""
    assert ClockSampleSeries(camera_id="cam_03").suggested_offset_ms() is None
    assert ClockSampleSeries(camera_id="cam_03").verified_at_utc is None


def test_a_series__fits_the_rate_at_which_a_clock_is_drifting() -> None:
    """One comparison cannot tell a constant offset from a drifting clock.

    The difference decides whether an operator corrects the camera once or
    replaces its time client, so the series is kept rather than a latest value.
    """
    series = ClockSampleSeries(camera_id="cam_03")
    for hour in range(5):
        series.record(
            ClockSample(
                camera_id="cam_03",
                observed_at_utc=NOW + timedelta(hours=hour),
                camera_time_utc=NOW + timedelta(hours=hour, milliseconds=500 * hour),
            )
        )

    assert series.drift_rate_ms_per_hour() == pytest.approx(500.0, rel=0.01)


def test_a_stable_clock__fits_a_rate_of_zero() -> None:
    """The control."""
    series = ClockSampleSeries(camera_id="cam_03")
    for hour in range(5):
        series.record(
            ClockSample(
                camera_id="cam_03",
                observed_at_utc=NOW + timedelta(hours=hour),
                camera_time_utc=NOW + timedelta(hours=hour, milliseconds=120),
            )
        )

    assert series.drift_rate_ms_per_hour() == pytest.approx(0.0, abs=1e-9)


def test_two_samples__do_not_produce_a_rate() -> None:
    """A line through two points is a confident claim from no evidence."""
    series = ClockSampleSeries(camera_id="cam_03")
    for hour in range(2):
        series.record(
            ClockSample(
                camera_id="cam_03",
                observed_at_utc=NOW + timedelta(hours=hour),
                camera_time_utc=NOW + timedelta(hours=hour, milliseconds=500 * hour),
            )
        )

    assert series.drift_rate_ms_per_hour() == 0.0


def test_samples_recorded_out_of_order__are_kept_in_observation_order() -> None:
    """The rate fit and the latest-offset read both depend on the ordering."""
    series = ClockSampleSeries(camera_id="cam_03")
    later = ClockSample(
        camera_id="cam_03",
        observed_at_utc=NOW + timedelta(hours=2),
        camera_time_utc=NOW + timedelta(hours=2, milliseconds=900),
    )
    earlier = ClockSample(
        camera_id="cam_03", observed_at_utc=NOW, camera_time_utc=NOW + timedelta(milliseconds=100)
    )

    series.record(later)
    series.record(earlier)

    assert series.latest_offset_ms == pytest.approx(900.0)
    assert series.verified_at_utc == later.observed_at_utc


# ---------------------------------------------------------------------------
# Sweeping several cameras
# ---------------------------------------------------------------------------


def test_probing__records_one_sample_per_camera() -> None:
    """The ordinary sweep."""
    clock = FixedClock(NOW)

    def probe(camera_id: str) -> tuple[datetime, float]:
        offsets = {"cam_01": 0, "cam_02": 500, "cam_03": -250}
        return (NOW + timedelta(milliseconds=offsets[camera_id]), 0.0)

    series = probe_cameras(["cam_01", "cam_02", "cam_03"], probe, clock)

    assert series["cam_02"].latest_offset_ms == pytest.approx(500.0)
    assert series["cam_03"].latest_offset_ms == pytest.approx(-250.0)


def test_an_unreachable_camera__does_not_abort_the_sweep() -> None:
    """A sweep of forty must not stop at the one that is down.

    The gap in its series is what the staleness check reads later, which is the
    right outcome: an unprobed camera is unverified, not healthy.
    """
    clock = FixedClock(NOW)

    def probe(camera_id: str) -> tuple[datetime, float]:
        if camera_id == "cam_02":
            msg = "connection refused"
            raise TimeoutError(msg)
        return (NOW, 0.0)

    series = probe_cameras(["cam_01", "cam_02", "cam_03"], probe, clock)

    assert series["cam_01"].samples
    assert series["cam_02"].samples == []
    assert series["cam_03"].samples


def test_probing_again__appends_to_the_existing_series() -> None:
    """A rate can only be fitted across sweeps, so history is carried forward."""
    first = probe_cameras(["cam_01"], lambda _camera: (NOW, 0.0), FixedClock(NOW))

    later = NOW + timedelta(hours=1)
    second = probe_cameras(
        ["cam_01"],
        lambda _camera: (later + timedelta(milliseconds=400), 0.0),
        FixedClock(later),
        series=first,
    )

    assert len(second["cam_01"].samples) == 2
    assert second["cam_01"].latest_offset_ms == pytest.approx(400.0)


def test_a_naive_camera_time__is_rejected() -> None:
    """A camera reporting local time without saying so is a configuration fault."""
    from multicam_tracker.exceptions import ValidationError

    with pytest.raises(ValidationError, match="Naive datetime"):
        probe_cameras(
            ["cam_01"],
            lambda _camera: (datetime(2026, 8, 24, 12, 0, 0), 0.0),
            FixedClock(NOW),
        )
