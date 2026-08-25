"""Unit tests for the temporal integrity gate.

The gate decides whether a route is worth assembling at all, and the severity
split is what it exists for: a stale verification is worth saying, while offsets
larger than the shortest transit make hop ordering itself unsupported and there
is no useful route to return.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.exceptions import TemporalIntegrityError
from multicam_tracker.models import TemporalIntegrity, TemporalSeverity
from multicam_tracker.timesync import (
    CameraTimeStatus,
    DriftAlert,
    ReliabilityTier,
    assert_temporal_integrity,
    check_temporal_integrity,
)
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
CLOCK = FixedClock(NOW)
RECENT = NOW - timedelta(hours=2)
CAMERAS = ["cam_01", "cam_02", "cam_03"]
SHORTEST_TRANSIT_SEC = 60.0


def _topology() -> object:
    """Return a chain whose shortest transit is 60 seconds.

    Returns:
        ``cam_01 <-> cam_02 <-> cam_03``.
    """
    return make_topology(
        CAMERAS,
        [
            ("cam_01", "cam_02", SHORTEST_TRANSIT_SEC, 300.0),
            ("cam_02", "cam_03", 90.0, 300.0),
        ],
        bidirectional=True,
    )


def _healthy(**overrides: object) -> dict[str, CameraTimeStatus]:
    """Return statuses for three recently verified, high-reliability cameras.

    Args:
        **overrides: Per-camera replacements.

    Returns:
        Statuses by camera id.
    """
    statuses = {
        camera: CameraTimeStatus(camera, verified_at_utc=RECENT, tier=ReliabilityTier.HIGH)
        for camera in CAMERAS
    }
    statuses.update(overrides)  # type: ignore[arg-type]
    return statuses


def test_a_fully_healthy_camera_set__passes_with_no_caveats() -> None:
    """The clean case must be genuinely clean, or every route carries noise."""
    verdict = check_temporal_integrity(CAMERAS, _healthy(), _topology(), CLOCK)

    assert verdict.verified
    assert verdict.caveats == []
    assert verdict.worst_severity is None
    assert "verification is current" in verdict.summary()


def test_a_camera_never_verified__produces_a_warning_naming_it() -> None:
    """No record is not the same as a clean record, and must not read like one."""
    verdict = check_temporal_integrity(
        CAMERAS, _healthy(cam_02=CameraTimeStatus("cam_02")), _topology(), CLOCK
    )

    assert not verdict.verified
    codes = [(caveat.code, caveat.camera_id) for caveat in verdict.caveats]
    assert ("stale_verification", "cam_02") in codes


def test_a_camera_absent_from_the_status_map__is_treated_as_unverified() -> None:
    """ "We have no record" is the honest reading of a missing entry."""
    statuses = _healthy()
    del statuses["cam_03"]

    verdict = check_temporal_integrity(CAMERAS, statuses, _topology(), CLOCK)

    assert any(caveat.camera_id == "cam_03" for caveat in verdict.caveats)


def test_a_verification_older_than_the_staleness_limit__warns() -> None:
    """A week-old check is not evidence about today's clock."""
    stale = CameraTimeStatus("cam_02", verified_at_utc=NOW - timedelta(hours=200))

    verdict = check_temporal_integrity(
        CAMERAS, _healthy(cam_02=stale), _topology(), CLOCK, staleness_hours=168.0
    )

    caveat = next(entry for entry in verdict.caveats if entry.camera_id == "cam_02")
    assert caveat.code == "stale_verification"
    assert caveat.severity is TemporalSeverity.WARNING
    assert caveat.context["staleness_hours"] == pytest.approx(200.0)


def test_a_verification_inside_the_limit__produces_no_caveat() -> None:
    """The boundary the staleness limit draws."""
    fresh = CameraTimeStatus("cam_02", verified_at_utc=NOW - timedelta(hours=167))

    verdict = check_temporal_integrity(
        CAMERAS, _healthy(cam_02=fresh), _topology(), CLOCK, staleness_hours=168.0
    )

    assert verdict.verified


def test_an_active_drift_alert__is_attached_as_a_caveat_carrying_its_detail() -> None:
    """The operator needs the alert's own explanation, not a generic one."""
    alert = DriftAlert(
        camera_id="cam_03",
        kind="constant_offset",
        magnitude_ms=45_000,
        rate_ms_per_hour=0.0,
        sample_count=6,
        detail="cam_03 reports +45000 ms away from what the topology expects",
    )

    verdict = check_temporal_integrity(
        CAMERAS,
        _healthy(cam_03=CameraTimeStatus("cam_03", verified_at_utc=RECENT, drift_alert=alert)),
        _topology(),
        CLOCK,
    )

    caveat = next(entry for entry in verdict.caveats if entry.code == "drift_alert")
    assert caveat.severity is TemporalSeverity.WARNING
    assert "45000 ms" in caveat.detail
    assert caveat.context["kind"] == "constant_offset"


@pytest.mark.parametrize(
    ("tier", "severity"),
    [
        (ReliabilityTier.MEDIUM, TemporalSeverity.INFO),
        (ReliabilityTier.LOW, TemporalSeverity.WARNING),
    ],
)
def test_a_lower_reliability_source__produces_a_caveat_of_the_matching_severity(
    tier: ReliabilityTier, severity: TemporalSeverity
) -> None:
    """A filename-derived timestamp changes what a route is worth as evidence."""
    verdict = check_temporal_integrity(
        CAMERAS,
        _healthy(cam_01=CameraTimeStatus("cam_01", verified_at_utc=RECENT, tier=tier)),
        _topology(),
        CLOCK,
    )

    caveat = next(entry for entry in verdict.caveats if entry.code == "low_reliability_source")
    assert caveat.severity is severity
    assert caveat.context["tier"] == tier.value


def test_offsets_wider_than_the_shortest_transit__block_outright() -> None:
    """Beyond this point hop ordering is not determined by the data.

    Two sightings closer together than the clock error can swap places, and a
    trajectory *is* an ordering -- so there is no useful route to return.
    """
    statuses = _healthy(
        cam_01=CameraTimeStatus("cam_01", offset_ms=0, verified_at_utc=RECENT),
        cam_03=CameraTimeStatus("cam_03", offset_ms=90_000, verified_at_utc=RECENT),
    )

    verdict = check_temporal_integrity(CAMERAS, statuses, _topology(), CLOCK)

    caveat = next(entry for entry in verdict.caveats if entry.code == "offset_spread")
    assert caveat.severity is TemporalSeverity.BLOCKING
    assert verdict.is_blocking
    assert caveat.context["smallest_travel_time_sec"] == SHORTEST_TRANSIT_SEC


def test_offsets_inside_the_shortest_transit__do_not_block() -> None:
    """Known, corrected offsets are not a problem; unknowable ones are."""
    statuses = _healthy(cam_03=CameraTimeStatus("cam_03", offset_ms=30_000, verified_at_utc=RECENT))

    verdict = check_temporal_integrity(CAMERAS, statuses, _topology(), CLOCK)

    assert not verdict.is_blocking


def test_a_single_camera__cannot_have_an_offset_spread() -> None:
    """Boundary: one clock is trivially consistent with itself."""
    verdict = check_temporal_integrity(
        ["cam_01"],
        {"cam_01": CameraTimeStatus("cam_01", offset_ms=900_000, verified_at_utc=RECENT)},
        _topology(),
        CLOCK,
    )

    assert not verdict.is_blocking


# ---------------------------------------------------------------------------
# The assertion form
# ---------------------------------------------------------------------------


def test_assert__returns_the_verdict_when_only_warnings_are_present() -> None:
    """Warnings belong on the trajectory, not in an exception handler."""
    verdict = assert_temporal_integrity(
        CAMERAS, _healthy(cam_02=CameraTimeStatus("cam_02")), _topology(), CLOCK
    )

    assert not verdict.verified
    assert verdict.caveats


def test_assert__raises_on_a_blocking_caveat_and_names_every_reason() -> None:
    """An operator fixing this needs all of them, not the first one."""
    statuses = _healthy(cam_03=CameraTimeStatus("cam_03", offset_ms=90_000, verified_at_utc=RECENT))

    with pytest.raises(TemporalIntegrityError, match="not comparable") as excinfo:
        assert_temporal_integrity(CAMERAS, statuses, _topology(), CLOCK)

    assert "offset_spread" in excinfo.value.context["codes"]


def test_the_verdict__survives_serialization_intact() -> None:
    """It is stored on the trajectory and read back by the review UI."""
    original = check_temporal_integrity(
        CAMERAS, _healthy(cam_02=CameraTimeStatus("cam_02")), _topology(), CLOCK
    )

    restored = TemporalIntegrity.from_json_dict(original.to_json_dict())

    assert restored == original
    assert restored.caveats[0].severity is TemporalSeverity.WARNING


def test_the_summary__lists_every_caveat_for_a_trajectory_listing() -> None:
    """One line an operator can read beside the route."""
    verdict = check_temporal_integrity(
        CAMERAS,
        _healthy(
            cam_01=CameraTimeStatus("cam_01"),
            cam_02=CameraTimeStatus("cam_02", verified_at_utc=RECENT, tier=ReliabilityTier.LOW),
        ),
        _topology(),
        CLOCK,
    )

    summary = verdict.summary()

    assert "cam_01" in summary
    assert "cam_02" in summary
