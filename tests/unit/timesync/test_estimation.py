"""Unit tests for offset estimation from reference passes.

Every case here injects a known offset and asks the estimator to recover it.
That is the only honest way to test a solver: comparing it against itself proves
nothing, and comparing it against real data has no answer key.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.timesync import ReferencePass, estimate_offsets
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
EXPECTED_SEC = 120.0
"""The transit time the topology considers typical for every link here."""

TOLERANCE_MS = 50.0
"""Recovery tolerance. Tight: with consistent data the solver should land on the
injected value almost exactly, and a loose tolerance would hide a real bias."""


def _topology() -> object:
    """Return a four-camera chain.

    Returns:
        ``cam_01 -> cam_02 -> cam_03 -> cam_04``.
    """
    return make_topology(
        ["cam_01", "cam_02", "cam_03", "cam_04"],
        [
            ("cam_01", "cam_02", 60.0, 300.0),
            ("cam_02", "cam_03", 60.0, 300.0),
            ("cam_03", "cam_04", 60.0, 300.0),
        ],
        bidirectional=True,
    )


def _passes(
    injected_offsets_sec: dict[str, float],
    *,
    count: int = 4,
    legs: tuple[tuple[str, str], ...] = (
        ("cam_01", "cam_02"),
        ("cam_02", "cam_03"),
        ("cam_03", "cam_04"),
    ),
) -> list[ReferencePass]:
    """Build reference passes for vehicles driving a chain with injected drift.

    Args:
        injected_offsets_sec: How far each camera's clock runs ahead of truth.
        count: How many reference vehicles drive the route.
        legs: Camera pairs each vehicle traverses, in order.

    Returns:
        One pass per vehicle per leg, with the injected drift applied to the
        reported times exactly as a drifting camera would.
    """
    passes: list[ReferencePass] = []
    for vehicle in range(count):
        base = START + timedelta(minutes=10 * vehicle)
        for leg_index, (origin, destination) in enumerate(legs):
            true_departure = base + timedelta(seconds=EXPECTED_SEC * leg_index)
            true_arrival = true_departure + timedelta(seconds=EXPECTED_SEC)
            passes.append(
                ReferencePass(
                    from_camera_id=origin,
                    to_camera_id=destination,
                    departure_utc=true_departure
                    + timedelta(seconds=injected_offsets_sec.get(origin, 0.0)),
                    arrival_utc=true_arrival
                    + timedelta(seconds=injected_offsets_sec.get(destination, 0.0)),
                    expected_sec=EXPECTED_SEC,
                    vehicle_id=f"ref_{vehicle}",
                )
            )
    return passes


def _scatter_sec(index: int) -> float:
    """Return a deterministic per-pass perturbation.

    The period is coprime with the number of legs on purpose. An earlier version
    used ``index % 3``, which gave every vehicle the same shift on the same leg
    -- leaving the system perfectly consistent and the residuals at zero, so the
    tests below were asserting nothing.

    Args:
        index: Position of the pass in the list.

    Returns:
        Seconds to add to the reported arrival, in ``[-30, 30]``.
    """
    return float((index * 37) % 61 - 30)


def test_a_single_injected_offset__is_recovered_within_tolerance() -> None:
    """The base case: one camera 45 seconds fast, everything else correct."""
    estimation = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    )

    # The correction is the negation of the error: a camera reading 45 s late
    # needs 45 s subtracted.
    assert estimation.offset_ms_for("cam_03") == pytest.approx(-45_000, abs=TOLERANCE_MS)


def test_several_cameras__are_solved_simultaneously() -> None:
    """The equations only constrain differences, so they must be solved together."""
    estimation = estimate_offsets(
        _passes({"cam_02": 12.0, "cam_03": 45.0, "cam_04": -8.0}),
        _topology(),
        reference_camera_id="cam_01",
    )

    assert estimation.offset_ms_for("cam_02") == pytest.approx(-12_000, abs=TOLERANCE_MS)
    assert estimation.offset_ms_for("cam_03") == pytest.approx(-45_000, abs=TOLERANCE_MS)
    assert estimation.offset_ms_for("cam_04") == pytest.approx(8_000, abs=TOLERANCE_MS)


def test_the_reference_camera__is_assigned_exactly_zero() -> None:
    """Its offset is a definition, not a measurement.

    Without the pin the system is underdetermined: adding an hour to every
    camera fits the data exactly as well.
    """
    estimation = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    )

    assert estimation.offset_ms_for("cam_01") == pytest.approx(0.0, abs=1e-6)
    assert estimation.estimates["cam_01"].is_reference


def test_cameras_with_no_drift__are_estimated_at_zero() -> None:
    """A correct clock must not be corrected."""
    estimation = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    )

    assert estimation.offset_ms_for("cam_02") == pytest.approx(0.0, abs=TOLERANCE_MS)


def test_consistent_data__produces_small_residuals() -> None:
    """Residuals are how an operator tells a good estimate from a fitted one."""
    estimation = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    )

    assert estimation.estimates["cam_03"].residual_rms_ms < TOLERANCE_MS


def test_inconsistent_data__produces_large_residuals() -> None:
    """The estimate still comes back, and says loudly that it does not fit."""
    passes = _passes({})
    noisy = [
        ReferencePass(
            from_camera_id=entry.from_camera_id,
            to_camera_id=entry.to_camera_id,
            departure_utc=entry.departure_utc,
            arrival_utc=entry.arrival_utc + timedelta(seconds=_scatter_sec(index)),
            expected_sec=entry.expected_sec,
            vehicle_id=entry.vehicle_id,
        )
        for index, entry in enumerate(passes)
    ]

    estimation = estimate_offsets(noisy, _topology(), reference_camera_id="cam_01")

    assert estimation.estimates["cam_02"].residual_rms_ms > 1_000


def test_a_single_outlier_pass__does_not_dominate_the_estimate() -> None:
    """One mismatched vehicle must not move every camera's clock.

    Plain least squares spreads a wild residual across the whole graph; the
    reweighting is what keeps it local.
    """
    passes = _passes({"cam_03": 45.0}, count=6)
    passes.append(
        ReferencePass(
            from_camera_id="cam_02",
            to_camera_id="cam_03",
            departure_utc=START,
            arrival_utc=START + timedelta(hours=3),
            expected_sec=EXPECTED_SEC,
            vehicle_id="mismatched",
        )
    )

    estimation = estimate_offsets(passes, _topology(), reference_camera_id="cam_01")

    assert estimation.offset_ms_for("cam_03") == pytest.approx(-45_000, abs=2_000)
    assert estimation.outlier_pass_count >= 1


def test_a_camera_with_too_few_passes__is_excluded_rather_than_guessed_at() -> None:
    """A guessed offset applied to real data is worse than none: it looks like
    a correction."""
    passes = _passes({"cam_02": 10.0}, count=4, legs=(("cam_01", "cam_02"),))
    passes.append(
        ReferencePass(
            from_camera_id="cam_02",
            to_camera_id="cam_03",
            departure_utc=START,
            arrival_utc=START + timedelta(seconds=EXPECTED_SEC),
            expected_sec=EXPECTED_SEC,
            vehicle_id="lonely",
        )
    )

    estimation = estimate_offsets(
        passes, _topology(), reference_camera_id="cam_01", min_reference_passes=3
    )

    assert "cam_03" in estimation.excluded_camera_ids
    assert estimation.offset_ms_for("cam_03") is None
    assert "not estimated" in estimation.describe()


def test_no_reference_passes_at_all__raises() -> None:
    """Entirely underdetermined; there is nothing to solve."""
    with pytest.raises(ValidationError, match="no reference passes"):
        estimate_offsets([], _topology())


def test_a_reference_camera_with_no_passes__raises() -> None:
    """Nothing anchors the solution to it, so the pin does not pin anything."""
    with pytest.raises(ValidationError, match="appears in no reference pass"):
        estimate_offsets(_passes({}), _topology(), reference_camera_id="cam_04_unused")


def test_a_pass_naming_an_unknown_camera__raises() -> None:
    """A typo in a camera id would otherwise produce a confident wrong graph."""
    passes = [
        ReferencePass(
            from_camera_id="cam_01",
            to_camera_id="cam_typo",
            departure_utc=START,
            arrival_utc=START + timedelta(seconds=EXPECTED_SEC),
            expected_sec=EXPECTED_SEC,
        )
    ]

    with pytest.raises(ValidationError, match="not in the topology"):
        estimate_offsets(passes, _topology())


def test_the_default_reference__is_the_best_constrained_camera() -> None:
    """Pinning a camera with one pass would anchor the solution to noise."""
    estimation = estimate_offsets(_passes({"cam_03": 45.0}), _topology())

    assert estimation.estimates[estimation.reference_camera_id].is_reference
    assert estimation.reference_camera_id in {"cam_02", "cam_03"}


def test_the_summary__names_each_camera_and_its_support() -> None:
    """An operator decides from this text whether to act on the estimate."""
    estimation = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    )

    summary = estimation.describe()

    assert "cam_03" in summary
    assert "reference passes" in summary
    assert "reference clock" in summary


def test_the_confidence_interval__widens_when_the_data_disagrees() -> None:
    """A wide interval is the signal not to act on an estimate yet."""
    tight = estimate_offsets(
        _passes({"cam_03": 45.0}), _topology(), reference_camera_id="cam_01"
    ).estimates["cam_03"]

    noisy_passes = [
        ReferencePass(
            from_camera_id=entry.from_camera_id,
            to_camera_id=entry.to_camera_id,
            departure_utc=entry.departure_utc,
            arrival_utc=entry.arrival_utc + timedelta(seconds=_scatter_sec(index)),
            expected_sec=entry.expected_sec,
            vehicle_id=entry.vehicle_id,
        )
        for index, entry in enumerate(_passes({"cam_03": 45.0}))
    ]
    loose = estimate_offsets(noisy_passes, _topology(), reference_camera_id="cam_01").estimates[
        "cam_03"
    ]

    tight_low, tight_high = tight.confidence_interval_ms
    loose_low, loose_high = loose.confidence_interval_ms

    assert (loose_high - loose_low) > (tight_high - tight_low)
