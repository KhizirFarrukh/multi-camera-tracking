"""The definitive proof that this stage does its job.

One pair of tests carries the whole argument: the same data, reconstructed
without clock correction and with it. Without, the route is wrong. With, it is
the ground truth. Both directions are asserted explicitly, because either one
alone proves nothing -- a reconstruction that is always right would pass the
second, and one that is always wrong would pass the first.

The drift is injected rather than taken from the shipped adversarial scenario
for the isolated pair, so the only difference between the two runs is the clock.
The adversarial scenario is exercised separately below, where the clone and the
decoys are also in play.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from multicam_tracker.clock import FixedClock
from multicam_tracker.models import Sighting, TemporalSeverity
from multicam_tracker.pathing import candidates_from_matches, reconstruct_trajectory
from multicam_tracker.synth import generate_from_file
from multicam_tracker.timesync import (
    CameraTimeStatus,
    ReferencePass,
    check_temporal_integrity,
    correct_sighting,
    detect_drift,
    estimate_offsets,
)
from multicam_tracker.topology import Topology, load_topology

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tests" / "fixtures" / "scenarios"
TARGET_ID = "a0000000-0000-4000-8000-000000000001"

DRIFTED_CAMERA = "cam_03"
INJECTED_DRIFT_SEC = 300.0
"""Five minutes.

Chosen because it exceeds the slack in the topology's travel-time windows. A
45-second drift on this network is absorbed by those windows and changes no
route at all -- which is a real finding, tested below, and the reason the
integrity gate's blocking threshold is tied to the *shortest transit* rather
than to any drift at all."""

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def topology() -> Topology:
    """Return the shipped topology.

    Returns:
        The graph the scenarios run over.
    """
    return load_topology(REPO_ROOT / "config" / "topology.yaml")


@pytest.fixture(scope="module")
def clean(topology: Topology) -> Any:
    """Return the clean scenario, whose route is unambiguous without drift.

    Args:
        topology: The graph.

    Returns:
        The generated dataset.
    """
    return generate_from_file(SCENARIO_DIR / "clean.yaml", topology=topology)


def _drift(sightings: dict[str, Sighting], camera_id: str, seconds: float) -> dict[str, Sighting]:
    """Shift one camera's reported times, leaving the correction field at zero.

    That is what an uncorrected drifting camera looks like in the data: both the
    raw and the corrected timestamp are wrong, and nothing records that they are.

    Args:
        sightings: Sightings by id.
        camera_id: The camera to shift.
        seconds: How far to shift it.

    Returns:
        A new mapping with the shift applied.
    """
    shift = timedelta(seconds=seconds)
    return {
        sighting_id: (
            sighting.model_copy(
                update={
                    "timestamp_utc": sighting.timestamp_utc + shift,
                    "raw_timestamp": sighting.raw_timestamp + shift,
                }
            )
            if sighting.camera_id == camera_id
            else sighting
        )
        for sighting_id, sighting in sightings.items()
    }


def _corrected(
    sightings: dict[str, Sighting], camera_id: str, offset_ms: int
) -> dict[str, Sighting]:
    """Apply a clock correction to one camera's sightings.

    Args:
        sightings: Sightings by id.
        camera_id: The camera to correct.
        offset_ms: The correction to apply.

    Returns:
        A new mapping with the correction applied.
    """
    return {
        sighting_id: (
            correct_sighting(sighting, offset_ms) if sighting.camera_id == camera_id else sighting
        )
        for sighting_id, sighting in sightings.items()
    }


def _reconstruct(dataset: Any, sightings: dict[str, Sighting], topology: Topology) -> Any:
    """Reconstruct the target's route from a given set of sightings.

    Args:
        dataset: The generated dataset, for matching.
        sightings: The sightings to reconstruct from, possibly drifted.
        topology: The graph.

    Returns:
        The reconstruction.
    """
    candidates = candidates_from_matches(
        dataset, dataset.ground_truth, TARGET_ID, topology=topology
    )
    return reconstruct_trajectory(
        TARGET_ID,
        candidates,
        sightings,
        topology,
        cameras=dict(topology.cameras_by_id),
        activity=list(sightings.values()),
    )


def _camera_order(result: Any) -> list[str]:
    """Return the camera sequence of a reconstruction.

    Args:
        result: The reconstruction.

    Returns:
        Camera ids in route order, or an empty list when no route was produced.
    """
    if result.trajectory is None:
        return []
    return [sighting.camera_id for sighting in result.trajectory.sightings]


def _truth_order(dataset: Any) -> list[str]:
    """Return the target's true camera sequence.

    Args:
        dataset: The generated dataset.

    Returns:
        Camera ids in ground-truth order.
    """
    by_id = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    return [
        by_id[sighting_id].camera_id
        for sighting_id in dataset.ground_truth.target.sighting_ids
        if sighting_id in by_id
    ]


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def test_without_correction__the_reconstructed_route_is_wrong(
    clean: Any, topology: Topology
) -> None:
    """Half of the definitive pair.

    Camera 3's clock is five minutes fast and nothing in the data says so. The
    reconstruction is not merely less confident -- it puts the vehicle at
    cam_04 before cam_03, which is a route it never took.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)

    result = _reconstruct(clean, drifted, topology)

    assert _camera_order(result) != _truth_order(clean)
    assert result.trajectory is not None


def test_with_correction__the_reconstructed_route_is_the_ground_truth(
    clean: Any, topology: Topology
) -> None:
    """The other half. Same data, same matcher, correction applied."""
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)
    corrected = _corrected(drifted, DRIFTED_CAMERA, -int(INJECTED_DRIFT_SEC * 1000))

    result = _reconstruct(clean, corrected, topology)

    assert _camera_order(result) == _truth_order(clean)


def test_correction__restores_the_confidence_the_drift_destroyed(
    clean: Any, topology: Topology
) -> None:
    """The score says so too, not only the ordering.

    An operator scanning a list of trajectories sees the number before they see
    the route, and the drifted one must not look healthy.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)
    corrected = _corrected(drifted, DRIFTED_CAMERA, -int(INJECTED_DRIFT_SEC * 1000))

    without = _reconstruct(clean, drifted, topology)
    with_correction = _reconstruct(clean, corrected, topology)
    untouched = _reconstruct(clean, baseline, topology)

    assert without.trajectory.overall_confidence < 0.5
    assert with_correction.trajectory.overall_confidence == pytest.approx(
        untouched.trajectory.overall_confidence
    )


def test_correction__removes_the_gaps_the_drift_manufactured(
    clean: Any, topology: Topology
) -> None:
    """A drifting clock invents unmonitored ground that was never there."""
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)
    corrected = _corrected(drifted, DRIFTED_CAMERA, -int(INJECTED_DRIFT_SEC * 1000))

    assert _reconstruct(clean, drifted, topology).gaps
    assert _reconstruct(clean, corrected, topology).gaps == []


def test_a_drift_smaller_than_the_travel_window_slack__changes_no_route(
    clean: Any, topology: Topology
) -> None:
    """The finding that sets the gate's blocking threshold.

    Forty-five seconds on this network is absorbed by the travel-time windows:
    the route is identical, so a gate that blocked on *any* drift would refuse
    to answer questions it could answer perfectly well. What matters is drift
    large enough to reorder hops, which is why the threshold is the shortest
    transit rather than zero.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    slightly_drifted = _drift(baseline, DRIFTED_CAMERA, 45.0)

    assert _camera_order(_reconstruct(clean, slightly_drifted, topology)) == _truth_order(clean)


# ---------------------------------------------------------------------------
# Recovering the offset from the data itself
# ---------------------------------------------------------------------------


def _reference_passes(dataset: Any, sightings: dict[str, Sighting], topology: Topology) -> list:
    """Build reference passes from the vehicles ground truth knows about.

    A reference pass is a vehicle observed at two cameras with a known route --
    which is exactly what the ground truth records, and exactly what an operator
    supplies in production by driving a known car past known cameras.

    Args:
        dataset: The generated dataset.
        sightings: The sightings to read times from.
        topology: The graph, for the expected transit.

    Returns:
        One pass per consecutive camera pair per vehicle.
    """
    passes = []
    for vehicle in dataset.ground_truth.vehicles:
        # Route order comes from what the operator knows the vehicle did, not
        # from the reported timestamps. Re-sorting by reported time would let a
        # drifting camera reorder the very legs the estimator is meant to
        # measure, and the drift would hide inside its own symptom.
        route = [
            sightings[sighting_id]
            for sighting_id in vehicle.sighting_ids
            if sighting_id in sightings
        ]
        for origin, destination in pairwise(route):
            link = topology.get_link(origin.camera_id, destination.camera_id)
            if link is None:
                continue
            passes.append(
                ReferencePass(
                    from_camera_id=origin.camera_id,
                    to_camera_id=destination.camera_id,
                    departure_utc=origin.timestamp_utc,
                    arrival_utc=destination.timestamp_utc,
                    expected_sec=(link.min_travel_time_sec + link.max_travel_time_sec) / 2,
                    vehicle_id=vehicle.vehicle_id,
                )
            )
    return passes


def test_the_estimator__attributes_the_injected_drift_to_the_drifted_camera(
    clean: Any, topology: Topology
) -> None:
    """Closing the loop: nothing told the estimator what was injected.

    An operator does not know the offset either -- that is the whole problem --
    so recovering it from ordinary passes is what makes correction possible on a
    camera that exposes no clock.

    The comparison is between the same estimate run on undrifted and drifted
    data, because the *absolute* offsets carry a bias from the expected transit
    times (see the test below). The change between the two runs does not: the
    only thing that differs is the clock.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)

    before = estimate_offsets(
        _reference_passes(clean, baseline, topology),
        topology,
        reference_camera_id="cam_01",
        min_reference_passes=1,
    )
    after = estimate_offsets(
        _reference_passes(clean, drifted, topology),
        topology,
        reference_camera_id="cam_01",
        min_reference_passes=1,
    )

    change = {
        camera: after.estimates[camera].offset_ms - estimate.offset_ms
        for camera, estimate in before.estimates.items()
    }

    assert change.pop(DRIFTED_CAMERA) == pytest.approx(-INJECTED_DRIFT_SEC * 1000, abs=1_000)
    assert all(value == pytest.approx(0.0, abs=1_000) for value in change.values())


def test_the_estimator__is_only_as_good_as_the_expected_transit_times(
    clean: Any, topology: Topology
) -> None:
    """Documents the limitation rather than leaving it to be discovered.

    Feeding the midpoint of a travel-time window as "expected" makes every
    camera absorb the difference between that midpoint and how the traffic
    actually moves, so the absolute offsets come back biased even on data with
    no drift at all. Differences are trustworthy; absolutes need transit
    estimates worth the name, which is why the direct clock probe is the
    preferred mechanism and this is the fallback.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}

    undrifted = estimate_offsets(
        _reference_passes(clean, baseline, topology),
        topology,
        reference_camera_id="cam_01",
        min_reference_passes=1,
    )

    biased = [
        abs(estimate.offset_ms)
        for camera, estimate in undrifted.estimates.items()
        if camera != "cam_01"
    ]
    assert max(biased) > 1_000


def test_drift_detection__flags_the_drifted_camera_and_not_its_neighbours(
    clean: Any, topology: Topology
) -> None:
    """A detector that flagged everything would be no detector at all."""
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)
    passes = _reference_passes(clean, drifted, topology)

    flagged = detect_drift(DRIFTED_CAMERA, passes, alert_ms=30_000)
    neighbour = detect_drift("cam_01", passes, alert_ms=30_000)

    assert flagged.has_alert
    assert not neighbour.has_alert


# ---------------------------------------------------------------------------
# The verdict travels with the route
# ---------------------------------------------------------------------------


def test_the_integrity_verdict__is_attached_to_the_trajectory(
    clean: Any, topology: Topology
) -> None:
    """A caveat in a log file is a caveat nobody acts on."""
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    passes = _reference_passes(
        clean, _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC), topology
    )
    alert = detect_drift(DRIFTED_CAMERA, passes, alert_ms=30_000).alert
    assert alert is not None

    cameras = sorted({sighting.camera_id for sighting in clean.sightings})
    verdict = check_temporal_integrity(
        cameras,
        {
            camera: CameraTimeStatus(
                camera,
                verified_at_utc=NOW - timedelta(hours=1),
                drift_alert=alert if camera == DRIFTED_CAMERA else None,
            )
            for camera in cameras
        },
        topology,
        FixedClock(NOW),
    )

    candidates = candidates_from_matches(clean, clean.ground_truth, TARGET_ID, topology=topology)
    result = reconstruct_trajectory(
        TARGET_ID,
        candidates,
        baseline,
        topology,
        cameras=dict(topology.cameras_by_id),
        temporal_integrity=verdict,
    )

    integrity = result.trajectory.temporal_integrity
    assert integrity is not None
    assert not integrity.verified
    assert integrity.worst_severity is TemporalSeverity.WARNING
    assert DRIFTED_CAMERA in integrity.summary()


def test_a_route_assembled_without_checking__says_so_rather_than_looking_verified(
    clean: Any, topology: Topology
) -> None:
    """``None`` is a third state, and collapsing it into "clean" would lie."""
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}

    result = _reconstruct(clean, baseline, topology)

    assert result.trajectory.temporal_integrity is None


# ---------------------------------------------------------------------------
# The shipped adversarial scenario
# ---------------------------------------------------------------------------


def test_the_adversarial_scenario__carries_its_injected_drift_uncorrected(
    topology: Topology,
) -> None:
    """The scenario stage 05 built for this stage to find.

    Its 45-second drift on cam_03 is recorded in the ground truth and absent
    from the data, which is what makes it a fair test of detection rather than
    of bookkeeping.
    """
    dataset = generate_from_file(SCENARIO_DIR / "adversarial.yaml", topology=topology)
    drift_events = [
        event for event in dataset.ground_truth.injections if event.kind == "clock_drift"
    ]

    assert drift_events
    assert drift_events[0].context["camera_id"] == DRIFTED_CAMERA
    assert all(
        sighting.clock_offset_applied_ms == 0
        for sighting in dataset.sightings
        if sighting.camera_id == DRIFTED_CAMERA
    )


def test_the_adversarial_scenario__the_estimator_isolates_its_injected_drift(
    topology: Topology,
) -> None:
    """The graph solution finds the injected offset even in messy data.

    A clone, decoys, an outage, and a 45-second drift, and the estimator still
    attributes exactly -45,000 ms to cam_03 and nothing to any other camera.
    That is what the graph buys: a camera late by X makes its incoming legs long
    by X *and* its outgoing legs short by X, and no other explanation fits both.
    """
    dataset = generate_from_file(SCENARIO_DIR / "adversarial.yaml", topology=topology)
    as_reported = {sighting.sighting_id: sighting for sighting in dataset.sightings}
    # Undo the injection to get the same data with a correct clock.
    without_drift = _drift(as_reported, DRIFTED_CAMERA, -45.0)

    before = estimate_offsets(
        _reference_passes(dataset, without_drift, topology),
        topology,
        reference_camera_id="cam_01",
        min_reference_passes=2,
    )
    after = estimate_offsets(
        _reference_passes(dataset, as_reported, topology),
        topology,
        reference_camera_id="cam_01",
        min_reference_passes=2,
    )

    change = {
        camera: after.estimates[camera].offset_ms - estimate.offset_ms
        for camera, estimate in before.estimates.items()
    }

    assert change.pop(DRIFTED_CAMERA) == pytest.approx(-45_000, abs=1_000)
    assert all(value == pytest.approx(0.0, abs=1_000) for value in change.values())


def test_a_forty_five_second_drift__is_below_the_detectors_noise_floor(
    topology: Topology,
) -> None:
    """The limit of the monitor, measured rather than assumed.

    Drift detection compares observed transits against an expected one, and the
    shipped topology's travel windows are minutes wide -- so the expectation fed
    to it carries more uncertainty than the 45-second offset being looked for.
    The detector correctly declines to call it, which is the behaviour that
    keeps its alerts worth reading.

    The estimator finds the same drift exactly (test above), because it uses the
    graph rather than a single expectation. That is the division of labour: the
    monitor watches for gross faults, the estimator measures.
    """
    dataset = generate_from_file(SCENARIO_DIR / "adversarial.yaml", topology=topology)
    sightings = {sighting.sighting_id: sighting for sighting in dataset.sightings}

    analysis = detect_drift(
        DRIFTED_CAMERA, _reference_passes(dataset, sightings, topology), alert_ms=20_000
    )

    assert analysis.sample_count > 0
    assert not analysis.has_alert
    assert analysis.residual_spread_ms > 45_000


def test_a_gross_drift__is_detected_even_against_that_noise_floor(
    clean: Any, topology: Topology
) -> None:
    """What the monitor is for: the fault nobody has to squint at.

    Five minutes is well clear of the scatter, and the detector says so.
    """
    baseline = {sighting.sighting_id: sighting for sighting in clean.sightings}
    drifted = _drift(baseline, DRIFTED_CAMERA, INJECTED_DRIFT_SEC)

    analysis = detect_drift(
        DRIFTED_CAMERA, _reference_passes(clean, drifted, topology), alert_ms=30_000
    )

    assert analysis.has_alert
    assert analysis.alert is not None
    assert analysis.alert.magnitude_ms > 0
