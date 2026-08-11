"""Unit tests for :mod:`multicam_tracker.models.trajectory`.

A malformed trajectory does not look malformed -- it looks like a confident,
plausible route that happens to be wrong. These structural checks are the reason
that cannot happen silently.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_tracker.models import CoverageGap, Trajectory, TrajectoryHop
from tests.fixtures.factories import make_sighting, make_trajectory

pytestmark = pytest.mark.unit


def _hop(origin_id: str, destination_id: str, **overrides: object) -> TrajectoryHop:
    """Build a hop between two sighting ids.

    Args:
        origin_id: Source sighting id.
        destination_id: Destination sighting id.
        **overrides: Field overrides.

    Returns:
        A validated hop.
    """
    fields: dict[str, object] = {
        "from_sighting_id": origin_id,
        "to_sighting_id": destination_id,
        "from_camera_id": "cam_01",
        "to_camera_id": "cam_02",
        "elapsed_sec": 90.0,
        "topology_plausible": True,
        "hop_confidence": 0.8,
    }
    fields.update(overrides)
    return TrajectoryHop(**fields)


# ---------------------------------------------------------------------------
# TrajectoryHop
# ---------------------------------------------------------------------------


def test_hop__valid_input__constructs() -> None:
    """The happy path."""
    hop = _hop("s1", "s2")

    assert hop.elapsed_sec == pytest.approx(90.0)
    assert hop.topology_plausible is True


def test_hop__zero_elapsed_time__is_accepted() -> None:
    """Boundary: two cameras with overlapping views can see a vehicle at once."""
    assert _hop("s1", "s2", elapsed_sec=0.0).elapsed_sec == 0.0


def test_hop__negative_elapsed_time__is_rejected() -> None:
    """Sightings are strictly ordered, so a hop never runs backwards."""
    with pytest.raises(ValidationError):
        _hop("s1", "s2", elapsed_sec=-1.0)


def test_hop__self_referencing_endpoints__is_rejected() -> None:
    """A hop connects two distinct sightings."""
    with pytest.raises(ValidationError, match="must differ"):
        _hop("s1", "s1")


@pytest.mark.parametrize("confidence", [0.0, 1.0], ids=["zero", "one"])
def test_hop__confidence_at_the_bounds__is_accepted(confidence: float) -> None:
    """Boundary: the closed interval [0, 1]."""
    assert _hop("s1", "s2", hop_confidence=confidence).hop_confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01], ids=["below", "above"])
def test_hop__confidence_out_of_range__is_rejected(confidence: float) -> None:
    """Confidence drives the UI's certainty encoding; it must be a real probability."""
    with pytest.raises(ValidationError):
        _hop("s1", "s2", hop_confidence=confidence)


def test_hop__implausible_flag__is_recorded_not_rejected() -> None:
    """An implausible transit may still be real -- a detour, a stop. It is scored, not dropped."""
    hop = _hop("s1", "s2", topology_plausible=False, hop_confidence=0.3)

    assert hop.topology_plausible is False


# ---------------------------------------------------------------------------
# Trajectory: structure
# ---------------------------------------------------------------------------


def test_trajectory__single_sighting_and_no_hops__is_valid() -> None:
    """Boundary: one sighting is a complete, if uninformative, trajectory."""
    only = make_sighting("cam_01", offset_sec=0)

    trajectory = make_trajectory(sightings=[only])

    assert trajectory.hops == []
    assert trajectory.duration_sec == 0.0


def test_trajectory__no_sightings__is_rejected() -> None:
    """Boundary: an empty trajectory asserts a route with no evidence for it."""
    with pytest.raises(ValidationError):
        make_trajectory(sightings=[])


def test_trajectory__multi_hop_route__is_valid() -> None:
    """The ordinary case: three cameras, two hops."""
    trajectory = make_trajectory(
        sightings=[
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_02", offset_sec=90),
            make_sighting("cam_03", offset_sec=240),
        ]
    )

    assert len(trajectory.hops) == 2
    assert trajectory.duration_sec == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# Trajectory: ordering
# ---------------------------------------------------------------------------


def test_trajectory__out_of_order_sightings__is_rejected_naming_the_index() -> None:
    """The index is what makes a 40-sighting trajectory debuggable."""
    first = make_sighting("cam_01", offset_sec=0)
    second = make_sighting("cam_02", offset_sec=200)
    third = make_sighting("cam_03", offset_sec=100)

    with pytest.raises(ValidationError) as excinfo:
        make_trajectory(sightings=[first, second, third])

    message = str(excinfo.value)
    assert "sightings[2]" in message
    assert "before" in message


def test_trajectory__identical_timestamps__is_rejected_as_not_strictly_ascending() -> None:
    """Two sightings at one instant have no defined order, so the hop between them has none."""
    first = make_sighting("cam_01", offset_sec=0)
    second = make_sighting("cam_02", offset_sec=0)

    with pytest.raises(ValidationError) as excinfo:
        make_trajectory(sightings=[first, second])

    assert "equal to" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Trajectory: hop consistency
# ---------------------------------------------------------------------------


def test_trajectory__too_few_hops__is_rejected() -> None:
    """Every adjacent pair needs exactly one hop."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]

    with pytest.raises(ValidationError, match="expected 1 hops"):
        make_trajectory(sightings=sightings, hops=[])


def test_trajectory__too_many_hops__is_rejected() -> None:
    """An extra hop would render a transition the evidence does not support."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]
    duplicated = [
        _hop(sightings[0].sighting_id, sightings[1].sighting_id),
        _hop(sightings[0].sighting_id, sightings[1].sighting_id),
    ]

    with pytest.raises(ValidationError, match="expected 1 hops"):
        make_trajectory(sightings=sightings, hops=duplicated)


def test_trajectory__hop_referencing_an_unknown_sighting__is_rejected() -> None:
    """A dangling reference would attribute an elapsed time to the wrong camera pair."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]

    with pytest.raises(ValidationError, match="from_sighting_id"):
        make_trajectory(
            sightings=sightings,
            hops=[_hop("not-a-real-sighting", sightings[1].sighting_id)],
        )


def test_trajectory__hop_with_the_wrong_destination__is_rejected() -> None:
    """The destination must be the *next* sighting, not merely a member of the list."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
        make_sighting("cam_03", offset_sec=240),
    ]
    skipping = [
        _hop(sightings[0].sighting_id, sightings[2].sighting_id),
        _hop(sightings[1].sighting_id, sightings[2].sighting_id),
    ]

    with pytest.raises(ValidationError, match="to_sighting_id"):
        make_trajectory(sightings=sightings, hops=skipping)


# ---------------------------------------------------------------------------
# Trajectory: declared bounds
# ---------------------------------------------------------------------------


def test_trajectory__start_time_not_matching_the_first_sighting__is_rejected() -> None:
    """The bounds are the timeline axis; drift from the data misrepresents it."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]

    with pytest.raises(ValidationError, match="start_time_utc"):
        make_trajectory(
            sightings=sightings, start_time_utc=sightings[0].timestamp_utc.replace(second=0)
        )


def test_trajectory__end_time_not_matching_the_last_sighting__is_rejected() -> None:
    """Same rule on the other end."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]

    with pytest.raises(ValidationError, match="end_time_utc"):
        make_trajectory(
            sightings=sightings, end_time_utc=sightings[-1].timestamp_utc.replace(second=0)
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01], ids=["below", "above"])
def test_trajectory__overall_confidence_out_of_range__is_rejected(confidence: float) -> None:
    """The headline number the operator reads must be a real probability."""
    with pytest.raises(ValidationError):
        make_trajectory(overall_confidence=confidence)


# ---------------------------------------------------------------------------
# Trajectory: computed properties
# ---------------------------------------------------------------------------


def test_trajectory__computed_properties__on_a_linear_route() -> None:
    """Distinct cameras, in order."""
    trajectory = make_trajectory(
        sightings=[
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_02", offset_sec=90),
            make_sighting("cam_03", offset_sec=240),
        ]
    )

    assert trajectory.duration_sec == pytest.approx(240.0)
    assert trajectory.camera_sequence == ["cam_01", "cam_02", "cam_03"]
    assert trajectory.distinct_camera_count == 3


def test_trajectory__repeated_camera__keeps_the_repeat_in_the_sequence() -> None:
    """A vehicle returning past a camera is a meaningful pattern, not noise to collapse."""
    trajectory = make_trajectory(
        sightings=[
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_02", offset_sec=90),
            make_sighting("cam_01", offset_sec=240),
        ]
    )

    assert trajectory.camera_sequence == ["cam_01", "cam_02", "cam_01"]
    assert trajectory.distinct_camera_count == 2


def test_trajectory__single_sighting__has_zero_duration_and_one_camera() -> None:
    """Boundary behaviour of the computed properties."""
    trajectory = make_trajectory(sightings=[make_sighting("cam_07", offset_sec=0)])

    assert trajectory.duration_sec == 0.0
    assert trajectory.camera_sequence == ["cam_07"]
    assert trajectory.distinct_camera_count == 1


# ---------------------------------------------------------------------------
# CoverageGap
# ---------------------------------------------------------------------------


def test_coverage_gap__valid_input__constructs() -> None:
    """A gap states plainly that the system does not know what happened."""
    gap = CoverageGap(
        from_sighting_id="s1",
        to_sighting_id="s2",
        elapsed_sec=1800.0,
        expected_max_sec=340.0,
        reason="elapsed time exceeds the slowest plausible transit for cam_03 -> cam_04",
    )

    assert gap.elapsed_sec > gap.expected_max_sec


def test_coverage_gap__empty_reason__is_rejected() -> None:
    """A gap with no stated reason is not actionable by an operator."""
    with pytest.raises(ValidationError):
        CoverageGap(
            from_sighting_id="s1",
            to_sighting_id="s2",
            elapsed_sec=10.0,
            expected_max_sec=5.0,
            reason="   ",
        )


def test_trajectory__gaps_default_to_empty() -> None:
    """A trajectory with full coverage carries an empty gap list, not None."""
    assert make_trajectory().gaps == []


def test_trajectory__attached_gaps__survive_a_round_trip() -> None:
    """Gaps are part of the answer and must serialize with it."""
    sightings = [
        make_sighting("cam_01", offset_sec=0),
        make_sighting("cam_02", offset_sec=90),
    ]
    gap = CoverageGap(
        from_sighting_id=sightings[0].sighting_id,
        to_sighting_id=sightings[1].sighting_id,
        elapsed_sec=90.0,
        expected_max_sec=60.0,
        reason="exceeded the plausible window",
    )
    trajectory = make_trajectory(sightings=sightings, gaps=[gap])

    assert Trajectory.from_json_dict(trajectory.to_json_dict()) == trajectory
