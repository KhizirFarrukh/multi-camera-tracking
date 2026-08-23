"""Unit tests for trajectory explanations.

A trajectory a human cannot interrogate is not usable evidence, so these tests
check the two questions an operator actually asks -- why is this in, why is that
out -- and that the answers survive a round trip through storage.
"""

from __future__ import annotations

import json

import pytest

from multicam_tracker.pathing import (
    ExclusionReason,
    TrajectoryExplanation,
    build_graph,
    explain_path,
)
from multicam_tracker.pathing.explanation import verify_path_score
from multicam_tracker.pathing.optimal_path import best_path
from tests.fixtures.pathing import (
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    prepared,
)

pytestmark = pytest.mark.unit


def _explain(entries: list[tuple[str, float, float]]) -> TrajectoryExplanation:
    """Reconstruct a route and explain it.

    Args:
        entries: ``(camera, offset_sec, score)`` per candidate.

    Returns:
        The explanation.
    """
    graph = build_graph(
        [prepared(camera, offset, score) for camera, offset, score in entries],
        chain_topology_with_outlier(),
        cameras=chain_cameras(),
    )
    return explain_path(graph, best_path(graph), ambiguity="a clear separation")


def test_every_included_sighting__has_a_recorded_reason() -> None:
    """No sighting appears in a route without saying how it got there."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)])

    assert len(explanation.included) == 3
    assert all(record.arrived_via for record in explanation.included)


def test_the_first_sighting__is_recorded_as_having_nothing_before_it() -> None:
    """It has no hop, and pretending otherwise would invent evidence."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    assert explanation.included[0].hop_confidence is None
    assert "nothing precedes it" in explanation.included[0].arrived_via
    assert explanation.included[1].hop_confidence is not None


def test_each_included_record__carries_the_evidence_behind_it() -> None:
    """The review UI shows the method and score beside every hop."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.85)])

    assert explanation.included[1].match_confidence == pytest.approx(0.85)
    assert explanation.included[1].match_method == "plate_exact"
    assert explanation.included[1].camera_id == "cam_02"


def test_excluded_candidates__are_explained_with_the_governing_constraint() -> None:
    """The question that matters most: why is that sighting not in the route?"""
    explanation = _explain(
        [("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), (FAR_CAMERA, 200.0, 0.95)]
    )

    assert explanation.excluded_total == 1
    assert explanation.excluded[0].reason is ExclusionReason.NO_PLAUSIBLE_CONNECTION
    assert explanation.excluded[0].detail


def test_excluded_candidates__are_ranked_by_confidence() -> None:
    """The high-confidence rejects are the ones an operator asks about."""
    explanation = _explain(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            (FAR_CAMERA, 200.0, 0.55),
            (FAR_CAMERA, 260.0, 0.95),
        ]
    )

    confidences = [record.match_confidence for record in explanation.excluded]
    assert confidences == sorted(confidences, reverse=True)


def test_the_explained_exclusions__are_capped_but_the_total_is_still_reported() -> None:
    """An explanation nobody can read is not an explanation.

    The count is kept so the operator knows how much was left unsaid.
    """
    entries = [("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)]
    entries += [(FAR_CAMERA, 200.0 + index, 0.5) for index in range(15)]
    graph = build_graph(
        [prepared(camera, offset, score) for camera, offset, score in entries],
        chain_topology_with_outlier(),
        cameras=chain_cameras(),
    )

    explanation = explain_path(graph, best_path(graph), max_exclusions=3)

    assert len(explanation.excluded) == 3
    assert explanation.excluded_total == 15


def test_the_objective__is_stated_in_words_on_every_explanation() -> None:
    """A reader should not need the source open to know what was maximised."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    assert "hop weights" in explanation.objective
    assert "unmonitored ground" in explanation.objective


def test_the_ambiguity_verdict__travels_with_the_explanation() -> None:
    """Whether the answer was close to call is part of the answer."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    assert explanation.ambiguity == "a clear separation"


def test_the_recorded_score__matches_the_published_objective() -> None:
    """The explanation must describe the algorithm the system actually runs."""
    graph = build_graph(
        [prepared(camera, offset, 0.9) for camera, offset in [("cam_01", 0.0), ("cam_02", 120.0)]],
        chain_topology_with_outlier(),
    )

    assert verify_path_score(graph, best_path(graph))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_the_explanation__survives_a_round_trip_through_json() -> None:
    """The audit trail outlives the process that produced it."""
    original = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), (FAR_CAMERA, 200.0, 0.95)])

    payload = json.loads(json.dumps(original.to_json_dict()))
    restored = TrajectoryExplanation.from_json_dict(payload)

    assert restored.to_json_dict() == original.to_json_dict()
    assert restored.excluded[0].reason is ExclusionReason.NO_PLAUSIBLE_CONNECTION


def test_the_serialized_form__is_json_native_throughout() -> None:
    """Anything that needs a custom encoder will not survive the audit store."""
    explanation = _explain([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])

    encoded = json.dumps(explanation.to_json_dict())

    assert json.loads(encoded)["included"][0]["camera_id"] == "cam_01"
