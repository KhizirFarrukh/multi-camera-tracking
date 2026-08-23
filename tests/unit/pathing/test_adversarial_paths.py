"""The false-positive rejection cases. This module is why the stage exists.

Greedy chronological chaining passes every ordinary test and fails exactly here:
one spurious match in the middle of a timeline, and the route teleports across
the city. These graphs are small enough that the correct answer is obvious by
inspection, which is the point -- if the engine disagrees with a case a reader
can verify by eye, the engine is wrong.

The false positive is placed at each of the three positions that behave
differently: the middle (a preceding node constrains it), the start (nothing
precedes it), and the end (nothing follows it). Boundary cases are where a
global objective is most likely to quietly degrade into a greedy one.
"""

from __future__ import annotations

import pytest

from multicam_tracker.pathing import (
    ExclusionReason,
    build_graph,
    explain_path,
    prepare_candidates,
    reconstruct_trajectory,
)
from multicam_tracker.pathing.optimal_path import best_path, path_score
from tests.fixtures.factories import make_sighting
from tests.fixtures.pathing import (
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    scored_candidate,
)

pytestmark = pytest.mark.unit

TARGET_ID = "t-0001"
STRONG = 0.95
"""A false positive is given a *higher* score than the genuine matches wherever
possible. Rejecting a weak false positive proves nothing -- the threshold would
have done it."""

GENUINE = 0.90


def _reconstruct(entries: list[tuple[str, float, float]]) -> object:
    """Reconstruct a route from ``(camera, offset_sec, score)`` triples.

    Args:
        entries: The candidates to feed the engine.

    Returns:
        The reconstruction.
    """
    topology = chain_topology_with_outlier()
    cameras = chain_cameras()
    sightings = [make_sighting(camera_id, offset_sec=offset) for camera_id, offset, _ in entries]
    candidates = [
        scored_candidate(sighting, score, target_id=TARGET_ID)
        for sighting, (_, _, score) in zip(sightings, entries, strict=True)
    ]
    return reconstruct_trajectory(
        TARGET_ID,
        candidates,
        {sighting.sighting_id: sighting for sighting in sightings},
        topology,
        cameras=cameras,
    )


def _cameras_on_route(result: object) -> list[str]:
    """Return the camera sequence of a reconstruction.

    Args:
        result: The reconstruction.

    Returns:
        Camera ids in route order.
    """
    trajectory = result.trajectory  # type: ignore[attr-defined]
    assert trajectory is not None
    return [sighting.camera_id for sighting in trajectory.sightings]


# ---------------------------------------------------------------------------
# The defining property
# ---------------------------------------------------------------------------


def test_false_positive__in_the_middle_of_a_perfect_chain__is_excluded() -> None:
    """The stage's defining test.

    A genuine chain runs cam_01 -> cam_02 -> cam_03. A high-scoring false
    positive sits 55 km away, between the second and third sightings. Reaching
    it and coming back costs two hops the topology cannot account for, so the
    route around it must win -- even though the decoy scores higher than every
    genuine match.
    """
    result = _reconstruct(
        [
            ("cam_01", 0.0, GENUINE),
            ("cam_02", 120.0, GENUINE),
            (FAR_CAMERA, 180.0, STRONG),
            ("cam_03", 240.0, GENUINE),
        ]
    )

    assert _cameras_on_route(result) == ["cam_01", "cam_02", "cam_03"]


def test_false_positive__at_the_start_of_the_timeline__is_excluded() -> None:
    """Boundary: nothing precedes it, so nothing constrains it locally.

    A greedy search seeded at the earliest candidate starts here and never
    recovers.
    """
    result = _reconstruct(
        [
            (FAR_CAMERA, 0.0, STRONG),
            ("cam_01", 120.0, GENUINE),
            ("cam_02", 240.0, GENUINE),
            ("cam_03", 360.0, GENUINE),
        ]
    )

    assert _cameras_on_route(result) == ["cam_01", "cam_02", "cam_03"]


def test_false_positive__at_the_end_of_the_timeline__is_excluded() -> None:
    """Boundary: nothing follows it, so only one bad hop is needed to reach it.

    The weaker of the two boundary cases, and the one a penalty tuned too low
    would let through.
    """
    result = _reconstruct(
        [
            ("cam_01", 0.0, GENUINE),
            ("cam_02", 120.0, GENUINE),
            ("cam_03", 240.0, GENUINE),
            (FAR_CAMERA, 360.0, STRONG),
        ]
    )

    assert _cameras_on_route(result) == ["cam_01", "cam_02", "cam_03"]


def test_false_positive__at_a_plausible_location__is_included() -> None:
    """The control.

    The same candidate, same score, moved onto the chain at a plausible time, is
    accepted. Without this the tests above would also pass on an engine that
    simply rejected everything.
    """
    result = _reconstruct(
        [
            ("cam_01", 0.0, GENUINE),
            ("cam_02", 120.0, GENUINE),
            ("cam_03", 240.0, GENUINE),
            ("cam_04", 360.0, STRONG),
        ]
    )

    assert _cameras_on_route(result) == ["cam_01", "cam_02", "cam_03", "cam_04"]


def test_two_consecutive_false_positives__do_not_beat_the_longer_genuine_path() -> None:
    """A decoy pair that is self-consistent must still lose.

    Two false positives at the far camera support each other perfectly -- same
    place, plausible interval -- so a search that scored sub-paths in isolation
    would prefer them. Scoring the whole route does not: three genuine sightings
    on a plausible chain outweigh two on an island.
    """
    result = _reconstruct(
        [
            ("cam_01", 0.0, GENUINE),
            (FAR_CAMERA, 60.0, STRONG),
            ("cam_02", 120.0, GENUINE),
            (FAR_CAMERA, 150.0, STRONG),
            ("cam_03", 240.0, GENUINE),
        ]
    )

    assert _cameras_on_route(result) == ["cam_01", "cam_02", "cam_03"]


def test_false_positive__is_recorded_with_a_reason_not_silently_dropped() -> None:
    """An operator asking "why is that sighting missing?" gets an answer.

    A search that quietly drops a candidate looks identical to one that never
    saw it, and the difference matters most for the candidate someone expected.
    """
    result = _reconstruct(
        [
            ("cam_01", 0.0, GENUINE),
            ("cam_02", 120.0, GENUINE),
            (FAR_CAMERA, 180.0, STRONG),
            ("cam_03", 240.0, GENUINE),
        ]
    )
    explanation = result.explanation  # type: ignore[attr-defined]

    assert explanation is not None
    assert len(explanation.excluded) == 1
    excluded = explanation.excluded[0]
    assert excluded.camera_id == FAR_CAMERA
    assert excluded.reason is ExclusionReason.NO_PLAUSIBLE_CONNECTION
    assert excluded.detail


# ---------------------------------------------------------------------------
# Why the objective produces that answer
# ---------------------------------------------------------------------------


def test_the_route_through_the_false_positive__scores_lower__by_construction() -> None:
    """States the arithmetic the previous tests rely on.

    If the objective ever changed so that detouring became profitable, this test
    fails with the two numbers side by side rather than leaving a reader to
    reverse-engineer why a route changed.
    """
    topology = chain_topology_with_outlier()
    cameras = chain_cameras()
    entries = [
        ("cam_01", 0.0, GENUINE),
        ("cam_02", 120.0, GENUINE),
        (FAR_CAMERA, 180.0, STRONG),
        ("cam_03", 240.0, GENUINE),
    ]
    sightings = [make_sighting(camera, offset_sec=offset) for camera, offset, _ in entries]
    candidates = [
        scored_candidate(sighting, score, target_id=TARGET_ID)
        for sighting, (_, _, score) in zip(sightings, entries, strict=True)
    ]
    nodes = prepare_candidates(
        candidates, {sighting.sighting_id: sighting for sighting in sightings}
    )
    graph = build_graph(nodes, topology, cameras=cameras)

    without_detour = path_score(graph, [0, 1, 3])
    chosen = best_path(graph)

    assert chosen.node_indices == [0, 1, 3]
    assert chosen.score == pytest.approx(without_detour)
    # The detour is not merely worse -- there is no route through it at all,
    # because the hops it would need imply an impossible speed and no edge was
    # ever created for them.
    assert not [edge for edge in graph.edges if edge.to_index == 2]


def test_explanation__names_the_constraint_that_did_the_work() -> None:
    """The audit trail has to say *which* rule excluded a candidate."""
    topology = chain_topology_with_outlier()
    cameras = chain_cameras()
    entries = [
        ("cam_01", 0.0, GENUINE),
        ("cam_02", 120.0, GENUINE),
        (FAR_CAMERA, 180.0, STRONG),
    ]
    sightings = [make_sighting(camera, offset_sec=offset) for camera, offset, _ in entries]
    candidates = [
        scored_candidate(sighting, score, target_id=TARGET_ID)
        for sighting, (_, _, score) in zip(sightings, entries, strict=True)
    ]
    nodes = prepare_candidates(
        candidates, {sighting.sighting_id: sighting for sighting in sightings}
    )
    graph = build_graph(nodes, topology, cameras=cameras)

    explanation = explain_path(graph, best_path(graph))

    assert [record.reason for record in explanation.excluded] == [
        ExclusionReason.NO_PLAUSIBLE_CONNECTION
    ]
