"""Unit tests for incremental extension.

The equivalence property is the one that matters: whatever order sightings
arrive in, the trajectory must equal what a full recomputation would have
produced. A live system that disagrees with its own batch reprocessing is worse
than one that is merely slow, and out-of-order delivery is normal rather than
exceptional.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.models import MatchCandidate, Sighting
from multicam_tracker.pathing import IncrementalReconstructor, reconstruct_trajectory
from tests.fixtures.factories import make_sighting
from tests.fixtures.pathing import (
    FAR_CAMERA,
    chain_cameras,
    chain_topology_with_outlier,
    scored_candidate,
)

pytestmark = pytest.mark.unit

TARGET_ID = "t-0001"
_HYPOTHESIS = settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


def _feed(entries: list[tuple[str, float, float]]) -> list[tuple[MatchCandidate, Sighting]]:
    """Build ``(candidate, sighting)`` pairs from triples.

    Args:
        entries: ``(camera, offset_sec, score)`` per candidate.

    Returns:
        The pairs, in the order given.
    """
    pairs = []
    for camera, offset, score in entries:
        sighting = make_sighting(camera, offset_sec=offset)
        pairs.append((scored_candidate(sighting, score, target_id=TARGET_ID), sighting))
    return pairs


def _reconstructor(**kwargs: object) -> IncrementalReconstructor:
    """Build a reconstructor over the chain topology.

    Args:
        **kwargs: Passed through to the constructor.

    Returns:
        The reconstructor.
    """
    return IncrementalReconstructor(
        TARGET_ID,
        chain_topology_with_outlier(),
        cameras=chain_cameras(),
        **kwargs,  # type: ignore[arg-type]
    )


def _route(result: object) -> list[str]:
    """Return the camera sequence of a reconstruction.

    Args:
        result: The reconstruction.

    Returns:
        Camera ids in route order.
    """
    trajectory = result.trajectory  # type: ignore[attr-defined]
    return [] if trajectory is None else [s.camera_id for s in trajectory.sightings]


def _full(pairs: list[tuple[MatchCandidate, Sighting]]) -> list[str]:
    """Reconstruct from scratch over the whole candidate set.

    Args:
        pairs: All candidates and their sightings.

    Returns:
        Camera ids in route order.
    """
    return _route(
        reconstruct_trajectory(
            TARGET_ID,
            [candidate for candidate, _ in pairs],
            {sighting.sighting_id: sighting for _, sighting in pairs},
            chain_topology_with_outlier(),
            cameras=chain_cameras(),
        )
    )


# ---------------------------------------------------------------------------
# Ordinary extension
# ---------------------------------------------------------------------------


def test_appending_a_later_plausible_sighting__extends_the_trajectory() -> None:
    """The ordinary live case."""
    pairs = _feed([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)])
    reconstructor = _reconstructor()
    for candidate, sighting in pairs:
        extension = reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    assert _route(extension.result) == ["cam_01", "cam_02"]
    assert extension.accepted_sighting_ids == [pairs[-1][1].sighting_id]


def test_appending_an_implausible_sighting__leaves_the_route_unchanged() -> None:
    """It is admitted to the graph and rejected by the search, which is recorded."""
    reconstructor = _reconstructor()
    for candidate, sighting in _feed([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)]):
        reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    decoy_candidate, decoy = _feed([(FAR_CAMERA, 200.0, 0.5)])[0]
    extension = reconstructor.extend([decoy_candidate], {decoy.sighting_id: decoy})

    assert _route(extension.result) == ["cam_01", "cam_02"]
    assert extension.result.explanation is not None
    assert extension.result.explanation.excluded_total == 1


def test_an_out_of_order_sighting__triggers_a_suffix_recomputation() -> None:
    """Appending it blindly would leave a route chosen without knowing it existed.

    The buffer is widened so this exercises the recomputation rather than the
    late-arrival rejection; the two are tested separately on purpose.
    """
    reconstructor = _reconstructor(reorder_buffer_sec=600.0)
    late_candidate, late = _feed([("cam_02", 60.0, 0.9)])[0]
    for candidate, sighting in _feed([("cam_01", 0.0, 0.9), ("cam_03", 240.0, 0.9)]):
        reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    extension = reconstructor.extend([late_candidate], {late.sighting_id: late})

    assert extension.recomputed_from_index == 1
    assert _route(extension.result) == ["cam_01", "cam_02", "cam_03"]


def test_a_sighting_beyond_the_reorder_buffer__is_rejected_and_reported() -> None:
    """A stream running minutes behind is a fault the operator needs to see.

    Silently dropping it hides the fault; silently accepting it rewrites a route
    that has already been acted on.
    """
    reconstructor = _reconstructor(reorder_buffer_sec=60.0)
    for candidate, sighting in _feed([("cam_01", 0.0, 0.9), ("cam_03", 600.0, 0.9)]):
        reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    stale_candidate, stale = _feed([("cam_02", 120.0, 0.9)])[0]
    extension = reconstructor.extend([stale_candidate], {stale.sighting_id: stale})

    assert [entry.sighting_id for entry in extension.rejected] == [stale.sighting_id]
    assert extension.rejected[0].behind_by_sec == pytest.approx(480.0)
    assert "reorder buffer" in extension.rejected[0].reason
    assert _route(extension.result) == ["cam_01", "cam_03"]


def test_a_sighting_inside_the_reorder_buffer__is_admitted() -> None:
    """The boundary the buffer exists to draw."""
    reconstructor = _reconstructor(reorder_buffer_sec=600.0)
    for candidate, sighting in _feed([("cam_01", 0.0, 0.9), ("cam_03", 600.0, 0.9)]):
        reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    late_candidate, late = _feed([("cam_02", 120.0, 0.9)])[0]
    extension = reconstructor.extend([late_candidate], {late.sighting_id: late})

    assert extension.rejected == []
    assert _route(extension.result) == ["cam_01", "cam_02", "cam_03"]


def test_the_first_extension__has_nothing_to_be_late_for() -> None:
    """Boundary: with no tail yet, no arrival can be out of order."""
    candidate, sighting = _feed([("cam_01", 0.0, 0.9)])[0]

    extension = _reconstructor().extend([candidate], {sighting.sighting_id: sighting})

    assert extension.rejected == []
    assert _route(extension.result) == ["cam_01"]


def test_extending_with_nothing__leaves_the_trajectory_alone() -> None:
    """An empty batch is the common case in a live loop."""
    reconstructor = _reconstructor()
    for candidate, sighting in _feed([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9)]):
        reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    extension = reconstructor.extend([], {})

    assert extension.recomputed_from_index is None
    assert _route(extension.result) == ["cam_01", "cam_02"]


# ---------------------------------------------------------------------------
# Equivalence with full recomputation
# ---------------------------------------------------------------------------


def test_incremental_and_full_recomputation__agree_on_an_in_order_stream() -> None:
    """The base case."""
    pairs = _feed(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_03", 240.0, 0.85),
            ("cam_04", 360.0, 0.9),
        ]
    )
    reconstructor = _reconstructor()
    for candidate, sighting in pairs:
        extension = reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    assert _route(extension.result) == _full(pairs)


def test_incremental_and_full_recomputation__agree_on_a_reversed_stream() -> None:
    """The worst realistic ordering: every arrival lands before the current tail."""
    pairs = _feed(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            ("cam_03", 240.0, 0.85),
            ("cam_04", 360.0, 0.9),
        ]
    )
    reconstructor = _reconstructor(reorder_buffer_sec=10_000.0)
    for candidate, sighting in reversed(pairs):
        extension = reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    assert _route(extension.result) == _full(pairs)


@_HYPOTHESIS
@given(seed=st.integers(min_value=0, max_value=500))
def test_incremental_and_full_recomputation__agree_for_arbitrary_arrival_orders(
    seed: int,
) -> None:
    """Property: the order sightings arrive in must not change the answer.

    Includes a decoy at the unreachable camera, so the orderings differ in what
    the search has to reject as well as in what it accepts.
    """
    pairs = _feed(
        [
            ("cam_01", 0.0, 0.9),
            ("cam_02", 120.0, 0.9),
            (FAR_CAMERA, 180.0, 0.95),
            ("cam_03", 240.0, 0.85),
            ("cam_04", 360.0, 0.9),
        ]
    )
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    reconstructor = _reconstructor(reorder_buffer_sec=10_000.0)
    for candidate, sighting in shuffled:
        extension = reconstructor.extend([candidate], {sighting.sighting_id: sighting})

    assert _route(extension.result) == _full(pairs)


def test_a_batch_arriving_at_once__matches_the_same_batch_arriving_one_by_one() -> None:
    """Batch size is a delivery detail and must not change the trajectory."""
    pairs = _feed([("cam_01", 0.0, 0.9), ("cam_02", 120.0, 0.9), ("cam_03", 240.0, 0.9)])

    at_once = _reconstructor()
    batched = at_once.extend(
        [candidate for candidate, _ in pairs],
        {sighting.sighting_id: sighting for _, sighting in pairs},
    )

    one_by_one = _reconstructor()
    for candidate, sighting in pairs:
        singly = one_by_one.extend([candidate], {sighting.sighting_id: sighting})

    assert _route(batched.result) == _route(singly.result)
