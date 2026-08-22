"""Unit tests for topology-constrained embedding search.

The pair that matters is the lookalike test run twice: once anchored, once not.
A decoy whose embedding is nearly identical to the target survives any threshold
that keeps the true match, and is removed only by the constraint. Running the
same decoy through both paths is what shows the constraint -- not the
threshold -- is doing the work.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import pytest
from structlog.testing import capture_logs

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching import find_embedding_matches
from multicam_tracker.matching.embedding_search import EmbeddingMatchResult
from multicam_tracker.models import ReviewStatus, Sighting, TimeWindow
from multicam_tracker.topology import Topology
from tests.fixtures.factories import make_camera, make_sighting, make_target, unit_vector
from tests.fixtures.fake_repositories import InMemoryStore, build_in_memory_repositories
from tests.fixtures.topologies import make_topology

pytestmark = pytest.mark.unit

MODEL_VERSION = "reid_v1"

CAMERAS = ("cam_01", "cam_02", "cam_03", "cam_09")
"""``cam_09`` is deliberately linked to nothing: it is where the lookalike sits."""

REFERENCE = unit_vector(seed=1)


def _near(base: list[float], seed: int, weight: float) -> list[float]:
    """Return a unit vector a controlled distance from ``base``.

    Args:
        base: The vector to perturb.
        seed: Seeds the perturbation direction.
        weight: How much of the perturbation to mix in. Smaller is more similar.

    Returns:
        A normalized vector near ``base``.
    """
    noise = unit_vector(seed=seed)
    mixed = [b + weight * n for b, n in zip(base, noise, strict=True)]
    norm = math.sqrt(math.fsum(value * value for value in mixed))
    return [value / norm for value in mixed]


def _topology() -> Topology:
    """Return a chain with one unreachable camera hanging off it.

    Returns:
        ``cam_01 -> cam_02 -> cam_03``, plus an isolated ``cam_09``.
    """
    return make_topology(
        list(CAMERAS),
        [("cam_01", "cam_02", 60.0, 300.0), ("cam_02", "cam_03", 60.0, 300.0)],
    )


class RecordingVectorRepository:
    """Delegates to a sighting repository, recording the vector-query arguments.

    The camera restriction is an argument, not a result: a repository that
    ignored it would return the rows the caller then filters anyway, so the
    narrowing has to be asserted where it is requested.

    Args:
        inner: The repository to delegate to.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.camera_ids: list[str] | None = None
        self.window: TimeWindow | None = None
        self.model_version: str | None = None

    def find_nearest_by_embedding(
        self,
        embedding: list[float],
        k: int,
        *,
        camera_ids: list[str] | None = None,
        window: TimeWindow | None = None,
        model_version: str | None = None,
    ) -> Any:
        """Record the restrictions, then delegate.

        Args:
            embedding: Query vector.
            k: Maximum results.
            camera_ids: Camera restriction under test.
            window: Time restriction under test.
            model_version: Model-version restriction under test.

        Returns:
            Whatever the wrapped repository returns.
        """
        self.camera_ids = camera_ids
        self.window = window
        self.model_version = model_version
        return self._inner.find_nearest_by_embedding(
            embedding,
            k,
            camera_ids=camera_ids,
            window=window,
            model_version=model_version,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else.

        Args:
            name: Attribute being accessed.

        Returns:
            The wrapped attribute.
        """
        return getattr(self._inner, name)


class IgnoresVersionFilter:
    """A repository that drops the model-version filter, as a broken one would.

    Args:
        inner: The repository to delegate to.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def find_nearest_by_embedding(
        self,
        embedding: list[float],
        k: int,
        *,
        camera_ids: list[str] | None = None,
        window: TimeWindow | None = None,
        model_version: str | None = None,
    ) -> Any:
        """Delegate without the version restriction.

        Args:
            embedding: Query vector.
            k: Maximum results.
            camera_ids: Camera restriction.
            window: Time restriction.
            model_version: Discarded, deliberately.

        Returns:
            Whatever the wrapped repository returns.
        """
        return self._inner.find_nearest_by_embedding(
            embedding, k, camera_ids=camera_ids, window=window
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else.

        Args:
            name: Attribute being accessed.

        Returns:
            The wrapped attribute.
        """
        return getattr(self._inner, name)


def _embedded(
    camera_id: str, offset_sec: float, embedding: list[float], **overrides: Any
) -> Sighting:
    """Build a sighting carrying an embedding.

    Args:
        camera_id: The observing camera.
        offset_sec: Seconds after the base instant.
        embedding: The vector to attach.
        **overrides: Further field overrides.

    Returns:
        A validated sighting.
    """
    fields: dict[str, Any] = {
        "embedding": embedding,
        "embedding_model_version": MODEL_VERSION,
    }
    fields.update(overrides)
    return make_sighting(camera_id, offset_sec=offset_sec, **fields)


class Scenario(NamedTuple):
    """The pieces of the shared fixture a test needs to reach for."""

    sightings: RecordingVectorRepository
    """The repository under test, wrapped to record its vector queries."""

    anchor: Sighting
    """The confirmed sighting the search starts from."""

    true_match: Sighting
    """A genuine later sighting of the same vehicle."""

    lookalike: Sighting
    """The decoy at the unreachable camera."""


def _scenario() -> Scenario:
    """Build the shared scenario.

    The lookalike is *more* similar to the reference than the true match is, so
    no threshold can separate them and only the topology can.

    Returns:
        The scenario.
    """
    store = InMemoryStore()
    repositories = build_in_memory_repositories(store)
    for camera_id in CAMERAS:
        repositories.cameras.upsert(make_camera(camera_id=camera_id))

    anchor = _embedded("cam_01", 0.0, REFERENCE)
    true_match = _embedded("cam_02", 120.0, _near(REFERENCE, 7, 0.30))
    lookalike = _embedded("cam_09", 120.0, _near(REFERENCE, 8, 0.05))

    for sighting in (anchor, true_match, lookalike):
        repositories.sightings.add(sighting)

    return Scenario(
        sightings=RecordingVectorRepository(repositories.sightings),
        anchor=anchor,
        true_match=true_match,
        lookalike=lookalike,
    )


def _ids(results: list[EmbeddingMatchResult]) -> set[str]:
    """Return the sighting ids in a result list.

    Args:
        results: Match results.

    Returns:
        The ids.
    """
    return {result.sighting.sighting_id for result in results}


# ---------------------------------------------------------------------------
# The constraint, measured against itself
# ---------------------------------------------------------------------------


def test_find__anchored__excludes_a_lookalike_at_an_unreachable_camera() -> None:
    """The claim the stage rests on: narrowing the space removes the decoy."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert scenario.true_match.sighting_id in _ids(results)
    assert scenario.lookalike.sighting_id not in _ids(results)


def test_find__unanchored__returns_the_same_lookalike__outranking_the_true_match() -> None:
    """The control. Without the anchor the decoy comes back, above the real match.

    Which shows the exclusion above came from the topology, not from a threshold
    that would have removed the decoy anyway: any cut-off keeping the true match
    keeps the decoy too.
    """
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target, scenario.sightings, _topology(), model_version=MODEL_VERSION
    )

    ranking = [result.sighting.sighting_id for result in results]
    assert scenario.lookalike.sighting_id in ranking
    assert ranking.index(scenario.lookalike.sighting_id) < ranking.index(
        scenario.true_match.sighting_id
    )


def test_find__anchored__asks_the_repository_for_a_narrowed_set() -> None:
    """The restriction must reach the query, not be applied after it."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert scenario.sightings.camera_ids is not None
    assert "cam_09" not in scenario.sightings.camera_ids
    assert "cam_01" in scenario.sightings.camera_ids, "repeat passes stay searchable"


def test_find__explicit_cameras__intersect_with_the_reachable_set() -> None:
    """A caller may narrow the search further; it cannot widen it."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        camera_ids=["cam_03", "cam_09"],
        model_version=MODEL_VERSION,
    )

    assert scenario.sightings.camera_ids == ["cam_03"]


def test_find__anchored__searches_forward_from_the_anchor_only() -> None:
    """The window opens at the anchor: the vehicle cannot arrive before it left."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    window = scenario.sightings.window
    assert window is not None
    assert window.start_utc == scenario.anchor.timestamp_utc


def test_find__unanchored__warns_that_the_query_is_weaker() -> None:
    """A caller must be able to tell a global search from a constrained one."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    with capture_logs() as entries:
        find_embedding_matches(target, scenario.sightings, _topology(), model_version=MODEL_VERSION)

    warnings = [entry for entry in entries if entry["log_level"] == "warning"]
    assert [entry["event"] for entry in warnings] == ["embedding_search_unconstrained"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_find__target_without_references__raises() -> None:
    """Searching by appearance with nothing to compare against is a caller bug."""
    scenario = _scenario()

    with pytest.raises(MatchingError, match="no reference embeddings"):
        find_embedding_matches(
            make_target(), scenario.sightings, _topology(), anchor_sighting=scenario.anchor
        )


def test_find__the_anchor_itself__is_not_returned_as_a_match() -> None:
    """It is the query, not a result."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert scenario.anchor.sighting_id not in _ids(results)


def test_find__a_mixed_version_table__filters_the_other_model_in_the_query() -> None:
    """During a re-embedding migration both versions are present on purpose.

    The other version is excluded where the rows are selected, so the search
    still works rather than returning a page of unusable candidates.
    """
    scenario = _scenario()
    superseded = _embedded("cam_02", 150.0, REFERENCE, embedding_model_version="reid_v2")
    scenario.sightings.add(superseded)
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert scenario.sightings.model_version == MODEL_VERSION
    assert superseded.sighting_id not in _ids(results)


def test_find__a_repository_ignoring_the_filter__still_cannot_compare_across_models() -> None:
    """Defence in depth. The scoring path checks even when the query did not.

    A cross-version similarity is not NaN or out of range -- it is 0.31, and a
    ranking built on it looks entirely reasonable while being noise. That has to
    be impossible by construction, not merely filtered out upstream.
    """
    scenario = _scenario()
    scenario.sightings.add(_embedded("cam_02", 150.0, REFERENCE, embedding_model_version="reid_v2"))
    target = make_target(reference_embeddings=[REFERENCE])

    with pytest.raises(MatchingError, match="different model versions"):
        find_embedding_matches(
            target,
            IgnoresVersionFilter(scenario.sightings),
            _topology(),
            anchor_sighting=scenario.anchor,
            model_version=MODEL_VERSION,
        )


def test_find__sightings_without_an_embedding__are_skipped_without_raising() -> None:
    """Most sightings carry no vector; that is ordinary, not an error."""
    scenario = _scenario()
    scenario.sightings.add(make_sighting("cam_02", offset_sec=140.0))
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert scenario.true_match.sighting_id in _ids(results)


def test_find__a_perfect_match_outside_the_arrival_window__is_excluded() -> None:
    """Similarity cannot buy its way past the timetable.

    The candidate here is the reference vector exactly -- similarity 1.0 -- at a
    reachable camera, arriving far later than any route allows.
    """
    scenario = _scenario()
    too_late = _embedded("cam_02", 4000.0, REFERENCE)
    scenario.sightings.add(too_late)
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert too_late.sighting_id not in _ids(results)


def test_find__nothing_clears_the_review_floor__returns_empty() -> None:
    """An empty result is the honest answer, not an error."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[unit_vector(seed=404)])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert results == []


def test_find__every_result_carries_an_adjudication_state() -> None:
    """Nothing is returned without saying whether a human still has to look."""
    scenario = _scenario()
    target = make_target(reference_embeddings=[REFERENCE])

    results = find_embedding_matches(
        target,
        scenario.sightings,
        _topology(),
        anchor_sighting=scenario.anchor,
        model_version=MODEL_VERSION,
    )

    assert results
    assert all(
        result.review_status in {ReviewStatus.AUTO_ACCEPTED, ReviewStatus.PENDING_REVIEW}
        for result in results
    )
