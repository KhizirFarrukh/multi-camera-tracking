"""Unit tests for :mod:`multicam_tracker.synth.embeddings`.

The separation test is the point: intra-vehicle similarity must clearly exceed
inter-vehicle similarity, and hard negatives must genuinely close that gap. A
"hard negative" that sat at cosine 0.1 would be decorative, and stage 07 would
be tuned against a problem that does not exist.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from multicam_tracker.synth import EmbeddingFactory, cosine_similarity, normalize

pytestmark = pytest.mark.unit

DIMENSION = 512


def _factory(seed: int = 5, *, intra: float = 0.15, hard: float = 0.15) -> EmbeddingFactory:
    """Build a factory with explicit parameters.

    Args:
        seed: Generator seed.
        intra: Intra-class sigma.
        hard: Hard-negative sigma.

    Returns:
        The factory.
    """
    return EmbeddingFactory(
        random.Random(seed),
        dimension=DIMENSION,
        intra_class_sigma=intra,
        hard_negative_sigma=hard,
    )


def _norm(vector: list[float]) -> float:
    """Return a vector's L2 norm.

    Args:
        vector: The vector to measure.

    Returns:
        Its magnitude.
    """
    return math.sqrt(sum(component * component for component in vector))


def test_normalize__produces_unit_length() -> None:
    """Every embedding in the system must be normalized; this is where that starts."""
    assert _norm(normalize([3.0, 4.0, 0.0])) == pytest.approx(1.0)


def test_normalize__zero_vector__returns_a_unit_vector() -> None:
    """Boundary: a zero vector has no direction, but must still be normalized."""
    result = normalize([0.0, 0.0, 0.0])

    assert _norm(result) == pytest.approx(1.0)


def test_cosine_similarity__identical_vectors__is_one() -> None:
    """The upper bound."""
    vector = normalize([1.0, 2.0, 3.0])

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity__opposite_vectors__is_minus_one() -> None:
    """The lower bound."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity__orthogonal_vectors__is_zero() -> None:
    """The 'unrelated vehicles' case."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity__zero_magnitude__is_zero_rather_than_undefined() -> None:
    """Boundary: no division by zero escapes to the caller."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_factory__base_vectors__are_normalized_and_of_the_right_dimension() -> None:
    """The two invariants the Sighting model enforces downstream."""
    factory = _factory()

    for _ in range(20):
        vector = factory.base_vector()
        assert len(vector) == DIMENSION
        assert _norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_factory__observations__are_normalized() -> None:
    """Adding noise then normalizing must not leave the vector off the unit sphere."""
    factory = _factory()
    base = factory.base_vector()

    for _ in range(20):
        assert _norm(factory.observation_of(base)) == pytest.approx(1.0, abs=1e-9)


def test_factory__intra_similarity__clearly_exceeds_inter_similarity() -> None:
    """The default configuration must be separable, or nothing downstream can work."""
    factory = _factory()
    base_a, base_b = factory.base_vector(), factory.base_vector()

    intra = [
        cosine_similarity(factory.observation_of(base_a), factory.observation_of(base_a))
        for _ in range(60)
    ]
    inter = [
        cosine_similarity(factory.observation_of(base_a), factory.observation_of(base_b))
        for _ in range(60)
    ]

    assert statistics.mean(intra) > statistics.mean(inter) + 0.5


def test_factory__raising_intra_sigma__lowers_intra_similarity() -> None:
    """The knob has to actually turn: this is how 'degraded' gets harder."""
    tight = _factory(intra=0.05)
    loose = _factory(intra=0.60)

    tight_base = tight.base_vector()
    loose_base = loose.base_vector()

    tight_scores = [
        cosine_similarity(tight_base, tight.observation_of(tight_base)) for _ in range(60)
    ]
    loose_scores = [
        cosine_similarity(loose_base, loose.observation_of(loose_base)) for _ in range(60)
    ]

    assert statistics.mean(tight_scores) > statistics.mean(loose_scores)


def test_factory__hard_negatives__sit_far_closer_than_random_decoys() -> None:
    """Simulating a same make, model, and colour vehicle."""
    factory = _factory(hard=0.15)
    target = factory.base_vector()

    hard = [cosine_similarity(target, factory.hard_negative_of(target)) for _ in range(40)]
    random_decoys = [cosine_similarity(target, factory.base_vector()) for _ in range(40)]

    assert statistics.mean(hard) > 0.85
    assert statistics.mean(random_decoys) < 0.25


def test_factory__hard_negatives__are_genuinely_hard() -> None:
    """At least one must clear a naive threshold.

    Otherwise the 'hard_negatives' scenario would be decorative and stage 07
    would be tuned against a problem that does not exist.
    """
    factory = _factory(hard=0.15)
    target = factory.base_vector()

    scores = [cosine_similarity(target, factory.hard_negative_of(target)) for _ in range(40)]

    assert max(scores) > 0.92, "no hard negative reaches the default auto-accept threshold"


def test_factory__smaller_hard_negative_sigma__is_harder() -> None:
    """The difficulty knob points the right way."""
    easy = _factory(hard=1.0)
    hard = _factory(hard=0.15)

    easy_base = easy.base_vector()
    hard_base = hard.base_vector()

    easy_scores = [
        cosine_similarity(easy_base, easy.hard_negative_of(easy_base)) for _ in range(40)
    ]
    hard_scores = [
        cosine_similarity(hard_base, hard.hard_negative_of(hard_base)) for _ in range(40)
    ]

    assert statistics.mean(hard_scores) > statistics.mean(easy_scores)


def test_factory__is_deterministic_for_a_given_seed() -> None:
    """Reproducibility applies to every stream, embeddings included."""
    assert _factory(seed=9).base_vector() == _factory(seed=9).base_vector()
