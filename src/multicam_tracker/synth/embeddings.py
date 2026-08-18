"""Synthetic re-id embeddings with controllable difficulty.

Two distributions decide whether re-id works: how similar a vehicle looks to
*itself* across cameras, and how similar it looks to *other* vehicles. Both are
controllable here, and they can be made to overlap on purpose.

``hard_negative_fraction`` is the parameter that matters. Decoys drawn uniformly
at random sit near-orthogonal to the target in high dimensions -- cosine
similarity around zero -- so any threshold separates them and re-id looks
flawless. Real traffic contains other silver hatchbacks. A hard negative is a
decoy drawn *near* the target's base vector, and without them stage 07 would be
tuned against a problem that does not exist.
"""

from __future__ import annotations

import math
import random

__all__ = ["EmbeddingFactory", "cosine_similarity", "normalize"]


def normalize(vector: list[float]) -> list[float]:
    """Return the L2-normalized form of a vector.

    Args:
        vector: The vector to scale.

    Returns:
        A vector of unit length. A zero vector is returned as a unit vector
        along the first axis, since it has no direction to preserve and every
        embedding in the system must be normalized.
    """
    norm = math.sqrt(math.fsum(component * component for component in vector))
    if norm == 0.0:
        return [1.0] + [0.0] * (len(vector) - 1)
    return [component / norm for component in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Similarity in ``[-1, 1]``, or ``0.0`` if either has zero magnitude.
    """
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class EmbeddingFactory:
    """Draws base vectors per vehicle and noisy observations per sighting.

    Args:
        rng: Seeded generator. All randomness flows through it.
        dimension: Vector width, matching the configured embedding dimension.
        intra_class_sigma: Per-component Gaussian noise added to a vehicle's
            base vector for each sighting. Higher means the same vehicle looks
            less like itself from camera to camera.
        hard_negative_sigma: Spread of hard negatives around the target's base
            vector. Smaller means harder.
    """

    def __init__(
        self,
        rng: random.Random,
        *,
        dimension: int,
        intra_class_sigma: float,
        hard_negative_sigma: float,
    ) -> None:
        self._rng = rng
        self._dimension = dimension
        # Both sigmas are *relative* to a unit vector's natural component scale,
        # which is 1/sqrt(dimension). Treating them as absolute per-component
        # values would make similarity depend on the vector width: a sigma that
        # barely perturbs a 64-dimensional vector would obliterate a
        # 512-dimensional one, and every scenario would need retuning whenever
        # stage 13 picked a different model.
        self._component_scale = 1.0 / math.sqrt(dimension)
        self._intra_class_sigma = intra_class_sigma * self._component_scale
        self._hard_negative_sigma = hard_negative_sigma * self._component_scale

    @property
    def dimension(self) -> int:
        """Return the configured vector width."""
        return self._dimension

    def base_vector(self) -> list[float]:
        """Draw a fresh vehicle identity vector.

        Returns:
            A unit vector drawn from an isotropic Gaussian, which in high
            dimensions puts independent draws close to orthogonal -- the
            "obviously different vehicle" case.
        """
        return normalize([self._rng.gauss(0.0, 1.0) for _ in range(self._dimension)])

    def hard_negative_of(self, base: list[float]) -> list[float]:
        """Draw a vector deliberately close to ``base``.

        Simulates a same make, model, and colour vehicle: the decoy re-id
        actually has to reject.

        Args:
            base: The vector to sit near.

        Returns:
            A unit vector at high cosine similarity to ``base``.
        """
        return normalize(
            [component + self._rng.gauss(0.0, self._hard_negative_sigma) for component in base]
        )

    def observation_of(self, base: list[float]) -> list[float]:
        """Draw one sighting's embedding for a vehicle.

        Args:
            base: The vehicle's identity vector.

        Returns:
            A unit vector near ``base``, differing by the intra-class noise that
            stands in for viewing angle, lighting, and occlusion.
        """
        return normalize(
            [component + self._rng.gauss(0.0, self._intra_class_sigma) for component in base]
        )
