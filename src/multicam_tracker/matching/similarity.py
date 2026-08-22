"""Vector similarity primitives.

Cosine similarity, assuming L2-normalized inputs -- and **validating** that
assumption rather than trusting it. A non-normalized embedding is an upstream
bug: an extractor emitting raw logits, or a vector concatenated from two models.
Silently renormalizing would hide it and quietly corrupt every similarity
computed afterwards, which is why stage 02 rejects such vectors at the model
boundary and this module rejects them again here.

numpy is used because re-id search runs over large candidate sets and a Python
loop would dominate the query. It carries no CV or model-weight dependency, so
the stage stays free of torch.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.models.base import EMBEDDING_NORM_TOLERANCE

__all__ = [
    "SimilarityHit",
    "as_matrix",
    "as_vector",
    "batch_cosine_similarity",
    "cosine_similarity",
    "top_k_similar",
]

_FLOAT_SLACK = 1e-9
"""Absorbs floating-point noise so a vector sitting exactly on the tolerance
boundary is accepted, matching the model-layer validator."""


class SimilarityHit(NamedTuple):
    """One candidate and its similarity to the query.

    The field is ``position`` rather than ``index`` because ``NamedTuple``
    inherits ``tuple.index``, and shadowing it would break the tuple protocol.
    """

    position: int
    """Offset of the candidate in the list that was searched."""

    similarity: float


def as_vector(
    values: list[float] | npt.NDArray[np.float64], *, field_name: str
) -> npt.NDArray[np.float64]:
    """Validate and convert one embedding to a numpy array.

    Args:
        values: The embedding.
        field_name: Name reported in error messages.

    Returns:
        The embedding as a float64 array.

    Raises:
        ValidationError: If the vector is empty, contains non-finite values, has
            zero magnitude, or is not L2-normalized. A zero vector would produce
            NaN rather than a similarity, which would propagate silently through
            a ranking.
    """
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValidationError(
            "Embedding must be a non-empty one-dimensional vector",
            {"field": field_name, "shape": list(array.shape)},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError("Embedding contains non-finite values", {"field": field_name})

    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValidationError(
            "Embedding has zero magnitude and no direction to compare",
            {"field": field_name},
        )
    if abs(norm - 1.0) > EMBEDDING_NORM_TOLERANCE + _FLOAT_SLACK:
        raise ValidationError(
            "Embedding is not L2-normalized. It is rejected rather than "
            "renormalized so the defect is visible at its source",
            {"field": field_name, "norm": norm, "tolerance": EMBEDDING_NORM_TOLERANCE},
        )

    return array


def as_matrix(
    rows: list[list[float]], *, field_name: str, expected_dim: int | None = None
) -> npt.NDArray[np.float64]:
    """Validate and stack several embeddings into a matrix.

    Args:
        rows: The embeddings.
        field_name: Name reported in error messages.
        expected_dim: Width every row must have.

    Returns:
        A ``(n, dim)`` float64 array. Shape ``(0, 0)`` for an empty input.

    Raises:
        ValidationError: If any row fails validation or the widths disagree.
    """
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)

    vectors = [
        as_vector(row, field_name=f"{field_name}[{index}]") for index, row in enumerate(rows)
    ]
    widths = {vector.size for vector in vectors}
    if len(widths) > 1:
        raise ValidationError(
            "Embeddings have inconsistent dimensions",
            {"field": field_name, "dimensions": sorted(widths)},
        )
    if expected_dim is not None and vectors[0].size != expected_dim:
        raise ValidationError(
            "Embeddings have the wrong dimension",
            {"field": field_name, "expected": expected_dim, "actual": vectors[0].size},
        )

    return np.vstack(vectors)


def cosine_similarity(
    left: list[float] | npt.NDArray[np.float64],
    right: list[float] | npt.NDArray[np.float64],
) -> float:
    """Return the cosine similarity of two L2-normalized embeddings.

    Args:
        left: First embedding.
        right: Second embedding.

    Returns:
        Similarity in ``[-1, 1]``. Clamped, because floating-point error can push
        the dot product of two unit vectors a hair outside the range and a
        similarity of 1.0000000002 would fail a downstream range check.

    Raises:
        ValidationError: If either vector is invalid or the widths differ.
    """
    a = as_vector(left, field_name="left")
    b = as_vector(right, field_name="right")

    if a.size != b.size:
        raise ValidationError(
            "Cannot compare embeddings of different dimensions",
            {"left_dimension": a.size, "right_dimension": b.size},
        )

    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def batch_cosine_similarity(
    query: list[float] | npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the similarity of one query against every row of a matrix.

    Args:
        query: The query embedding.
        matrix: Candidate embeddings, one per row, already validated.

    Returns:
        A one-dimensional array of similarities, one per row. Empty for an empty
        matrix.

    Raises:
        ValidationError: If the query is invalid or its width differs from the
            matrix width.
    """
    vector = as_vector(query, field_name="query")

    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)

    if matrix.shape[1] != vector.size:
        raise ValidationError(
            "Cannot compare a query against candidates of a different dimension",
            {"query_dimension": vector.size, "candidate_dimension": int(matrix.shape[1])},
        )

    return np.clip(matrix @ vector, -1.0, 1.0)


def top_k_similar(
    query: list[float] | npt.NDArray[np.float64],
    candidates: list[list[float]],
    k: int,
) -> list[SimilarityHit]:
    """Return the ``k`` most similar candidates, best first.

    Args:
        query: The query embedding.
        candidates: Candidate embeddings.
        k: Maximum number of results. Asking for more than exist returns all of
            them rather than raising -- a caller ranking a short list should not
            have to special-case it.

    Returns:
        At most ``k`` hits, sorted by descending similarity with ties broken by
        index so the ordering is reproducible.

    Raises:
        ValidationError: If the query or any candidate is invalid.
    """
    if k <= 0 or not candidates:
        return []

    matrix = as_matrix(candidates, field_name="candidates")
    scores = batch_cosine_similarity(query, matrix)

    order = sorted(range(scores.size), key=lambda index: (-scores[index], index))
    return [SimilarityHit(position=index, similarity=float(scores[index])) for index in order[:k]]
