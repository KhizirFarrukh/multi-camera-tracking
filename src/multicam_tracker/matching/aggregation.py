"""Aggregating similarity across a target's several reference embeddings.

A target accumulates references as humans confirm matches -- the same vehicle
from different angles, in different light. Comparing a candidate against that
set needs a rule for turning several similarities into one.

**``max`` is the default, and the choice matters.** A vehicle photographed from
the front and the rear produces two references that are barely similar to each
other. Averaging them yields a vector resembling neither, so a genuine rear-view
sighting scores poorly against the mean while scoring highly against the rear
reference. ``max`` asks the question that is actually meaningful: *does this
look like the target from any angle we have seen?*

The cost is precision. ``max`` rises monotonically as references accumulate, so
one bad reference permanently inflates every score. That is why
:mod:`multicam_tracker.matching.references` refuses to add a reference from an
unconfirmed match, and why pruning keeps a diverse subset rather than a recent
one.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching.similarity import as_matrix, as_vector, batch_cosine_similarity

__all__ = ["AggregationStrategy", "aggregate_similarity"]


class AggregationStrategy(StrEnum):
    """How to combine similarity across several reference embeddings."""

    MAX = "max"
    """Highest similarity to any reference. Most permissive, and the default."""

    MEAN = "mean"
    """Similarity to the averaged reference vector. Most conservative."""

    TOPK_MEAN = "topk_mean"
    """Mean of the k highest similarities. A middle ground."""


def aggregate_similarity(
    query: list[float] | npt.NDArray[np.float64],
    references: list[list[float]],
    *,
    strategy: AggregationStrategy = AggregationStrategy.MAX,
    top_k: int = 2,
) -> float:
    """Reduce a candidate's similarity across a reference set to one number.

    All three strategies coincide for a single reference, so a target with one
    confirmed sighting behaves identically however this is configured.

    Args:
        query: The candidate embedding.
        references: The target's reference embeddings.
        strategy: How to combine. See the class docstring for the trade-off.
        top_k: How many similarities ``topk_mean`` averages. ``1`` makes it
            equivalent to ``max``.

    Returns:
        A similarity in ``[-1, 1]``.

    Raises:
        MatchingError: If the reference set is empty, or ``top_k`` is not
            positive. An empty set cannot produce a similarity, and returning a
            neutral 0.0 would look like a legitimate weak match.
        ValidationError: If any embedding is invalid or the widths differ.
    """
    if not references:
        raise MatchingError(
            "Cannot aggregate similarity against an empty reference set",
            {"strategy": strategy.value},
        )
    if top_k < 1:
        raise MatchingError("top_k must be at least 1", {"top_k": top_k})

    matrix = as_matrix(references, field_name="references")
    vector = as_vector(query, field_name="query")

    if strategy is AggregationStrategy.MEAN:
        # Average the reference vectors, then compare once. The averaged vector
        # is renormalized because averaging unit vectors does not preserve unit
        # length, and the similarity primitives require it.
        averaged = matrix.mean(axis=0)
        norm = float(np.linalg.norm(averaged))
        if norm == 0.0:
            raise MatchingError(
                "Reference embeddings average to a zero vector, which has no "
                "direction to compare against",
                {"reference_count": int(matrix.shape[0])},
            )
        return float(np.clip(np.dot(averaged / norm, vector), -1.0, 1.0))

    similarities = batch_cosine_similarity(vector, matrix)

    if strategy is AggregationStrategy.MAX:
        return float(similarities.max())

    count = min(top_k, similarities.size)
    return float(np.sort(similarities)[-count:].mean())
