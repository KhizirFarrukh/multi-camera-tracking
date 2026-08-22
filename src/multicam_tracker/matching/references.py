"""Managing a target's reference embedding set.

References accumulate as humans confirm matches, and the set is what every
future appearance search is measured against. Two failure modes matter, and both
are guarded here.

**Poisoning.** Because ``max`` aggregation rises monotonically as references
accumulate, a single wrong reference permanently inflates the similarity of
everything resembling it -- and the damage compounds, because the inflated scores
produce more wrong confirmations. So a reference may only come from a sighting a
human actually **confirmed**. A ``pending_review`` match is a question, not an
answer, and admitting one would let the system confirm its own guesses.

**Collapse.** Keeping the most recent references clusters them: consecutive
sightings on one route show the same vehicle from nearly the same angle, so a
recency-pruned set describes one viewpoint in triplicate. Pruning therefore keeps
a *diverse* subset by greedy max-min-distance selection, so the set spans
viewpoints instead of stacking on one.
"""

from __future__ import annotations

import numpy as np

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching.similarity import as_matrix, cosine_similarity
from multicam_tracker.matching.versioning import require_same_model_version
from multicam_tracker.models import ReviewStatus, Sighting, Target

__all__ = ["add_reference_embedding", "prune_references", "select_diverse"]


def add_reference_embedding(
    target: Target,
    sighting: Sighting,
    review_status: ReviewStatus,
    *,
    max_references: int | None = None,
    model_version: str | None = None,
) -> Target:
    """Add a confirmed sighting's embedding to a target's reference set.

    Args:
        target: The target to extend.
        sighting: The sighting whose appearance to remember.
        review_status: The adjudication state of the match that produced it.
            Only ``CONFIRMED`` and ``AUTO_ACCEPTED`` are admissible.
        max_references: Cap after which the set is pruned. Defaults to config.
        model_version: Version the existing references were embedded with.
            Defaults to the target's first reference, if any.

    Returns:
        A new target with the reference added and the set pruned to size.
        The input is not mutated -- domain models are values.

    Raises:
        MatchingError: If the match was not confirmed, if the sighting carries
            no embedding, or if the embedding came from a different model
            version than the existing references.
    """
    from multicam_tracker.config import get_settings

    if review_status not in {ReviewStatus.CONFIRMED, ReviewStatus.AUTO_ACCEPTED}:
        raise MatchingError(
            "Only a confirmed match may become a reference embedding. Admitting "
            "an unreviewed one would let the system confirm its own guesses, and "
            "max aggregation would make the error permanent",
            {
                "target_id": target.target_id,
                "sighting_id": sighting.sighting_id,
                "review_status": review_status.value,
            },
        )

    if sighting.embedding is None:
        raise MatchingError(
            "Sighting carries no embedding to use as a reference",
            {"target_id": target.target_id, "sighting_id": sighting.sighting_id},
        )

    existing = list(target.reference_embeddings or [])
    if existing:
        require_same_model_version(model_version, sighting.embedding_model_version)

    cap = (
        max_references
        if max_references is not None
        else get_settings().thresholds.embedding_max_references
    )

    return target.model_copy(
        update={"reference_embeddings": select_diverse([*existing, list(sighting.embedding)], cap)}
    )


def prune_references(target: Target, max_references: int | None = None) -> Target:
    """Reduce a target's reference set to a diverse subset of the configured size.

    Args:
        target: The target to prune.
        max_references: Maximum references to keep. Defaults to config.

    Returns:
        A new target with at most ``max_references`` references. A set already
        at or below the limit is returned unchanged, including its ordering, so
        pruning is a no-op rather than a reshuffle.
    """
    from multicam_tracker.config import get_settings

    references = target.reference_embeddings
    if not references:
        return target

    cap = (
        max_references
        if max_references is not None
        else get_settings().thresholds.embedding_max_references
    )
    if len(references) <= cap:
        return target

    return target.model_copy(update={"reference_embeddings": select_diverse(references, cap)})


def select_diverse(references: list[list[float]], limit: int) -> list[list[float]]:
    """Choose a spread-out subset by greedy max-min-distance selection.

    Starts from the first reference, then repeatedly adds whichever remaining
    vector is *least* similar to everything already chosen. That maximises the
    minimum pairwise distance, which is exactly the property a reference set
    wants: coverage of distinct viewpoints rather than repeated views of one.

    Args:
        references: The candidate embeddings.
        limit: How many to keep.

    Returns:
        Up to ``limit`` references. Returned unchanged when the set is already
        small enough.

    Raises:
        MatchingError: If ``limit`` is not positive.
        ValidationError: If any embedding is invalid.
    """
    if limit < 1:
        raise MatchingError("Reference limit must be at least 1", {"limit": limit})
    if len(references) <= limit:
        return [list(reference) for reference in references]

    matrix = as_matrix(references, field_name="references")
    chosen = [0]

    while len(chosen) < limit:
        # Similarity of every candidate to its nearest already-chosen reference.
        # The best next pick is whichever has the lowest such value: it adds the
        # most new coverage.
        closeness = np.max(matrix @ matrix[chosen].T, axis=1)
        closeness[chosen] = np.inf
        chosen.append(int(np.argmin(closeness)))

    return [list(matrix[index]) for index in chosen]


def minimum_pairwise_similarity(references: list[list[float]]) -> float:
    """Return the lowest similarity between any two references.

    A diversity measure: the higher this is, the more the set clusters. Used by
    tests to prove greedy selection beats recency selection rather than assuming
    it.

    Args:
        references: The embeddings to measure.

    Returns:
        The minimum pairwise cosine similarity, or ``1.0`` for a set of fewer
        than two.
    """
    if len(references) < 2:
        return 1.0

    return min(
        cosine_similarity(references[i], references[j])
        for i in range(len(references))
        for j in range(i + 1, len(references))
    )
