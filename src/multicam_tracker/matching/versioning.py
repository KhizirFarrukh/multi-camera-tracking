"""Embedding model-version safety.

Two embeddings are only comparable if the same model produced them. Vectors from
different models occupy unrelated spaces, so their cosine similarity is a number
with no meaning -- and, fatally, it is a *plausible* number. It will not be NaN
or out of range; it will be 0.31, and a ranking built on it will look entirely
reasonable while being noise.

So cross-version comparison is prevented by construction rather than warned
about. Every similarity path checks first, and the repository query filters by
version so mismatched candidates never reach the comparison at all.

**Re-embedding migration path.** When the model is upgraded:

1. Deploy the new model writing a new ``embedding_model_version``.
2. Backfill historical sightings by re-running extraction over stored
   thumbnails, writing the new version alongside the old rows.
3. Switch searches to the new version once backfill covers the retention window.
4. Drop the old vectors after the retention TTL expires them.

Steps 1 and 2 overlap deliberately: during backfill the two versions coexist,
and this module is what stops them being compared.
"""

from __future__ import annotations

from collections.abc import Iterable

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.models import Sighting

__all__ = [
    "distinct_model_versions",
    "require_same_model_version",
    "sightings_for_version",
]


def require_same_model_version(left: str | None, right: str | None) -> str | None:
    """Verify two embeddings came from the same model.

    Args:
        left: First embedding's model version.
        right: Second embedding's model version.

    Returns:
        The shared version, or ``None`` when neither declares one.

    Raises:
        MatchingError: If the two differ, naming both. Also raised when only one
            side declares a version: an unlabelled vector cannot be assumed to
            match, and assuming it would defeat the whole guard.
    """
    if left == right:
        return left

    raise MatchingError(
        "Embeddings from different model versions are not comparable; their "
        "similarity would be a plausible-looking number with no meaning",
        {"left_model_version": left, "right_model_version": right},
    )


def distinct_model_versions(sightings: Iterable[Sighting]) -> set[str | None]:
    """Return every embedding model version present among some sightings.

    Args:
        sightings: The sightings to inspect. Those without an embedding are
            ignored, since they carry no vector to be incomparable.

    Returns:
        The distinct versions, which may include ``None``.
    """
    return {
        sighting.embedding_model_version for sighting in sightings if sighting.embedding is not None
    }


def sightings_for_version(
    sightings: Iterable[Sighting], model_version: str | None
) -> list[Sighting]:
    """Filter sightings to those embedded by one model version.

    Args:
        sightings: The sightings to filter.
        model_version: The version to keep.

    Returns:
        Only sightings carrying an embedding from that version.
    """
    return [
        sighting
        for sighting in sightings
        if sighting.embedding is not None and sighting.embedding_model_version == model_version
    ]
