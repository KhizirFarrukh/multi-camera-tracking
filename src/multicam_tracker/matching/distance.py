"""Edit distance for plate comparison.

Damerau-Levenshtein rather than plain Levenshtein because OCR swaps adjacent
characters, and a transposition costing 2 would push a one-error read outside a
distance-2 threshold that should have caught it.

Specifically this is the *optimal string alignment* variant, in which no
substring is edited more than once. It is the standard restricted form and
matches the failure it models: real OCR transposes a pair, it does not shuffle a
plate through a sequence of overlapping swaps.

Two variants exist for one reason. The unweighted distance answers "how many
edits apart are these strings", which is what a human-facing explanation needs.
The **weighted** distance charges less for a substitution inside a confusion
group, because a 0 read as O is far more likely than a 0 read as W -- and a
matcher that treated them identically would either admit the W case or reject
the O case.
"""

from __future__ import annotations

from multicam_tracker.matching.folding import same_confusion_group

__all__ = [
    "ARBITRARY_SUBSTITUTION_COST",
    "damerau_levenshtein",
    "weighted_damerau_levenshtein",
]

ARBITRARY_SUBSTITUTION_COST = 1.0
"""Cost of substituting a character for one it is not confusable with."""


def damerau_levenshtein(left: str, right: str, *, max_distance: int | None = None) -> int:
    """Return the restricted Damerau-Levenshtein distance between two strings.

    Args:
        left: First string.
        right: Second string.
        max_distance: Early-exit cutoff. When the true distance would exceed it,
            the search stops and ``max_distance + 1`` is returned instead of the
            real value -- the caller only asked whether it was within the
            threshold, and computing an exact large distance across a big
            candidate set is wasted work.

    Returns:
        The edit distance, or ``max_distance + 1`` when the cutoff was hit.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if max_distance is not None and abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))

    for i, left_char in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if i > 1 and j > 1 and left_char == right[j - 2] and left[i - 2] == right_char:
                current[j] = min(current[j], previous_previous[j - 2] + 1)

        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1

        previous_previous = previous
        previous = current

    return previous[-1]


def weighted_damerau_levenshtein(
    left: str,
    right: str,
    *,
    confusion_substitution_cost: float = 0.5,
    max_distance: float | None = None,
) -> float:
    """Return the confusion-aware edit distance between two strings.

    Identical to :func:`damerau_levenshtein` except that substituting a
    character for one in the same confusion group costs
    ``confusion_substitution_cost`` instead of 1.0.

    Because every operation costs at most as much as in the unweighted variant,
    the result never exceeds the unweighted distance. A property test asserts
    that relation across arbitrary inputs.

    Args:
        left: First string.
        right: Second string.
        confusion_substitution_cost: Cost of an in-group substitution. Lower
            makes the matcher more tolerant of the confusions OCR actually
            makes, without loosening it for arbitrary ones.
        max_distance: Early-exit cutoff, with the same semantics as the
            unweighted variant: exceeding it returns a value strictly above it
            rather than the true distance.

    Returns:
        The weighted distance, or a value above ``max_distance`` when the cutoff
        was hit.
    """
    if left == right:
        return 0.0
    if not left:
        return float(len(right))
    if not right:
        return float(len(left))

    # Insertions and deletions still cost 1.0, so a length gap alone already
    # exceeds a cutoff smaller than it.
    if max_distance is not None and abs(len(left) - len(right)) > max_distance:
        return max_distance + 1.0

    previous_previous: list[float] = []
    previous = [float(index) for index in range(len(right) + 1)]

    for i, left_char in enumerate(left, start=1):
        current = [float(i)] + [0.0] * len(right)
        for j, right_char in enumerate(right, start=1):
            if left_char == right_char:
                cost = 0.0
            elif same_confusion_group(left_char, right_char):
                cost = confusion_substitution_cost
            else:
                cost = ARBITRARY_SUBSTITUTION_COST

            current[j] = min(
                previous[j] + 1.0,
                current[j - 1] + 1.0,
                previous[j - 1] + cost,
            )
            if i > 1 and j > 1 and left_char == right[j - 2] and left[i - 2] == right_char:
                current[j] = min(current[j], previous_previous[j - 2] + 1.0)

        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1.0

        previous_previous = previous
        previous = current

    return previous[-1]
