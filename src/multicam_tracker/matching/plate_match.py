"""Classifying one candidate plate reading against a target plate.

Precision matters more than recall here. A false positive puts an innocent
vehicle on a stolen-car trajectory, and that error is expensive in a way a miss
is not: a missed sighting leaves a gap an operator can see, while a wrong one
looks exactly like a real hit. So the classifier is deliberately conservative and
the uncertain middle is routed to human review rather than auto-accepted.

Classification order matters:

1. **Length gate first.** A four-character read and a seven-character plate are
   not the same plate however the edit distance works out, and checking it first
   also short-circuits the expensive comparison across a large candidate set.
2. **Exact.** Identical normalized strings.
3. **Folded-equal.** Same folded form, different normalized form: the classic
   OCR confusion, reported as fuzzy with the distance measured on the
   *normalized* strings so the number describes the real difference.
4. **Weighted distance within threshold.** Everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from multicam_tracker.matching.distance import (
    damerau_levenshtein,
    weighted_damerau_levenshtein,
)
from multicam_tracker.matching.folding import fold_ambiguous

__all__ = ["PlateMatchMethod", "PlateMatchResult", "classify_plate_match"]


class PlateMatchMethod(StrEnum):
    """How, if at all, a candidate plate matched the target."""

    PLATE_EXACT = "plate_exact"
    PLATE_FUZZY = "plate_fuzzy"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class PlateMatchResult:
    """The verdict on one candidate plate."""

    method: PlateMatchMethod
    edit_distance: int
    weighted_distance: float
    folded_equal: bool
    length_delta: int

    @property
    def is_match(self) -> bool:
        """Return whether the candidate matched at all."""
        return self.method is not PlateMatchMethod.NO_MATCH

    def __str__(self) -> str:
        """Return a short human-readable form."""
        return (
            f"{self.method.value} (distance={self.edit_distance}, "
            f"weighted={self.weighted_distance:.2f}, folded_equal={self.folded_equal})"
        )


_NO_MATCH = PlateMatchResult(
    method=PlateMatchMethod.NO_MATCH,
    edit_distance=-1,
    weighted_distance=float("inf"),
    folded_equal=False,
    length_delta=0,
)
"""Returned when there is nothing to compare. The sentinel distances are
deliberately out of range so a caller that ignores ``method`` and reads
``edit_distance`` gets an obviously wrong number rather than a plausible one."""


def classify_plate_match(
    candidate: str | None,
    target: str | None,
    *,
    max_weighted_distance: float | None = None,
    max_length_delta: int | None = None,
    confusion_substitution_cost: float | None = None,
) -> PlateMatchResult:
    """Classify a candidate plate against a target plate.

    Both inputs must already be normalized. This function does not normalize:
    doing so silently would hide a caller that forgot, and comparing a raw read
    against a normalized target is a bug worth surfacing.

    Args:
        candidate: The observed plate, or ``None`` for an unreadable one.
        target: The plate being searched for, or ``None``.
        max_weighted_distance: Fuzzy cutoff. Defaults to the configured value.
        max_length_delta: Length gate. Defaults to the configured value.
        confusion_substitution_cost: In-group substitution cost. Defaults to the
            configured value.

    Returns:
        The classification, including both distances so a caller can explain the
        decision as well as act on it.
    """
    from multicam_tracker.config import get_settings

    if not candidate or not target:
        return _NO_MATCH

    thresholds = get_settings().thresholds
    weighted_cutoff = (
        max_weighted_distance
        if max_weighted_distance is not None
        else thresholds.plate_fuzzy_max_weighted_distance
    )
    length_cutoff = (
        max_length_delta if max_length_delta is not None else thresholds.plate_max_length_delta
    )
    substitution_cost = (
        confusion_substitution_cost
        if confusion_substitution_cost is not None
        else thresholds.plate_confusion_substitution_cost
    )

    length_delta = abs(len(candidate) - len(target))
    if length_delta > length_cutoff:
        return PlateMatchResult(
            method=PlateMatchMethod.NO_MATCH,
            edit_distance=length_delta,
            weighted_distance=float(length_delta),
            folded_equal=False,
            length_delta=length_delta,
        )

    if candidate == target:
        return PlateMatchResult(
            method=PlateMatchMethod.PLATE_EXACT,
            edit_distance=0,
            weighted_distance=0.0,
            folded_equal=True,
            length_delta=0,
        )

    folded_equal = fold_ambiguous(candidate) == fold_ambiguous(target)

    # The distance is measured on the normalized strings even when the folded
    # forms agree, so the reported number describes the difference a human would
    # see rather than the difference after the ambiguity was collapsed away.
    edit = damerau_levenshtein(candidate, target)
    weighted = weighted_damerau_levenshtein(
        candidate, target, confusion_substitution_cost=substitution_cost
    )

    if folded_equal or weighted <= weighted_cutoff:
        return PlateMatchResult(
            method=PlateMatchMethod.PLATE_FUZZY,
            edit_distance=edit,
            weighted_distance=weighted,
            folded_equal=folded_equal,
            length_delta=length_delta,
        )

    return PlateMatchResult(
        method=PlateMatchMethod.NO_MATCH,
        edit_distance=edit,
        weighted_distance=weighted,
        folded_equal=False,
        length_delta=length_delta,
    )
