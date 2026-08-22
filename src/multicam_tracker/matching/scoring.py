"""Turning a classification and an OCR confidence into one score.

The score is what thresholds are applied to, so its shape decides which matches
are auto-accepted, which go to a human, and which are dropped.

Two monotonicity properties are asserted by tests because a violation would be
invisible in aggregate metrics and wrong in individual cases:

* raising OCR confidence never lowers the score;
* raising edit distance never raises it.

The OCR confidence enters **multiplicatively**. A perfect string match on a read
the OCR itself barely believes is not strong evidence, and an additive
combination would let a confident method weight paper over a hopeless read.
Multiplying also gives the right boundary behaviour: confidence zero scores
zero, whatever the method.
"""

from __future__ import annotations

from multicam_tracker.matching.plate_match import PlateMatchMethod, PlateMatchResult

__all__ = ["score_plate_match"]


def score_plate_match(
    result: PlateMatchResult,
    ocr_confidence: float | None,
    *,
    exact_weight: float | None = None,
    fuzzy_weight: float | None = None,
    distance_penalty_per_unit: float | None = None,
) -> float:
    """Score a plate match in ``[0, 1]``.

    ::

        score = method_weight
              * max(0, 1 - penalty_per_unit * weighted_distance)
              * ocr_confidence

    Args:
        result: The classification.
        ocr_confidence: What the OCR reported for this read. ``None`` means
            there was no readable plate, which scores zero -- there is nothing
            to have matched.
        exact_weight: Base weight for an exact match. Defaults to config.
        fuzzy_weight: Base weight for a fuzzy match. Defaults to config.
        distance_penalty_per_unit: Score lost per unit of weighted distance.
            Defaults to config.

    Returns:
        A score in ``[0, 1]``. Zero for a non-match, for a missing confidence,
        and for a confidence of zero.
    """
    from multicam_tracker.config import get_settings

    if result.method is PlateMatchMethod.NO_MATCH or ocr_confidence is None:
        return 0.0

    thresholds = get_settings().thresholds
    resolved_exact = (
        exact_weight if exact_weight is not None else thresholds.plate_exact_method_weight
    )
    resolved_fuzzy = (
        fuzzy_weight if fuzzy_weight is not None else thresholds.plate_fuzzy_method_weight
    )
    resolved_penalty = (
        distance_penalty_per_unit
        if distance_penalty_per_unit is not None
        else thresholds.plate_distance_penalty_per_unit
    )

    method_weight = (
        resolved_exact if result.method is PlateMatchMethod.PLATE_EXACT else resolved_fuzzy
    )

    # An infinite weighted distance only reaches here through a hand-built
    # result; clamping keeps the arithmetic finite rather than producing a nan.
    distance = min(result.weighted_distance, 1e6) if result.weighted_distance >= 0 else 0.0
    distance_factor = max(0.0, 1.0 - resolved_penalty * distance)

    confidence = max(0.0, min(1.0, ocr_confidence))
    return max(0.0, min(1.0, method_weight * distance_factor * confidence))
