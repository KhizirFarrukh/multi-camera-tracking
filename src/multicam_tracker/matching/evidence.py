"""Combining plate and appearance evidence into one verdict.

The governing rule: **an embedding-only match can never outrank a plate match.**
Re-id is weaker evidence and the system must not let it masquerade as stronger.
Many vehicles share a make, model, and colour; almost none share a plate.

That is enforced structurally rather than by tuning. A visual-only score is
capped at ``embedding_only_score_ceiling``, which sits below the lowest score any
valid plate match can produce, so no combination of inputs can invert the
ordering. A test asserts the property across randomized inputs.

Four cases:

============================  ===========================================
plate strong, embedding strong  plate method, confidence boosted
plate absent, embedding strong  embedding method, score capped
plate strong, embedding weak    **conflicting** -> forced to human review
plate weak, embedding strong    **conflicting** -> forced to human review
============================  ===========================================

Disagreement is the interesting case. Two independent signals pointing different
ways is *more* alarming than one weak signal, because one of them is wrong and
nothing in the data says which. Averaging them would produce a middling score
that hides the contradiction; forcing review surfaces it.
"""

from __future__ import annotations

from dataclasses import dataclass

from multicam_tracker.matching.plate_match import PlateMatchMethod, PlateMatchResult
from multicam_tracker.matching.scoring import score_plate_match
from multicam_tracker.models import MatchCandidate, MatchMethod, ReviewStatus

__all__ = ["CombinedEvidence", "combine_evidence"]


@dataclass(frozen=True)
class CombinedEvidence:
    """One verdict drawing on both identification paths."""

    method: MatchMethod | None
    """``None`` when neither path produced any evidence at all."""

    score: float
    review_status: ReviewStatus
    conflicting: bool = False
    disagreement: str = ""
    plate_score: float = 0.0
    embedding_similarity: float | None = None

    @property
    def is_match(self) -> bool:
        """Return whether any evidence supported a match."""
        return self.method is not None

    def to_candidate(
        self, *, sighting_id: str, target_id: str, edit_distance: int | None
    ) -> MatchCandidate:
        """Convert to the persisted match-candidate model.

        Args:
            sighting_id: The matched sighting.
            target_id: The target it was matched to.
            edit_distance: Plate edit distance, or ``None`` for a visual-only
                match.

        Returns:
            The domain model, ready to store.

        Raises:
            ValueError: If there was no evidence to record.
        """
        if self.method is None:
            msg = "cannot build a MatchCandidate from evidence that supports no match"
            raise ValueError(msg)

        # The model requires plate_exact to carry distance 0 and plate_fuzzy to
        # carry at least 1, so a visual-only match records no distance at all.
        distance = edit_distance if self.method is not MatchMethod.EMBEDDING else None

        return MatchCandidate(
            sighting_id=sighting_id,
            target_id=target_id,
            match_method=self.method,
            match_score=self.score,
            plate_edit_distance=distance,
            embedding_similarity=self.embedding_similarity,
            review_status=self.review_status,
        )


_NO_EVIDENCE = CombinedEvidence(
    method=None, score=0.0, review_status=ReviewStatus.REJECTED, conflicting=False
)
"""Returned when neither path found anything.

A rejected zero-score verdict rather than an exception: the caller is scanning
many sightings, most of which match nothing, and that is the ordinary case
rather than an error.
"""


def combine_evidence(
    plate_result: PlateMatchResult | None,
    ocr_confidence: float | None,
    embedding_similarity: float | None,
    *,
    auto_accept_min: float | None = None,
    embedding_auto_accept: float | None = None,
    score_ceiling: float | None = None,
    agreement_boost: float | None = None,
    disagreement_similarity: float | None = None,
) -> CombinedEvidence:
    """Combine plate and appearance evidence into a single verdict.

    Args:
        plate_result: Classification from the plate path, or ``None``.
        ocr_confidence: What the OCR reported, or ``None`` when unreadable.
        embedding_similarity: Aggregated appearance similarity, or ``None`` when
            no embedding was available.
        auto_accept_min: Plate score at or above which no human is needed.
            Defaults to config.
        embedding_auto_accept: Similarity at or above which appearance evidence
            counts as strong. Defaults to config.
        score_ceiling: Cap on a visual-only score. Defaults to config.
        agreement_boost: Added when both paths agree. Defaults to config.
        disagreement_similarity: Below this, appearance actively contradicts a
            strong plate match. Defaults to config.

    Returns:
        The combined verdict. Never raises for missing evidence -- a sighting
        that matches nothing is the ordinary case.
    """
    from multicam_tracker.config import get_settings

    thresholds = get_settings().thresholds
    plate_accept = (
        auto_accept_min
        if auto_accept_min is not None
        else thresholds.plate_auto_accept_min_confidence
    )
    embedding_accept = (
        embedding_auto_accept
        if embedding_auto_accept is not None
        else thresholds.embedding_auto_accept_min_similarity
    )
    ceiling = (
        score_ceiling if score_ceiling is not None else thresholds.embedding_only_score_ceiling
    )
    boost = agreement_boost if agreement_boost is not None else thresholds.embedding_agreement_boost
    contradicts_below = (
        disagreement_similarity
        if disagreement_similarity is not None
        else thresholds.embedding_disagreement_similarity
    )

    has_plate = plate_result is not None and plate_result.method is not PlateMatchMethod.NO_MATCH
    plate_score = (
        score_plate_match(plate_result, ocr_confidence) if has_plate and plate_result else 0.0
    )
    plate_is_strong = plate_score >= plate_accept

    has_embedding = embedding_similarity is not None
    embedding_is_strong = has_embedding and embedding_similarity >= embedding_accept  # type: ignore[operator]

    if not has_plate and not has_embedding:
        return _NO_EVIDENCE

    # --- disagreement ------------------------------------------------------
    if has_plate and has_embedding and plate_is_strong and embedding_similarity < contradicts_below:  # type: ignore[operator]
        return CombinedEvidence(
            method=_plate_method(plate_result),
            score=min(plate_score, plate_accept - 1e-9),
            review_status=ReviewStatus.PENDING_REVIEW,
            conflicting=True,
            disagreement=(
                f"plate evidence is strong ({plate_score:.2f}) but appearance "
                f"similarity is only {embedding_similarity:.2f}; one of the two "
                f"signals is wrong and the data does not say which"
            ),
            plate_score=plate_score,
            embedding_similarity=embedding_similarity,
        )

    if (
        has_plate
        and has_embedding
        and embedding_is_strong
        and not plate_is_strong
        and plate_score > 0.0
    ):
        return CombinedEvidence(
            method=_plate_method(plate_result),
            score=min(max(plate_score, ceiling), plate_accept - 1e-9),
            review_status=ReviewStatus.PENDING_REVIEW,
            conflicting=True,
            disagreement=(
                f"appearance similarity is high ({embedding_similarity:.2f}) but "
                f"the plate evidence is weak ({plate_score:.2f}); the two signals "
                f"do not corroborate each other"
            ),
            plate_score=plate_score,
            embedding_similarity=embedding_similarity,
        )

    # --- plate present, appearance absent or corroborating -----------------
    if has_plate:
        score = plate_score
        if embedding_is_strong:
            score = min(1.0, plate_score + boost)
        return CombinedEvidence(
            method=_plate_method(plate_result),
            score=score,
            review_status=(
                ReviewStatus.AUTO_ACCEPTED if score >= plate_accept else ReviewStatus.PENDING_REVIEW
            ),
            plate_score=plate_score,
            embedding_similarity=embedding_similarity,
        )

    # --- appearance only ---------------------------------------------------
    # Capped so a visual-only match cannot outrank any plate match, and never
    # auto-accepted: appearance alone is corroborating evidence, not proof.
    return CombinedEvidence(
        method=MatchMethod.EMBEDDING,
        score=min(ceiling, max(0.0, embedding_similarity or 0.0)),
        review_status=ReviewStatus.PENDING_REVIEW,
        plate_score=0.0,
        embedding_similarity=embedding_similarity,
    )


def _plate_method(result: PlateMatchResult | None) -> MatchMethod:
    """Map a plate classification to the persisted match method.

    Args:
        result: The plate classification.

    Returns:
        ``PLATE_EXACT`` or ``PLATE_FUZZY``.
    """
    if result is not None and result.method is PlateMatchMethod.PLATE_EXACT:
        return MatchMethod.PLATE_EXACT
    return MatchMethod.PLATE_FUZZY
