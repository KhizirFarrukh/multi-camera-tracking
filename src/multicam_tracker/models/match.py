"""The match candidate model: an assertion that a sighting is the target.

The validators here exist to stop internally inconsistent match records from
ever reaching storage. A record claiming ``plate_exact`` while carrying an edit
distance of 2 is not a slightly-wrong match -- it is evidence that whichever
code produced it has a bug, and letting it persist would corrupt every accuracy
measurement taken afterwards.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from multicam_tracker.models.base import MCTBaseModel
from multicam_tracker.models.enums import MatchMethod, ReviewStatus

__all__ = ["MatchCandidate"]


class MatchCandidate(MCTBaseModel):
    """A scored link between one sighting and one target."""

    sighting_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    match_method: MatchMethod
    match_score: float = Field(
        ge=0.0, le=1.0, description="Method-specific raw similarity, normalized to [0, 1]"
    )
    plate_edit_distance: int | None = Field(
        default=None,
        ge=0,
        description="Levenshtein distance on the ambiguity-folded plate forms",
    )
    embedding_similarity: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Cosine similarity"
    )
    review_status: ReviewStatus

    @model_validator(mode="after")
    def _evidence_matches_the_declared_method(self) -> MatchCandidate:
        """Require the evidence fields the declared method implies.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the record's evidence contradicts its
                ``match_method``, or if a manual decision is left unresolved.
        """
        if self.match_method is MatchMethod.PLATE_EXACT and self.plate_edit_distance != 0:
            msg = (
                f"match_method 'plate_exact' requires plate_edit_distance == 0; "
                f"got {self.plate_edit_distance}. A non-zero distance means the "
                f"match is 'plate_fuzzy'"
            )
            raise ValueError(msg)

        if self.match_method is MatchMethod.PLATE_FUZZY and (
            self.plate_edit_distance is None or self.plate_edit_distance < 1
        ):
            msg = (
                f"match_method 'plate_fuzzy' requires plate_edit_distance >= 1; "
                f"got {self.plate_edit_distance}. A distance of 0 on the folded "
                f"forms is still a fuzzy match only if the normalized forms "
                f"differ, in which case the distance is >= 1"
            )
            raise ValueError(msg)

        if self.match_method is MatchMethod.EMBEDDING and self.embedding_similarity is None:
            msg = (
                "match_method 'embedding' requires embedding_similarity; without it "
                "the match carries no evidence at all"
            )
            raise ValueError(msg)

        if self.match_method is MatchMethod.MANUAL and self.review_status not in {
            ReviewStatus.CONFIRMED,
            ReviewStatus.REJECTED,
        }:
            msg = (
                f"match_method 'manual' records a human decision, so review_status "
                f"must be 'confirmed' or 'rejected'; got '{self.review_status.value}'"
            )
            raise ValueError(msg)

        return self
