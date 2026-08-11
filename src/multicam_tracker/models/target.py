"""The target model: the object being searched for."""

from __future__ import annotations

import uuid

from pydantic import Field, field_validator, model_validator

from multicam_tracker.models.base import MCTBaseModel, UtcDatetime, validate_embedding

__all__ = ["Target"]


class Target(MCTBaseModel):
    """An object under search, identified by plate, appearance, or both."""

    target_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(min_length=1, description="Human name, e.g. 'Stolen Blue Sedan'")
    plate_query: str | None = Field(default=None, description="Normalized plate being searched for")
    reference_embeddings: list[list[float]] | None = Field(
        default=None, description="Known-good L2-normalized embeddings for this target"
    )
    created_at: UtcDatetime
    active: bool = True

    @field_validator("reference_embeddings")
    @classmethod
    def _reference_embeddings_are_normalized(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        """Validate every reference embedding's dimension and L2 norm.

        Args:
            value: The reference embeddings, or ``None``.

        Returns:
            The validated embeddings, or ``None``.

        Raises:
            ValueError: If any embedding has the wrong dimension or is not
                L2-normalized. The message names the offending index, since a
                target may carry many references and finding the bad one by hand
                is tedious.
        """
        if value is None:
            return None
        return [
            validate_embedding(embedding, field_name=f"reference_embeddings[{index}]")
            for index, embedding in enumerate(value)
        ]

    @model_validator(mode="after")
    def _target_is_searchable(self) -> Target:
        """Require at least one identifying signal.

        Returns:
            The validated instance.

        Raises:
            ValueError: If neither a plate query nor any reference embedding is
                present. Such a target matches nothing by construction, and
                accepting one would produce an empty trajectory that looks like
                "the vehicle was never seen" rather than "the search was never
                specified".
        """
        has_plate = bool(self.plate_query and self.plate_query.strip())
        has_embeddings = bool(self.reference_embeddings)

        if not has_plate and not has_embeddings:
            msg = (
                "a target needs at least one of plate_query or reference_embeddings; "
                "with neither it is unsearchable and would silently match nothing"
            )
            raise ValueError(msg)
        return self

    @property
    def has_plate_query(self) -> bool:
        """Return whether a non-empty plate query is configured."""
        return bool(self.plate_query and self.plate_query.strip())

    @property
    def has_reference_embeddings(self) -> bool:
        """Return whether at least one reference embedding is configured."""
        return bool(self.reference_embeddings)
