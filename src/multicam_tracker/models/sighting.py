"""The sighting model: one detection of one object by one camera at one instant.

This is the atomic record of the entire system. Everything downstream --
matching, path reconstruction, the timeline, the audit trail -- is a function of
sightings, so the invariants enforced here are the ones every later stage gets
to assume.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from multicam_tracker.models.base import MCTBaseModel, UtcDatetime, validate_embedding
from multicam_tracker.models.enums import ObjectClass

__all__ = ["BBOX_LENGTH", "TIMESTAMP_CONSISTENCY_TOLERANCE_MS", "Sighting"]

BBOX_LENGTH = 4
"""A bounding box is exactly ``[x1, y1, x2, y2]``."""

TIMESTAMP_CONSISTENCY_TOLERANCE_MS = 1.0
"""Permitted disagreement between ``timestamp_utc`` and the corrected raw timestamp."""


class Sighting(MCTBaseModel):
    """A single detection of an object by one camera at one timestamp."""

    sighting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = Field(min_length=1, description="References Camera.camera_id")

    timestamp_utc: UtcDatetime = Field(
        description="Corrected timestamp, after the camera's clock offset is applied"
    )
    raw_timestamp: UtcDatetime = Field(description="Timestamp as reported by the source")
    clock_offset_applied_ms: int = Field(
        default=0,
        description="Offset added to raw_timestamp to produce timestamp_utc",
    )

    object_class: ObjectClass = ObjectClass.UNKNOWN
    detection_confidence: float = Field(ge=0.0, le=1.0)
    bbox: Annotated[list[int], Field(min_length=BBOX_LENGTH, max_length=BBOX_LENGTH)] = Field(
        description="[x1, y1, x2, y2] in source-frame pixel coordinates"
    )
    frame_index: int = Field(ge=0, description="Index of the frame within its source")

    plate_text_raw: str | None = Field(default=None, description="OCR output before normalization")
    plate_text_normalized: str | None = Field(
        default=None, description="Uppercase, alphanumeric only (see plate_normalization)"
    )
    plate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    embedding: list[float] | None = Field(
        default=None, description="L2-normalized re-id vector of the configured dimension"
    )
    embedding_model_version: str | None = None

    thumbnail_path: str | None = Field(
        default=None, description="Relative path to the cropped object image"
    )
    source_id: str = Field(
        min_length=1, description="Identifies the video file or stream that produced this sighting"
    )
    created_at: UtcDatetime = Field(
        description=(
            "Row insertion time. Required rather than auto-generated: the project forbids "
            "calling datetime.now() outside an injected Clock, and a default_factory here "
            "would be exactly that call, hidden inside the model."
        )
    )

    @field_validator("bbox")
    @classmethod
    def _bbox_is_a_positive_area_box(cls, value: list[int]) -> list[int]:
        """Reject boxes with negative coordinates or non-positive area.

        Args:
            value: The four bbox coordinates.

        Returns:
            The validated coordinates.

        Raises:
            ValueError: If any coordinate is negative, or if the box is inverted
                or degenerate. A zero-area box crops to an empty image, which
                would fail far downstream in the thumbnail writer instead of
                here where the cause is obvious.
        """
        x1, y1, x2, y2 = value

        if any(coordinate < 0 for coordinate in value):
            msg = f"bbox coordinates must be non-negative; got {value}"
            raise ValueError(msg)

        if x2 <= x1:
            msg = f"bbox requires x2 > x1; got x1={x1}, x2={x2}"
            raise ValueError(msg)
        if y2 <= y1:
            msg = f"bbox requires y2 > y1; got y1={y1}, y2={y2}"
            raise ValueError(msg)

        return value

    @field_validator("embedding")
    @classmethod
    def _embedding_is_normalized(cls, value: list[float] | None) -> list[float] | None:
        """Validate the embedding's dimension and L2 norm when one is present.

        Args:
            value: The embedding, or ``None``.

        Returns:
            The validated embedding, or ``None``.

        Raises:
            ValueError: If the dimension is wrong or the vector is not
                L2-normalized. It is never silently renormalized.
        """
        if value is None:
            return None
        return validate_embedding(value, field_name="embedding")

    @model_validator(mode="after")
    def _timestamp_matches_the_corrected_raw_value(self) -> Sighting:
        """Verify ``timestamp_utc == raw_timestamp + clock_offset_applied_ms``.

        The three fields are redundant by design: storing the raw value, the
        offset, and the result means a drift correction can be audited and
        recomputed later (stage 09). Redundancy is only useful while it stays
        consistent, which is what this check enforces.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the three fields disagree by more than
                :data:`TIMESTAMP_CONSISTENCY_TOLERANCE_MS`.
        """
        expected = self.raw_timestamp + timedelta(milliseconds=self.clock_offset_applied_ms)
        drift_ms = abs((self.timestamp_utc - expected).total_seconds()) * 1000

        if drift_ms > TIMESTAMP_CONSISTENCY_TOLERANCE_MS:
            msg = (
                f"timestamp_utc ({self.timestamp_utc.isoformat()}) does not equal "
                f"raw_timestamp ({self.raw_timestamp.isoformat()}) plus "
                f"clock_offset_applied_ms ({self.clock_offset_applied_ms}); "
                f"expected {expected.isoformat()}, differs by {drift_ms:.3f} ms"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _normalized_plate_carries_a_confidence(self) -> Sighting:
        """Require a confidence wherever a normalized plate is recorded.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``plate_text_normalized`` is set without
                ``plate_confidence``. Matching weights every plate read by its
                confidence; a plate with no confidence has no defined weight and
                would have to be either dropped or silently trusted.
        """
        if self.plate_text_normalized is not None and self.plate_confidence is None:
            msg = (
                f"plate_confidence is required when plate_text_normalized is set "
                f"(got plate_text_normalized={self.plate_text_normalized!r})"
            )
            raise ValueError(msg)
        return self

    @property
    def has_plate(self) -> bool:
        """Return whether this sighting carries a normalized plate reading."""
        return self.plate_text_normalized is not None

    @property
    def has_embedding(self) -> bool:
        """Return whether this sighting carries a re-id embedding."""
        return self.embedding is not None
