"""Base model configuration, the UTC datetime type, and shared validators.

Every domain model inherits :class:`MCTBaseModel`, which is strict on purpose:
unknown fields are rejected rather than dropped, assignment is re-validated, and
timestamps must be timezone-aware.

``extra="forbid"`` is the load-bearing choice. A typo'd field name in a fixture,
an API payload, or a database row would otherwise be silently discarded, and the
resulting model would look valid while missing data. Forbidding extras turns
that into a construction-time error naming the offending key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, ValidationInfo

__all__ = [
    "EMBEDDING_NORM_TOLERANCE",
    "MCTBaseModel",
    "UtcDatetime",
    "default_embedding_dim",
    "validate_embedding",
]

EMBEDDING_NORM_TOLERANCE = 1e-3
"""Permitted deviation of an embedding's L2 norm from 1.0."""

_FLOAT_SLACK = 1e-9
"""Absorbs floating-point noise so a vector built to sit exactly on the tolerance
boundary is accepted rather than rejected by a last-bit rounding error."""

_MICROSECONDS_PER_MILLISECOND = 1000


def _to_utc_millisecond(value: datetime, info: ValidationInfo) -> datetime:
    """Reject naive datetimes and normalize aware ones to UTC millisecond precision.

    Truncation to milliseconds is what makes serialization lossless: the wire
    format carries millisecond precision (global contract, ``Sighting``), so a
    value carrying microseconds would not survive a round trip. Truncating on
    the way in means the model and its serialized form always agree.

    Args:
        value: The parsed datetime.
        info: Validation context, used to name the field in error messages.

    Returns:
        An aware UTC datetime truncated to millisecond precision.

    Raises:
        ValueError: If ``value`` carries no timezone information.
    """
    field_name = info.field_name or "datetime"
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = (
            f"field '{field_name}' received a naive datetime "
            f"({value.isoformat()}); a timezone-aware UTC datetime is required"
        )
        raise ValueError(msg)
    as_utc = value.astimezone(UTC)
    whole_milliseconds = as_utc.microsecond // _MICROSECONDS_PER_MILLISECOND
    return as_utc.replace(microsecond=whole_milliseconds * _MICROSECONDS_PER_MILLISECOND)


def _serialize_utc(value: datetime) -> str:
    """Render an aware datetime as ISO-8601 with a ``Z`` suffix and milliseconds.

    Args:
        value: The datetime to render.

    Returns:
        A string of the form ``2026-08-10T14:22:11.500Z``.
    """
    as_utc = value.astimezone(UTC)
    milliseconds = as_utc.microsecond // _MICROSECONDS_PER_MILLISECOND
    return f"{as_utc.strftime('%Y-%m-%dT%H:%M:%S')}.{milliseconds:03d}Z"


UtcDatetime = Annotated[
    datetime,
    AfterValidator(_to_utc_millisecond),
    PlainSerializer(_serialize_utc, return_type=str, when_used="json"),
]
"""Timezone-aware UTC datetime at millisecond precision.

Accepts any aware datetime or ISO-8601 string with an offset and normalizes it
to UTC. Rejects naive values outright: a naive datetime could mean local time,
camera time, or UTC, and guessing wrong corrupts every elapsed-time calculation
downstream.
"""


def default_embedding_dim() -> int:
    """Return the configured re-id embedding dimension.

    Read from settings rather than hardcoded so the dimension follows whichever
    model stage 13 ends up using.

    Returns:
        The expected number of components in an embedding vector.
    """
    from multicam_tracker.config import get_settings

    return get_settings().vision.embedding_dim


def validate_embedding(
    vector: Sequence[float],
    *,
    field_name: str,
    expected_dim: int | None = None,
) -> list[float]:
    """Validate an embedding's dimension and L2 norm.

    A non-normalized vector is **rejected, never silently renormalized**.
    Renormalizing would paper over the real defect -- an extractor emitting raw
    logits, or a vector concatenated from two models -- and every cosine
    similarity computed against it afterwards would be quietly wrong.

    Args:
        vector: The embedding components.
        field_name: Field name reported in error messages.
        expected_dim: Required length. Defaults to the configured dimension.

    Returns:
        The embedding as a list of floats.

    Raises:
        ValueError: If the dimension is wrong, or the L2 norm deviates from 1.0
            by more than :data:`EMBEDDING_NORM_TOLERANCE`.
    """
    dimension = expected_dim if expected_dim is not None else default_embedding_dim()

    if len(vector) != dimension:
        msg = f"field '{field_name}' has embedding dimension {len(vector)}; expected {dimension}"
        raise ValueError(msg)

    norm = math.sqrt(math.fsum(float(component) * float(component) for component in vector))
    if abs(norm - 1.0) > EMBEDDING_NORM_TOLERANCE + _FLOAT_SLACK:
        msg = (
            f"field '{field_name}' embedding is not L2-normalized: norm={norm:.6f}, "
            f"expected 1.0 +/- {EMBEDDING_NORM_TOLERANCE}. It is rejected rather "
            f"than renormalized so the defect is visible at its source"
        )
        raise ValueError(msg)

    return [float(component) for component in vector]


class MCTBaseModel(BaseModel):
    """Base class for every domain model.

    Configuration rationale:

    ``extra="forbid"``
        A misspelled key is an error, not silently dropped data.
    ``validate_assignment=True``
        Invariants hold for the object's whole life, not only at construction.
    ``str_strip_whitespace=True``
        Plate text and identifiers arriving from OCR or JSON often carry
        incidental whitespace.
    ``use_enum_values=False``
        Fields keep their enum type in Python and serialize to strings only at
        the JSON boundary, so comparisons stay type-safe.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Datetimes become ISO-8601 strings with a ``Z`` suffix and millisecond
        precision, enums become their string values, and embeddings become plain
        float lists.

        Returns:
            A dictionary containing only JSON-native types.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> Self:
        """Reconstruct a model from the output of :meth:`to_json_dict`.

        Args:
            data: A mapping produced by :meth:`to_json_dict`, or any equivalent
                JSON-decoded payload.

        Returns:
            The validated model.

        Raises:
            pydantic.ValidationError: If the payload violates any model
                invariant.
        """
        return cls.model_validate(dict(data))
