"""Unit tests for :mod:`multicam_tracker.models.base`.

Covers the UTC datetime type, the shared embedding validator, and the base model
configuration. Everything else in the domain inherits these behaviours, so a gap
here is a gap in every model at once.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from multicam_tracker.models.base import (
    EMBEDDING_NORM_TOLERANCE,
    MCTBaseModel,
    UtcDatetime,
    default_embedding_dim,
    validate_embedding,
)
from tests.fixtures.factories import unit_vector

pytestmark = pytest.mark.unit

INSTANT = datetime(2026, 8, 10, 14, 22, 11, 500000, tzinfo=UTC)


class Sample(MCTBaseModel):
    """Throwaway model exercising the base configuration."""

    moment: UtcDatetime
    label: str
    count: int = 0


# ---------------------------------------------------------------------------
# UtcDatetime
# ---------------------------------------------------------------------------


def test_utc_datetime__aware_utc_value__is_accepted_unchanged() -> None:
    """A value already in UTC passes through with the same instant."""
    assert Sample(moment=INSTANT, label="x").moment == INSTANT


def test_utc_datetime__aware_non_utc_value__is_converted_preserving_the_instant() -> None:
    """Conversion moves the offset, not the moment in time."""
    tashkent = timezone(timedelta(hours=5))
    local = datetime(2026, 8, 10, 19, 22, 11, 500000, tzinfo=tashkent)

    result = Sample(moment=local, label="x").moment

    assert result == INSTANT
    assert result.tzinfo is UTC


def test_utc_datetime__naive_value__is_rejected_naming_the_field() -> None:
    """The message must identify which field was naive, not just that one was."""
    with pytest.raises(ValidationError) as excinfo:
        Sample(moment=datetime(2026, 8, 10, 14, 22, 11), label="x")

    message = str(excinfo.value)
    assert "moment" in message
    assert "naive datetime" in message


def test_utc_datetime__iso_string_with_offset__is_parsed_and_normalized() -> None:
    """JSON payloads arrive as strings; they normalize the same way."""
    result = Sample(moment="2026-08-10T19:22:11.500+05:00", label="x").moment

    assert result == INSTANT
    assert result.tzinfo is UTC


def test_utc_datetime__iso_string_with_z_suffix__is_parsed() -> None:
    """The serialized form round-trips back through the same type."""
    assert Sample(moment="2026-08-10T14:22:11.500Z", label="x").moment == INSTANT


def test_utc_datetime__naive_iso_string__is_rejected() -> None:
    """A string without an offset is as ambiguous as a naive datetime object."""
    with pytest.raises(ValidationError, match="naive datetime"):
        Sample(moment="2026-08-10T14:22:11.500", label="x")


def test_utc_datetime__sub_millisecond_precision__is_truncated() -> None:
    """Truncation on the way in is what makes serialization lossless."""
    precise = datetime(2026, 8, 10, 14, 22, 11, 500987, tzinfo=UTC)

    assert Sample(moment=precise, label="x").moment.microsecond == 500000


def test_utc_datetime__truncation__rounds_down_never_up() -> None:
    """Boundary: 999 microseconds is still millisecond zero, not millisecond one."""
    edge = datetime(2026, 8, 10, 14, 22, 11, 999, tzinfo=UTC)

    assert Sample(moment=edge, label="x").moment.microsecond == 0


def test_utc_datetime__json_serialization__uses_z_suffix_and_milliseconds() -> None:
    """The wire format is ISO-8601 with a Z suffix and exactly three decimals."""
    assert Sample(moment=INSTANT, label="x").to_json_dict()["moment"] == "2026-08-10T14:22:11.500Z"


def test_utc_datetime__python_dump__keeps_the_datetime_object() -> None:
    """Only the JSON boundary stringifies; in-process code keeps real datetimes."""
    assert Sample(moment=INSTANT, label="x").model_dump()["moment"] == INSTANT


# ---------------------------------------------------------------------------
# MCTBaseModel configuration
# ---------------------------------------------------------------------------


def test_base_model__unknown_field__is_rejected() -> None:
    """extra='forbid' turns a typo'd key into an error instead of lost data."""
    with pytest.raises(ValidationError) as excinfo:
        Sample(moment=INSTANT, label="x", labell="typo")

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())


def test_base_model__whitespace_around_strings__is_stripped() -> None:
    """OCR and JSON payloads routinely carry incidental whitespace."""
    assert Sample(moment=INSTANT, label="  ABC1234  ").label == "ABC1234"


def test_base_model__assignment__is_revalidated() -> None:
    """Invariants hold for the object's whole life, not only at construction."""
    sample = Sample(moment=INSTANT, label="x")

    with pytest.raises(ValidationError):
        sample.count = "not an int"


def test_base_model__assignment_of_a_naive_datetime__is_rejected() -> None:
    """The UTC guarantee survives mutation."""
    sample = Sample(moment=INSTANT, label="x")

    with pytest.raises(ValidationError, match="naive datetime"):
        sample.moment = datetime(2026, 8, 10, 14, 22, 11)


def test_base_model__round_trip__produces_an_equal_model() -> None:
    """to_json_dict and from_json_dict are inverses."""
    sample = Sample(moment=INSTANT, label="x", count=3)

    assert Sample.from_json_dict(sample.to_json_dict()) == sample


# ---------------------------------------------------------------------------
# validate_embedding
# ---------------------------------------------------------------------------


def test_default_embedding_dim__reads_the_configured_value() -> None:
    """The dimension follows settings so it can track whichever model stage 13 picks."""
    assert default_embedding_dim() == 512


def test_validate_embedding__normalized_vector__is_accepted() -> None:
    """A unit vector of the configured dimension passes."""
    vector = unit_vector(seed=1)

    assert validate_embedding(vector, field_name="embedding") == pytest.approx(vector)


def test_validate_embedding__wrong_dimension__names_expected_and_actual() -> None:
    """The message must be actionable without opening a debugger."""
    with pytest.raises(ValueError) as excinfo:
        validate_embedding([0.0] * 128, field_name="embedding", expected_dim=512)

    message = str(excinfo.value)
    assert "128" in message
    assert "512" in message


def test_validate_embedding__unnormalized_vector__is_rejected_not_renormalized() -> None:
    """Silent renormalization would hide the real defect in the extractor."""
    doubled = [component * 2 for component in unit_vector(seed=2)]

    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_embedding(doubled, field_name="embedding")


def test_validate_embedding__zero_vector__is_rejected() -> None:
    """Boundary: a zero vector has norm 0 and no meaningful cosine similarity."""
    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_embedding([0.0] * 512, field_name="embedding")


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["above", "below"])
def test_validate_embedding__exactly_at_the_tolerance_boundary__is_accepted(
    direction: float,
) -> None:
    """The tolerance is inclusive on both sides."""
    scale = 1.0 + direction * EMBEDDING_NORM_TOLERANCE
    scaled = [component * scale for component in unit_vector(seed=3)]

    assert validate_embedding(scaled, field_name="embedding") == pytest.approx(scaled)


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["above", "below"])
def test_validate_embedding__just_outside_the_tolerance__is_rejected(direction: float) -> None:
    """One notch past the boundary fails, so the boundary is real."""
    scale = 1.0 + direction * EMBEDDING_NORM_TOLERANCE * 2
    scaled = [component * scale for component in unit_vector(seed=4)]

    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_embedding(scaled, field_name="embedding")


def test_validate_embedding__empty_vector__is_rejected() -> None:
    """Boundary: an empty vector fails on dimension before norm is considered."""
    with pytest.raises(ValueError, match="dimension 0"):
        validate_embedding([], field_name="embedding")


def test_unit_vector__factory__produces_a_normalized_vector() -> None:
    """The test factory itself must satisfy the invariant it feeds into models."""
    vector = unit_vector(seed=7, dim=64)

    assert math.isclose(math.sqrt(sum(c * c for c in vector)), 1.0, abs_tol=1e-12)
