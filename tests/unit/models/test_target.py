"""Unit tests for :mod:`multicam_tracker.models.target`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_tracker.models import Target
from tests.fixtures.factories import BASE_INSTANT, make_target, unit_vector

pytestmark = pytest.mark.unit


def test_target__plate_query_only__is_valid() -> None:
    """The primary case: a stolen vehicle identified by its plate."""
    target = make_target(plate_query="ABC1234", reference_embeddings=None)

    assert target.has_plate_query is True
    assert target.has_reference_embeddings is False


def test_target__reference_embeddings_only__is_valid() -> None:
    """The fallback case: appearance is known but the plate is not."""
    target = make_target(plate_query=None, reference_embeddings=[unit_vector(seed=1)])

    assert target.has_plate_query is False
    assert target.has_reference_embeddings is True


def test_target__both_signals__is_valid() -> None:
    """Strongest case: plate for precision, embeddings for recall."""
    target = make_target(
        plate_query="ABC1234",
        reference_embeddings=[unit_vector(seed=1), unit_vector(seed=2)],
    )

    assert target.has_plate_query is True
    assert len(target.reference_embeddings or []) == 2


def test_target__neither_signal__is_rejected_with_an_explanation() -> None:
    """An unsearchable target would return an empty result that looks like a miss."""
    with pytest.raises(ValidationError, match="unsearchable"):
        make_target(plate_query=None, reference_embeddings=None)


@pytest.mark.parametrize("plate", [None, "", "   "], ids=["none", "empty", "whitespace"])
def test_target__blank_plate_and_no_embeddings__is_rejected(plate: str | None) -> None:
    """A whitespace-only plate is absent, not present-but-blank."""
    with pytest.raises(ValidationError, match="unsearchable"):
        make_target(plate_query=plate, reference_embeddings=None)


def test_target__empty_embedding_list_and_no_plate__is_rejected() -> None:
    """Boundary: an empty list carries no more information than None."""
    with pytest.raises(ValidationError, match="unsearchable"):
        make_target(plate_query=None, reference_embeddings=[])


@pytest.mark.parametrize("label", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_target__blank_label__is_rejected(label: str) -> None:
    """The label is what an operator sees in the review queue."""
    with pytest.raises(ValidationError):
        make_target(label=label)


def test_target__label_with_surrounding_whitespace__is_stripped() -> None:
    """Incidental whitespace from a form field is not part of the name."""
    assert make_target(label="  Stolen Blue Sedan  ").label == "Stolen Blue Sedan"


def test_target__reference_embedding_of_the_wrong_dimension__names_the_index() -> None:
    """A target may carry many references; the message must say which one is bad."""
    with pytest.raises(ValidationError) as excinfo:
        make_target(
            reference_embeddings=[unit_vector(seed=1), unit_vector(seed=2, dim=128)],
        )

    message = str(excinfo.value)
    assert "reference_embeddings[1]" in message
    assert "128" in message


def test_target__unnormalized_reference_embedding__is_rejected() -> None:
    """The same rule as Sighting.embedding: rejected, never renormalized."""
    scaled = [component * 2.0 for component in unit_vector(seed=3)]

    with pytest.raises(ValidationError, match="not L2-normalized"):
        make_target(reference_embeddings=[scaled])


def test_target__defaults__are_active_with_a_generated_id() -> None:
    """A newly created target is active and needs no caller-supplied id."""
    target = make_target()

    assert target.active is True
    assert target.target_id


def test_target__naive_created_at__is_rejected() -> None:
    """No model accepts a naive datetime anywhere."""
    with pytest.raises(ValidationError, match="naive datetime"):
        make_target(created_at=BASE_INSTANT.replace(tzinfo=None))


def test_target__unknown_extra_field__is_rejected() -> None:
    """extra='forbid' applies to every domain model."""
    with pytest.raises(ValidationError) as excinfo:
        Target(label="X", plate_query="ABC1234", created_at=BASE_INSTANT, plate="ABC1234")

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())
