"""Unit tests for :mod:`multicam_tracker.models.enums`.

The contract fixes both the member sets and their string values; a rename would
silently break every stored row and JSON payload, so both are asserted
explicitly rather than inferred.
"""

from __future__ import annotations

from enum import Enum

import pytest

from multicam_tracker.models.enums import MatchMethod, ObjectClass, ReviewStatus, SourceType

pytestmark = pytest.mark.unit

ALL_ENUMS = [ObjectClass, MatchMethod, ReviewStatus, SourceType]


@pytest.mark.parametrize("enum_type", ALL_ENUMS, ids=lambda cls: cls.__name__)
def test_enums__every_member__is_a_plain_string(enum_type: type[Enum]) -> None:
    """String enums compare equal to their values at every boundary."""
    assert issubclass(enum_type, str)
    for member in enum_type:
        assert member == member.value
        assert isinstance(member.value, str)


@pytest.mark.parametrize(
    ("enum_type", "expected"),
    [
        (ObjectClass, {"car", "truck", "bus", "motorcycle", "unknown"}),
        (MatchMethod, {"plate_exact", "plate_fuzzy", "embedding", "manual"}),
        (ReviewStatus, {"auto_accepted", "pending_review", "confirmed", "rejected"}),
        (SourceType, {"recorded_file", "live_stream"}),
    ],
    ids=lambda value: getattr(value, "__name__", "values"),
)
def test_enums__member_values__match_the_contract(
    enum_type: type[Enum], expected: set[str]
) -> None:
    """The contract's string values are the storage format; they cannot drift."""
    assert {member.value for member in enum_type} == expected


@pytest.mark.parametrize("enum_type", ALL_ENUMS, ids=lambda cls: cls.__name__)
def test_enums__unknown_value__is_rejected(enum_type: type[Enum]) -> None:
    """Constructing from an unrecognised string fails rather than inventing a member."""
    with pytest.raises(ValueError, match="is not a valid"):
        enum_type("not_a_real_member")


def test_object_class__has_an_unknown_member() -> None:
    """The detector must be able to say it does not know, rather than guess 'car'."""
    assert ObjectClass.UNKNOWN.value == "unknown"


def test_review_status__auto_accepted_and_confirmed_are_distinct() -> None:
    """The audit trail must distinguish 'no human looked' from 'a human agreed'."""
    assert ReviewStatus.AUTO_ACCEPTED is not ReviewStatus.CONFIRMED
