"""Unit tests for :mod:`multicam_tracker.db.folding`.

The fold is defined once and consumed twice -- by the SQL generated column and by
the in-memory fake. If the two ever disagreed, fuzzy matching would return
different candidates in tests than in production, so the derived ``translate()``
arguments are asserted here alongside the behaviour.
"""

from __future__ import annotations

import pytest

from multicam_tracker.db.folding import (
    AMBIGUITY_FOLDING,
    FOLD_REPLACEMENT,
    FOLD_SOURCE,
    fold_plate,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("confusable", "canonical"),
    sorted(AMBIGUITY_FOLDING.items()),
    ids=lambda value: value,
)
def test_fold_plate__each_confusable_character__maps_to_its_canonical_form(
    confusable: str, canonical: str
) -> None:
    """Every pair from the contract's plate_normalization rules is applied."""
    assert fold_plate(confusable) == canonical


def test_fold_plate__covers_exactly_the_contract_pairs() -> None:
    """The fold set is fixed by the contract; adding one silently would change
    which plates are considered candidates for each other."""
    assert AMBIGUITY_FOLDING == {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
    }


def test_fold_plate__realistic_ocr_confusion__collapses_to_one_key() -> None:
    """The motivating case: one plate read two ways at two cameras."""
    assert fold_plate("ABC1234") == fold_plate("A8C1Z34")


def test_fold_plate__genuinely_different_plates__do_not_collapse() -> None:
    """Folding must widen the search, not erase the distinction it searches on."""
    assert fold_plate("ABC1234") != fold_plate("XYZ9999")


def test_fold_plate__already_folded_input__is_unchanged() -> None:
    """Idempotent: folding a folded form is a no-op, since digits are the targets."""
    once = fold_plate("ABC1234")
    assert once is not None
    assert fold_plate(once) == once


def test_fold_plate__none__returns_none() -> None:
    """Most sightings have no plate; the column is nullable and so is the fold."""
    assert fold_plate(None) is None


def test_fold_plate__empty_string__returns_empty_string() -> None:
    """Boundary: an empty plate folds to an empty plate, not to None."""
    assert fold_plate("") == ""


def test_fold_plate__unaffected_characters__pass_through() -> None:
    """Only the listed confusables change."""
    assert fold_plate("ACDEF") == "ACDEF"


def test_fold_plate__does_not_mutate_digits_into_letters() -> None:
    """The fold runs one way. Digits are the canonical side and stay put."""
    assert fold_plate("0123456789") == "0123456789"


def test_fold_arguments__are_positionally_aligned() -> None:
    """The SQL translate() arguments are derived, never hand-written.

    Hand-writing them is exactly how the generated column and the Python fold
    would drift apart.
    """
    assert len(FOLD_SOURCE) == len(FOLD_REPLACEMENT)
    for source, replacement in zip(FOLD_SOURCE, FOLD_REPLACEMENT, strict=True):
        assert AMBIGUITY_FOLDING[source] == replacement


def test_fold_arguments__match_the_migration_expression() -> None:
    """The values baked into the migration are the ones this module defines."""
    assert FOLD_SOURCE == "OQILSBZG"
    assert FOLD_REPLACEMENT == "00115826"
