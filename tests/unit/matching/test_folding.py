"""Unit tests for :mod:`multicam_tracker.matching.folding`.

Every documented confusion pair gets its own explicit case. A pair silently
dropped from the map would not fail anything else -- it would just quietly stop
matching, and the recall loss would look like OCR noise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching import (
    assert_schema_fold_matches,
    confusion_groups,
    fold_ambiguous,
    folding_map,
    same_confusion_group,
)
from multicam_tracker.matching.normalize import load_confusion_config

pytestmark = pytest.mark.unit

CONTRACT_PAIRS = [
    ("O", "0"),
    ("Q", "0"),
    ("I", "1"),
    ("L", "1"),
    ("S", "5"),
    ("B", "8"),
    ("Z", "2"),
    ("G", "6"),
]


@pytest.mark.parametrize(("confusable", "representative"), CONTRACT_PAIRS, ids=lambda v: v)
def test_fold_ambiguous__each_contract_pair__folds_to_its_representative(
    confusable: str, representative: str
) -> None:
    """One explicit case per pair in the global contract."""
    assert fold_ambiguous(confusable) == representative


@pytest.mark.parametrize(("confusable", "representative"), CONTRACT_PAIRS, ids=lambda v: v)
def test_fold_ambiguous__both_members_of_a_pair__land_on_one_key(
    confusable: str, representative: str
) -> None:
    """The point of folding: two readings of one plate compare equal."""
    assert fold_ambiguous(f"AB{confusable}12") == fold_ambiguous(f"AB{representative}12")


@pytest.mark.parametrize("char", ["A", "C", "D", "E", "F", "H", "3", "4", "7", "9"])
def test_fold_ambiguous__characters_outside_any_group__are_unchanged(char: str) -> None:
    """Folding must not collapse plates that are genuinely different."""
    assert fold_ambiguous(char) == char


def test_fold_ambiguous__is_idempotent() -> None:
    """Representatives fold to themselves, so folding twice changes nothing."""
    once = fold_ambiguous("ABC1234")
    assert once is not None
    assert fold_ambiguous(once) == once


@pytest.mark.parametrize(
    "plate",
    ["ABC1234", "OQILSBZG", "", "1", "ZZZZZZZZZZ"],
    ids=["normal", "all-confusable", "empty", "single", "long"],
)
def test_fold_ambiguous__preserves_string_length(plate: str) -> None:
    """One character maps to one character, so positions stay comparable."""
    folded = fold_ambiguous(plate)
    assert folded is not None
    assert len(folded) == len(plate)


def test_fold_ambiguous__none__returns_none() -> None:
    """An unreadable plate has no folded form."""
    assert fold_ambiguous(None) is None


def test_fold_ambiguous__does_not_normalize() -> None:
    """This folds; it does not uppercase or strip. Mixing the two would hide a
    caller that skipped normalization."""
    assert fold_ambiguous("abc") == "abc"


def test_folding_map__matches_the_shipped_config() -> None:
    """The map is config, not a constant, so it can be tuned per region."""
    assert folding_map() == dict(CONTRACT_PAIRS)


def test_folding_map__config_file_overrides_the_default(tmp_path: Path) -> None:
    """A region whose plates never use O can drop that group and gain precision."""
    config = tmp_path / "confusion_map.yaml"
    config.write_text("folding:\n  X: 'Y'\nregional: {}\n", encoding="utf-8")
    folding_map.cache_clear()
    load_confusion_config.cache_clear()

    assert folding_map(str(config)) == {"X": "Y"}
    assert fold_ambiguous("XO", path=str(config)) == "YO", "O is no longer folded"

    folding_map.cache_clear()
    load_confusion_config.cache_clear()


# ---------------------------------------------------------------------------
# Confusion groups
# ---------------------------------------------------------------------------


def test_confusion_groups__collect_every_member_of_a_group() -> None:
    """I, L and 1 are all one another's plausible misreads."""
    groups = confusion_groups()

    assert groups["1"] == frozenset({"1", "I", "L"})
    assert groups["0"] == frozenset({"0", "O", "Q"})


def test_confusion_groups__omit_characters_in_no_group() -> None:
    """Absence is the answer for a character nothing is confused with."""
    assert "A" not in confusion_groups()


@pytest.mark.parametrize(("left", "right"), [("0", "O"), ("O", "0"), ("I", "L"), ("8", "B")])
def test_same_confusion_group__confusable_pairs__are_recognised(left: str, right: str) -> None:
    """Symmetric, and covers pairs that share a representative without mapping
    to each other."""
    assert same_confusion_group(left, right) is True


@pytest.mark.parametrize(("left", "right"), [("0", "W"), ("A", "B"), ("3", "4")])
def test_same_confusion_group__unrelated_pairs__are_not(left: str, right: str) -> None:
    """A 0 read as W is not a confusion the model should forgive."""
    assert same_confusion_group(left, right) is False


def test_same_confusion_group__a_character_with_itself__is_false() -> None:
    """That is an equality, not a substitution.

    The weighted distance charges zero for equality and a reduced cost for an
    in-group substitution; conflating them would make the two indistinguishable.
    """
    assert same_confusion_group("0", "0") is False


# ---------------------------------------------------------------------------
# Schema synchronisation
# ---------------------------------------------------------------------------


def test_assert_schema_fold_matches__shipped_config__is_in_sync() -> None:
    """The database prefilter and in-process matching must agree.

    They disagree by *missing* candidates rather than mis-scoring them, which
    leaves no trace in the results -- so this is asserted rather than trusted.
    """
    assert_schema_fold_matches()


def test_assert_schema_fold_matches__detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard has to actually fire, not be vacuously true."""
    import multicam_tracker.db.folding as db_folding

    monkeypatch.setattr(db_folding, "AMBIGUITY_FOLDING", {"O": "0"})

    with pytest.raises(MatchingError, match="differs from the one compiled"):
        assert_schema_fold_matches()


def test_fold_ambiguous__agrees_with_the_database_fold() -> None:
    """Same input, same output, through two independent implementations."""
    from multicam_tracker.db.folding import fold_plate

    for plate in ("ABC1234", "OQILSBZG", "XYZ9999", "A8C1Z34"):
        assert fold_ambiguous(plate) == fold_plate(plate)
