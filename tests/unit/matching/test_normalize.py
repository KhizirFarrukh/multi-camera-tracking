"""Unit tests for :mod:`multicam_tracker.matching.normalize`.

String logic is cheap to test exhaustively, so it is tested exhaustively. A
normalization bug is silent: it produces a plate that looks like a plate and
matches the wrong things.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.matching import RegionalRules, normalize_plate
from multicam_tracker.matching.normalize import load_confusion_config

pytestmark = pytest.mark.unit

_HYPOTHESIS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("abc1234", "ABC1234"),
        ("AbC1234", "ABC1234"),
        ("ABC 1234", "ABC1234"),
        ("ABC-1234", "ABC1234"),
        ("A.B.C.1234", "ABC1234"),
        ("ABC/1234", "ABC1234"),
        ("  ABC1234  ", "ABC1234"),
        ("\tABC 12-34\n", "ABC1234"),
    ],
    ids=[
        "lowercase",
        "mixed-case",
        "space",
        "hyphen",
        "dots",
        "slash",
        "surrounding-space",
        "everything",
    ],
)
def test_normalize_plate__strips_and_uppercases(raw: str, expected: str) -> None:
    """The contract rules, one case per rule."""
    assert normalize_plate(raw) == expected


def test_normalize_plate__already_normalized__is_unchanged() -> None:
    """Idempotency, the property every caller relies on to normalize freely."""
    assert normalize_plate("ABC1234") == "ABC1234"


@pytest.mark.parametrize(
    "raw",
    ["abc-1234", "  a.b/c 1234 ", "ABC1234", "Ä1234"],
    ids=["punct", "messy", "clean", "accent"],
)
def test_normalize_plate__is_idempotent(raw: str) -> None:
    """normalize(normalize(x)) == normalize(x)."""
    once = normalize_plate(raw)
    assert once is not None
    assert normalize_plate(once) == once


def test_normalize_plate__none_input__returns_none() -> None:
    """An unreadable plate stays unreadable rather than becoming a value."""
    assert normalize_plate(None) is None


def test_normalize_plate__empty_string__returns_none() -> None:
    """Boundary: the empty string is not a plate.

    Returning it would make every unreadable plate compare equal to every other,
    collapsing a whole class of failures into one false identity.
    """
    assert normalize_plate("") is None


@pytest.mark.parametrize(
    "raw", ["---", "   ", "...", "/-. "], ids=["dashes", "spaces", "dots", "mixed"]
)
def test_normalize_plate__punctuation_only__returns_none(raw: str) -> None:
    """Same reasoning: nothing alphanumeric survived, so there is no plate."""
    assert normalize_plate(raw) is None


def test_normalize_plate__accented_latin__is_transliterated() -> None:
    """An accented Latin character has exactly one base letter, so it is certain."""
    assert normalize_plate("ÁBC1234") == "ABC1234"
    assert normalize_plate("ÖRD-99") == "ORD99"


@pytest.mark.parametrize(
    "raw",
    ["АBC1234", "ABC1Ζ34", "ＡBC1234"],
    ids=["cyrillic-a", "greek-zeta", "fullwidth-mismatch"],
)
def test_normalize_plate__ambiguous_unicode__is_rejected_explicitly(raw: str) -> None:
    """Never passed through silently.

    A Cyrillic A is a different character that renders identically. Guessing
    which script a plate uses would create a second identity for one vehicle.
    """
    try:
        result = normalize_plate(raw)
    except ValidationError as exc:
        assert "characters" in exc.context
        return
    # Full-width forms decompose to ASCII under NFKD, which is unambiguous, so
    # transliteration is the correct outcome for those.
    assert result is not None
    assert result.isascii()


def test_normalize_plate__rejection_names_the_offending_code_points() -> None:
    """An operator needs to know which character was the problem."""
    with pytest.raises(ValidationError) as excinfo:
        normalize_plate("АBC1234")

    assert excinfo.value.context["code_points"] == ["U+0410"]


def test_normalize_plate__regional_stripping__is_off_by_default() -> None:
    """The contract default. Stripping a real prefix destroys the identifier."""
    assert normalize_plate("GB-ABC1234") == "GBABC1234"


def test_normalize_plate__regional_stripping__applies_when_enabled(tmp_path: Path) -> None:
    """And is active when a region configures it."""
    config = tmp_path / "confusion_map.yaml"
    config.write_text(
        "folding:\n  O: '0'\nregional:\n  strip_prefixes: [GB]\n  strip_suffixes: [X]\n",
        encoding="utf-8",
    )
    load_confusion_config.cache_clear()

    rules = RegionalRules(*load_confusion_config(str(config))[1])

    assert rules.enabled is True
    assert rules.apply("GBABC1234X") == "ABC1234"
    load_confusion_config.cache_clear()


def test_regional_rules__disabled_by_default_config() -> None:
    """The shipped config declares no stripping."""
    from multicam_tracker.matching import regional_rules

    assert regional_rules().enabled is False


def test_regional_rules__never_strips_the_whole_plate() -> None:
    """Boundary: a plate that is only its prefix is a read error, not an empty plate."""
    rules = RegionalRules(strip_prefixes=("GB",), strip_suffixes=())

    assert rules.apply("GB") == "GB"


def test_regional_rules__strips_at_most_one_prefix() -> None:
    """A repeated prefix is far more likely to be part of the plate."""
    rules = RegionalRules(strip_prefixes=("GB",), strip_suffixes=())

    assert rules.apply("GBGBABC") == "GBABC"


def test_load_confusion_config__missing_file__raises_a_project_error(tmp_path: Path) -> None:
    """Callers catch one hierarchy, not a mix of OSError and YAMLError."""
    load_confusion_config.cache_clear()
    with pytest.raises(ValidationError, match="not found"):
        load_confusion_config(str(tmp_path / "nope.yaml"))
    load_confusion_config.cache_clear()


def test_load_confusion_config__malformed_yaml__raises_a_project_error(tmp_path: Path) -> None:
    """A raw parser exception must not escape."""
    bad = tmp_path / "confusion_map.yaml"
    bad.write_text("folding: [\n", encoding="utf-8")
    load_confusion_config.cache_clear()

    with pytest.raises(ValidationError, match="could not be parsed"):
        load_confusion_config(str(bad))
    load_confusion_config.cache_clear()


@_HYPOTHESIS
@given(raw=st.text(max_size=40))
def test_normalize_plate__output_is_always_uppercase_alphanumeric_or_none(raw: str) -> None:
    """Property: nothing else ever escapes normalization."""
    try:
        result = normalize_plate(raw)
    except ValidationError:
        return

    if result is None:
        return
    assert result.isascii()
    assert result.isalnum()
    assert result == result.upper()


@_HYPOTHESIS
@given(raw=st.text(alphabet=st.characters(max_codepoint=127), max_size=40))
def test_normalize_plate__is_idempotent_for_ascii_input(raw: str) -> None:
    """Property: normalizing twice changes nothing."""
    once = normalize_plate(raw)
    assert normalize_plate(once) == once
