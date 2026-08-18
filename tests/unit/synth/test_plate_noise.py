"""Unit tests for :mod:`multicam_tracker.synth.plate_noise`.

The correlation test is the one that matters most. If reported confidence did
not fall with corruption severity, stage 06 could hit good precision by ignoring
confidence entirely -- and would then collapse on real OCR, which does report
low confidence on the reads it gets wrong.
"""

from __future__ import annotations

import random
import statistics

import pytest

from multicam_tracker.synth import NoiseProfile, corrupt_plate, edit_distance
from multicam_tracker.synth.plate_noise import CONFUSION_WEIGHTS, mutate_plate_to_distance

pytestmark = pytest.mark.unit

TRUE_PLATE = "ABC1234"
SAMPLE_SIZE = 600


def _profile(**overrides: object) -> NoiseProfile:
    """Build a noise profile with the given overrides.

    Args:
        **overrides: Fields to set.

    Returns:
        The profile.
    """
    return NoiseProfile(**overrides)


def _readings(profile: NoiseProfile, count: int = SAMPLE_SIZE, seed: int = 11) -> list:
    """Draw many readings of the same plate.

    Args:
        profile: Noise configuration.
        count: How many readings.
        seed: Generator seed.

    Returns:
        The readings.
    """
    rng = random.Random(seed)
    return [corrupt_plate(rng, TRUE_PLATE, profile) for _ in range(count)]


# ---------------------------------------------------------------------------
# Edit distance (own implementation, deliberately not shared with stage 06)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("ABC1234", "ABC1234", 0),
        ("ABC1234", "A8C1234", 1),
        ("ABC1234", "A8C1Z34", 2),
        ("ABC1234", "ABC123", 1),
        ("ABC1234", "XABC1234", 1),
        ("", "ABC", 3),
        ("ABC", "", 3),
    ],
    ids=["identical", "one-sub", "two-subs", "deletion", "insertion", "empty-left", "empty-right"],
)
def test_edit_distance__matches_known_values(left: str, right: str, expected: int) -> None:
    """Boundary cases included, since the empty string is a real OCR outcome."""
    assert edit_distance(left, right) == expected


# ---------------------------------------------------------------------------
# Corruption probability boundaries
# ---------------------------------------------------------------------------


def test_corrupt_plate__probability_zero__never_changes_the_plate() -> None:
    """Boundary: the 'clean' scenario depends on this being exact, not approximate."""
    readings = _readings(_profile(corruption_probability=0.0))

    assert all(reading.text == TRUE_PLATE for reading in readings)
    assert all(reading.is_clean for reading in readings)


def test_corrupt_plate__probability_one__always_corrupts() -> None:
    """Boundary on the other end.

    The plate here contains confusable characters and is long enough to drop
    from, so every failure mode has something to work with.
    """
    readings = _readings(_profile(corruption_probability=1.0))

    assert all(not reading.is_clean for reading in readings)
    assert all(reading.text != TRUE_PLATE for reading in readings)


def test_corrupt_plate__clean_read__still_carries_a_confidence() -> None:
    """A correct read is not a certain one; OCR always reports a number."""
    reading = _readings(_profile(corruption_probability=0.0), count=1)[0]

    assert reading.confidence is not None
    assert 0.0 <= reading.confidence <= 1.0


# ---------------------------------------------------------------------------
# Individual failure modes
# ---------------------------------------------------------------------------


def test_corrupt_plate__substitutions__only_use_the_confusion_set() -> None:
    """A random character substitution would be a failure mode OCR does not have."""
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=1.0,
        dropout_weight=0.0,
        read_failure_weight=0.0,
        spurious_weight=0.0,
    )

    for reading in _readings(profile, count=200):
        assert reading.text is not None
        assert len(reading.text) == len(TRUE_PLATE)
        for observed, truth in zip(reading.text, TRUE_PLATE, strict=True):
            if observed != truth:
                assert observed in {char for char, _ in CONFUSION_WEIGHTS[truth]}


def test_corrupt_plate__dropout__produces_a_strictly_shorter_string() -> None:
    """Occlusion removes characters; it never adds or replaces them."""
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=0.0,
        dropout_weight=1.0,
        read_failure_weight=0.0,
        spurious_weight=0.0,
    )

    for reading in _readings(profile, count=200):
        assert reading.text is not None
        assert 0 < len(reading.text) < len(TRUE_PLATE)


def test_corrupt_plate__read_failure__yields_no_text_but_keeps_the_sighting() -> None:
    """The vehicle was still seen. This is the case that forces re-id to carry the match."""
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=0.0,
        dropout_weight=0.0,
        read_failure_weight=1.0,
        spurious_weight=0.0,
    )

    for reading in _readings(profile, count=50):
        assert reading.text is None
        assert reading.confidence is None
        assert reading.kind == "read_failure"


def test_corrupt_plate__spurious__produces_a_longer_string() -> None:
    """A sticker or frame border read as plate text."""
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=0.0,
        dropout_weight=0.0,
        read_failure_weight=0.0,
        spurious_weight=1.0,
    )

    for reading in _readings(profile, count=200):
        assert reading.text is not None
        assert len(reading.text) > len(TRUE_PLATE)


def test_corrupt_plate__all_four_modes__occur_in_a_mixed_profile() -> None:
    """A generator that silently only ever substituted would flatter the matcher."""
    kinds = {reading.kind for reading in _readings(_profile(corruption_probability=1.0))}

    assert kinds == {"substitution", "dropout", "read_failure", "spurious"}


# ---------------------------------------------------------------------------
# Severity, confidence, and bounds
# ---------------------------------------------------------------------------


def test_corrupt_plate__confidence__is_negatively_correlated_with_severity() -> None:
    """Asserted as a correlation coefficient over a large sample, not by eye."""
    readings = [
        reading
        for reading in _readings(_profile(corruption_probability=1.0), count=1200)
        if reading.confidence is not None
    ]
    severities = [reading.severity for reading in readings]
    confidences = [reading.confidence for reading in readings]

    correlation = statistics.correlation(severities, confidences)

    assert correlation < -0.5, f"confidence must track quality; got r={correlation:.3f}"


def test_corrupt_plate__confidence__never_leaves_the_unit_interval() -> None:
    """A heavy penalty must floor rather than go negative."""
    profile = _profile(corruption_probability=1.0, confidence_penalty_per_severity=5.0)

    for reading in _readings(profile, count=300):
        if reading.confidence is not None:
            assert 0.0 <= reading.confidence <= 1.0


def test_corrupt_plate__edit_distance__never_exceeds_the_configured_maximum() -> None:
    """The cap is what keeps corrupted plates inside fuzzy-matching range."""
    profile = _profile(
        corruption_probability=1.0,
        max_substitutions=2,
        max_dropped=2,
        max_spurious=2,
        read_failure_weight=0.0,
    )

    for reading in _readings(profile, count=400):
        assert reading.text is not None
        assert edit_distance(TRUE_PLATE, reading.text) <= profile.max_edit_distance


def test_corrupt_plate__severity__equals_the_edit_distance_for_text_reads() -> None:
    """Severity is what drives confidence, so it must be the real distance."""
    for reading in _readings(_profile(corruption_probability=1.0), count=300):
        if reading.text is not None and not reading.is_clean:
            assert reading.severity == float(edit_distance(TRUE_PLATE, reading.text))


def test_corrupt_plate__plate_with_no_confusable_characters__reports_a_clean_read() -> None:
    """Honest fallback.

    Inventing a different corruption would make the requested failure-mode mix a
    lie, and a test asserting on that mix would silently pass.
    """
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=1.0,
        dropout_weight=0.0,
        read_failure_weight=0.0,
        spurious_weight=0.0,
    )
    rng = random.Random(3)

    reading = corrupt_plate(rng, "HHHH", profile)

    assert reading.is_clean
    assert reading.text == "HHHH"


def test_corrupt_plate__single_character_plate__cannot_be_dropped_to_nothing() -> None:
    """Boundary: an empty read is a read failure, a different mode with its own weight."""
    profile = _profile(
        corruption_probability=1.0,
        substitution_weight=0.0,
        dropout_weight=1.0,
        read_failure_weight=0.0,
        spurious_weight=0.0,
    )
    rng = random.Random(3)

    reading = corrupt_plate(rng, "A", profile)

    assert reading.text == "A"


# ---------------------------------------------------------------------------
# Near-miss plate construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance", [1, 2, 3], ids=["one", "two", "three"])
def test_mutate_plate_to_distance__produces_exactly_that_distance(distance: int) -> None:
    """Near-miss decoys stress fuzzy matching at a known threshold, not a guessed one."""
    rng = random.Random(distance)

    for _ in range(50):
        mutated = mutate_plate_to_distance(rng, TRUE_PLATE, distance)
        assert edit_distance(TRUE_PLATE, mutated) == distance


def test_mutate_plate_to_distance__zero__returns_the_plate_unchanged() -> None:
    """Boundary: distance zero is the plate itself, which is plate cloning."""
    assert mutate_plate_to_distance(random.Random(1), TRUE_PLATE, 0) == TRUE_PLATE


def test_mutate_plate_to_distance__beyond_the_plate_length__is_rejected() -> None:
    """Substitution alone cannot exceed the length; asking for it is a caller bug."""
    with pytest.raises(ValueError, match="cannot change"):
        mutate_plate_to_distance(random.Random(1), "AB", 5)


def test_mutate_plate_to_distance__negative__is_rejected() -> None:
    """Boundary on the other side."""
    with pytest.raises(ValueError, match="non-negative"):
        mutate_plate_to_distance(random.Random(1), TRUE_PLATE, -1)


def test_corrupt_plate__is_deterministic_for_a_given_seed() -> None:
    """Every stream in the generator has to be reproducible, including this one."""
    profile = _profile(corruption_probability=0.7)

    first = [r.text for r in _readings(profile, count=50, seed=42)]
    second = [r.text for r in _readings(profile, count=50, seed=42)]

    assert first == second
