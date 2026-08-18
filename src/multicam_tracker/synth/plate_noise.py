"""The OCR noise model.

Real ALPR does not fail by adding uniform random noise. It fails in four
characteristic ways, and a generator that models only the first would make fuzzy
matching look far better than it is:

1. **Substitution** -- a character read as its lookalike. Weighted by which
   confusions actually happen: ``0/O`` far more often than ``6/G``.
2. **Dropout** -- part of the plate occluded by a tow bar, dirt, or the frame
   edge, producing a strictly shorter string.
3. **Read failure** -- nothing legible at all. The plate is ``None`` but the
   vehicle was still seen, so the embedding survives. This is the case that
   forces re-id to carry the match.
4. **Spurious characters** -- a bumper sticker or frame border read as text,
   producing a longer string.

Reported confidence tracks severity. That coupling is load-bearing: if a
generator emitted high confidence on mangled reads, stage 06 could reach good
precision by ignoring confidence entirely, and would then collapse on real data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from multicam_tracker.synth.scenario import NoiseProfile

__all__ = [
    "CONFUSION_WEIGHTS",
    "SPURIOUS_ALPHABET",
    "PlateReading",
    "corrupt_plate",
    "edit_distance",
    "mutate_plate_to_distance",
]

CONFUSION_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "0": [("O", 6.0), ("D", 1.0), ("Q", 0.5)],
    "O": [("0", 6.0), ("Q", 1.0), ("D", 0.5)],
    "1": [("I", 5.0), ("L", 3.0), ("7", 0.5)],
    "I": [("1", 5.0), ("L", 2.0)],
    "L": [("1", 3.0), ("I", 2.0)],
    "5": [("S", 4.0), ("6", 0.5)],
    "S": [("5", 4.0)],
    "8": [("B", 3.0), ("6", 0.5), ("0", 0.5)],
    "B": [("8", 3.0), ("6", 0.5)],
    "2": [("Z", 3.0), ("7", 0.5)],
    "Z": [("2", 3.0), ("7", 0.5)],
    "6": [("G", 2.0), ("5", 0.5), ("8", 0.5)],
    "G": [("6", 2.0), ("C", 0.5)],
}
"""Character -> plausible misreads with relative frequency.

Drawn from the ambiguity pairs in the global contract's ``plate_normalization``,
weighted so the common confusions dominate. A uniform model would produce a
distribution of errors no real OCR engine makes.
"""

SPURIOUS_ALPHABET = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
"""Characters a stray sticker or frame border can be misread as."""


@dataclass(frozen=True)
class PlateReading:
    """What the OCR stage reported for one plate, and how wrong it is."""

    text: str | None
    """The observed plate, or ``None`` for a complete read failure."""

    confidence: float | None
    """Reported confidence, or ``None`` when there is no reading at all."""

    kind: str | None
    """``substitution``, ``dropout``, ``read_failure``, ``spurious``, or ``None``
    when the read was clean."""

    severity: float
    """Edit operations away from the truth. Zero for a clean read."""

    detail: str = ""

    @property
    def is_clean(self) -> bool:
        """Return whether the reading matches the true plate exactly."""
        return self.kind is None


def edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance between two strings.

    Implemented here rather than pulled in as a dependency because stage 06 owns
    the matching implementation, and the generator must not share code with the
    thing it is used to evaluate -- a shared bug would cancel itself out and the
    evaluation would look perfect.

    Args:
        left: First string.
        right: Second string.

    Returns:
        The minimum number of insertions, deletions, and substitutions.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    """Pick one option in proportion to its weight.

    Args:
        rng: Seeded generator.
        options: ``(value, weight)`` pairs with at least one positive weight.

    Returns:
        The chosen value.
    """
    total = sum(weight for _, weight in options)
    target = rng.random() * total
    running = 0.0
    for value, weight in options:
        running += weight
        if target < running:
            return value
    return options[-1][0]


def _substitute(rng: random.Random, plate: str, profile: NoiseProfile) -> tuple[str, int]:
    """Replace confusable characters with their lookalikes.

    Args:
        rng: Seeded generator.
        plate: The true plate.
        profile: Noise configuration.

    Returns:
        ``(corrupted_plate, substitutions_applied)``. Zero substitutions when the
        plate contains no confusable characters at all.
    """
    positions = [index for index, char in enumerate(plate) if char in CONFUSION_WEIGHTS]
    if not positions:
        return plate, 0

    count = min(rng.randint(1, profile.max_substitutions), len(positions))
    chosen = rng.sample(sorted(positions), count)

    characters = list(plate)
    for index in chosen:
        characters[index] = _weighted_choice(rng, CONFUSION_WEIGHTS[characters[index]])
    return "".join(characters), count


def _drop(rng: random.Random, plate: str, profile: NoiseProfile) -> tuple[str, int]:
    """Remove characters, as an occlusion would.

    Args:
        rng: Seeded generator.
        plate: The true plate.
        profile: Noise configuration.

    Returns:
        ``(corrupted_plate, characters_dropped)``. At least one character is
        always kept, since a plate read as the empty string is a read failure,
        which is a different mode with its own weight.
    """
    droppable = min(profile.max_dropped, len(plate) - 1)
    if droppable < 1:
        return plate, 0

    count = rng.randint(1, droppable)
    dropped = set(rng.sample(range(len(plate)), count))
    return "".join(char for index, char in enumerate(plate) if index not in dropped), count


def _add_spurious(rng: random.Random, plate: str, profile: NoiseProfile) -> tuple[str, int]:
    """Insert stray characters, as a sticker or frame border would.

    Args:
        rng: Seeded generator.
        plate: The true plate.
        profile: Noise configuration.

    Returns:
        ``(corrupted_plate, characters_added)``.
    """
    count = rng.randint(1, profile.max_spurious)
    characters = list(plate)
    for _ in range(count):
        position = rng.randint(0, len(characters))
        characters.insert(position, rng.choice(SPURIOUS_ALPHABET))
    return "".join(characters), count


def corrupt_plate(rng: random.Random, plate: str, profile: NoiseProfile) -> PlateReading:
    """Produce one simulated OCR reading of a plate.

    Args:
        rng: Seeded generator. All randomness flows through it so output stays
            reproducible.
        plate: The vehicle's true plate.
        profile: Noise configuration.

    Returns:
        The reading, including which failure mode was applied and how severe it
        was. Confidence falls with severity.
    """
    if profile.corruption_probability <= 0.0 or rng.random() >= profile.corruption_probability:
        return PlateReading(
            text=plate,
            confidence=rng.uniform(profile.clean_confidence_min, profile.clean_confidence_max),
            kind=None,
            severity=0.0,
        )

    modes = [
        ("substitution", profile.substitution_weight),
        ("dropout", profile.dropout_weight),
        ("read_failure", profile.read_failure_weight),
        ("spurious", profile.spurious_weight),
    ]
    kind = _weighted_choice(rng, modes)

    if kind == "read_failure":
        return PlateReading(
            text=None,
            confidence=None,
            kind="read_failure",
            severity=float(len(plate)),
            detail="plate illegible; the vehicle was still detected and embedded",
        )

    if kind == "substitution":
        corrupted, applied = _substitute(rng, plate, profile)
        detail = f"{applied} character(s) read as a lookalike"
    elif kind == "dropout":
        corrupted, applied = _drop(rng, plate, profile)
        detail = f"{applied} character(s) occluded"
    else:
        corrupted, applied = _add_spurious(rng, plate, profile)
        detail = f"{applied} spurious character(s) read from outside the plate"

    if applied == 0 or corrupted == plate:
        # The chosen mode had nothing to work with -- a plate with no confusable
        # characters, or one too short to drop from. Reporting it as a clean read
        # is the honest outcome; inventing a different corruption would make the
        # requested failure-mode mix a lie.
        return PlateReading(
            text=plate,
            confidence=rng.uniform(profile.clean_confidence_min, profile.clean_confidence_max),
            kind=None,
            severity=0.0,
        )

    severity = float(edit_distance(plate, corrupted))
    confidence = max(
        profile.min_confidence,
        rng.uniform(profile.clean_confidence_min, profile.clean_confidence_max)
        - profile.confidence_penalty_per_severity * severity,
    )

    return PlateReading(
        text=corrupted,
        confidence=confidence,
        kind=kind,
        severity=severity,
        detail=detail,
    )


def mutate_plate_to_distance(rng: random.Random, plate: str, distance: int) -> str:
    """Produce a plate exactly ``distance`` substitutions from ``plate``.

    Used to build near-miss decoys at a *known* distance, so fuzzy matching is
    stressed at the threshold rather than wherever random plates happen to land.

    Substitutions only -- never insertions or deletions -- so the resulting
    Levenshtein distance is exactly the requested one.

    Args:
        rng: Seeded generator.
        plate: The plate to move away from.
        distance: Number of characters to change.

    Returns:
        The mutated plate.

    Raises:
        ValueError: If ``distance`` exceeds the plate length, which cannot be
            achieved by substitution alone.
    """
    if distance < 0:
        msg = f"distance must be non-negative; got {distance}"
        raise ValueError(msg)
    if distance > len(plate):
        msg = f"cannot change {distance} characters of a {len(plate)}-character plate"
        raise ValueError(msg)
    if distance == 0:
        return plate

    positions = rng.sample(range(len(plate)), distance)
    characters = list(plate)
    for index in positions:
        alternatives = [char for char in SPURIOUS_ALPHABET if char != characters[index]]
        characters[index] = rng.choice(alternatives)
    return "".join(characters)
