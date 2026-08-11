"""Plate ambiguity folding.

OCR confuses characters that look alike. A plate read as ``ABC1234`` at one
camera may read as ``A8C1Z34`` at the next, and an exact-match index would treat
those as unrelated. Folding collapses each confusable group to one
representative so both spellings land on the same index key, turning fuzzy
prefiltering into an ordinary equality lookup.

The fold is defined once here and used in two places that must never disagree:

* the ``plate_folded`` generated column in Postgres, whose SQL ``translate()``
  arguments are derived from :data:`AMBIGUITY_FOLDING` rather than written by
  hand;
* the in-memory repository fake, so its query results match Postgres exactly.

Folding is for *comparison only*. It never mutates ``plate_text_normalized``,
which keeps the value actually read from the plate (global contract,
``plate_normalization``). A distance of 0 on folded forms but non-zero on
normalized forms is still classified ``plate_fuzzy``, never ``plate_exact``.

Stage 06 owns the full normalization and matching pipeline. Only the folding is
here, because the database index that depends on it is created in this stage.
"""

from __future__ import annotations

__all__ = [
    "AMBIGUITY_FOLDING",
    "FOLD_REPLACEMENT",
    "FOLD_SOURCE",
    "fold_plate",
]

AMBIGUITY_FOLDING: dict[str, str] = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}
"""Confusable character -> canonical representative.

Taken from the global contract's ``plate_normalization.rules``. Digits are the
representatives because a digit is the more common true reading on most plate
formats, so the folded form usually equals the correct one.
"""

FOLD_SOURCE = "".join(AMBIGUITY_FOLDING)
"""Characters to replace, as the ``from`` argument of SQL ``translate()``."""

FOLD_REPLACEMENT = "".join(AMBIGUITY_FOLDING.values())
"""Replacements, positionally aligned with :data:`FOLD_SOURCE`."""

_FOLD_TABLE = str.maketrans(AMBIGUITY_FOLDING)


def fold_plate(plate: str | None) -> str | None:
    """Collapse ambiguous characters in an already-normalized plate.

    Args:
        plate: A normalized plate (uppercase, alphanumeric), or ``None``.

    Returns:
        The folded form, or ``None`` when ``plate`` is ``None``. The input is
        not otherwise modified: this function folds, it does not normalize.
    """
    if plate is None:
        return None
    return plate.translate(_FOLD_TABLE)
