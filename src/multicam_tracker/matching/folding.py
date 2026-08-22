"""Ambiguity folding.

Collapses each confusion group to one representative so two OCR readings of the
same plate land on the same comparison key. That turns fuzzy candidate lookup
into an ordinary indexed equality query.

**Folding is for comparison only.** The folded form is never written to
``plate_text_normalized`` and never displayed as the plate. Showing an operator
``A8C1234`` when the vehicle actually reads ``ABC1234`` would put a wrong plate
in front of a human decision, and the global contract forbids discarding the
value that was really read.

The map lives in ``config/confusion_map.yaml`` so it can be tuned per region --
a region whose plates never use the letter O can drop that group and gain
precision. It is also baked into the ``plate_folded`` generated column by
migration 0001, and :func:`assert_schema_fold_matches` exists so a config edit
that would silently desynchronise the two fails loudly instead.
"""

from __future__ import annotations

from functools import lru_cache

from multicam_tracker.exceptions import MatchingError
from multicam_tracker.matching.normalize import load_confusion_config

__all__ = [
    "assert_schema_fold_matches",
    "confusion_groups",
    "fold_ambiguous",
    "folding_map",
    "same_confusion_group",
]


@lru_cache(maxsize=4)
def folding_map(path: str | None = None) -> dict[str, str]:
    """Return the confusable-character map from config.

    Args:
        path: Confusion-map file to read. Defaults to the checkout config.

    Returns:
        Confusable character to canonical representative.
    """
    pairs, _ = load_confusion_config(path)
    return dict(pairs)


@lru_cache(maxsize=4)
def _translation_table(path: str | None = None) -> dict[int, str]:
    """Return a ``str.translate`` table for the configured map.

    Args:
        path: Confusion-map file to read.

    Returns:
        A translation table.
    """
    return str.maketrans(folding_map(path))


def fold_ambiguous(normalized: str | None, *, path: str | None = None) -> str | None:
    """Collapse confusable characters in an already-normalized plate.

    Length-preserving: every mapping is one character to one character, so the
    folded form can be compared position by position with the original.

    Args:
        normalized: An already-normalized plate, or ``None``.
        path: Confusion-map file to read.

    Returns:
        The folded form, or ``None`` when the input was ``None``. The input is
        not otherwise modified -- this folds, it does not normalize.
    """
    if normalized is None:
        return None
    return normalized.translate(_translation_table(path))


@lru_cache(maxsize=4)
def confusion_groups(path: str | None = None) -> dict[str, frozenset[str]]:
    """Return, for each character, the set of characters it can be confused with.

    Args:
        path: Confusion-map file to read.

    Returns:
        Character to its confusion group, including the character itself and the
        group representative. Characters in no group are absent.
    """
    mapping = folding_map(path)
    members: dict[str, set[str]] = {}

    for source, representative in mapping.items():
        members.setdefault(representative, {representative}).add(source)

    groups: dict[str, frozenset[str]] = {}
    for group in members.values():
        frozen = frozenset(group)
        for member in group:
            groups[member] = frozen
    return groups


def same_confusion_group(left: str, right: str, *, path: str | None = None) -> bool:
    """Return whether two characters are plausible misreads of each other.

    Args:
        left: First character.
        right: Second character.
        path: Confusion-map file to read.

    Returns:
        ``True`` when both belong to one confusion group. A character is not
        reported as confusable with itself -- that is an equality, not a
        substitution, and the weighted distance treats the two differently.
    """
    if left == right:
        return False
    groups = confusion_groups(path)
    group = groups.get(left)
    return group is not None and right in group


def assert_schema_fold_matches() -> None:
    """Verify the configured map matches the one baked into the database schema.

    Migration 0001 compiled the fold into a ``translate()`` expression on the
    ``plate_folded`` generated column. If the config drifts from it, the
    database prefilter and in-process matching disagree -- and they disagree by
    *missing* candidates rather than mis-scoring them, which leaves no trace in
    the results.

    Raises:
        MatchingError: If the two differ, naming what changed and what to do
            about it.
    """
    from multicam_tracker.db.folding import AMBIGUITY_FOLDING

    configured = folding_map()
    if configured != AMBIGUITY_FOLDING:
        only_config = sorted(set(configured.items()) - set(AMBIGUITY_FOLDING.items()))
        only_schema = sorted(set(AMBIGUITY_FOLDING.items()) - set(configured.items()))
        raise MatchingError(
            "The configured confusion map differs from the one compiled into the "
            "plate_folded column; the database prefilter would silently miss "
            "candidates. Write a migration that rebuilds the column, or revert "
            "the config change",
            {"only_in_config": only_config, "only_in_schema": only_schema},
        )
