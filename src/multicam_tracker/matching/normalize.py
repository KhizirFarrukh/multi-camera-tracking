"""Plate normalization.

Turns whatever OCR produced into the canonical form everything else compares
against. Pure and side-effect free apart from reading the cached confusion-map
file.

Two rules are worth stating up front because they are easy to get wrong.

**Unicode is transliterated where unambiguous and rejected otherwise.** An
accented Latin character decomposes to its base letter, which is a certain
reading. A Cyrillic capital A looks identical to a Latin one but is a different
character, and guessing which a plate carries would silently create two
identities for one vehicle. So it raises.

**An empty result is None, not the empty string.** The empty string compares
equal to itself, so every plate that normalized to nothing would match every
other one -- a whole class of unreadable plates collapsing into a single false
identity.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from multicam_tracker.exceptions import ValidationError

__all__ = [
    "RegionalRules",
    "default_confusion_map_file",
    "load_confusion_config",
    "normalize_plate",
    "regional_rules",
]

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent


def default_confusion_map_file() -> Path:
    """Locate the confusion map for a source checkout.

    Returns:
        Path to ``config/confusion_map.yaml``, falling back to the working
        directory when the package is installed outside a checkout.
    """
    checkout = _REPO_ROOT / "config" / "confusion_map.yaml"
    if checkout.is_file():
        return checkout
    return Path.cwd() / "config" / "confusion_map.yaml"


@lru_cache(maxsize=4)
def load_confusion_config(
    path: str | None = None,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Read the confusion map and regional rules from YAML.

    Cached, and returned as tuples so the result is hashable and no caller can
    mutate it into affecting every other caller.

    Args:
        path: File to read. Defaults to the checkout config.

    Returns:
        ``(folding_pairs, (strip_prefixes, strip_suffixes))``.

    Raises:
        ValidationError: If the file is missing, unparseable, or malformed.
    """
    resolved = Path(path) if path else default_confusion_map_file()
    if not resolved.is_file():
        raise ValidationError("Confusion map file not found", {"path": str(resolved)})

    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(
            "Confusion map could not be parsed", {"path": str(resolved), "reason": str(exc)}
        ) from exc

    if not isinstance(raw, dict):
        raise ValidationError(
            "Confusion map must contain a YAML mapping",
            {"path": str(resolved), "parsed_type": type(raw).__name__},
        )

    folding = raw.get("folding") or {}
    if not isinstance(folding, dict):
        raise ValidationError("The folding section must be a mapping", {"path": str(resolved)})

    regional = raw.get("regional") or {}
    prefixes = tuple(str(value).upper() for value in regional.get("strip_prefixes", []))
    suffixes = tuple(str(value).upper() for value in regional.get("strip_suffixes", []))
    pairs = tuple((str(key).upper(), str(value).upper()) for key, value in folding.items())

    return pairs, (prefixes, suffixes)


class RegionalRules:
    """Optional prefix and suffix stripping for a plate format.

    Off by default. Stripping a prefix that is genuinely part of the plate
    destroys the identifier the whole system keys on, and does so silently.

    Args:
        strip_prefixes: Prefixes to remove, longest match first.
        strip_suffixes: Suffixes to remove, longest match first.
    """

    def __init__(
        self, strip_prefixes: tuple[str, ...] = (), strip_suffixes: tuple[str, ...] = ()
    ) -> None:
        self.strip_prefixes = tuple(sorted(strip_prefixes, key=len, reverse=True))
        self.strip_suffixes = tuple(sorted(strip_suffixes, key=len, reverse=True))

    @property
    def enabled(self) -> bool:
        """Return whether any stripping rule is configured."""
        return bool(self.strip_prefixes or self.strip_suffixes)

    def apply(self, plate: str) -> str:
        """Strip one configured prefix and one suffix from a normalized plate.

        Never strips the whole string: a plate that is nothing but its prefix is
        a read error, not an empty plate.

        Args:
            plate: The normalized plate.

        Returns:
            The stripped plate, or the input unchanged when no rule applies.
        """
        result = plate
        for prefix in self.strip_prefixes:
            if prefix and result.startswith(prefix) and len(result) > len(prefix):
                result = result[len(prefix) :]
                break
        for suffix in self.strip_suffixes:
            if suffix and result.endswith(suffix) and len(result) > len(suffix):
                result = result[: -len(suffix)]
                break
        return result


def regional_rules(path: str | None = None) -> RegionalRules:
    """Return the configured regional stripping rules.

    Args:
        path: Confusion-map file to read from.

    Returns:
        The rules, disabled unless the config declares any.
    """
    _, (prefixes, suffixes) = load_confusion_config(path)
    return RegionalRules(prefixes, suffixes)


def _to_ascii(raw: str) -> str:
    """Transliterate unambiguous Unicode to ASCII, rejecting the rest.

    Args:
        raw: The input string.

    Returns:
        The decomposed string with combining marks removed.

    Raises:
        ValidationError: If an alphanumeric character survives that is not
            ASCII. Homoglyphs from another script cannot be resolved without
            guessing, and a wrong guess creates a second identity for one
            vehicle.
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))

    offenders = sorted({char for char in stripped if char.isalnum() and not char.isascii()})
    if offenders:
        raise ValidationError(
            "Plate contains non-ASCII characters that cannot be transliterated "
            "unambiguously; resolving them by guesswork would create a second "
            "identity for one vehicle",
            {
                "characters": offenders,
                "code_points": [f"U+{ord(char):04X}" for char in offenders],
                "raw": raw,
            },
        )
    return stripped


def normalize_plate(raw: str | None, *, apply_regional: bool = False) -> str | None:
    """Normalize an OCR plate reading to its canonical comparison form.

    Applies exactly the global contract rules: uppercase, strip every
    non-alphanumeric character, and optionally strip a configured regional
    prefix or suffix.

    Idempotent by construction -- the output contains only uppercase ASCII
    alphanumerics, which the same pipeline leaves untouched.

    Args:
        raw: The OCR output, which may be ``None`` for an unreadable plate.
        apply_regional: Apply configured prefix/suffix stripping. Off by
            default, matching the contract.

    Returns:
        The normalized plate, or ``None`` when the input was ``None`` or
        contained no alphanumeric characters at all.

    Raises:
        ValidationError: If the input contains non-ASCII alphanumerics that
            cannot be transliterated unambiguously.
    """
    if raw is None:
        return None

    kept = "".join(char for char in _to_ascii(raw) if char.isalnum()).upper()
    if not kept:
        return None

    if apply_regional:
        kept = regional_rules().apply(kept)

    return kept or None
