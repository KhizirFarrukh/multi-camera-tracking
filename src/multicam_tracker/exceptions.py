"""Project-wide exception hierarchy.

Every error raised inside :mod:`multicam_tracker` derives from
:class:`MulticamTrackerError`, so a caller can catch the entire family with one
``except`` clause and still discriminate on the specific subclass when it cares.
Bare ``Exception`` is never raised (global contract,
``cross_cutting_requirements.errors``).

Each exception carries an optional ``context`` mapping. Context is the
structured detail a human needs to act on the error -- the offending field name,
the camera id, the threshold that was violated -- kept out of the message string
so it stays machine-readable for the structured logger.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "AuthorizationError",
    "ConfigurationError",
    "IngestError",
    "MatchingError",
    "MulticamTrackerError",
    "PathReconstructionError",
    "StorageError",
    "TopologyError",
    "ValidationError",
    "VisionError",
]


class MulticamTrackerError(Exception):
    """Root of the project exception hierarchy.

    Args:
        message: Human-readable description of what went wrong.
        context: Optional structured detail (field names, identifiers,
            thresholds) rendered into ``str()`` and available to log processors
            via the :attr:`context` attribute.
    """

    def __init__(self, message: str, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:
        """Render ``message``, appending context only when there is some.

        Returns:
            The message alone when :attr:`context` is empty -- no trailing
            separator, no empty parentheses -- otherwise the message followed by
            the context rendered as ``key=value`` pairs in sorted key order.
        """
        if not self.context:
            return self.message
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({rendered})"

    def __repr__(self) -> str:
        """Return an unambiguous representation including the context mapping."""
        return f"{type(self).__name__}(message={self.message!r}, context={self.context!r})"


class ConfigurationError(MulticamTrackerError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised when settings fail to load or validate, when a required setting has
    no value, or when a config file referenced by settings cannot be read.
    """


class ValidationError(MulticamTrackerError):
    """Input failed a domain validation rule.

    Distinct from :class:`pydantic.ValidationError`, which signals a schema
    violation at a model boundary. This one signals a rule the domain imposes on
    already-well-formed data -- a naive datetime where UTC is required, an
    embedding of the wrong dimension, a bbox with inverted coordinates.
    """


class StorageError(MulticamTrackerError):
    """A persistence operation failed (stage 03 onward)."""


class TopologyError(MulticamTrackerError):
    """The camera topology is invalid or a query against it cannot be answered.

    Covers unknown camera ids, malformed travel-time windows, and links that
    reference cameras absent from the graph (stage 04 onward).
    """


class MatchingError(MulticamTrackerError):
    """A matching operation failed (stages 06-07)."""


class IngestError(MulticamTrackerError):
    """A video source could not be opened, read, or sampled (stage 10 onward)."""


class VisionError(MulticamTrackerError):
    """Detection, OCR, or embedding extraction failed (stages 11-13)."""


class PathReconstructionError(MulticamTrackerError):
    """A trajectory could not be assembled from the supplied sightings (stage 08)."""


class AuthorizationError(MulticamTrackerError):
    """An actor attempted an operation they are not permitted to perform.

    Write and confirm operations require an authenticated actor identity for the
    audit trail (global contract, ``operational_and_ethical_constraints``).
    """
