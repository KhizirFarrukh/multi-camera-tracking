"""Enumerations shared across the domain.

All inherit from ``(str, Enum)`` so a member compares equal to its string value
and serializes as a plain string -- which matters at three boundaries: JSON
payloads, database columns, and the string enums named in the global contract's
``canonical_data_contracts``.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["MatchMethod", "ObjectClass", "ReviewStatus", "SourceType"]


class ObjectClass(str, Enum):
    """Class of the detected object, as reported by the detector (stage 11)."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    UNKNOWN = "unknown"


class MatchMethod(str, Enum):
    """How a sighting was linked to a target.

    The ordering here is the ordering of evidential strength: an exact plate
    read beats a fuzzy one, which beats visual appearance. ``MANUAL`` sits
    outside that scale -- it records a human decision, which is why stage 18
    treats it as terminal.
    """

    PLATE_EXACT = "plate_exact"
    PLATE_FUZZY = "plate_fuzzy"
    EMBEDDING = "embedding"
    MANUAL = "manual"


class ReviewStatus(str, Enum):
    """Adjudication state of a match candidate.

    ``AUTO_ACCEPTED`` means the match cleared the auto-accept threshold without
    a human looking at it. ``CONFIRMED`` means a human looked and agreed. The
    two are kept distinct because the audit trail must be able to say which one
    happened (global contract, ``operational_and_ethical_constraints``).
    """

    AUTO_ACCEPTED = "auto_accepted"
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class SourceType(str, Enum):
    """Origin of the video that produced a sighting."""

    RECORDED_FILE = "recorded_file"
    LIVE_STREAM = "live_stream"
