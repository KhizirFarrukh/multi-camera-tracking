"""Applying a camera's clock offset, once, at one boundary.

The Sighting contract carries three fields that must always agree:
``raw_timestamp`` as the source reported it, ``clock_offset_applied_ms``, and
``timestamp_utc`` as the corrected instant. Correction happens **here**, at
ingestion, and nowhere else. Scattering it through downstream code is how a
value gets corrected twice -- and a double-corrected timestamp is not obviously
wrong, it is merely somewhere else.

Idempotency follows from the same discipline: applying correction to an already
corrected sighting recomputes from ``raw_timestamp`` rather than shifting
``timestamp_utc`` again, so running ingestion twice over the same record cannot
move it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from multicam_tracker.clock import Clock, ensure_utc
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import Sighting

__all__ = [
    "OffsetChange",
    "apply_offset",
    "apply_offset_change",
    "correct_sighting",
    "recompute_offsets",
]

logger = get_logger(__name__)


def apply_offset(raw_timestamp: datetime, offset_ms: int) -> datetime:
    """Return the corrected instant for one raw timestamp.

    Args:
        raw_timestamp: The instant as the source reported it.
        offset_ms: Milliseconds to add. Positive when the camera's clock runs
            slow, negative when it runs fast.

    Returns:
        The corrected instant in UTC.

    Raises:
        ValidationError: If ``raw_timestamp`` is naive.
    """
    return ensure_utc(raw_timestamp, field_name="raw_timestamp") + timedelta(milliseconds=offset_ms)


def correct_sighting(sighting: Sighting, offset_ms: int) -> Sighting:
    """Return the sighting with its camera's clock offset applied.

    Recomputes from ``raw_timestamp`` rather than adjusting ``timestamp_utc``,
    which is what makes running this twice harmless.

    Args:
        sighting: The sighting to correct.
        offset_ms: The camera's offset in milliseconds.

    Returns:
        A new sighting whose three timestamp fields agree.
    """
    return sighting.model_copy(
        update={
            "timestamp_utc": apply_offset(sighting.raw_timestamp, offset_ms),
            "clock_offset_applied_ms": offset_ms,
        }
    )


@dataclass(frozen=True)
class OffsetChange:
    """The record of one camera's offset being changed.

    Held as its own type because the change has to reach the audit log with both
    values: "the offset is now -45000 ms" is not answerable after the fact
    without knowing what it was before.
    """

    camera_id: str
    previous_offset_ms: int
    new_offset_ms: int
    sightings_updated: int
    changed_at_utc: datetime
    reason: str = ""
    affected_trajectory_ids: tuple[str, ...] = ()
    """Routes assembled from the sightings that just moved.

    Recorded because correcting a clock does not merely rewrite rows -- it
    invalidates the conclusions drawn from them, and an operator cannot be
    expected to work out which ones by hand."""

    @property
    def shift_ms(self) -> int:
        """Return how far every affected timestamp moved."""
        return self.new_offset_ms - self.previous_offset_ms

    def audit_entry(self) -> dict[str, object]:
        """Return the change as an audit-log payload.

        Returns:
            A JSON-native mapping naming both values and the blast radius.
        """
        return {
            "event": "camera_clock_offset_changed",
            "camera_id": self.camera_id,
            "previous_offset_ms": self.previous_offset_ms,
            "new_offset_ms": self.new_offset_ms,
            "shift_ms": self.shift_ms,
            "sightings_updated": self.sightings_updated,
            "changed_at_utc": self.changed_at_utc.isoformat(),
            "reason": self.reason,
            "affected_trajectory_ids": list(self.affected_trajectory_ids),
        }


def recompute_offsets(
    camera_id: str,
    new_offset_ms: int,
    sightings: list[Sighting],
    clock: Clock,
    *,
    previous_offset_ms: int | None = None,
    reason: str = "",
) -> tuple[list[Sighting], OffsetChange]:
    """Recompute every stored sighting for one camera under a new offset.

    Pure: the caller owns the transaction. This returns the rewritten sightings
    and the change record, and the repository writes both or neither. Doing the
    persistence here would spread transaction control across two layers and make
    the atomicity guarantee impossible to state.

    Args:
        camera_id: The camera whose clock is being corrected.
        new_offset_ms: The offset to apply from now on.
        sightings: Every stored sighting for that camera. Sightings from other
            cameras are returned untouched, so a caller may pass a mixed batch.
        clock: Supplies the change timestamp.
        previous_offset_ms: The offset in force before. Defaults to whatever the
            sightings themselves record, which is the truth for data already
            written.
        reason: Why the offset changed, for the audit trail.

    Returns:
        ``(rewritten_sightings, change)``.
    """
    affected = [sighting for sighting in sightings if sighting.camera_id == camera_id]
    resolved_previous = (
        previous_offset_ms
        if previous_offset_ms is not None
        else (affected[0].clock_offset_applied_ms if affected else 0)
    )

    rewritten = [
        correct_sighting(sighting, new_offset_ms) if sighting.camera_id == camera_id else sighting
        for sighting in sightings
    ]

    change = OffsetChange(
        camera_id=camera_id,
        previous_offset_ms=resolved_previous,
        new_offset_ms=new_offset_ms,
        sightings_updated=len(affected),
        changed_at_utc=clock.now_utc(),
        reason=reason,
    )

    entry = change.audit_entry()
    # The payload names its own event so an audit consumer reading stored
    # entries sees the same field as one reading the log stream; structlog takes
    # the event positionally, so it is popped rather than passed twice.
    logger.info(str(entry.pop("event")), **entry)
    return rewritten, change


def apply_offset_change(
    sighting_repo: Any,
    trajectory_repo: Any,
    camera_id: str,
    new_offset_ms: int,
    clock: Clock,
    *,
    previous_offset_ms: int = 0,
    reason: str = "",
) -> OffsetChange:
    """Correct one camera's clock across everything already stored.

    Three things happen together, inside the caller's transaction: the camera's
    sightings are recomputed, the trajectories built from them are flagged as
    needing recomputation, and the change is recorded. All three or none -- a
    correction that moved the sightings but not the flag would leave routes that
    silently no longer follow from their evidence.

    The transaction itself belongs to the caller. Committing here would take a
    decision that is not this function's to make, and would make it impossible
    to correct several cameras atomically.

    Args:
        sighting_repo: Repository whose sightings are rewritten.
        trajectory_repo: Repository whose trajectories are flagged.
        camera_id: The camera whose clock is being corrected.
        new_offset_ms: The offset to apply from now on.
        clock: Supplies the change timestamp.
        previous_offset_ms: The offset in force before, for the audit record.
        reason: Why the offset changed.

    Returns:
        The change record, naming both offsets and everything affected.

    Raises:
        StorageError: If either write fails.
    """
    rewritten = sighting_repo.apply_clock_offset(camera_id, new_offset_ms)
    flagged = trajectory_repo.flag_for_recomputation(camera_id)

    change = OffsetChange(
        camera_id=camera_id,
        previous_offset_ms=previous_offset_ms,
        new_offset_ms=new_offset_ms,
        sightings_updated=rewritten,
        changed_at_utc=clock.now_utc(),
        reason=reason,
        affected_trajectory_ids=tuple(flagged),
    )

    entry = change.audit_entry()
    logger.info(str(entry.pop("event")), **entry)
    return change
