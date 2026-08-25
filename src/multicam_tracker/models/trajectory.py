"""Trajectory models: the ordered route a target took across cameras.

A trajectory is the system's final answer, so its invariants are the strictest
in the codebase. The structural checks below -- strict time ordering, hop count,
hop endpoints matching adjacent sightings -- exist because a malformed
trajectory does not look malformed. It looks like a confident, plausible route
that happens to be wrong, which is the single worst failure mode this system
has (global contract via stage 20: no failure mode may produce a confident wrong
trajectory).
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from multicam_tracker.models.base import MCTBaseModel, UtcDatetime
from multicam_tracker.models.sighting import Sighting

__all__ = [
    "CoverageGap",
    "TemporalCaveat",
    "TemporalIntegrity",
    "TemporalSeverity",
    "Trajectory",
    "TrajectoryHop",
]


class TrajectoryHop(MCTBaseModel):
    """A transition between two consecutive sightings of the same target."""

    from_sighting_id: str = Field(min_length=1)
    to_sighting_id: str = Field(min_length=1)
    from_camera_id: str = Field(min_length=1)
    to_camera_id: str = Field(min_length=1)
    elapsed_sec: float = Field(
        ge=0.0, description="Seconds between the two sightings; never negative by construction"
    )
    topology_plausible: bool = Field(
        description="Whether elapsed_sec fell inside the CameraLink travel-time window"
    )
    hop_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _endpoints_differ(self) -> TrajectoryHop:
        """Reject a hop from a sighting to itself.

        Returns:
            The validated instance.

        Raises:
            ValueError: If both endpoints name the same sighting.
        """
        if self.from_sighting_id == self.to_sighting_id:
            msg = (
                f"from_sighting_id and to_sighting_id must differ; both are "
                f"'{self.from_sighting_id}'"
            )
            raise ValueError(msg)
        return self


class CoverageGap(MCTBaseModel):
    """A stretch of a trajectory where the target went unobserved for too long.

    Populated by stage 08. A gap is not an error -- it is the system stating
    plainly that it does not know what happened between two sightings, which is
    more honest than interpolating a route through cameras that saw nothing.
    """

    from_sighting_id: str = Field(min_length=1)
    to_sighting_id: str = Field(min_length=1)
    elapsed_sec: float = Field(ge=0.0)
    expected_max_sec: float = Field(
        ge=0.0, description="Longest transit the topology considered plausible"
    )
    reason: str = Field(min_length=1, description="Why this stretch was flagged")


class TemporalSeverity(StrEnum):
    """How badly a timing problem undermines a route.

    ``StrEnum`` rather than the ``(str, Enum)`` used in
    :mod:`multicam_tracker.models.enums`: that form was specified verbatim for
    the contract's canonical entities, this enum is a stage 09 addition, and
    ``StrEnum`` is what the project's lint rules prefer.
    """

    INFO = "info"
    """Worth stating; the route stands."""

    WARNING = "warning"
    """The route may be wrong in ways the confidence score does not capture."""

    BLOCKING = "blocking"
    """Hop ordering itself is unreliable. A route built from this is not evidence."""


class TemporalCaveat(MCTBaseModel):
    """One reason the timestamps behind a route may not be comparable."""

    code: str = Field(
        min_length=1,
        description="stale_verification | drift_alert | low_reliability_source | offset_spread",
    )
    severity: TemporalSeverity
    camera_id: str | None = Field(
        default=None, description="The camera at fault, when the caveat names one"
    )
    detail: str = Field(min_length=1, description="Human-readable explanation for an operator")
    context: dict[str, Any] = Field(default_factory=dict)


class TemporalIntegrity(MCTBaseModel):
    """The verdict on whether a route's timestamps can be trusted.

    Attached to the trajectory rather than logged, because an operator reading a
    route needs to see the caveat beside it. A warning that lives only in a log
    file is a warning nobody acts on.
    """

    verified: bool = Field(
        description="True when every camera in the query passed with no caveats at all"
    )
    checked_camera_ids: list[str] = Field(default_factory=list)
    caveats: list[TemporalCaveat] = Field(default_factory=list)

    @property
    def is_blocking(self) -> bool:
        """Return whether any caveat makes the route unusable as evidence."""
        return any(caveat.severity is TemporalSeverity.BLOCKING for caveat in self.caveats)

    @property
    def worst_severity(self) -> TemporalSeverity | None:
        """Return the most serious caveat's severity, or ``None`` when clean."""
        order = [TemporalSeverity.BLOCKING, TemporalSeverity.WARNING, TemporalSeverity.INFO]
        for severity in order:
            if any(caveat.severity is severity for caveat in self.caveats):
                return severity
        return None

    def summary(self) -> str:
        """Return a one-line summary for a trajectory listing.

        Returns:
            A sentence an operator can act on.
        """
        if self.verified:
            return "clock verification is current for every camera on this route"
        return "; ".join(caveat.detail for caveat in self.caveats)


class Trajectory(MCTBaseModel):
    """The ordered sequence of a target's sightings plus the hops between them."""

    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str = Field(min_length=1)
    sightings: list[Sighting] = Field(
        min_length=1, description="Sorted strictly ascending by timestamp_utc"
    )
    hops: list[TrajectoryHop] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    start_time_utc: UtcDatetime
    end_time_utc: UtcDatetime
    gaps: list[CoverageGap] = Field(default_factory=list)
    requires_recomputation: bool = Field(
        default=False,
        description=(
            "Set when a camera's clock offset changed after this route was assembled. "
            "The sightings underneath it have moved, so the conclusion no longer follows "
            "from the evidence -- and an operator cannot be expected to work out which "
            "routes were affected by hand."
        ),
    )
    temporal_integrity: TemporalIntegrity | None = Field(
        default=None,
        description=(
            "The stage 09 verdict on whether these timestamps are comparable. None when "
            "the route was assembled without checking, which is itself worth knowing: "
            "an unchecked route is not the same as a verified one."
        ),
    )

    @model_validator(mode="after")
    def _sightings_are_strictly_ascending(self) -> Trajectory:
        """Require strictly increasing sighting timestamps.

        Strict, not merely non-decreasing: two sightings sharing a timestamp
        have no defined order, so the hop between them would have an arbitrary
        direction and a zero elapsed time that no travel-time window accepts.

        Returns:
            The validated instance.

        Raises:
            ValueError: If any sighting is not strictly later than its
                predecessor. The message names the offending index.
        """
        for index in range(1, len(self.sightings)):
            previous = self.sightings[index - 1]
            current = self.sightings[index]
            if current.timestamp_utc <= previous.timestamp_utc:
                is_tie = current.timestamp_utc == previous.timestamp_utc
                relation = "equal to" if is_tie else "before"
                msg = (
                    f"sightings must be sorted strictly ascending by timestamp_utc; "
                    f"sightings[{index}] ({current.timestamp_utc.isoformat()}) is "
                    f"{relation} sightings[{index - 1}] "
                    f"({previous.timestamp_utc.isoformat()})"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _hops_bridge_consecutive_sightings(self) -> Trajectory:
        """Require exactly one hop between each adjacent pair of sightings.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the hop count is wrong, or if any hop's endpoints do
                not match the sightings it is supposed to bridge. A hop pointing
                at the wrong sighting would attribute an elapsed time to the
                wrong camera pair, and the topology check applied to it would be
                meaningless.
        """
        expected_hop_count = max(0, len(self.sightings) - 1)
        if len(self.hops) != expected_hop_count:
            msg = (
                f"expected {expected_hop_count} hops for {len(self.sightings)} sightings; "
                f"got {len(self.hops)}"
            )
            raise ValueError(msg)

        for index, hop in enumerate(self.hops):
            origin = self.sightings[index]
            destination = self.sightings[index + 1]

            if hop.from_sighting_id != origin.sighting_id:
                msg = (
                    f"hops[{index}].from_sighting_id ('{hop.from_sighting_id}') must "
                    f"reference sightings[{index}] ('{origin.sighting_id}')"
                )
                raise ValueError(msg)

            if hop.to_sighting_id != destination.sighting_id:
                msg = (
                    f"hops[{index}].to_sighting_id ('{hop.to_sighting_id}') must "
                    f"reference sightings[{index + 1}] ('{destination.sighting_id}')"
                )
                raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def _bounds_match_the_endpoints(self) -> Trajectory:
        """Require the declared time bounds to match the first and last sightings.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``start_time_utc`` or ``end_time_utc`` disagrees with
                the corresponding sighting. The bounds are what the UI renders
                on the timeline axis; if they drift from the data, the plot
                misrepresents the evidence.
        """
        first = self.sightings[0].timestamp_utc
        last = self.sightings[-1].timestamp_utc

        if self.start_time_utc != first:
            msg = (
                f"start_time_utc ({self.start_time_utc.isoformat()}) must equal the first "
                f"sighting's timestamp_utc ({first.isoformat()})"
            )
            raise ValueError(msg)

        if self.end_time_utc != last:
            msg = (
                f"end_time_utc ({self.end_time_utc.isoformat()}) must equal the last "
                f"sighting's timestamp_utc ({last.isoformat()})"
            )
            raise ValueError(msg)

        return self

    @property
    def duration_sec(self) -> float:
        """Return the elapsed seconds from the first sighting to the last.

        Returns:
            ``0.0`` for a single-sighting trajectory.
        """
        return (self.end_time_utc - self.start_time_utc).total_seconds()

    @property
    def camera_sequence(self) -> list[str]:
        """Return the camera ids in observation order, repeats included.

        Repeats are kept deliberately: a vehicle passing the same camera twice
        is a meaningful pattern (a loop, a return trip), and collapsing it would
        erase that.

        Returns:
            One camera id per sighting, in chronological order.
        """
        return [sighting.camera_id for sighting in self.sightings]

    @property
    def distinct_camera_count(self) -> int:
        """Return how many different cameras observed the target."""
        return len(set(self.camera_sequence))
