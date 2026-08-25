"""The gate: may these timestamps be compared at all?

Called before path reconstruction. The premise of the entire system is that
timestamps from independent cameras are comparable, and this is the one place
that premise is checked rather than assumed.

Four things can undermine it, and they differ in how badly:

``stale_verification``
    A camera's clock has not been checked recently enough. A warning, not a
    refusal: an unverified clock is usually a correct clock, and refusing to
    answer would make the system unusable in exactly the deployments that need
    it most.

``drift_alert``
    Drift detection has an open alert. A warning by default and a refusal by
    configuration, because an operator who knows a camera is 45 seconds fast can
    still read a route usefully -- as long as the route says so.

``low_reliability_source``
    A timestamp came from a filename or an OCR overlay. Worth stating on the
    trajectory, because it changes what the route is worth as evidence.

``offset_spread``
    **Blocking.** Known offsets differ by more than the smallest travel time in
    the topology. At that point two sightings can be reordered by the error
    alone, so the hop ordering -- the thing a trajectory *is* -- is not
    determined by the data. There is no useful route to return, and returning
    one anyway is the failure the whole stage exists to prevent.

The verdict is attached to the trajectory rather than logged. A caveat in a log
file is a caveat nobody acts on; a caveat beside the route is one an operator
reads at the moment it matters.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta

from multicam_tracker.clock import Clock
from multicam_tracker.exceptions import TemporalIntegrityError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import (
    TemporalCaveat,
    TemporalIntegrity,
    TemporalSeverity,
)
from multicam_tracker.timesync.drift import DriftAlert
from multicam_tracker.timesync.sources import ReliabilityTier
from multicam_tracker.topology import Topology

__all__ = ["CameraTimeStatus", "assert_temporal_integrity", "check_temporal_integrity"]

logger = get_logger(__name__)

_MS_PER_SECOND = 1000.0


class CameraTimeStatus:
    """What is known about one camera's clock.

    Args:
        camera_id: The camera.
        offset_ms: The correction currently applied to its timestamps.
        verified_at_utc: When its clock was last checked, or ``None`` if never.
        tier: Reliability of its timestamp source.
        drift_alert: An open drift alert, if any.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        offset_ms: int = 0,
        verified_at_utc: datetime | None = None,
        tier: ReliabilityTier = ReliabilityTier.HIGH,
        drift_alert: DriftAlert | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.offset_ms = offset_ms
        self.verified_at_utc = verified_at_utc
        self.tier = tier
        self.drift_alert = drift_alert

    def staleness_hours(self, now_utc: datetime) -> float | None:
        """Return how long since this camera's clock was verified.

        Args:
            now_utc: The current instant.

        Returns:
            Hours since verification, or ``None`` when it never was.
        """
        if self.verified_at_utc is None:
            return None
        return (now_utc - self.verified_at_utc).total_seconds() / 3600.0


def _smallest_travel_time_sec(topology: Topology, camera_ids: list[str]) -> float | None:
    """Return the shortest minimum travel time among the queried cameras.

    This is the scale at which a clock error starts reordering hops: two
    sightings closer together than the offset spread can swap places, and the
    route becomes a claim the data does not support.

    Args:
        topology: The camera graph.
        camera_ids: Cameras in the query.

    Returns:
        The smallest ``min_travel_time_sec`` on any link between them, or
        ``None`` when no such link exists.
    """
    relevant = set(camera_ids)
    times = [
        link.min_travel_time_sec
        for camera_id in camera_ids
        if camera_id in topology
        for link in topology.neighbors(camera_id)
        if link.to_camera_id in relevant
    ]
    return min(times) if times else None


def check_temporal_integrity(
    camera_ids: list[str],
    statuses: Mapping[str, CameraTimeStatus],
    topology: Topology,
    clock: Clock,
    *,
    staleness_hours: float | None = None,
) -> TemporalIntegrity:
    """Judge whether a set of cameras' timestamps may be compared.

    Args:
        camera_ids: Cameras the query will touch.
        statuses: What is known about each camera's clock. A camera absent from
            the mapping is treated as never verified, which is the honest
            reading of "we have no record".
        topology: The camera graph, for the smallest travel time.
        clock: Supplies the current instant.
        staleness_hours: How long a verification stays good. Defaults to config.

    Returns:
        The verdict, with one caveat per problem found.
    """
    from multicam_tracker.config import get_settings

    settings = get_settings().timesync
    stale_after = (
        staleness_hours if staleness_hours is not None else settings.verification_staleness_hours
    )
    now = clock.now_utc()

    caveats: list[TemporalCaveat] = []

    for camera_id in camera_ids:
        status = statuses.get(camera_id) or CameraTimeStatus(camera_id)
        staleness = status.staleness_hours(now)

        if staleness is None:
            caveats.append(
                TemporalCaveat(
                    code="stale_verification",
                    severity=TemporalSeverity.WARNING,
                    camera_id=camera_id,
                    detail=(
                        f"{camera_id} has no record of its clock ever being verified, so "
                        f"nothing rules out an offset large enough to reorder its hops"
                    ),
                )
            )
        elif staleness > stale_after:
            caveats.append(
                TemporalCaveat(
                    code="stale_verification",
                    severity=TemporalSeverity.WARNING,
                    camera_id=camera_id,
                    detail=(
                        f"{camera_id} was last clock-verified {staleness:.0f} hours ago, "
                        f"past the {stale_after:.0f}-hour staleness limit"
                    ),
                    context={"staleness_hours": round(staleness, 1)},
                )
            )

        if status.drift_alert is not None:
            caveats.append(
                TemporalCaveat(
                    code="drift_alert",
                    severity=(
                        TemporalSeverity.BLOCKING
                        if settings.block_on_drift_alert
                        else TemporalSeverity.WARNING
                    ),
                    camera_id=camera_id,
                    detail=status.drift_alert.describe(),
                    context={
                        "kind": status.drift_alert.kind,
                        "magnitude_ms": round(status.drift_alert.magnitude_ms, 1),
                        "rate_ms_per_hour": round(status.drift_alert.rate_ms_per_hour, 1),
                    },
                )
            )

        if status.tier is not ReliabilityTier.HIGH:
            caveats.append(
                TemporalCaveat(
                    code="low_reliability_source",
                    severity=(
                        TemporalSeverity.WARNING
                        if status.tier is ReliabilityTier.LOW
                        else TemporalSeverity.INFO
                    ),
                    camera_id=camera_id,
                    detail=(
                        f"{camera_id} timestamps come from a {status.tier.value}-reliability "
                        f"source, which can be wrong without any signal that it is"
                    ),
                    context={"tier": status.tier.value},
                )
            )

    caveats.extend(_offset_spread_caveats(camera_ids, statuses, topology))

    verdict = TemporalIntegrity(
        verified=not caveats,
        checked_camera_ids=list(camera_ids),
        caveats=caveats,
    )

    if caveats:
        logger.warning(
            "temporal_integrity_caveats",
            camera_ids=list(camera_ids),
            codes=[caveat.code for caveat in caveats],
            blocking=verdict.is_blocking,
        )
    return verdict


def _offset_spread_caveats(
    camera_ids: list[str],
    statuses: Mapping[str, CameraTimeStatus],
    topology: Topology,
) -> list[TemporalCaveat]:
    """Check whether known offsets are large enough to reorder hops.

    Args:
        camera_ids: Cameras in the query.
        statuses: What is known about each camera's clock.
        topology: The camera graph.

    Returns:
        A single blocking caveat when the spread exceeds the smallest travel
        time, or an empty list.
    """
    offsets = [
        (statuses[camera_id].offset_ms if camera_id in statuses else 0) for camera_id in camera_ids
    ]
    if len(offsets) < 2:
        return []

    spread_ms = max(offsets) - min(offsets)
    smallest_travel_sec = _smallest_travel_time_sec(topology, camera_ids)
    if smallest_travel_sec is None or spread_ms <= smallest_travel_sec * _MS_PER_SECOND:
        return []

    return [
        TemporalCaveat(
            code="offset_spread",
            severity=TemporalSeverity.BLOCKING,
            camera_id=None,
            detail=(
                f"clock offsets across these cameras span {spread_ms / _MS_PER_SECOND:.1f}s, "
                f"more than the {smallest_travel_sec:.1f}s shortest transit between them. "
                f"Two sightings can be reordered by the clock error alone, so any hop "
                f"ordering built from this is not supported by the data"
            ),
            context={
                "offset_spread_ms": spread_ms,
                "smallest_travel_time_sec": smallest_travel_sec,
            },
        )
    ]


def assert_temporal_integrity(
    camera_ids: list[str],
    statuses: Mapping[str, CameraTimeStatus],
    topology: Topology,
    clock: Clock,
    *,
    staleness_hours: float | None = None,
) -> TemporalIntegrity:
    """Check integrity and raise when the result makes a route meaningless.

    Args:
        camera_ids: Cameras the query will touch.
        statuses: What is known about each camera's clock.
        topology: The camera graph.
        clock: Supplies the current instant.
        staleness_hours: How long a verification stays good. Defaults to config.

    Returns:
        The verdict, when it is one a caller can proceed on. Warnings are
        returned rather than raised: they belong on the trajectory, not in an
        exception handler.

    Raises:
        TemporalIntegrityError: If any caveat is blocking. The message carries
            every blocking detail, since an operator fixing this needs all of
            them rather than the first.
    """
    verdict = check_temporal_integrity(
        camera_ids, statuses, topology, clock, staleness_hours=staleness_hours
    )

    blocking = [
        caveat for caveat in verdict.caveats if caveat.severity is TemporalSeverity.BLOCKING
    ]
    if blocking:
        raise TemporalIntegrityError(
            "Timestamps across these cameras are not comparable: "
            + "; ".join(caveat.detail for caveat in blocking),
            {
                "camera_ids": list(camera_ids),
                "codes": [caveat.code for caveat in blocking],
            },
        )

    return verdict


def stale_after(verified_at_utc: datetime, staleness_hours: float) -> datetime:
    """Return when a verification stops counting as current.

    Args:
        verified_at_utc: When the clock was verified.
        staleness_hours: How long a verification stays good.

    Returns:
        The instant at which it goes stale.
    """
    return verified_at_utc + timedelta(hours=staleness_hours)
