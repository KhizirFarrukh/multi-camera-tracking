"""Plate conflict detection: spotting a plate that cannot belong to one vehicle.

When two sightings share a plate but sit on distant cameras closer together in
time than the fastest possible drive between them, one vehicle cannot account
for both. Something else is true: the plate is cloned, one read is wrong, or a
camera clock has drifted.

Whichever it is, the answer is **not** to build a trajectory. Left undetected,
path reconstruction would happily emit a route in which a vehicle teleports --
a confident, plausible-looking, wrong answer, which is the failure mode this
system exists to avoid. So the conflict is surfaced for a human instead.

Two sightings on the *same* camera are never a conflict: a vehicle lingering in
frame or circling back is ordinary, and stage 04's ``is_same_pass`` is the tool
for that case.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime

from multicam_tracker.models import Sighting
from multicam_tracker.topology import Topology, shortest_transit_sec

__all__ = ["PlateConflict", "detect_plate_conflicts"]


@dataclass(frozen=True)
class PlateConflict:
    """Two sightings of one plate that no single vehicle could have produced."""

    plate: str
    earlier_sighting_id: str
    later_sighting_id: str
    from_camera_id: str
    to_camera_id: str
    elapsed_sec: float
    minimum_transit_sec: float
    """Fastest transit the topology permits between the two cameras, over any
    path. ``inf`` when no route exists at all."""

    @property
    def deficit_sec(self) -> float:
        """Return how many seconds short of possible the observed gap was."""
        if self.minimum_transit_sec == float("inf"):
            return float("inf")
        return self.minimum_transit_sec - self.elapsed_sec

    def describe(self) -> str:
        """Explain the conflict in one sentence.

        Returns:
            A message naming both cameras, the observed gap, and the minimum
            possible one -- enough for an operator to judge whether to suspect a
            cloned plate, a misread, or a drifting clock.
        """
        if self.minimum_transit_sec == float("inf"):
            return (
                f"plate {self.plate} was seen on {self.from_camera_id} and "
                f"{self.to_camera_id} {self.elapsed_sec:.0f}s apart, but the topology "
                f"declares no route between them at all"
            )
        return (
            f"plate {self.plate} was seen on {self.from_camera_id} and "
            f"{self.to_camera_id} {self.elapsed_sec:.0f}s apart, {self.deficit_sec:.0f}s "
            f"faster than the {self.minimum_transit_sec:.0f}s minimum possible transit; "
            f"one vehicle cannot account for both"
        )


def _ordered(sightings: list[Sighting]) -> list[Sighting]:
    """Sort sightings chronologically, ties broken by id.

    Args:
        sightings: The sightings to order.

    Returns:
        A new sorted list.
    """
    return sorted(sightings, key=lambda item: (item.timestamp_utc, item.sighting_id))


def _elapsed(earlier: datetime, later: datetime) -> float:
    """Return the seconds between two instants.

    Args:
        earlier: First instant.
        later: Second instant.

    Returns:
        The gap in seconds.
    """
    return (later - earlier).total_seconds()


def detect_plate_conflicts(
    plate: str,
    sightings: list[Sighting],
    topology: Topology,
    *,
    include_unlinked: bool = False,
) -> list[PlateConflict]:
    """Find pairs of sightings that one vehicle could not have produced.

    Every pair is examined, not just consecutive ones. A cloned plate driving a
    parallel route can produce a sequence in which each neighbouring pair looks
    fine while a wider pair is impossible.

    Args:
        plate: The plate these sightings share, used in the report.
        sightings: Sightings carrying that plate.
        topology: Graph supplying the minimum possible transit times.
        include_unlinked: Also report pairs with no route between them at all.
            Off by default: an undeclared route is far more often a surveying
            gap than a cloned plate, and reporting every such pair would bury
            the real conflicts.

    Returns:
        One conflict per impossible pair, ordered by the earlier sighting.
    """
    ordered = _ordered(sightings)
    conflicts: list[PlateConflict] = []
    known = set(topology.camera_ids)

    for earlier, later in itertools.combinations(ordered, 2):
        if earlier.camera_id == later.camera_id:
            continue
        if earlier.camera_id not in known or later.camera_id not in known:
            continue

        elapsed = _elapsed(earlier.timestamp_utc, later.timestamp_utc)
        minimum = shortest_transit_sec(topology, earlier.camera_id, later.camera_id)

        if minimum is None:
            if include_unlinked:
                conflicts.append(
                    PlateConflict(
                        plate=plate,
                        earlier_sighting_id=earlier.sighting_id,
                        later_sighting_id=later.sighting_id,
                        from_camera_id=earlier.camera_id,
                        to_camera_id=later.camera_id,
                        elapsed_sec=elapsed,
                        minimum_transit_sec=float("inf"),
                    )
                )
            continue

        if elapsed < minimum:
            conflicts.append(
                PlateConflict(
                    plate=plate,
                    earlier_sighting_id=earlier.sighting_id,
                    later_sighting_id=later.sighting_id,
                    from_camera_id=earlier.camera_id,
                    to_camera_id=later.camera_id,
                    elapsed_sec=elapsed,
                    minimum_transit_sec=minimum,
                )
            )

    return conflicts
