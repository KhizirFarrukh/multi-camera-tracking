"""The ground-truth artifact: what actually happened.

Everything stages 06-08 measure themselves against is here. Precision and recall
are only meaningful because this file records, for every synthetic sighting,
which vehicle really produced it -- something no amount of real footage can tell
you with certainty.

It also records every corruption and every injection, so a test can assert on a
*specific* failure mode ("the degraded scenario really did produce read failures")
rather than on an aggregate that could be right for the wrong reason.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from multicam_tracker.models.base import MCTBaseModel, UtcDatetime

__all__ = ["CorruptionEvent", "GroundTruth", "InjectionEvent", "VehicleTruth"]


class CorruptionEvent(MCTBaseModel):
    """One plate reading that OCR got wrong, and how."""

    sighting_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, description="substitution | dropout | read_failure | spurious")
    true_plate: str = Field(min_length=1)
    observed_plate: str | None = Field(default=None, description="None for a complete read failure")
    severity: float = Field(
        ge=0.0,
        description="How badly mangled, in edit operations. Drives the reported "
        "confidence so low-quality reads carry low confidence, as real OCR does.",
    )
    detail: str = ""


class InjectionEvent(MCTBaseModel):
    """One deliberate adversarial event."""

    kind: str = Field(
        min_length=1,
        description="clock_drift | outage | duplicate | plate_clone | boundary_hop",
    )
    detail: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class VehicleTruth(MCTBaseModel):
    """Everything true about one synthetic vehicle."""

    vehicle_id: str = Field(min_length=1)
    true_plate: str = Field(min_length=1)
    is_target: bool = False
    is_decoy: bool = False
    is_hard_negative: bool = False
    sighting_ids: list[str] = Field(
        default_factory=list, description="This vehicle's sightings, in true time order"
    )


class GroundTruth(MCTBaseModel):
    """The complete answer key for one generated dataset."""

    scenario_name: str = Field(min_length=1)
    seed: int
    generated_at_utc: UtcDatetime
    vehicles: list[VehicleTruth] = Field(default_factory=list)
    sighting_to_vehicle: dict[str, str] = Field(default_factory=dict)
    true_timestamps: dict[str, UtcDatetime] = Field(
        default_factory=dict,
        description=(
            "The instant a sighting really happened, which differs from the "
            "recorded timestamp wherever clock drift was injected. Stage 09 is "
            "scored against this."
        ),
    )
    corruptions: list[CorruptionEvent] = Field(default_factory=list)
    injections: list[InjectionEvent] = Field(default_factory=list)

    def vehicle_for(self, sighting_id: str) -> str | None:
        """Return the vehicle that produced a sighting.

        Args:
            sighting_id: The sighting to look up.

        Returns:
            Its vehicle id, or ``None`` if the sighting is unknown.
        """
        return self.sighting_to_vehicle.get(sighting_id)

    def truth_for(self, vehicle_id: str) -> VehicleTruth | None:
        """Return one vehicle's truth record.

        Args:
            vehicle_id: The vehicle to look up.

        Returns:
            Its record, or ``None``.
        """
        return next((entry for entry in self.vehicles if entry.vehicle_id == vehicle_id), None)

    @property
    def target(self) -> VehicleTruth | None:
        """Return the search target's truth record, if the scenario has one."""
        return next((entry for entry in self.vehicles if entry.is_target), None)

    def corruptions_of_kind(self, kind: str) -> list[CorruptionEvent]:
        """Return every corruption of one failure mode.

        Args:
            kind: The failure mode to filter on.

        Returns:
            The matching events.
        """
        return [event for event in self.corruptions if event.kind == kind]

    def injections_of_kind(self, kind: str) -> list[InjectionEvent]:
        """Return every injection of one kind.

        Args:
            kind: The injection kind to filter on.

        Returns:
            The matching events.
        """
        return [event for event in self.injections if event.kind == kind]

    @property
    def read_failure_rate(self) -> float:
        """Return the share of sightings whose plate was unreadable.

        Returns:
            A fraction in ``[0, 1]``, or ``0.0`` for an empty dataset. The
            'degraded' regression scenario asserts on this directly.
        """
        if not self.sighting_to_vehicle:
            return 0.0
        return len(self.corruptions_of_kind("read_failure")) / len(self.sighting_to_vehicle)

    def write_json(self, path: Path | str) -> Path:
        """Write the ground truth to a JSON file.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return resolved

    @classmethod
    def read_json(cls, path: Path | str) -> GroundTruth:
        """Read a ground truth back from JSON.

        Args:
            path: The file to read.

        Returns:
            The reloaded object, equal to the one written.
        """
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def order_by_time(sighting_ids: list[str], timestamps: dict[str, datetime]) -> list[str]:
    """Sort sighting ids by their true instant.

    Args:
        sighting_ids: The ids to order.
        timestamps: True instants, keyed by sighting id.

    Returns:
        The ids in ascending time order, ties broken by id so the result is
        deterministic.
    """
    return sorted(sighting_ids, key=lambda identifier: (timestamps[identifier], identifier))
