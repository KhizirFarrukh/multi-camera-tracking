"""Declarative description of a synthetic dataset.

A scenario is data, not code: it serializes to YAML, gets committed, and becomes
a regression fixture. Stages 06-08 tune thresholds against these files and a
later change that quietly degrades matching shows up as a scenario whose
measured precision moved.

Everything that affects generation lives here, because the determinism guarantee
is "same scenario file plus same seed produces identical output". A parameter
read from anywhere else would break that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.models import ObjectClass
from multicam_tracker.models.base import MCTBaseModel, UtcDatetime

__all__ = [
    "AdversarialSpec",
    "ClockDriftInjection",
    "EmbeddingProfile",
    "NoiseProfile",
    "OutageInjection",
    "Scenario",
    "TrafficSpec",
    "VehicleSpec",
    "load_scenario",
    "save_scenario",
]


class NoiseProfile(MCTBaseModel):
    """How badly OCR mangles plate readings.

    The weights are relative, not probabilities: a corruption event picks one
    failure mode in proportion to them. Real ALPR fails in all four ways, and a
    generator that only ever substitutes characters would make fuzzy matching
    look far better than it is.
    """

    corruption_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Chance that a given plate reading is corrupted at all",
    )
    substitution_weight: float = Field(default=6.0, ge=0.0)
    dropout_weight: float = Field(default=2.0, ge=0.0)
    read_failure_weight: float = Field(default=1.5, ge=0.0)
    spurious_weight: float = Field(default=1.0, ge=0.0)

    max_substitutions: int = Field(
        default=2,
        ge=1,
        description="Cap on substituted characters, bounding the edit distance",
    )
    max_dropped: int = Field(default=2, ge=1)
    max_spurious: int = Field(default=2, ge=1)

    clean_confidence_min: float = Field(default=0.88, ge=0.0, le=1.0)
    clean_confidence_max: float = Field(default=0.99, ge=0.0, le=1.0)
    confidence_penalty_per_severity: float = Field(
        default=0.22,
        ge=0.0,
        description=(
            "Confidence lost per unit of corruption severity. Real OCR reports "
            "low confidence on the reads it gets wrong; a generator whose "
            "confidences did not track quality would let matching cheat."
        ),
    )
    min_confidence: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_and_bounds_are_usable(self) -> NoiseProfile:
        """Reject an unusable profile.

        Returns:
            The validated profile.

        Raises:
            ValueError: If every weight is zero while corruption is enabled, or
                the clean-confidence range is inverted.
        """
        total = (
            self.substitution_weight
            + self.dropout_weight
            + self.read_failure_weight
            + self.spurious_weight
        )
        if self.corruption_probability > 0 and total <= 0:
            msg = "corruption_probability > 0 requires at least one non-zero failure-mode weight"
            raise ValueError(msg)
        if self.clean_confidence_max < self.clean_confidence_min:
            msg = "clean_confidence_max must be >= clean_confidence_min"
            raise ValueError(msg)
        return self

    @property
    def total_weight(self) -> float:
        """Return the sum of the failure-mode weights."""
        return (
            self.substitution_weight
            + self.dropout_weight
            + self.read_failure_weight
            + self.spurious_weight
        )

    @property
    def max_edit_distance(self) -> int:
        """Return the largest edit distance any corruption can produce."""
        return max(self.max_substitutions, self.max_dropped, self.max_spurious)


class EmbeddingProfile(MCTBaseModel):
    """How synthetic re-id vectors are drawn.

    Args are chosen so the two similarity distributions the matcher has to
    separate -- same vehicle across cameras, different vehicles -- are both
    controllable, and can be made to overlap on purpose.
    """

    dimension: int = Field(
        default=512,
        ge=8,
        description=(
            "Must equal the configured vision.embedding_dim: the Sighting model "
            "rejects any other width, so a mismatch here would fail at the first "
            "generated sighting."
        ),
    )
    intra_class_sigma: float = Field(
        default=0.25,
        ge=0.0,
        description=(
            "Noise added to a vehicle's base vector per sighting, *relative* to a "
            "unit vector's component scale. Higher means the same vehicle looks "
            "less like itself across cameras. Dimension-independent, so a "
            "scenario does not need retuning when the embedding width changes."
        ),
    )
    hard_negative_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of decoys drawn near the target rather than at random",
    )
    hard_negative_sigma: float = Field(
        default=0.15,
        gt=0.0,
        description=(
            "Spread of hard negatives around the target's base vector, on the same "
            "relative scale. Smaller means harder. These simulate same make, "
            "model, and colour vehicles -- the decoys that actually defeat re-id. "
            "The default lands them around cosine 0.93, just *above* the "
            "contract's 0.92 auto-accept threshold, so some are wrongly "
            "auto-accepted on appearance alone and only topology can reject them. "
            "A value that kept them safely below the threshold would make the "
            "scenario decorative."
        ),
    )
    model_version: str = Field(default="synthetic@v1")


class TrafficSpec(MCTBaseModel):
    """Background traffic: the haystack the target hides in."""

    decoy_vehicles: int = Field(default=0, ge=0)
    decoy_route_length: int = Field(default=3, ge=1)
    near_miss_edit_distances: list[int] = Field(
        default_factory=list,
        description=(
            "One extra decoy per entry, with a plate exactly this far from the "
            "target's. Stresses fuzzy matching at a known distance rather than "
            "hoping a random plate lands nearby."
        ),
    )

    @model_validator(mode="after")
    def _near_miss_distances_are_positive(self) -> TrafficSpec:
        """Reject a near-miss distance of zero.

        Returns:
            The validated spec.

        Raises:
            ValueError: If any requested distance is not positive. Distance zero
                is plate cloning, which is an adversarial injection with its own
                switch rather than an accident of traffic generation.
        """
        if any(distance < 1 for distance in self.near_miss_edit_distances):
            msg = (
                "near-miss edit distances must be >= 1; distance 0 is plate cloning, "
                "enabled via adversarial.clone_target_plate"
            )
            raise ValueError(msg)
        return self


class ClockDriftInjection(MCTBaseModel):
    """A camera whose clock is wrong by a fixed amount."""

    camera_id: str = Field(min_length=1)
    offset_sec: float = Field(
        description="Added to this camera's reported timestamps. Positive means the "
        "camera reports later than the truth."
    )


class OutageInjection(MCTBaseModel):
    """A window during which a camera recorded nothing."""

    camera_id: str = Field(min_length=1)
    start_utc: UtcDatetime
    end_utc: UtcDatetime

    @model_validator(mode="after")
    def _window_is_ordered(self) -> OutageInjection:
        """Reject an empty or inverted outage.

        Returns:
            The validated injection.

        Raises:
            ValueError: If the window ends before it starts.
        """
        if self.end_utc <= self.start_utc:
            msg = "outage end_utc must be strictly after start_utc"
            raise ValueError(msg)
        return self


class AdversarialSpec(MCTBaseModel):
    """Deliberate injections that break naive implementations."""

    clock_drift: list[ClockDriftInjection] = Field(default_factory=list)
    outages: list[OutageInjection] = Field(default_factory=list)
    duplicate_passes: int = Field(
        default=0,
        ge=0,
        description="Extra near-identical sightings of a pass already generated",
    )
    duplicate_offset_sec: float = Field(default=0.4, gt=0.0)
    clone_target_plate: bool = Field(
        default=False,
        description="Give one decoy the target's exact plate. Plate matching alone "
        "cannot separate them; only appearance and topology can.",
    )
    boundary_hops: bool = Field(
        default=False,
        description="Force one hop at exactly min_travel_time and one at exactly "
        "max_travel_time, so the inclusive-boundary convention is exercised by data.",
    )
    shuffle_emission_order: bool = Field(
        default=False,
        description="Emit records out of chronological order, as a real ingest queue does",
    )


class VehicleSpec(MCTBaseModel):
    """One vehicle and the route it drives."""

    vehicle_id: str = Field(min_length=1)
    plate: str = Field(min_length=1, description="The true plate, before any OCR noise")
    route: list[str] = Field(min_length=1, description="Ordered camera ids")
    departure_utc: UtcDatetime
    object_class: ObjectClass = ObjectClass.CAR
    speed_profile: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Where in each travel-time window this vehicle sits. 0 is the "
        "fastest plausible transit, 1 the slowest.",
    )
    is_target: bool = Field(
        default=False, description="Whether this is the vehicle a search is looking for"
    )


class Scenario(MCTBaseModel):
    """A complete, reproducible description of a synthetic dataset."""

    name: str = Field(min_length=1)
    description: str = ""
    topology_file: Path = Path("config/topology.yaml")
    seed: int = Field(default=0, description="Default seed; the CLI may override it")
    jitter_sec: float = Field(
        default=5.0,
        ge=0.0,
        description="Uniform jitter applied to each transit, clamped to stay inside "
        "the link's plausible window so generated data is topology-consistent "
        "by construction",
    )
    ingest_delay_sec: float = Field(
        default=1.0,
        ge=0.0,
        description="Gap between a sighting's timestamp and its created_at. Fixed "
        "rather than random so output stays byte-identical.",
    )
    vehicles: list[VehicleSpec] = Field(min_length=1)
    noise: NoiseProfile = Field(default_factory=NoiseProfile)
    embeddings: EmbeddingProfile = Field(default_factory=EmbeddingProfile)
    traffic: TrafficSpec = Field(default_factory=TrafficSpec)
    adversarial: AdversarialSpec = Field(default_factory=AdversarialSpec)

    @model_validator(mode="after")
    def _vehicle_ids_are_unique(self) -> Scenario:
        """Reject duplicate vehicle ids.

        Returns:
            The validated scenario.

        Raises:
            ValueError: If two vehicles share an id, which would make the ground
                truth's assignment map ambiguous.
        """
        seen = [vehicle.vehicle_id for vehicle in self.vehicles]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            msg = f"duplicate vehicle_id in scenario: {duplicates}"
            raise ValueError(msg)
        return self

    @property
    def target(self) -> VehicleSpec | None:
        """Return the first vehicle flagged as the search target, if any."""
        return next((vehicle for vehicle in self.vehicles if vehicle.is_target), None)


def load_scenario(path: Path | str) -> Scenario:
    """Read a scenario from YAML.

    Args:
        path: Path to the scenario file.

    Returns:
        The validated scenario.

    Raises:
        ValidationError: If the file is missing, unparseable, or does not
            describe a valid scenario. Wrapped so callers catch one hierarchy
            rather than a mix of OSError, YAMLError, and pydantic errors.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise ValidationError("Scenario file not found", {"path": str(resolved)})

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(
            "Scenario file could not be parsed", {"path": str(resolved), "reason": str(exc)}
        ) from exc

    if not isinstance(raw, dict):
        raise ValidationError(
            "Scenario file must contain a YAML mapping",
            {"path": str(resolved), "parsed_type": type(raw).__name__},
        )

    return Scenario.model_validate(raw)


def save_scenario(scenario: Scenario, path: Path | str) -> Path:
    """Write a scenario to YAML.

    Args:
        scenario: The scenario to serialize.
        path: Destination file.

    Returns:
        The path written.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = scenario.to_json_dict()
    resolved.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    return resolved
