"""The generator: scenario in, dataset plus answer key out.

Determinism is the contract. The same scenario and seed must produce identical
output, because stages 06-08 tune thresholds against these datasets and a
generator that drifted would turn every regression test into noise.

Two things make that hold. Every source of randomness is a seeded
:class:`random.Random`, never the module-level functions. And each component
draws from its *own* stream, derived from the master seed by a stable hash of the
component name -- so adding a component later does not shift the numbers every
existing component would have drawn.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from multicam_tracker.db.repositories.protocols import SightingRepository
from multicam_tracker.exceptions import ValidationError
from multicam_tracker.models import ObjectClass, Sighting
from multicam_tracker.synth import adversarial as adv
from multicam_tracker.synth.embeddings import EmbeddingFactory
from multicam_tracker.synth.ground_truth import (
    CorruptionEvent,
    GroundTruth,
    InjectionEvent,
    VehicleTruth,
    order_by_time,
)
from multicam_tracker.synth.plate_noise import corrupt_plate
from multicam_tracker.synth.routes import RoutePass, simulate_route
from multicam_tracker.synth.scenario import Scenario, VehicleSpec, load_scenario
from multicam_tracker.synth.traffic import DecoyVehicle, generate_decoys
from multicam_tracker.topology import Topology, load_topology

__all__ = [
    "SyntheticDataset",
    "generate",
    "generate_from_file",
    "load_dataset_into",
    "stream_seed",
]

GENERATION_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
"""Fixed stamp for ``GroundTruth.generated_at_utc``.

A wall-clock reading would make two runs of the same scenario differ, breaking
the byte-identical guarantee for the sake of a field nothing reads.
"""


def stream_seed(master_seed: int, component: str) -> int:
    """Derive a component's seed from the master seed.

    Uses CRC32 of the component name rather than :func:`hash`, whose string
    hashing is randomized per process and would destroy reproducibility.

    Args:
        master_seed: The scenario's seed.
        component: Stable name of the component, e.g. ``"plate_noise"``.

    Returns:
        A seed for that component's own generator.
    """
    return (master_seed * 1_000_003) ^ zlib.crc32(component.encode("utf-8"))


@dataclass
class SyntheticDataset:
    """A generated dataset and the answer key that goes with it."""

    scenario_name: str
    seed: int
    sightings: list[Sighting] = field(default_factory=list)
    ground_truth: GroundTruth = field(
        default_factory=lambda: GroundTruth(
            scenario_name="empty", seed=0, generated_at_utc=GENERATION_EPOCH
        )
    )
    camera_ids: list[str] = field(default_factory=list)

    def sightings_of(self, vehicle_id: str) -> list[Sighting]:
        """Return one vehicle's sightings, in recorded time order.

        Args:
            vehicle_id: The vehicle to filter on.

        Returns:
            Its sightings.
        """
        wanted = {
            sighting_id
            for sighting_id, owner in self.ground_truth.sighting_to_vehicle.items()
            if owner == vehicle_id
        }
        return adv.sort_by_time([s for s in self.sightings if s.sighting_id in wanted])

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the whole dataset.

        Returns:
            A JSON-native mapping containing the sightings and the ground truth.
        """
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "camera_ids": list(self.camera_ids),
            "sightings": [sighting.to_json_dict() for sighting in self.sightings],
            "ground_truth": self.ground_truth.to_json_dict(),
        }


@dataclass
class _Vehicle:
    """Internal record joining a spec, its base vector, and its truth entry."""

    vehicle_id: str
    plate: str
    route: list[str]
    departure_utc: datetime
    speed_profile: float
    object_class: ObjectClass
    is_target: bool = False
    is_decoy: bool = False
    is_hard_negative: bool = False


def _vehicles_from(scenario: Scenario, decoys: list[DecoyVehicle]) -> list[_Vehicle]:
    """Merge the scenario's declared vehicles with generated decoys.

    Args:
        scenario: The scenario being generated.
        decoys: Background traffic.

    Returns:
        Every vehicle to simulate, declared ones first.
    """
    declared = [
        _Vehicle(
            vehicle_id=spec.vehicle_id,
            plate=spec.plate,
            route=list(spec.route),
            departure_utc=spec.departure_utc,
            speed_profile=spec.speed_profile,
            object_class=spec.object_class,
            is_target=spec.is_target,
        )
        for spec in scenario.vehicles
    ]
    generated = [
        _Vehicle(
            vehicle_id=decoy.vehicle_id,
            plate=decoy.plate,
            route=list(decoy.route),
            departure_utc=decoy.departure_utc,
            speed_profile=decoy.speed_profile,
            object_class=ObjectClass.CAR,
            is_decoy=True,
            is_hard_negative=decoy.is_hard_negative,
        )
        for decoy in decoys
    ]
    return [*declared, *generated]


def _bbox(rng: random.Random) -> list[int]:
    """Draw a plausible detection box.

    Args:
        rng: Seeded generator.

    Returns:
        ``[x1, y1, x2, y2]`` with positive area.
    """
    x1 = rng.randint(0, 1200)
    y1 = rng.randint(0, 600)
    return [x1, y1, x1 + rng.randint(60, 400), y1 + rng.randint(40, 300)]


def _span(scenario: Scenario) -> tuple[datetime, float]:
    """Return the departure window decoys are spread across.

    Args:
        scenario: The scenario being generated.

    Returns:
        ``(earliest_departure, width_seconds)``. The width has a floor so a
        single-vehicle scenario still spreads its traffic rather than departing
        everything at one instant.
    """
    departures = [vehicle.departure_utc for vehicle in scenario.vehicles]
    start = min(departures)
    width = max((max(departures) - start).total_seconds(), 1800.0)
    return start, width


def generate(
    scenario: Scenario, *, seed: int | None = None, topology: Topology | None = None
) -> SyntheticDataset:
    """Generate a dataset and its ground truth from a scenario.

    Args:
        scenario: The declarative description.
        seed: Overrides ``scenario.seed``. The seed actually used is recorded in
            the ground truth.
        topology: Pre-loaded graph. Loaded from ``scenario.topology_file`` when
            omitted.

    Returns:
        The dataset, its sightings, and the answer key.

    Raises:
        TopologyError: If the topology cannot be loaded, or a route asks for a
            transit the graph does not permit.
    """
    from multicam_tracker.models.base import default_embedding_dim

    configured_dim = default_embedding_dim()
    if scenario.embeddings.dimension != configured_dim:
        raise ValidationError(
            "Scenario embedding dimension does not match the configured one; every "
            "generated Sighting would be rejected at validation",
            {
                "scenario": scenario.name,
                "scenario_dimension": scenario.embeddings.dimension,
                "configured_dimension": configured_dim,
            },
        )

    resolved_seed = scenario.seed if seed is None else seed
    graph = topology if topology is not None else load_topology(scenario.topology_file)

    route_rng = random.Random(stream_seed(resolved_seed, "routes"))
    traffic_rng = random.Random(stream_seed(resolved_seed, "traffic"))
    plate_rng = random.Random(stream_seed(resolved_seed, "plate_noise"))
    vector_rng = random.Random(stream_seed(resolved_seed, "embeddings"))
    detect_rng = random.Random(stream_seed(resolved_seed, "detection"))
    id_rng = random.Random(stream_seed(resolved_seed, "identifiers"))
    inject_rng = random.Random(stream_seed(resolved_seed, "adversarial"))

    target_spec: VehicleSpec | None = scenario.target
    span_start, span_width = _span(scenario)

    decoys = generate_decoys(
        traffic_rng,
        graph,
        scenario,
        target_plate=target_spec.plate if target_spec else None,
        span_start=span_start,
        span_seconds=span_width,
    )
    vehicles = _vehicles_from(scenario, decoys)

    factory = EmbeddingFactory(
        vector_rng,
        dimension=scenario.embeddings.dimension,
        intra_class_sigma=scenario.embeddings.intra_class_sigma,
        hard_negative_sigma=scenario.embeddings.hard_negative_sigma,
    )

    target_base: list[float] | None = None
    if target_spec is not None:
        target_base = factory.base_vector()

    sightings: list[Sighting] = []
    truths: list[VehicleTruth] = []
    assignment: dict[str, str] = {}
    true_times: dict[str, datetime] = {}
    corruptions: list[CorruptionEvent] = []

    for vehicle in vehicles:
        if vehicle.is_target and target_base is not None:
            base = target_base
        elif vehicle.is_hard_negative and target_base is not None:
            base = factory.hard_negative_of(target_base)
        else:
            base = factory.base_vector()

        passes = simulate_route(
            route_rng,
            graph,
            vehicle.route,
            vehicle.departure_utc,
            speed_profile=vehicle.speed_profile,
            jitter_sec=scenario.jitter_sec,
            boundary_hops=scenario.adversarial.boundary_hops and vehicle.is_target,
        )

        for frame_index, route_pass in enumerate(passes):
            sighting = _build_sighting(
                route_pass,
                vehicle=vehicle,
                frame_index=frame_index,
                scenario=scenario,
                embedding=factory.observation_of(base),
                reading_rng=plate_rng,
                detect_rng=detect_rng,
                id_rng=id_rng,
                corruptions=corruptions,
            )
            sightings.append(sighting)
            assignment[sighting.sighting_id] = vehicle.vehicle_id
            true_times[sighting.sighting_id] = route_pass.timestamp_utc

        # Truth entries are assembled at the end, once injections have added and
        # removed sightings. Building them here would leave a duplicate present
        # in the assignment map but missing from its vehicle's trajectory.

    injections = []
    sightings, drift_events = adv.apply_clock_drift(sightings, scenario.adversarial)
    injections.extend(drift_events)

    sightings, outage_events = adv.apply_outages(sightings, scenario.adversarial)
    injections.extend(outage_events)

    sightings, duplicate_events = adv.inject_duplicates(inject_rng, sightings, scenario.adversarial)
    for event in duplicate_events:
        original_id = str(event.context["original_sighting_id"])
        copy_id = str(event.context["duplicate_sighting_id"])
        assignment[copy_id] = assignment[original_id]
        true_times[copy_id] = true_times[original_id]
    injections.extend(duplicate_events)

    if scenario.adversarial.clone_target_plate and target_spec is not None:
        injections.append(
            InjectionEvent(
                kind="plate_clone",
                detail=(
                    f"a decoy wears the target's plate {target_spec.plate!r}; "
                    f"plate matching alone cannot separate the two vehicles"
                ),
                context={"plate": target_spec.plate, "vehicle_ids": ["clone_00"]},
            )
        )

    if scenario.adversarial.boundary_hops and target_spec is not None:
        injections.append(
            InjectionEvent(
                kind="boundary_hop",
                detail="the target's first hop sits exactly at min_travel_time and its "
                "second exactly at max_travel_time",
                context={"vehicle_id": target_spec.vehicle_id},
            )
        )

    # The answer key describes what was actually emitted, not what was planned:
    # an outage removed some sightings and duplicate injection added others.
    surviving = {sighting.sighting_id for sighting in sightings}
    assignment = {key: value for key, value in assignment.items() if key in surviving}
    true_times = {key: value for key, value in true_times.items() if key in surviving}
    corruptions = [event for event in corruptions if event.sighting_id in surviving]

    owned: dict[str, list[str]] = {vehicle.vehicle_id: [] for vehicle in vehicles}
    for sighting_id, owner in assignment.items():
        owned[owner].append(sighting_id)

    truths = [
        VehicleTruth(
            vehicle_id=vehicle.vehicle_id,
            true_plate=vehicle.plate,
            is_target=vehicle.is_target,
            is_decoy=vehicle.is_decoy,
            is_hard_negative=vehicle.is_hard_negative,
            sighting_ids=order_by_time(owned[vehicle.vehicle_id], true_times),
        )
        for vehicle in vehicles
    ]

    sightings = adv.sort_by_time(sightings)
    sightings, order_events = adv.shuffle_emission_order(
        inject_rng, sightings, scenario.adversarial
    )
    injections.extend(order_events)

    ground_truth = GroundTruth(
        scenario_name=scenario.name,
        seed=resolved_seed,
        generated_at_utc=GENERATION_EPOCH,
        vehicles=truths,
        sighting_to_vehicle=assignment,
        true_timestamps=true_times,
        corruptions=corruptions,
        injections=injections,
    )

    return SyntheticDataset(
        scenario_name=scenario.name,
        seed=resolved_seed,
        sightings=sightings,
        ground_truth=ground_truth,
        camera_ids=graph.camera_ids,
    )


def _build_sighting(
    route_pass: RoutePass,
    *,
    vehicle: _Vehicle,
    frame_index: int,
    scenario: Scenario,
    embedding: list[float],
    reading_rng: random.Random,
    detect_rng: random.Random,
    id_rng: random.Random,
    corruptions: list[CorruptionEvent],
) -> Sighting:
    """Turn one route pass into a sighting, applying OCR noise.

    Args:
        route_pass: Where and when the vehicle was seen.
        vehicle: The vehicle being simulated.
        frame_index: Position along its route, used as the frame index.
        scenario: Supplies the noise profile and ingest delay.
        embedding: This sighting's re-id vector.
        reading_rng: Stream for plate corruption.
        detect_rng: Stream for detection confidence and bbox.
        id_rng: Stream for identifiers.
        corruptions: Accumulator the corruption log is appended to.

    Returns:
        The validated sighting.
    """
    sighting_id = adv.deterministic_uuid(id_rng)
    reading = corrupt_plate(reading_rng, vehicle.plate, scenario.noise)

    if not reading.is_clean:
        corruptions.append(
            CorruptionEvent(
                sighting_id=sighting_id,
                vehicle_id=vehicle.vehicle_id,
                kind=reading.kind or "unknown",
                true_plate=vehicle.plate,
                observed_plate=reading.text,
                severity=reading.severity,
                detail=reading.detail,
            )
        )

    return Sighting(
        sighting_id=sighting_id,
        camera_id=route_pass.camera_id,
        timestamp_utc=route_pass.timestamp_utc,
        raw_timestamp=route_pass.timestamp_utc,
        clock_offset_applied_ms=0,
        object_class=vehicle.object_class,
        detection_confidence=round(detect_rng.uniform(0.55, 0.99), 4),
        bbox=_bbox(detect_rng),
        frame_index=frame_index,
        plate_text_raw=reading.text,
        plate_text_normalized=reading.text,
        plate_confidence=reading.confidence,
        embedding=embedding,
        embedding_model_version=scenario.embeddings.model_version,
        thumbnail_path=f"storage/thumbnails/{route_pass.camera_id}/{sighting_id}.jpg",
        source_id=f"{route_pass.camera_id}_synthetic",
        created_at=route_pass.timestamp_utc + timedelta(seconds=scenario.ingest_delay_sec),
    )


def generate_from_file(
    scenario_path: Path | str,
    *,
    seed: int | None = None,
    topology: Topology | None = None,
) -> SyntheticDataset:
    """Load a scenario file and generate its dataset.

    Args:
        scenario_path: Path to the scenario YAML.
        seed: Overrides the scenario's own seed.
        topology: Pre-loaded graph, avoiding a re-read per scenario.

    Returns:
        The generated dataset.

    Raises:
        ValidationError: If the scenario file is missing or invalid.
        TopologyError: If the topology cannot be loaded or a route is impossible.
    """
    return generate(load_scenario(scenario_path), seed=seed, topology=topology)


def load_dataset_into(
    dataset: SyntheticDataset,
    sighting_repo: SightingRepository,
    *,
    ignore_conflicts: bool = True,
) -> int:
    """Insert a generated dataset's sightings into a repository.

    Works against the in-memory fakes and against Postgres alike, because both
    satisfy the same protocol. Conflicts are ignored by default so a demo script
    can be re-run without first clearing the table.

    The cameras must already exist -- ``sync_topology_to_db`` puts them there --
    since a sighting carries a foreign key to its camera.

    Args:
        dataset: The generated dataset.
        sighting_repo: Where to write.
        ignore_conflicts: Skip sightings whose id is already stored.

    Returns:
        The number of rows actually inserted.

    Raises:
        StorageError: If a camera is unknown or the write fails.
    """
    return sighting_repo.add_batch(dataset.sightings, ignore_conflicts=ignore_conflicts)
