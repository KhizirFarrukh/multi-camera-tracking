"""Synthetic sighting generation with known ground truth.

The highest-leverage piece of the project. With a ground-truth generator, the
matching and pathing engines of stages 06-08 can be *measured* -- precision and
recall become numbers, thresholds get tuned empirically, and a regression is
visible the moment it appears. None of that is possible on real footage, where
nobody ever truly knows the right answer.

The realism is the point. A generator that only ever substituted a character in
a plate, or drew decoys uniformly at random, would make the matcher look
excellent and then collapse on real data. So the noise model reproduces all
four ways ALPR actually fails, and decoys can be drawn deliberately close to the
target.
"""

from __future__ import annotations

from multicam_tracker.synth.embeddings import EmbeddingFactory, cosine_similarity, normalize
from multicam_tracker.synth.generator import (
    SyntheticDataset,
    generate,
    generate_from_file,
    load_dataset_into,
    stream_seed,
)
from multicam_tracker.synth.ground_truth import (
    CorruptionEvent,
    GroundTruth,
    InjectionEvent,
    VehicleTruth,
)
from multicam_tracker.synth.plate_noise import (
    PlateReading,
    corrupt_plate,
    edit_distance,
    mutate_plate_to_distance,
)
from multicam_tracker.synth.routes import RoutePass, leg_window, random_route, simulate_route
from multicam_tracker.synth.scenario import (
    AdversarialSpec,
    ClockDriftInjection,
    EmbeddingProfile,
    NoiseProfile,
    OutageInjection,
    Scenario,
    TrafficSpec,
    VehicleSpec,
    load_scenario,
    save_scenario,
)
from multicam_tracker.synth.traffic import DecoyVehicle, generate_decoys, random_plate

__all__ = [
    "AdversarialSpec",
    "ClockDriftInjection",
    "CorruptionEvent",
    "DecoyVehicle",
    "EmbeddingFactory",
    "EmbeddingProfile",
    "GroundTruth",
    "InjectionEvent",
    "NoiseProfile",
    "OutageInjection",
    "PlateReading",
    "RoutePass",
    "Scenario",
    "SyntheticDataset",
    "TrafficSpec",
    "VehicleSpec",
    "VehicleTruth",
    "corrupt_plate",
    "cosine_similarity",
    "edit_distance",
    "generate",
    "generate_decoys",
    "generate_from_file",
    "leg_window",
    "load_dataset_into",
    "load_scenario",
    "mutate_plate_to_distance",
    "normalize",
    "random_plate",
    "random_route",
    "save_scenario",
    "simulate_route",
    "stream_seed",
]
