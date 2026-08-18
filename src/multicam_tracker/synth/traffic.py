"""Background traffic: the haystack.

A target that is the only vehicle in the dataset is trivially found. Decoys make
the measurement mean something, and two kinds of decoy do most of the work:

* **Near-miss plates** sit at a *known* edit distance from the target's, so
  stage 06's fuzzy threshold is stressed exactly at the boundary rather than
  wherever random plates happen to land.
* **Hard negatives** look like the target, so stage 07's embedding threshold
  faces the case that actually defeats re-id in the field.

Decoy routes are drawn by walking the topology, so they are as plausible as the
target's. A decoy on an impossible route would be rejected by the travel-time
constraint alone and would add no difficulty at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from multicam_tracker.synth.plate_noise import SPURIOUS_ALPHABET, mutate_plate_to_distance
from multicam_tracker.synth.routes import random_route
from multicam_tracker.synth.scenario import Scenario
from multicam_tracker.topology import Topology

__all__ = ["DecoyVehicle", "generate_decoys", "random_plate"]

PLATE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
PLATE_DIGITS = "0123456789"


@dataclass(frozen=True)
class DecoyVehicle:
    """One background vehicle."""

    vehicle_id: str
    plate: str
    route: list[str]
    departure_utc: datetime
    speed_profile: float
    is_hard_negative: bool = False
    near_miss_distance: int | None = None
    is_plate_clone: bool = False


def random_plate(rng: random.Random) -> str:
    """Draw a plate in a common three-letter, four-digit format.

    Args:
        rng: Seeded generator.

    Returns:
        A plate such as ``KRT8421``. Visually confusable letters are still in
        the alphabet -- excluding them would quietly make OCR noise easier to
        undo than it is in reality.
    """
    letters = "".join(rng.choice(PLATE_LETTERS) for _ in range(3))
    digits = "".join(rng.choice(PLATE_DIGITS) for _ in range(4))
    return letters + digits


def generate_decoys(
    rng: random.Random,
    topology: Topology,
    scenario: Scenario,
    *,
    target_plate: str | None,
    span_start: datetime,
    span_seconds: float,
) -> list[DecoyVehicle]:
    """Build the scenario's background traffic.

    Args:
        rng: Seeded generator.
        topology: The graph decoys drive over.
        scenario: Supplies the traffic and embedding profiles.
        target_plate: The plate near-miss and clone decoys are built from.
            Near-miss generation is skipped when there is no target.
        span_start: Earliest departure.
        span_seconds: Width of the departure window.

    Returns:
        The decoys, in a deterministic order.

    Raises:
        TopologyError: If the topology has no traversable route.
    """
    traffic = scenario.traffic
    decoys: list[DecoyVehicle] = []
    taken_plates = {target_plate} if target_plate else set()

    hard_negative_count = round(traffic.decoy_vehicles * scenario.embeddings.hard_negative_fraction)

    for index in range(traffic.decoy_vehicles):
        plate = random_plate(rng)
        # Retry rather than accept a collision: an accidental clone would look
        # like a matching bug, and clones are supposed to be an explicit
        # adversarial injection.
        while plate in taken_plates:
            plate = random_plate(rng)
        taken_plates.add(plate)

        decoys.append(
            DecoyVehicle(
                vehicle_id=f"decoy_{index:04d}",
                plate=plate,
                route=random_route(rng, topology, traffic.decoy_route_length),
                departure_utc=span_start + timedelta(seconds=rng.uniform(0.0, span_seconds)),
                speed_profile=rng.uniform(0.2, 0.8),
                is_hard_negative=index < hard_negative_count,
            )
        )

    if target_plate:
        for index, distance in enumerate(traffic.near_miss_edit_distances):
            plate = mutate_plate_to_distance(rng, target_plate, distance)
            while plate in taken_plates:
                plate = mutate_plate_to_distance(rng, target_plate, distance)
            taken_plates.add(plate)

            decoys.append(
                DecoyVehicle(
                    vehicle_id=f"nearmiss_{index:02d}_d{distance}",
                    plate=plate,
                    route=random_route(rng, topology, traffic.decoy_route_length),
                    departure_utc=span_start + timedelta(seconds=rng.uniform(0.0, span_seconds)),
                    speed_profile=rng.uniform(0.2, 0.8),
                    near_miss_distance=distance,
                )
            )

    if scenario.adversarial.clone_target_plate and target_plate:
        decoys.append(
            DecoyVehicle(
                vehicle_id="clone_00",
                plate=target_plate,
                route=random_route(rng, topology, traffic.decoy_route_length),
                departure_utc=span_start + timedelta(seconds=rng.uniform(0.0, span_seconds)),
                speed_profile=rng.uniform(0.2, 0.8),
                is_plate_clone=True,
            )
        )

    return decoys


def unused_alphabet_char(rng: random.Random, exclude: str) -> str:
    """Return a plate character not in ``exclude``.

    Args:
        rng: Seeded generator.
        exclude: Characters to avoid.

    Returns:
        A single character.
    """
    options = [char for char in SPURIOUS_ALPHABET if char not in exclude]
    return rng.choice(options)
