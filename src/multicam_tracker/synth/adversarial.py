"""Deliberate injections that break naive implementations.

Each of these corresponds to something that genuinely happens to camera networks
and that a system built only against clean data will get wrong:

* **Clock drift** -- a camera reports the wrong time, so its sightings appear
  out of order and every hop through it looks implausible. Stage 09 must detect
  and correct it, and can only be scored against a dataset where the true
  instant is recorded separately from the reported one.
* **Outage** -- a camera saw nothing for a window. Path reconstruction must
  produce a coverage gap rather than inventing a route.
* **Duplicates** -- one pass recorded twice. Dedup must collapse them without
  collapsing a genuine revisit.
* **Plate cloning** -- two vehicles wearing one plate. Plate matching alone
  cannot separate them; only appearance and topology can.
* **Boundary timestamps** -- hops landing exactly on a window edge, so the
  inclusive-boundary convention is exercised by data rather than only by a unit
  test.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from multicam_tracker.models import Sighting
from multicam_tracker.synth.ground_truth import InjectionEvent
from multicam_tracker.synth.scenario import AdversarialSpec

__all__ = [
    "apply_clock_drift",
    "apply_outages",
    "deterministic_uuid",
    "inject_duplicates",
    "shuffle_emission_order",
]


def deterministic_uuid(rng: random.Random) -> str:
    """Return a UUID4-shaped identifier drawn from a seeded generator.

    ``uuid.uuid4()`` reads the OS entropy pool and would make output differ
    between runs, which the determinism guarantee forbids.

    Args:
        rng: Seeded generator.

    Returns:
        The identifier in canonical string form.
    """
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def apply_clock_drift(
    sightings: list[Sighting], spec: AdversarialSpec
) -> tuple[list[Sighting], list[InjectionEvent]]:
    """Shift a camera's reported timestamps away from the truth.

    Both ``raw_timestamp`` and ``timestamp_utc`` move, and
    ``clock_offset_applied_ms`` stays at zero. That is the point: the drift is
    *uncorrected* in the data, which is what leaves stage 09 something to find.
    The true instants are preserved separately in the ground truth.

    Args:
        sightings: The generated sightings.
        spec: Supplies the drift injections.

    Returns:
        The sightings with drift applied, and one event per injection.
    """
    if not spec.clock_drift:
        return sightings, []

    offsets = {entry.camera_id: entry.offset_sec for entry in spec.clock_drift}
    events = [
        InjectionEvent(
            kind="clock_drift",
            detail=(
                f"camera {entry.camera_id} reports {entry.offset_sec:+.1f}s away from "
                f"true time; the data carries no correction"
            ),
            context={"camera_id": entry.camera_id, "offset_sec": entry.offset_sec},
        )
        for entry in spec.clock_drift
    ]

    drifted: list[Sighting] = []
    for sighting in sightings:
        offset = offsets.get(sighting.camera_id)
        if offset is None:
            drifted.append(sighting)
            continue
        shifted = sighting.timestamp_utc + timedelta(seconds=offset)
        drifted.append(
            sighting.model_copy(update={"timestamp_utc": shifted, "raw_timestamp": shifted})
        )

    return drifted, events


def apply_outages(
    sightings: list[Sighting], spec: AdversarialSpec
) -> tuple[list[Sighting], list[InjectionEvent]]:
    """Drop sightings that fall inside a camera's outage window.

    Args:
        sightings: The generated sightings.
        spec: Supplies the outage windows.

    Returns:
        The surviving sightings and one event per outage, recording how many
        were removed so a test can assert the outage actually bit.
    """
    if not spec.outages:
        return sightings, []

    surviving = list(sightings)
    events: list[InjectionEvent] = []

    for outage in spec.outages:
        removed = [
            sighting
            for sighting in surviving
            if sighting.camera_id == outage.camera_id
            and outage.start_utc <= sighting.timestamp_utc < outage.end_utc
        ]
        surviving = [sighting for sighting in surviving if sighting not in removed]
        events.append(
            InjectionEvent(
                kind="outage",
                detail=(
                    f"camera {outage.camera_id} recorded nothing between "
                    f"{outage.start_utc.isoformat()} and {outage.end_utc.isoformat()}; "
                    f"{len(removed)} sighting(s) removed"
                ),
                context={
                    "camera_id": outage.camera_id,
                    "start_utc": outage.start_utc.isoformat(),
                    "end_utc": outage.end_utc.isoformat(),
                    "removed": len(removed),
                    "removed_sighting_ids": [s.sighting_id for s in removed],
                },
            )
        )

    return surviving, events


def inject_duplicates(
    rng: random.Random, sightings: list[Sighting], spec: AdversarialSpec
) -> tuple[list[Sighting], list[InjectionEvent]]:
    """Record some passes twice, as a double-triggered detector would.

    The copy gets a distinct ``sighting_id`` and a slightly later timestamp;
    everything else matches. Dedup has to collapse these without collapsing a
    genuine revisit, which is why the offset is far below the same-pass gap.

    Args:
        rng: Seeded generator.
        sightings: The generated sightings.
        spec: Supplies the duplicate count and offset.

    Returns:
        The sightings plus their duplicates, and one event per duplicate.
    """
    if spec.duplicate_passes <= 0 or not sightings:
        return sightings, []

    count = min(spec.duplicate_passes, len(sightings))
    chosen = rng.sample(range(len(sightings)), count)
    duplicated: list[Sighting] = []
    events: list[InjectionEvent] = []

    for index in sorted(chosen):
        original = sightings[index]
        shifted = original.timestamp_utc + timedelta(seconds=spec.duplicate_offset_sec)
        copy = original.model_copy(
            update={
                "sighting_id": deterministic_uuid(rng),
                "timestamp_utc": shifted,
                "raw_timestamp": shifted
                + timedelta(milliseconds=-original.clock_offset_applied_ms),
                "frame_index": original.frame_index + 1,
            }
        )
        duplicated.append(copy)
        events.append(
            InjectionEvent(
                kind="duplicate",
                detail=(
                    f"pass on {original.camera_id} recorded twice, "
                    f"{spec.duplicate_offset_sec:.2f}s apart"
                ),
                context={
                    "camera_id": original.camera_id,
                    "original_sighting_id": original.sighting_id,
                    "duplicate_sighting_id": copy.sighting_id,
                },
            )
        )

    return [*sightings, *duplicated], events


def shuffle_emission_order(
    rng: random.Random, sightings: list[Sighting], spec: AdversarialSpec
) -> tuple[list[Sighting], list[InjectionEvent]]:
    """Emit records out of chronological order.

    A real ingest queue does not deliver in time order: cameras buffer, networks
    retry, batch jobs finish late. Anything that assumes the input stream is
    sorted breaks here rather than in production.

    Args:
        rng: Seeded generator.
        sightings: The generated sightings.
        spec: Supplies the switch.

    Returns:
        The sightings, shuffled if requested, and an event recording it.
    """
    if not spec.shuffle_emission_order or len(sightings) < 2:
        return sightings, []

    shuffled = list(sightings)
    rng.shuffle(shuffled)
    return shuffled, [
        InjectionEvent(
            kind="out_of_order",
            detail="records are emitted in arrival order, not chronological order",
            context={"count": len(shuffled)},
        )
    ]


def sort_by_time(sightings: list[Sighting]) -> list[Sighting]:
    """Return sightings in chronological order, ties broken by id.

    Args:
        sightings: The sightings to order.

    Returns:
        A new sorted list.
    """
    return sorted(sightings, key=lambda item: (item.timestamp_utc, item.sighting_id))


def latest_timestamp(sightings: list[Sighting]) -> datetime | None:
    """Return the latest timestamp in a set of sightings.

    Args:
        sightings: The sightings to scan.

    Returns:
        The maximum ``timestamp_utc``, or ``None`` when empty.
    """
    return max((item.timestamp_utc for item in sightings), default=None)
