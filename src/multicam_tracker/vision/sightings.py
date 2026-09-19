"""Turning a completed track into the one record the rest of the system stores.

A track is forty views of a vehicle; a :class:`~multicam_tracker.models.sighting.Sighting`
is the single assertion "this vehicle was at this camera at this instant". This
module is the seam between them, and every choice it makes is a claim that will
be read back as fact months later.

**The timestamp comes from the frame nearest the track's temporal midpoint.**
Not the first frame, not the last, and the reason is not aesthetics. A track
begins when the vehicle is entering frame -- typically clipped by the edge,
partly occluded, at the periphery of the lens -- and ends the same way leaving.
The midpoint is where the vehicle is most fully in view, and it is the instant
least sensitive to exactly when the detector happened to acquire and lose the
track. Dating a sighting from the first frame would systematically place every
vehicle slightly early, and by an amount that varies with the camera's field of
view, which is precisely the kind of bias that survives every test and corrupts
travel-time plausibility across the whole topology.

**The box comes from that same frame.** The timestamp and the box must describe
one instant or they describe nothing.

**The confidence is the median across the track.** See
:attr:`~multicam_tracker.vision.track.VehicleTrack.aggregate_confidence` for why
a median and not a mean or a maximum.

**The thumbnail may come from a different frame.** The sighting is dated and
located from the midpoint frame, but the image is cropped from the
highest-ranked one, because that image exists to be read -- by OCR in stage 12,
by a human in stage 18 -- and the clearest view is the one worth keeping. The
two frames are usually the same and sometimes not, so the sighting records the
midpoint frame's index and anyone comparing the two should expect them to
differ.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from multicam_tracker.clock import Clock
from multicam_tracker.exceptions import VisionError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models.sighting import Sighting
from multicam_tracker.vision.thumbnails import ThumbnailWriter
from multicam_tracker.vision.track import TrackObservation, VehicleTrack

__all__ = ["track_to_sighting", "track_to_sightings"]

logger = get_logger(__name__)


def _new_id() -> str:
    """Return a fresh sighting identifier.

    Returns:
        A UUID4 string.
    """
    return str(uuid.uuid4())


def _sub_track(track: VehicleTrack, observations: Sequence[TrackObservation]) -> VehicleTrack:
    """Return a track covering only part of another track's observations.

    Used by the long-track split. The identifier gains a suffix rather than
    being reused unchanged: two sightings that both claim to come from track
    ``cam_03-00007`` are indistinguishable in an audit, and the whole point of
    splitting is that they are different moments.

    Args:
        track: The track being split.
        observations: The observations in this slice, in frame order.

    Returns:
        A track over that slice, carrying whichever of the parent track's
        ranked frames fall inside it, in the parent's order. The parent ranking
        is reused rather than recomputed because recomputing it here would need
        the configured weights, and a module that quietly substituted default
        weights would silently ignore an operator who had tuned them.
    """
    window = {observation.frame_index for observation in observations}
    ranked = [entry for entry in track.ranked if entry.observation.frame_index in window]
    return VehicleTrack(
        track_id=f"{track.track_id}+{observations[0].frame_index}",
        camera_id=track.camera_id,
        source_id=track.source_id,
        observations=tuple(observations),
        ranked=tuple(ranked),
        frame_size=track.frame_size,
        clock_offset_ms=track.clock_offset_ms,
    )


def track_to_sighting(
    track: VehicleTrack,
    *,
    clock: Clock,
    thumbnail_writer: ThumbnailWriter | None = None,
    id_factory: Callable[[], str] = _new_id,
) -> Sighting:
    """Build exactly one sighting from one completed track.

    Args:
        track: The finished track.
        clock: Supplies ``created_at``. Injected rather than called directly
            because nothing in this project reads the wall clock on its own.
        thumbnail_writer: Writes the crop, if imagery is wanted. ``None`` emits
            a sighting with no ``thumbnail_path``, which is what the
            arithmetic-only tests and a pipeline with storage disabled want.
        id_factory: Supplies the sighting identifier. Overridable so a test can
            make the thumbnail path deterministic.

    Returns:
        The sighting, already validated by the stage 02 model.

    Raises:
        VisionError: If the selected box clamps to nothing against the frame
            bounds. That means a box lay entirely outside the frame it came
            from, which is an upstream coordinate-mapping fault -- exactly the
            failure stage 10 exists to prevent -- and inventing a one-pixel box
            to keep going would hide it.
    """
    observation = track.midpoint_observation
    detection = observation.detection

    width, height = track.frame_size
    if width > 0 and height > 0:
        detection = detection.clamped_to(width, height)

    if detection.area <= 0:
        raise VisionError(
            "The selected detection box has no area inside the frame; a box "
            "entirely outside its own frame indicates a coordinate-mapping fault",
            {
                "track_id": track.track_id,
                "camera_id": track.camera_id,
                "bbox": list(observation.detection.bbox),
                "frame_size": list(track.frame_size),
            },
        )

    offset_ms = round(
        (observation.timestamp_utc - observation.raw_timestamp).total_seconds() * 1000
    )
    sighting_id = id_factory()

    thumbnail_path: str | None = None
    best = track.best
    if thumbnail_writer is not None and best is not None and best.observation.crop is not None:
        thumbnail_path = thumbnail_writer.write(
            best.observation.crop,
            best.observation.detection.bbox,
            camera_id=track.camera_id,
            timestamp_utc=observation.timestamp_utc,
            sighting_id=sighting_id,
            already_cropped=True,
        )

    x1, y1, x2, y2 = detection.bbox
    sighting = Sighting(
        sighting_id=sighting_id,
        camera_id=track.camera_id,
        timestamp_utc=observation.timestamp_utc,
        raw_timestamp=observation.raw_timestamp,
        clock_offset_applied_ms=offset_ms,
        object_class=track.object_class,
        detection_confidence=track.aggregate_confidence,
        bbox=[x1, y1, x2, y2],
        frame_index=observation.frame_index,
        thumbnail_path=thumbnail_path,
        source_id=track.source_id,
        created_at=clock.now_utc(),
    )

    logger.debug(
        "sighting_from_track",
        track_id=track.track_id,
        sighting_id=sighting.sighting_id,
        camera_id=track.camera_id,
        frame_index=sighting.frame_index,
        hits=track.hit_count,
        thumbnail_frame_index=None if best is None else best.observation.frame_index,
    )
    return sighting


def track_to_sightings(
    track: VehicleTrack,
    *,
    clock: Clock,
    thumbnail_writer: ThumbnailWriter | None = None,
    split_interval_sec: float | None = None,
    id_factory: Callable[[], str] = _new_id,
) -> list[Sighting]:
    """Build one sighting per track, or several for a very long one.

    A vehicle parked in view for ten minutes is one track, and collapsing it to
    one instantaneous sighting misrepresents it in both directions: it claims
    the vehicle was there at one moment rather than throughout, and it gives the
    path reconstructor a single point where it should see a stationary period.
    Splitting at a configured interval says what actually happened.

    Off by default, because for a vehicle that merely drives past -- which is
    almost all of them -- splitting would manufacture several sightings of one
    transit and make a single pass look like a vehicle circling.

    Args:
        track: The finished track.
        clock: Supplies ``created_at``.
        thumbnail_writer: Writes the crops, if imagery is wanted.
        split_interval_sec: Emit one sighting per this many seconds of track
            duration. ``None`` emits exactly one.
        id_factory: Supplies the sighting identifiers.

    Returns:
        The sightings, in ascending time order. Always at least one.
    """
    if split_interval_sec is None or track.duration_sec <= split_interval_sec:
        return [
            track_to_sighting(
                track,
                clock=clock,
                thumbnail_writer=thumbnail_writer,
                id_factory=id_factory,
            )
        ]

    windows: list[list[TrackObservation]] = []
    window_start = track.first_timestamp_utc
    current: list[TrackObservation] = []

    for observation in track.observations:
        elapsed = (observation.timestamp_utc - window_start).total_seconds()
        if current and elapsed >= split_interval_sec:
            windows.append(current)
            current = []
            window_start = observation.timestamp_utc
        current.append(observation)

    if current:
        windows.append(current)

    logger.info(
        "long_track_split",
        track_id=track.track_id,
        camera_id=track.camera_id,
        duration_sec=round(track.duration_sec, 3),
        interval_sec=split_interval_sec,
        sightings=len(windows),
    )

    return [
        track_to_sighting(
            _sub_track(track, window),
            clock=clock,
            thumbnail_writer=thumbnail_writer,
            id_factory=id_factory,
        )
        for window in windows
    ]
