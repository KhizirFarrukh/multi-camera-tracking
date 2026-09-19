"""One vehicle's pass through one camera, as a single object.

The unit this whole stage exists to produce. A car crossing one camera at 5 fps
generates forty detections; without something to bind them together, that is
forty sightings of forty different vehicles as far as everything downstream can
tell -- forty rows, forty thumbnails, forty matching candidates, and a
trajectory that appears to visit the same camera forty times.

A :class:`VehicleTrack` is the answer: one object holding every detection of one
pass, out of which exactly one sighting is built.

**Memory is bounded by design.** A track holds a :class:`TrackObservation` per
frame, which is a handful of numbers, and a cropped image for only the
highest-ranked few. That split is what lets a vehicle sit in view for ten
minutes without the tracker accumulating six hundred full-resolution crops. The
cap is :attr:`~multicam_tracker.config.DetectionSettings.max_retained_frames`,
and :mod:`~multicam_tracker.vision.best_frames` decides which frames survive it.

**Crops are owned copies.** A decoder hands out a view into a buffer it reuses
for the next frame (see :class:`~multicam_tracker.ingest.frame.Frame`), so a
crop kept as a view would change underneath the track. The tracker copies on the
way in; this module documents why, because the symptom -- a thumbnail showing a
vehicle from a later frame -- reads as a cropping bug rather than an ownership
one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from statistics import median

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import VisionError
from multicam_tracker.models.enums import ObjectClass
from multicam_tracker.vision.detector_protocol import Detection

__all__ = [
    "RankedObservation",
    "TrackObservation",
    "TrackStatus",
    "VehicleTrack",
]


class TrackStatus(StrEnum):
    """Where a track is in its life.

    ``TENTATIVE`` is the state that stops a single false positive becoming a
    sighting: a track that never reaches the confirmation threshold is discarded
    when it dies, and nothing downstream ever hears about it.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class TrackObservation:
    """One frame's view of a tracked vehicle.

    Args:
        frame_index: Index of the frame within its source. Never renumbered, so
            a reference to "frame 240" means the same thing everywhere.
        raw_timestamp: Capture time as the source reported it.
        timestamp_utc: Capture time after the camera clock offset is applied.
        detection: The box, in original source-frame coordinates.
        crop: A padded, owned copy of the vehicle region, or ``None`` for an
            observation whose image was not retained. Most observations carry
            ``None``; see the module docstring.
        sharpness: Laplacian variance of the crop, computed once when the crop
            was taken. Stored rather than recomputed because the image it
            describes may already have been discarded.
    """

    frame_index: int
    raw_timestamp: datetime
    timestamp_utc: datetime
    detection: Detection
    crop: npt.NDArray[np.uint8] | None = None
    sharpness: float = 0.0

    @property
    def has_image(self) -> bool:
        """Return whether this observation still carries its cropped image."""
        return self.crop is not None

    def without_image(self) -> TrackObservation:
        """Return the same observation with its image released.

        Used when an observation falls out of the retained set. The numbers stay
        -- the track's history must remain complete -- and only the pixels go.

        Returns:
            An observation identical but for :attr:`crop`.
        """
        return TrackObservation(
            frame_index=self.frame_index,
            raw_timestamp=self.raw_timestamp,
            timestamp_utc=self.timestamp_utc,
            detection=self.detection,
            crop=None,
            sharpness=self.sharpness,
        )


@dataclass(frozen=True)
class RankedObservation:
    """An observation with its best-frame score and the parts that made it.

    The components are kept rather than just the total because a ranking nobody
    can explain is a ranking nobody can tune. When an operator asks why a
    blurred frame was chosen for OCR, the answer has to be available.

    Args:
        observation: The frame being scored.
        score: The composite score, in ``[0, 1]`` before the edge penalty and
            clamped to ``[0, 1]`` after it.
        components: Each criterion's normalized contribution before weighting.
    """

    observation: TrackObservation
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleTrack:
    """Every detection of one vehicle by one camera, as one object.

    Args:
        track_id: Stable within one tracker run. Not a global identifier: the
            sighting built from this track carries that.
        camera_id: Which camera saw it.
        source_id: Which file or stream the frames came from.
        observations: Every frame the vehicle was detected in, ascending by
            frame index. Complete: the ones whose images were dropped are still
            here.
        ranked: The retained observations, highest-ranked first. Empty when the
            tracker was run without image retention.
        frame_size: ``(width, height)`` of the source frames, needed to clamp
            boxes and to compute centrality after the fact.
        clock_offset_ms: Offset that was applied to produce
            :attr:`TrackObservation.timestamp_utc`, carried so the sighting can
            record it and stage 09 can recompute it later.

    Raises:
        VisionError: If the track is empty or its observations are not in
            ascending frame order.
    """

    track_id: str
    camera_id: str
    source_id: str
    observations: tuple[TrackObservation, ...]
    ranked: tuple[RankedObservation, ...] = ()
    frame_size: tuple[int, int] = (0, 0)
    clock_offset_ms: int = 0

    def __post_init__(self) -> None:
        """Validate the track is non-empty and time-ordered.

        Raises:
            VisionError: If there are no observations, or the frame indices do
                not ascend. Out-of-order observations would make the midpoint
                rule in :mod:`~multicam_tracker.vision.sightings` select an
                arbitrary frame, which is the kind of defect that produces a
                plausible timestamp nobody can trace.
        """
        if not self.observations:
            raise VisionError(
                "A vehicle track must contain at least one observation",
                {"track_id": self.track_id, "camera_id": self.camera_id},
            )

        indices = [observation.frame_index for observation in self.observations]
        if indices != sorted(indices):
            raise VisionError(
                "Track observations must ascend by frame index",
                {"track_id": self.track_id, "frame_indices": indices},
            )

    @property
    def first_frame_index(self) -> int:
        """Return the frame the vehicle was first detected in."""
        return self.observations[0].frame_index

    @property
    def last_frame_index(self) -> int:
        """Return the frame the vehicle was last detected in."""
        return self.observations[-1].frame_index

    @property
    def first_timestamp_utc(self) -> datetime:
        """Return the corrected capture time of the first detection."""
        return self.observations[0].timestamp_utc

    @property
    def last_timestamp_utc(self) -> datetime:
        """Return the corrected capture time of the last detection."""
        return self.observations[-1].timestamp_utc

    @property
    def duration_sec(self) -> float:
        """Return how long the vehicle was in view, in seconds."""
        return (self.last_timestamp_utc - self.first_timestamp_utc).total_seconds()

    @property
    def hit_count(self) -> int:
        """Return how many frames the vehicle was detected in."""
        return len(self.observations)

    @property
    def detections(self) -> tuple[Detection, ...]:
        """Return every detection in the track, in frame order."""
        return tuple(observation.detection for observation in self.observations)

    @property
    def best(self) -> RankedObservation | None:
        """Return the highest-ranked retained observation, if any.

        Returns:
            The best frame, or ``None`` when no image was retained -- which is
            an honest answer rather than falling back to the first frame, since
            a caller wanting imagery has to handle its absence either way.
        """
        return self.ranked[0] if self.ranked else None

    @property
    def aggregate_confidence(self) -> float:
        """Return the track's detection confidence, as the median across frames.

        The median rather than the mean or the maximum, and the choice is load
        bearing. A track begins and ends with the vehicle half out of frame,
        where the detector is legitimately unsure; those frames drag a mean down
        and say nothing about how confident the system is that a vehicle passed.
        The maximum has the opposite problem -- one lucky frame promotes a track
        of marginal detections to near-certainty.

        Returns:
            The median confidence, in ``[0, 1]``.
        """
        return float(median(o.detection.confidence for o in self.observations))

    @property
    def object_class(self) -> ObjectClass:
        """Return the track's class, by confidence-weighted vote across frames.

        Weighted rather than a plain count because a detector that calls a
        vehicle a truck in six uncertain frames and a car in five confident ones
        is telling you something a majority vote discards. Ties break toward the
        class seen in the most frames, then alphabetically, so the result is
        deterministic.

        Returns:
            The winning class, or :attr:`ObjectClass.UNKNOWN` when the track has
            no confident opinion at all.
        """
        weights: defaultdict[ObjectClass, float] = defaultdict(float)
        counts: Counter[ObjectClass] = Counter()
        for observation in self.observations:
            detection = observation.detection
            weights[detection.object_class] += detection.confidence
            counts[detection.object_class] += 1

        if not weights:  # pragma: no cover - a track always has observations
            return ObjectClass.UNKNOWN

        best_class = max(
            weights,
            key=lambda candidate: (weights[candidate], counts[candidate], candidate.value),
        )
        return best_class

    def observation_nearest(self, instant: datetime) -> TrackObservation:
        """Return the observation captured closest to an instant.

        Ties -- two frames equidistant from the target -- resolve to the earlier
        one, so the choice does not depend on iteration order.

        Args:
            instant: The time to search around.

        Returns:
            The closest observation.
        """
        return min(
            self.observations,
            key=lambda observation: (
                abs((observation.timestamp_utc - instant).total_seconds()),
                observation.frame_index,
            ),
        )

    @property
    def midpoint_observation(self) -> TrackObservation:
        """Return the observation nearest the track's temporal midpoint.

        This is the frame the emitted sighting is dated from and boxed from; see
        :mod:`~multicam_tracker.vision.sightings` for why.

        Returns:
            The observation closest to halfway between first and last.
        """
        span = self.last_timestamp_utc - self.first_timestamp_utc
        midpoint = self.first_timestamp_utc + span / 2
        return self.observation_nearest(midpoint)

    def retained_image_count(self) -> int:
        """Return how many cropped images this track is holding.

        Returns:
            The count, which the tracker asserts stays within its configured
            cap -- the invariant that keeps a long clip from growing without
            bound.
        """
        return sum(1 for observation in self.observations if observation.has_image)

    def __repr__(self) -> str:
        """Return a representation that does not print any imagery."""
        return (
            f"VehicleTrack(track_id={self.track_id!r}, camera_id={self.camera_id!r}, "
            f"frames={self.first_frame_index}-{self.last_frame_index}, "
            f"hits={self.hit_count}, class={self.object_class.value}, "
            f"confidence={self.aggregate_confidence:.3f})"
        )
