"""Binding one vehicle's detections together across frames.

A detector has no memory. Shown forty consecutive frames of one car it reports
forty independent boxes, and without something to say "these are all the same
vehicle" the system stores forty sightings, crops forty thumbnails, and builds a
trajectory that appears to visit one camera forty times in eight seconds.

This module is that something: a SORT-style tracker that predicts where each
known vehicle should be next, matches detections to those predictions, and emits
one :class:`~multicam_tracker.vision.track.VehicleTrack` per pass.

**Why a motion model and not just overlap.** Matching on raw box overlap works
until two vehicles pass each other, at which point their boxes overlap each
other as much as they overlap themselves and the identities swap. A Kalman
filter carrying each track's velocity keeps predicting them *through* the
crossing, so the overlap that matters is with where each vehicle should be, not
where it was. The crossing case is the one this design exists for, and it is
tested directly.

**Why optimal assignment and not greedy matching.** Given two tracks and two
detections, greedy matching takes the single best pair first and leaves the
other track whatever remains -- which can be the wrong detection even when a
different pairing is better overall. That is precisely an identity swap, and it
happens exactly when two vehicles are close together, which is exactly when a
swap matters. So the matching is solved as a minimum-cost assignment problem
rather than picked off in order. The unit tests include a matrix where greedy
gets it wrong.

**Why age is counted in updates, not frame indices.** The tracker is fed sampled
frames (stage 10), so consecutive updates can be six source frames apart. The
gap that matters is the gap in what the tracker *saw*, and counting source
indices would make the occlusion allowance depend on the sampling rate.

**Memory.** A track keeps a few numbers per frame and a cropped image for only
its best few, so a vehicle parked in view for ten minutes costs kilobytes rather
than gigabytes. See :mod:`~multicam_tracker.vision.track`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from multicam_tracker.config import Settings
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.logging_config import get_logger
from multicam_tracker.vision.best_frames import (
    BestFrameWeights,
    laplacian_variance,
    rank_observations,
)
from multicam_tracker.vision.detector_protocol import Detection, iou
from multicam_tracker.vision.thumbnails import crop_with_padding
from multicam_tracker.vision.track import (
    TrackObservation,
    TrackStatus,
    VehicleTrack,
)

__all__ = [
    "KalmanBoxTracker",
    "SingleCameraTracker",
    "TrackerConfig",
    "TrackerStats",
    "solve_min_cost_assignment",
]

logger = get_logger(__name__)

_FORBIDDEN_COST = 1.0e6
"""Cost standing in for a pair that may not be matched.

A finite value rather than infinity on purpose: the assignment solver subtracts
potentials from costs, and an infinity there propagates into every row it
touches. A cost far above any real one has the same effect on the solution and
none of the arithmetic hazards; forbidden pairs are stripped from the result
afterwards."""


def solve_min_cost_assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return the pairing of rows to columns with the lowest total cost.

    The Jonker-Volgenant shortest-augmenting-path method, in ``O(n^3)``. Written
    out rather than taken from SciPy because SciPy is not otherwise a dependency
    of this project, and this is the only place it would be needed.

    Args:
        cost: A rectangular matrix. Rows and columns need not be equal in
            number; the smaller side is fully assigned.

    Returns:
        ``(row, column)`` pairs, sorted by row. Every row is paired when there
        are at least as many columns as rows, and vice versa.
    """
    rows = len(cost)
    if rows == 0 or len(cost[0]) == 0:
        return []
    columns = len(cost[0])

    transposed = rows > columns
    if transposed:
        matrix = [[float(cost[r][c]) for r in range(rows)] for c in range(columns)]
        n, m = columns, rows
    else:
        matrix = [[float(value) for value in row] for row in cost]
        n, m = rows, columns

    # One-indexed throughout: index 0 is the algorithm's sentinel row/column,
    # which is what lets the augmenting path terminate without a separate flag.
    potentials_row = [0.0] * (n + 1)
    potentials_col = [0.0] * (m + 1)
    assigned_row_of_col = [0] * (m + 1)
    previous_col = [0] * (m + 1)

    for row_index in range(1, n + 1):
        assigned_row_of_col[0] = row_index
        current_col = 0
        slack = [math.inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[current_col] = True
            current_row = assigned_row_of_col[current_col]
            delta = math.inf
            next_col = 0

            for column in range(1, m + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[current_row - 1][column - 1]
                    - potentials_row[current_row]
                    - potentials_col[column]
                )
                if reduced < slack[column]:
                    slack[column] = reduced
                    previous_col[column] = current_col
                if slack[column] < delta:
                    delta = slack[column]
                    next_col = column

            for column in range(m + 1):
                if used[column]:
                    potentials_row[assigned_row_of_col[column]] += delta
                    potentials_col[column] -= delta
                else:
                    slack[column] -= delta

            current_col = next_col
            if assigned_row_of_col[current_col] == 0:
                break

        while current_col != 0:
            source = previous_col[current_col]
            assigned_row_of_col[current_col] = assigned_row_of_col[source]
            current_col = source

    pairs: list[tuple[int, int]] = []
    for column in range(1, m + 1):
        row = assigned_row_of_col[column]
        if row == 0:
            continue
        pairs.append((column - 1, row - 1) if transposed else (row - 1, column - 1))

    return sorted(pairs)


class KalmanBoxTracker:
    """A constant-velocity motion model for one box.

    State is ``[cx, cy, scale, ratio, dcx, dcy, dscale]``: centre, area, aspect
    ratio, and the rates of change of the first three. Aspect ratio is modelled
    as constant, because a vehicle does not change shape -- what changes is how
    much of the frame it fills and where it is.

    The matrices are SORT's. They are not tuned to this project and should not
    be presented as if they were: they encode "position is observed fairly
    precisely, velocity is not observed at all and starts very uncertain", which
    is true of any box tracker.

    Args:
        bbox: The box this track starts from.
    """

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self._transition = np.eye(7, dtype=np.float64)
        for axis in range(3):
            self._transition[axis, axis + 4] = 1.0

        self._observation = np.zeros((4, 7), dtype=np.float64)
        for axis in range(4):
            self._observation[axis, axis] = 1.0

        self._covariance = np.eye(7, dtype=np.float64) * 10.0
        # Velocities are unobservable at birth, so they start almost entirely
        # uncertain. Without this the filter trusts an initial velocity of zero
        # and lags a full frame behind every moving vehicle.
        self._covariance[4:, 4:] *= 1000.0

        self._process_noise = np.eye(7, dtype=np.float64)
        self._process_noise[4:, 4:] *= 0.01
        self._process_noise[6, 6] *= 0.01

        self._measurement_noise = np.eye(4, dtype=np.float64)
        self._measurement_noise[2:, 2:] *= 10.0

        self._state = np.zeros((7, 1), dtype=np.float64)
        self._state[:4, 0] = self._to_measurement(bbox)

    @staticmethod
    def _to_measurement(bbox: tuple[int, int, int, int]) -> npt.NDArray[np.float64]:
        """Convert a box to the filter's measurement space.

        Args:
            bbox: ``(x1, y1, x2, y2)``.

        Returns:
            ``[centre_x, centre_y, area, aspect_ratio]``.
        """
        x1, y1, x2, y2 = bbox
        width = max(float(x2 - x1), 1e-6)
        height = max(float(y2 - y1), 1e-6)
        return np.array(
            [x1 + width / 2.0, y1 + height / 2.0, width * height, width / height],
            dtype=np.float64,
        )

    @staticmethod
    def _to_bbox(state: npt.NDArray[np.float64]) -> tuple[int, int, int, int]:
        """Convert the filter's state back to a box.

        Args:
            state: The seven-element state column.

        Returns:
            ``(x1, y1, x2, y2)``, clamped to non-negative integers.
        """
        centre_x, centre_y, area, ratio = (float(value) for value in state[:4, 0])
        area = max(area, 0.0)
        ratio = max(ratio, 1e-6)
        width = math.sqrt(area * ratio)
        height = 0.0 if width <= 0.0 else area / width

        return (
            max(0, round(centre_x - width / 2.0)),
            max(0, round(centre_y - height / 2.0)),
            max(0, round(centre_x + width / 2.0)),
            max(0, round(centre_y + height / 2.0)),
        )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return the current estimated box."""
        return self._to_bbox(self._state)

    def predict(self) -> tuple[int, int, int, int]:
        """Advance the state one step and return the predicted box.

        Returns:
            Where this vehicle is expected to be in the next frame.
        """
        # A negative area predicted by a shrinking track is meaningless; zeroing
        # the rate is cheaper than letting the box invert and then repairing it.
        if self._state[6, 0] + self._state[2, 0] <= 0:
            self._state[6, 0] = 0.0

        self._state = self._transition @ self._state
        self._covariance = (
            self._transition @ self._covariance @ self._transition.T + self._process_noise
        )
        return self.bbox

    def update(self, bbox: tuple[int, int, int, int]) -> None:
        """Correct the state with an observed box.

        Args:
            bbox: The detection this track was matched to.
        """
        measurement = self._to_measurement(bbox).reshape(4, 1)
        residual = measurement - self._observation @ self._state
        residual_covariance = (
            self._observation @ self._covariance @ self._observation.T + self._measurement_noise
        )
        gain = self._covariance @ self._observation.T @ np.linalg.inv(residual_covariance)

        self._state = self._state + gain @ residual
        self._covariance = (np.eye(7) - gain @ self._observation) @ self._covariance


@dataclass
class TrackerConfig:
    """How the tracker associates, confirms, and retains.

    Args:
        min_iou: Overlap below which a detection may not be matched to a track.
        max_age_frames: Consecutive updates a track survives without a match.
        min_hits: Detections a track needs before it may emit a sighting.
        max_retained_frames: Cropped images kept per track.
        crop_padding_fraction: Context kept around each retained crop.
        weights: Best-frame ranking weights.
        retain_crops: Whether to keep imagery at all. Off makes the tracker pure
            arithmetic, which is what the association tests want.
    """

    min_iou: float = 0.3
    max_age_frames: int = 5
    min_hits: int = 3
    max_retained_frames: int = 5
    crop_padding_fraction: float = 0.1
    weights: BestFrameWeights = field(default_factory=BestFrameWeights)
    retain_crops: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> TrackerConfig:
        """Build the configured tracker parameters.

        Args:
            settings: Loaded settings.

        Returns:
            Parameters from ``thresholds.yaml`` and the ``detection`` section.
        """
        return cls(
            min_iou=settings.thresholds.track_association_min_iou,
            max_age_frames=settings.thresholds.track_max_age_frames,
            min_hits=settings.thresholds.track_min_hits_to_confirm,
            max_retained_frames=settings.detection.max_retained_frames,
            crop_padding_fraction=settings.detection.crop_padding_fraction,
            weights=BestFrameWeights.from_settings(settings),
        )


@dataclass
class TrackerStats:
    """What the tracker did, for the throughput and quality metrics."""

    updates: int = 0
    detections_seen: int = 0
    associations: int = 0
    tracks_started: int = 0
    tracks_emitted: int = 0
    tracks_discarded_unconfirmed: int = 0

    @property
    def association_rate(self) -> float:
        """Return the share of detections that continued an existing track.

        A low rate on a busy camera means tracks are fragmenting -- usually a
        sampling cadence too slow for the traffic speed, which shows up as
        duplicate sightings rather than as an error.

        Returns:
            ``0.0`` when nothing has been seen.
        """
        return 0.0 if self.detections_seen == 0 else self.associations / self.detections_seen

    def as_dict(self) -> dict[str, float]:
        """Return the counters flattened for a metrics exporter.

        Returns:
            A flat mapping.
        """
        return {
            "updates": self.updates,
            "detections_seen": self.detections_seen,
            "associations": self.associations,
            "tracks_started": self.tracks_started,
            "tracks_emitted": self.tracks_emitted,
            "tracks_discarded_unconfirmed": self.tracks_discarded_unconfirmed,
            "association_rate": round(self.association_rate, 4),
        }


class _LiveTrack:
    """A track the tracker is still following.

    Args:
        track_id: Its stable identifier.
        detection: The detection that started it.
        config: The tracker configuration.
    """

    def __init__(self, track_id: str, detection: Detection, config: TrackerConfig) -> None:
        self.track_id = track_id
        self.config = config
        self.kalman = KalmanBoxTracker(detection.bbox)
        self.observations: list[TrackObservation] = []
        self.retained_indices: list[int] = []
        self.hits = 0
        self.frames_since_update = 0
        self.status = TrackStatus.TENTATIVE
        self.frame_size = (0, 0)
        self.clock_offset_ms = 0

    @property
    def is_confirmed(self) -> bool:
        """Return whether this track has earned the right to emit a sighting."""
        return self.status is TrackStatus.CONFIRMED

    def observe(self, frame: Frame, detection: Detection) -> None:
        """Record one matched detection.

        Args:
            frame: The frame it came from, in original source coordinates.
            detection: The matched detection.
        """
        self.kalman.update(detection.bbox)
        self.hits += 1
        self.frames_since_update = 0
        self.frame_size = (frame.width, frame.height)
        self.clock_offset_ms = round(
            (frame.timestamp_utc - frame.raw_timestamp).total_seconds() * 1000
        )

        crop: npt.NDArray[np.uint8] | None = None
        sharpness = 0.0
        if self.config.retain_crops:
            crop = crop_with_padding(
                frame.image,
                detection.bbox,
                padding_fraction=self.config.crop_padding_fraction,
            )
            sharpness = laplacian_variance(crop)

        observation = TrackObservation(
            frame_index=frame.frame_index,
            raw_timestamp=frame.raw_timestamp,
            timestamp_utc=frame.timestamp_utc,
            detection=detection,
            crop=crop,
            sharpness=sharpness,
        )
        self.observations.append(observation)

        if crop is not None:
            self.retained_indices.append(len(self.observations) - 1)
            self._evict_worst_if_over_cap()

        if self.hits >= self.config.min_hits and self.status is TrackStatus.TENTATIVE:
            self.status = TrackStatus.CONFIRMED

    def _evict_worst_if_over_cap(self) -> None:
        """Drop the lowest-ranked retained image once the cap is exceeded.

        A sliding top-K rather than a ranking over the whole track, and the
        difference is worth knowing: a frame evicted early cannot come back even
        if every later frame turns out to be worse. That is the price of a fixed
        memory bound, and it is the right trade -- the alternative is holding
        every crop of a ten-minute track to be sure.
        """
        if len(self.retained_indices) <= self.config.max_retained_frames:
            return

        ranked = rank_observations(
            [self.observations[index] for index in self.retained_indices],
            self.config.weights,
            self.frame_size,
        )
        worst_frame_index = ranked[-1].observation.frame_index
        for position, index in enumerate(self.retained_indices):
            if self.observations[index].frame_index == worst_frame_index:
                self.observations[index] = self.observations[index].without_image()
                self.retained_indices.pop(position)
                return

    def finish(self, camera_id: str, source_id: str) -> VehicleTrack:
        """Close the track and return it as an emitted object.

        Args:
            camera_id: Which camera saw it.
            source_id: Which file or stream the frames came from.

        Returns:
            The finished track, with its retained frames ranked.
        """
        ranked = rank_observations(
            [self.observations[index] for index in self.retained_indices],
            self.config.weights,
            self.frame_size,
        )
        return VehicleTrack(
            track_id=self.track_id,
            camera_id=camera_id,
            source_id=source_id,
            observations=tuple(self.observations),
            ranked=tuple(ranked),
            frame_size=self.frame_size,
            clock_offset_ms=self.clock_offset_ms,
        )


class SingleCameraTracker:
    """Associates detections across frames from one camera.

    Fed frames in capture order together with that frame's detections, and
    hands back completed tracks as they end. Nothing is emitted until a track
    has ended, because the best frame of a pass is not known until the pass is
    over.

    Args:
        camera_id: Which camera these frames are from.
        source_id: Which file or stream they came from.
        config: Association, confirmation, and retention parameters.
    """

    def __init__(self, camera_id: str, source_id: str, config: TrackerConfig | None = None) -> None:
        self.camera_id = camera_id
        self.source_id = source_id
        self.config = config or TrackerConfig()
        self.stats = TrackerStats()
        self._live: list[_LiveTrack] = []
        self._next_ordinal = 0

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the tracks currently being followed."""
        return tuple(track.track_id for track in self._live)

    @property
    def retained_image_count(self) -> int:
        """Return how many cropped images the tracker is holding in total.

        The number the memory-bound test watches: it must never exceed
        ``max_retained_frames`` times the number of live tracks, whatever the
        length of the clip.
        """
        return sum(len(track.retained_indices) for track in self._live)

    def update(self, frame: Frame, detections: Sequence[Detection]) -> list[VehicleTrack]:
        """Associate one frame's detections and return any tracks that ended.

        Args:
            frame: The source frame, in original coordinates.
            detections: Its detections, already filtered and already mapped back
                to source coordinates.

        Returns:
            Tracks terminated by this update, confirmed ones only. A track that
            never reached the confirmation threshold is discarded and counted --
            it is a detector that fired once or twice in the same place, and
            emitting it would put a phantom vehicle in the record.
        """
        self.stats.updates += 1
        self.stats.detections_seen += len(detections)

        predictions = [track.kalman.predict() for track in self._live]
        matches, unmatched_detections = self._associate(predictions, detections)

        for track_position, detection_index in matches:
            self._live[track_position].observe(frame, detections[detection_index])
            self.stats.associations += 1

        matched_tracks = {position for position, _ in matches}
        for position, track in enumerate(self._live):
            if position not in matched_tracks:
                track.frames_since_update += 1

        for detection_index in unmatched_detections:
            self._start_track(frame, detections[detection_index])

        return self._retire_stale()

    def flush(self) -> list[VehicleTrack]:
        """End every remaining track and return the confirmed ones.

        Called when the source runs out. Without it, every vehicle still in view
        at the end of a clip is silently lost -- which on a five-minute clip is
        every vehicle still on the road.

        Returns:
            The confirmed tracks that were still live.
        """
        emitted: list[VehicleTrack] = []
        for track in self._live:
            finished = self._close(track)
            if finished is not None:
                emitted.append(finished)
        self._live = []
        return emitted

    # -- internals ---------------------------------------------------------

    def _associate(
        self,
        predictions: Sequence[tuple[int, int, int, int]],
        detections: Sequence[Detection],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Match predicted track boxes to detections.

        Args:
            predictions: Where each live track is expected to be.
            detections: This frame's detections.

        Returns:
            ``(matches, unmatched_detection_indices)`` where each match is
            ``(track_position, detection_index)``.
        """
        if not predictions or not detections:
            return [], list(range(len(detections)))

        cost: list[list[float]] = []
        for prediction in predictions:
            row: list[float] = []
            for detection in detections:
                overlap = iou(prediction, detection.bbox)
                row.append(1.0 - overlap if overlap >= self.config.min_iou else _FORBIDDEN_COST)
            cost.append(row)

        matches = [
            (track_position, detection_index)
            for track_position, detection_index in solve_min_cost_assignment(cost)
            if cost[track_position][detection_index] < _FORBIDDEN_COST
        ]

        matched_detections = {index for _, index in matches}
        unmatched = [index for index in range(len(detections)) if index not in matched_detections]
        return matches, unmatched

    def _start_track(self, frame: Frame, detection: Detection) -> None:
        """Begin following a detection nothing else accounted for.

        Args:
            frame: The frame it was found in.
            detection: The unmatched detection.
        """
        track_id = f"{self.camera_id}-{self._next_ordinal:05d}"
        self._next_ordinal += 1
        track = _LiveTrack(track_id, detection, self.config)
        track.observe(frame, detection)
        self._live.append(track)
        self.stats.tracks_started += 1

    def _retire_stale(self) -> list[VehicleTrack]:
        """Terminate tracks that have gone unmatched for too long.

        A track survives up to and including :attr:`TrackerConfig.max_age_frames`
        consecutive updates without a match; the update after that ends it. The
        inclusivity is stated because it is exactly what the occlusion tests
        assert at the boundary.

        Returns:
            The confirmed tracks that ended.
        """
        emitted: list[VehicleTrack] = []
        surviving: list[_LiveTrack] = []

        for track in self._live:
            if track.frames_since_update > self.config.max_age_frames:
                finished = self._close(track)
                if finished is not None:
                    emitted.append(finished)
            else:
                surviving.append(track)

        self._live = surviving
        return emitted

    def _close(self, track: _LiveTrack) -> VehicleTrack | None:
        """Finish a track, returning it only if it was ever confirmed.

        Args:
            track: The track to close.

        Returns:
            The emitted track, or ``None`` when it never reached the
            confirmation threshold.
        """
        was_confirmed = track.is_confirmed
        track.status = TrackStatus.TERMINATED
        if not was_confirmed:
            self.stats.tracks_discarded_unconfirmed += 1
            logger.debug(
                "track_discarded_unconfirmed",
                track_id=track.track_id,
                camera_id=self.camera_id,
                hits=track.hits,
                required=self.config.min_hits,
            )
            return None

        self.stats.tracks_emitted += 1
        finished = track.finish(self.camera_id, self.source_id)
        logger.debug(
            "track_emitted",
            track_id=finished.track_id,
            camera_id=self.camera_id,
            frames=[finished.first_frame_index, finished.last_frame_index],
            hits=finished.hit_count,
        )
        return finished
