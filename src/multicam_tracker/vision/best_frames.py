"""Deciding which frame of a pass is worth keeping.

A vehicle crossing a camera is seen forty times and read once. Which of those
forty frames the plate is read from decides whether the plate is read at all --
so this ranking, more than any threshold in the OCR stage, determines plate
accuracy. That makes it worth spelling out rather than burying in a
``max(key=...)``.

Four criteria, each scored in ``[0, 1]``, weighted by configuration:

**Detection confidence.** The detector's own opinion. A frame it was unsure
about is usually one where the vehicle is partly occluded or motion-blurred --
exactly the frames OCR fails on.

**Sharpness**, as the variance of the Laplacian over the crop. A blurred image
has little high-frequency content and therefore a low Laplacian variance. This
is the criterion that separates two frames a detector is equally happy with, and
it is measured on the crop rather than the frame because a sharp background
behind a blurred vehicle is not what anyone wants selected.

**Area**, normalized against the largest box in the same track. A closer vehicle
covers more pixels and carries more plate detail. Normalized within the track
rather than against the frame, because every vehicle is a small fraction of a
frame and an absolute ratio would make this criterion do nothing.

**Centrality.** A vehicle in the middle of the frame is fully visible, is
usually facing the camera more squarely, and suffers less lens distortion than
one at the corner.

And one penalty: a box touching the frame edge has its score discounted,
because a vehicle the frame has cut in half is missing part of the very thing it
would be selected for. The discount is multiplicative so that a track seen only
at the frame edge still has a meaningfully ordered best frame.

**Normalization is within the supplied set.** Two criteria are relative, so a
ranking depends on what it is ranking against. That is correct for choosing
among one track's frames, which is the only thing this is used for, and it is
why the score is not comparable between tracks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from multicam_tracker.config import Settings
from multicam_tracker.vision.track import RankedObservation, TrackObservation

__all__ = [
    "BestFrameWeights",
    "centrality",
    "laplacian_variance",
    "rank_observations",
]

_GREY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)
"""BGR luminance weights. Frames are BGR, so blue comes first."""

_LAPLACIAN_MIN_SIDE = 3
"""Smallest crop a 3x3 Laplacian can be evaluated on."""

_COLOUR_NDIM = 3
"""A colour image has three axes; a greyscale crop has two."""


def laplacian_variance(image: npt.NDArray[np.uint8]) -> float:
    """Return the variance of the Laplacian of an image, as a sharpness score.

    Implemented with numpy rather than ``cv2.Laplacian`` on purpose: this runs
    once per detection on a small crop, and a hard OpenCV dependency here would
    make the tracker -- which is otherwise pure arithmetic -- untestable in the
    core install.

    Args:
        image: A BGR crop.

    Returns:
        The variance, which is unbounded above and normalized by the caller.
        ``0.0`` for an image too small to convolve, which is honest: a 2x2 crop
        carries no information about focus.
    """
    if image.size == 0:
        return 0.0
    if image.shape[0] < _LAPLACIAN_MIN_SIDE or image.shape[1] < _LAPLACIAN_MIN_SIDE:
        return 0.0

    grey = image.astype(np.float32)
    grey = grey @ _GREY_WEIGHTS if grey.ndim == _COLOUR_NDIM else grey

    # The four-neighbour Laplacian, evaluated on the interior. Border pixels are
    # dropped rather than padded: padding invents an edge at the crop boundary,
    # which every crop would then score highly on.
    interior = grey[1:-1, 1:-1]
    response = grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * interior
    return float(np.var(response))


def centrality(centre: tuple[float, float], frame_size: tuple[int, int]) -> float:
    """Return how central a point is within a frame, in ``[0, 1]``.

    Args:
        centre: ``(x, y)`` of the box centre.
        frame_size: ``(width, height)`` of the frame.

    Returns:
        ``1.0`` at the exact centre, falling to ``0.0`` at the corners. ``0.0``
        for a frame of unknown size -- unknown is not central, and defaulting to
        1.0 would silently hand every frame full marks on this criterion.
    """
    width, height = frame_size
    if width <= 0 or height <= 0:
        return 0.0

    half_diagonal = float(np.hypot(width / 2.0, height / 2.0))
    if half_diagonal == 0.0:  # pragma: no cover - guarded by the size check
        return 0.0

    offset = float(np.hypot(centre[0] - width / 2.0, centre[1] - height / 2.0))
    return max(0.0, 1.0 - offset / half_diagonal)


@dataclass(frozen=True)
class BestFrameWeights:
    """How much each criterion counts.

    The four weights sum to one, which configuration enforces at startup, so the
    composite score stays in ``[0, 1]`` and means the same thing wherever it is
    displayed.

    Args:
        confidence: Weight on the detector's confidence.
        sharpness: Weight on Laplacian variance, normalized within the set.
        area: Weight on box area, normalized within the set.
        centrality: Weight on distance from the frame centre.
        edge_penalty: Fraction of the score removed when the box touches the
            frame edge. ``0.5`` halves it.
    """

    confidence: float = 0.35
    sharpness: float = 0.30
    area: float = 0.20
    centrality: float = 0.15
    edge_penalty: float = 0.5

    @classmethod
    def from_settings(cls, settings: Settings) -> BestFrameWeights:
        """Build the configured weights.

        Args:
            settings: Loaded settings.

        Returns:
            The weights from ``thresholds.yaml``.
        """
        thresholds = settings.thresholds
        return cls(
            confidence=thresholds.best_frame_confidence_weight,
            sharpness=thresholds.best_frame_sharpness_weight,
            area=thresholds.best_frame_area_weight,
            centrality=thresholds.best_frame_centrality_weight,
            edge_penalty=thresholds.best_frame_edge_penalty,
        )


def _touches_edge(observation: TrackObservation, frame_size: tuple[int, int]) -> bool:
    """Return whether this observation's box reaches the frame border.

    Two sources agree here, and both are needed. The filters set
    :attr:`~multicam_tracker.vision.detector_protocol.Detection.touches_edge`
    under the keep-but-flag policy, using a configured margin -- but filtering is
    optional, and ranking must not silently lose a criterion when it is turned
    off. So a zero-margin geometric test backs it up: literally touching the
    border is unambiguous, whatever the policy.

    Args:
        observation: The observation under test.
        frame_size: ``(width, height)``, or ``(0, 0)`` when unknown.

    Returns:
        ``True`` when either source says the box is cut off.
    """
    if observation.detection.touches_edge:
        return True

    width, height = frame_size
    if width <= 0 or height <= 0:
        return False

    x1, y1, x2, y2 = observation.detection.bbox
    return x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1


def rank_observations(
    observations: Sequence[TrackObservation],
    weights: BestFrameWeights,
    frame_size: tuple[int, int],
) -> list[RankedObservation]:
    """Rank one track's frames, best first.

    Args:
        observations: The frames to rank. Normalization is relative to this set,
            so the ranking of a subset is not guaranteed to match its ranking
            within the whole track.
        weights: How much each criterion counts.
        frame_size: ``(width, height)`` of the source frames, for centrality and
            the edge test.

    Returns:
        One :class:`~multicam_tracker.vision.track.RankedObservation` per input,
        sorted by descending score. Ties break toward the earlier frame index,
        so the result is deterministic for identical inputs -- which matters,
        because a ranking that reorders between runs makes every downstream
        comparison flaky.
    """
    if not observations:
        return []

    max_area = max(float(observation.detection.area) for observation in observations)
    max_sharpness = max(float(observation.sharpness) for observation in observations)

    ranked: list[RankedObservation] = []
    for observation in observations:
        detection = observation.detection
        components = {
            "confidence": detection.confidence,
            "sharpness": (
                0.0 if max_sharpness <= 0.0 else float(observation.sharpness) / max_sharpness
            ),
            "area": 0.0 if max_area <= 0.0 else float(detection.area) / max_area,
            "centrality": centrality(detection.centroid, frame_size),
        }
        score = (
            weights.confidence * components["confidence"]
            + weights.sharpness * components["sharpness"]
            + weights.area * components["area"]
            + weights.centrality * components["centrality"]
        )
        if _touches_edge(observation, frame_size):
            # Multiplicative rather than subtractive, and the difference is not
            # cosmetic. Subtracting a fixed 0.5 drives most edge frames to the
            # clamp at zero, where they all tie and the ranking among them is
            # decided by frame index -- which is exactly the wrong answer for a
            # track that is entirely at the frame edge, where one of those
            # frames really is the best available. A discount preserves their
            # order and keeps the score inside [0, 1] without a clamp.
            score *= 1.0 - weights.edge_penalty
            components["edge_discount"] = weights.edge_penalty

        ranked.append(
            RankedObservation(
                observation=observation,
                score=min(1.0, max(0.0, score)),
                components=components,
            )
        )

    ranked.sort(key=lambda entry: (-entry.score, entry.observation.frame_index))
    return ranked
