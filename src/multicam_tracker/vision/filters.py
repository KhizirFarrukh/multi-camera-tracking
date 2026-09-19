"""Throwing away detections that cannot be vehicles, and counting what went.

A detector reports everything it is willing to call a car, including specks in
the distance, boxes covering the whole frame, and objects on the neighbouring
property that this camera has no business recording. Each of those becomes a
sighting, a thumbnail, a row, and a candidate for matching if it is not stopped
here.

**Every rejection is counted by reason.** That is the part that matters more
than the filtering. A pipeline that quietly discards three quarters of its
detections and reports healthy is exactly the failure this project exists to
avoid: the operator sees a sparse trajectory and concludes the vehicle was not
there, when in fact the aspect-ratio window was misconfigured. The counters turn
that into a number someone can look at.

**Boundaries are inclusive.** A detection is kept when its area is *at least*
the minimum and *at most* the maximum, and when its aspect ratio lies within the
closed window. Stated once here and tested at the exact values, because "at the
threshold" is where a filter is argued about and a convention nobody wrote down
is one that changes by accident.

**Every filter is independently configurable, and off by default.** A field left
as ``None`` disables that filter. The default :class:`DetectionFilters` is
therefore a pass-through, and :meth:`DetectionFilters.from_settings` is what
turns the configured ones on. That ordering is deliberate: a test constructs
exactly the filter it is testing, and production gets all of them from config.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from multicam_tracker.config import Settings
from multicam_tracker.ingest.preprocess import RegionOfInterest, polygon_mask
from multicam_tracker.logging_config import get_logger
from multicam_tracker.vision.detector_protocol import Detection

__all__ = [
    "DetectionFilters",
    "EdgePolicy",
    "FilterStats",
    "RejectionReason",
    "RoiMode",
]

logger = get_logger(__name__)


class RejectionReason(StrEnum):
    """Why a detection was discarded.

    A string enum because these end up as metric label values and as keys in a
    logged payload, where a bare integer would need a lookup table nobody has.
    """

    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    ASPECT_RATIO = "aspect_ratio"
    OUTSIDE_ROI = "outside_roi"
    EDGE_TOUCHING = "edge_touching"


class EdgePolicy(StrEnum):
    """What to do with a vehicle the frame cuts in half.

    The trade is real and has no universally right answer, which is why it is a
    policy rather than a rule. A box clipped by the frame edge usually has an
    unreadable plate and a re-id embedding computed from half a car, so keeping
    it pollutes matching. Dropping it loses the only evidence that something
    passed -- and a vehicle that enters and leaves at the frame edge is every
    vehicle, at the start and end of its track.

    ``FLAG`` is the default because it preserves the record while letting the
    best-frame ranking push those frames down, which is where the decision
    actually belongs.
    """

    KEEP = "keep"
    FLAG = "flag"
    DROP = "drop"


class RoiMode(StrEnum):
    """How region-of-interest membership is decided.

    ``CENTROID`` asks where the vehicle is; ``OVERLAP`` asks how much of it is
    inside. Centroid is the default because it is stable as a vehicle moves
    through the region boundary -- an overlap fraction crosses its threshold
    twice on the way past, which flickers a track in and out of existence.
    """

    CENTROID = "centroid"
    OVERLAP = "overlap"


@dataclass
class FilterStats:
    """What the filters did, for the metrics that make the loss visible."""

    seen: int = 0
    accepted: int = 0
    flagged_edge: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    @property
    def rejected_total(self) -> int:
        """Return how many detections were discarded, for any reason."""
        return int(sum(self.rejected.values()))

    @property
    def rejection_ratio(self) -> float:
        """Return the share of detections discarded.

        Returns:
            ``0.0`` when nothing has been seen. A filter that has processed
            nothing has rejected nothing, and reporting a ratio of one would
            read as a total outage.
        """
        return 0.0 if self.seen == 0 else self.rejected_total / self.seen

    def record_rejection(self, reason: RejectionReason) -> None:
        """Count one rejection under its reason.

        Args:
            reason: Why the detection was discarded.
        """
        self.rejected[reason.value] += 1

    def as_dict(self) -> dict[str, float]:
        """Return the counters flattened for a metrics exporter.

        Returns:
            A flat mapping, with one ``rejected_<reason>`` key per reason that
            has fired. Reasons that never fired are omitted rather than reported
            as zero, so a new reason appearing in the output means something
            new happened.
        """
        flat: dict[str, float] = {
            "seen": self.seen,
            "accepted": self.accepted,
            "flagged_edge": self.flagged_edge,
            "rejected_total": self.rejected_total,
            "rejection_ratio": round(self.rejection_ratio, 4),
        }
        for reason, count in sorted(self.rejected.items()):
            flat[f"rejected_{reason}"] = count
        return flat


@dataclass
class DetectionFilters:
    """Post-detection filtering, each criterion independently configurable.

    Every bound is optional, and ``None`` means that filter does not run. A
    default-constructed instance therefore passes every detection through
    unchanged, which is what a test that is not testing filtering wants.

    Args:
        min_area_px: Smallest box area kept, inclusive.
        max_area_fraction: Largest box area kept as a fraction of the frame,
            inclusive.
        min_aspect_ratio: Narrowest width-to-height ratio kept, inclusive.
        max_aspect_ratio: Widest ratio kept, inclusive.
        roi: Region the vehicle must be in.
        roi_mode: How membership is decided.
        roi_min_overlap: Fraction of the box that must be inside, in overlap
            mode.
        edge_policy: What to do with a box touching the frame edge.
        edge_margin_px: How close to a border counts as touching it.
    """

    min_area_px: int | None = None
    max_area_fraction: float | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    roi: RegionOfInterest | None = None
    roi_mode: RoiMode = RoiMode.CENTROID
    roi_min_overlap: float = 0.5
    edge_policy: EdgePolicy = EdgePolicy.KEEP
    edge_margin_px: int = 0
    _roi_masks: dict[tuple[int, int], npt.NDArray[np.bool_]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, roi: RegionOfInterest | None = None
    ) -> DetectionFilters:
        """Build the production filter set from configuration.

        Args:
            settings: Loaded settings. Bounds come from ``thresholds``; policy
                and mode come from the ``detection`` section.
            roi: The camera region, which is per-camera and so cannot come from
                a global setting.

        Returns:
            A filter set with every configured filter enabled.
        """
        thresholds = settings.thresholds
        detection = settings.detection
        return cls(
            min_area_px=thresholds.detection_min_bbox_area_px,
            max_area_fraction=thresholds.detection_max_bbox_area_fraction,
            min_aspect_ratio=thresholds.detection_min_aspect_ratio,
            max_aspect_ratio=thresholds.detection_max_aspect_ratio,
            roi=roi,
            roi_mode=RoiMode(detection.roi_mode),
            roi_min_overlap=thresholds.detection_roi_min_overlap,
            edge_policy=EdgePolicy(detection.edge_policy),
            edge_margin_px=detection.edge_margin_px,
        )

    @property
    def is_pass_through(self) -> bool:
        """Return whether this configuration would reject nothing."""
        return (
            self.min_area_px is None
            and self.max_area_fraction is None
            and self.min_aspect_ratio is None
            and self.max_aspect_ratio is None
            and self.roi is None
            and self.edge_policy in (EdgePolicy.KEEP, EdgePolicy.FLAG)
        )

    def apply(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
        stats: FilterStats | None = None,
    ) -> list[Detection]:
        """Return the detections worth keeping, counting what was not.

        Args:
            detections: What the detector reported.
            frame_width: Source frame width, for the area and edge tests.
            frame_height: Source frame height.
            stats: Counters to accumulate into. Supply the same instance across
                a whole run; a fresh one per frame makes the ratios meaningless.

        Returns:
            The surviving detections, in input order. Under
            :attr:`EdgePolicy.FLAG` a surviving edge-touching detection is
            returned with :attr:`Detection.touches_edge` set.
        """
        counters = stats if stats is not None else FilterStats()
        kept: list[Detection] = []

        for detection in detections:
            counters.seen += 1
            reason = self._rejection_for(detection, frame_width, frame_height)
            if reason is not None:
                counters.record_rejection(reason)
                logger.debug(
                    "detection_rejected",
                    reason=reason.value,
                    bbox=list(detection.bbox),
                    confidence=round(detection.confidence, 4),
                )
                continue

            survivor = detection
            if self.edge_policy is EdgePolicy.FLAG and self._touches_edge(
                detection, frame_width, frame_height
            ):
                survivor = replace(detection, touches_edge=True)
                counters.flagged_edge += 1

            counters.accepted += 1
            kept.append(survivor)

        return kept

    # -- individual criteria ------------------------------------------------

    def _rejection_for(
        self, detection: Detection, frame_width: int, frame_height: int
    ) -> RejectionReason | None:
        """Return the first reason this detection fails, if any.

        First rather than all: a detection is discarded once, and attributing it
        to the first criterion it fails keeps the counters summing to the number
        of rejections. Counting a box under three reasons at once would make
        ``rejected_total`` exceed ``seen`` and the ratio exceed one.

        Args:
            detection: The detection under test.
            frame_width: Source frame width.
            frame_height: Source frame height.

        Returns:
            The reason, or ``None`` when the detection survives.
        """
        if self.min_area_px is not None and detection.area < self.min_area_px:
            return RejectionReason.TOO_SMALL

        if self.max_area_fraction is not None:
            frame_area = frame_width * frame_height
            if frame_area > 0 and detection.area > self.max_area_fraction * frame_area:
                return RejectionReason.TOO_LARGE

        ratio = detection.aspect_ratio
        if self.min_aspect_ratio is not None and ratio < self.min_aspect_ratio:
            return RejectionReason.ASPECT_RATIO
        if self.max_aspect_ratio is not None and ratio > self.max_aspect_ratio:
            return RejectionReason.ASPECT_RATIO

        if self.roi is not None and not self._inside_roi(detection, frame_width, frame_height):
            return RejectionReason.OUTSIDE_ROI

        if self.edge_policy is EdgePolicy.DROP and self._touches_edge(
            detection, frame_width, frame_height
        ):
            return RejectionReason.EDGE_TOUCHING

        return None

    def _touches_edge(self, detection: Detection, frame_width: int, frame_height: int) -> bool:
        """Return whether the box reaches the frame border.

        Args:
            detection: The detection under test.
            frame_width: Source frame width.
            frame_height: Source frame height.

        Returns:
            ``True`` when any side of the box lies within
            :attr:`edge_margin_px` of the corresponding border.
        """
        x1, y1, x2, y2 = detection.bbox
        margin = self.edge_margin_px
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= frame_width - 1 - margin
            or y2 >= frame_height - 1 - margin
        )

    def _inside_roi(self, detection: Detection, frame_width: int, frame_height: int) -> bool:
        """Return whether the detection counts as inside the region.

        Args:
            detection: The detection under test.
            frame_width: Source frame width.
            frame_height: Source frame height.

        Returns:
            ``True`` when the detection satisfies the configured membership
            rule.
        """
        assert self.roi is not None
        mask = self._mask_for(frame_width, frame_height)

        if self.roi_mode is RoiMode.CENTROID:
            centre_x, centre_y = detection.centroid
            column = int(centre_x)
            row = int(centre_y)
            if not (0 <= row < frame_height and 0 <= column < frame_width):
                return False
            return bool(mask[row, column])

        x1, y1, x2, y2 = detection.clamped_to(frame_width, frame_height).bbox
        if x2 <= x1 or y2 <= y1:
            return False
        window = mask[y1:y2, x1:x2]
        inside_fraction = float(window.mean()) if window.size else 0.0
        return inside_fraction >= self.roi_min_overlap

    def _mask_for(self, width: int, height: int) -> npt.NDArray[np.bool_]:
        """Return the cached region mask for one frame size.

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            A boolean mask, ``True`` inside the region.
        """
        key = (width, height)
        if key not in self._roi_masks:
            assert self.roi is not None
            self._roi_masks[key] = polygon_mask(self.roi.polygon, width, height)
        return self._roi_masks[key]
