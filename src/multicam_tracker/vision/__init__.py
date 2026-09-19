"""Vehicle detection and single-camera tracking (stage 11).

Turns frames into sightings. One vehicle crossing one camera produces **one**
sighting backed by every frame it appeared in -- not one sighting per frame,
which is what a detector alone would give and what would make every later stage
meaningless.

The path a frame takes::

    Frame  ->  Preprocessor        (stage 10: mask, rotate, resize)
           ->  Detector            boxes, in the PROCESSED frame
           ->  in_source_coordinates   boxes, in the ORIGINAL frame
           ->  DetectionFilters    specks, whole-frame boxes, outside the ROI
           ->  SingleCameraTracker detections bound into tracks over time
           ->  VehicleTrack        emitted when the vehicle leaves
           ->  track_to_sighting   one row, one thumbnail

Four rules this package keeps, each tested:

**Boxes end up in original source-frame coordinates.** The detector sees a
preprocessed image; everything stored is in the coordinates of the frame as
captured. A box left in processed space is a plausible box in the wrong place,
and nothing downstream can tell.

**One pass, one sighting.** A track is not emitted until it ends, because which
frame of a pass is worth keeping cannot be known until the pass is over.

**Nothing is discarded silently.** Every filtered detection is counted by
reason, and every track dropped before confirmation is counted too.

**No weights are needed to run any of this.** The detector is an interface, and
:class:`~multicam_tracker.vision.fake_detector.FakeDetector` and
:class:`~multicam_tracker.vision.fixture_detector.FixtureDetector` implement it
without a model, which is what keeps stages 12-20 testable in CI.
"""

from __future__ import annotations

from multicam_tracker.vision.best_frames import (
    BestFrameWeights,
    centrality,
    laplacian_variance,
    rank_observations,
)
from multicam_tracker.vision.detector_protocol import (
    BaseDetector,
    Detection,
    Detector,
    iou,
)
from multicam_tracker.vision.factory import create_detector
from multicam_tracker.vision.fake_detector import FakeDetector
from multicam_tracker.vision.filters import (
    DetectionFilters,
    EdgePolicy,
    FilterStats,
    RejectionReason,
    RoiMode,
)
from multicam_tracker.vision.fixture_detector import (
    FIXTURE_FORMAT_VERSION,
    FixtureDetector,
    build_fixture_payload,
    write_fixture,
)
from multicam_tracker.vision.instrumentation import DetectionMetrics, StageTimer
from multicam_tracker.vision.sightings import track_to_sighting, track_to_sightings
from multicam_tracker.vision.thumbnails import (
    ThumbnailWriter,
    crop_with_padding,
    encode_jpeg,
    resize_image,
)
from multicam_tracker.vision.track import (
    RankedObservation,
    TrackObservation,
    TrackStatus,
    VehicleTrack,
)
from multicam_tracker.vision.tracker import (
    KalmanBoxTracker,
    SingleCameraTracker,
    TrackerConfig,
    TrackerStats,
    solve_min_cost_assignment,
)
from multicam_tracker.vision.yolo_detector import COCO_VEHICLE_CLASSES, YoloDetector

__all__ = [
    "COCO_VEHICLE_CLASSES",
    "FIXTURE_FORMAT_VERSION",
    "BaseDetector",
    "BestFrameWeights",
    "Detection",
    "DetectionFilters",
    "DetectionMetrics",
    "Detector",
    "EdgePolicy",
    "FakeDetector",
    "FilterStats",
    "FixtureDetector",
    "KalmanBoxTracker",
    "RankedObservation",
    "RejectionReason",
    "RoiMode",
    "SingleCameraTracker",
    "StageTimer",
    "ThumbnailWriter",
    "TrackObservation",
    "TrackStatus",
    "TrackerConfig",
    "TrackerStats",
    "VehicleTrack",
    "YoloDetector",
    "build_fixture_payload",
    "centrality",
    "create_detector",
    "crop_with_padding",
    "encode_jpeg",
    "iou",
    "laplacian_variance",
    "rank_observations",
    "resize_image",
    "solve_min_cost_assignment",
    "track_to_sighting",
    "track_to_sightings",
    "write_fixture",
]
