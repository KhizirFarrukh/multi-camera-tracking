"""Video ingestion: turning any source into timestamped frames.

One interface over a recorded file, a live RTSP stream, and a generated test
pattern, so everything downstream consumes :class:`Frame` objects and nothing
else. That is what lets stages 11-13 be tested in CI with no media files, no
network, and no GPU: :class:`SyntheticVideoSource` produces frames with an
answer key.

The stage's four standing rules, each enforced somewhere in this package:

**Coordinates are always original source-frame coordinates.** Preprocessing
returns the mapping that undoes it, and every emitted box goes back through
that mapping. A box in resized-and-rotated space is a perfectly plausible box in
the wrong place, and nothing downstream can tell.

**Timestamps are never interpolated.** A sampled frame carries the capture time
of the frame that was decoded. Smoothing a cadence would move a vehicle to a
moment nobody observed.

**Resources are released on every exit path.** Normal exit, exception, and
abandoning a generator half way -- all three release the decoder, and all three
are tested.

**Latency beats completeness on live sources.** A bounded buffer drops the
oldest frames rather than blocking the decoder, and counts every drop, because a
system silently discarding half its input while reporting healthy is worse than
one visibly falling behind.
"""

from __future__ import annotations

from multicam_tracker.ingest.file_source import DecoderBackend, FileVideoSource
from multicam_tracker.ingest.frame import Frame, as_readonly
from multicam_tracker.ingest.live_source import BackoffPolicy, LiveStreamSource, StreamMetrics
from multicam_tracker.ingest.motion import MotionGate, MotionStats, downsample_grey
from multicam_tracker.ingest.multi_source import MultiSourceReader, SourceHealth, SourceStatus
from multicam_tracker.ingest.preprocess import (
    CoordinateMapping,
    Preprocessor,
    RegionOfInterest,
    polygon_mask,
)
from multicam_tracker.ingest.protocol import (
    BaseVideoSource,
    ConnectionState,
    SourceKind,
    SourceProperties,
    VideoSource,
)
from multicam_tracker.ingest.samplers import (
    AdaptiveSampler,
    EveryNthFrame,
    FrameSampler,
    KeyframeOnlySampler,
    TargetFpsSampler,
)
from multicam_tracker.ingest.synthetic_source import FaultInjection, SyntheticVideoSource

__all__ = [
    "AdaptiveSampler",
    "BackoffPolicy",
    "BaseVideoSource",
    "ConnectionState",
    "CoordinateMapping",
    "DecoderBackend",
    "EveryNthFrame",
    "FaultInjection",
    "FileVideoSource",
    "Frame",
    "FrameSampler",
    "KeyframeOnlySampler",
    "LiveStreamSource",
    "MotionGate",
    "MotionStats",
    "MultiSourceReader",
    "Preprocessor",
    "RegionOfInterest",
    "SourceHealth",
    "SourceKind",
    "SourceProperties",
    "SourceStatus",
    "StreamMetrics",
    "SyntheticVideoSource",
    "TargetFpsSampler",
    "VideoSource",
    "as_readonly",
    "downsample_grey",
    "polygon_mask",
]
