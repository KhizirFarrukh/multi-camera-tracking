"""Time synchronization: making timestamps from independent cameras comparable.

Small in code, large in consequence. The entire system rests on one assumption
-- that a sighting at 14:02 on one camera and 14:05 on another are five minutes
apart in reality -- and if camera 3's clock is 90 seconds fast, every route
built from it is confidently wrong in a way no downstream logic can detect. A
matching engine cannot see it. A path search cannot see it. The trajectory looks
exactly like a correct one.

So this package does four things:

1. **Derives** timestamps exactly, with rational arithmetic rather than float
   accumulation, and refuses to guess at local times that daylight saving makes
   ambiguous or impossible.
2. **Corrects** raw timestamps at a single boundary, once, so a value cannot be
   shifted twice by two well-meaning callers.
3. **Measures** each camera's offset -- directly by probing its clock where the
   device exposes one, and by inference from reference vehicle passes where it
   does not -- and separates a constant offset from a clock that is still
   drifting.
4. **Refuses** to proceed when the offsets are large enough that hop ordering
   itself is unreliable, and attaches its verdict to every trajectory so an
   operator sees the caveat beside the route rather than only the route.

The fourth is the one that matters most. Everything else here can be got wrong
and recovered from; a route presented without its temporal caveat is the failure
this stage exists to make impossible.
"""

from __future__ import annotations

from multicam_tracker.timesync.derivation import (
    COMMON_RATIONAL_FPS,
    FrameTiming,
    derive_timestamp,
    fps_as_fraction,
)
from multicam_tracker.timesync.drift import DriftAlert, DriftAnalysis, detect_drift
from multicam_tracker.timesync.estimation import (
    CameraOffsetEstimate,
    OffsetEstimation,
    ReferencePass,
    estimate_offsets,
)
from multicam_tracker.timesync.integrity import (
    CameraTimeStatus,
    assert_temporal_integrity,
    check_temporal_integrity,
)
from multicam_tracker.timesync.monitoring import (
    ClockProbe,
    ClockSample,
    ClockSampleSeries,
    probe_cameras,
)
from multicam_tracker.timesync.offsets import (
    OffsetChange,
    apply_offset,
    apply_offset_change,
    correct_sighting,
    recompute_offsets,
)
from multicam_tracker.timesync.sources import (
    FileMetadataSource,
    FilenameSource,
    ManualOffsetSource,
    OverlayOcrSource,
    ReliabilityTier,
    StreamClockSource,
    TimestampSource,
    select_source,
)
from multicam_tracker.timesync.timezones import resolve_local_time, zone_for
from multicam_tracker.timesync.watermark import (
    LatePolicy,
    WatermarkMetrics,
    WatermarkTracker,
    WatermarkVerdict,
)

__all__ = [
    "COMMON_RATIONAL_FPS",
    "CameraOffsetEstimate",
    "CameraTimeStatus",
    "ClockProbe",
    "ClockSample",
    "ClockSampleSeries",
    "DriftAlert",
    "DriftAnalysis",
    "FileMetadataSource",
    "FilenameSource",
    "FrameTiming",
    "LatePolicy",
    "ManualOffsetSource",
    "OffsetChange",
    "OffsetEstimation",
    "OverlayOcrSource",
    "ReferencePass",
    "ReliabilityTier",
    "StreamClockSource",
    "TimestampSource",
    "WatermarkMetrics",
    "WatermarkTracker",
    "WatermarkVerdict",
    "apply_offset",
    "apply_offset_change",
    "assert_temporal_integrity",
    "check_temporal_integrity",
    "correct_sighting",
    "derive_timestamp",
    "detect_drift",
    "estimate_offsets",
    "fps_as_fraction",
    "probe_cameras",
    "recompute_offsets",
    "resolve_local_time",
    "select_source",
    "zone_for",
]
