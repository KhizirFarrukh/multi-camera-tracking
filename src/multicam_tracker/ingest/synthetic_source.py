"""Frames from nothing but a seed.

This is the source the rest of the vision work is tested against. Stages 11-13
need frames with known content at known times, and getting those from real video
means committing media, decoding it in CI, and accepting that the "expected"
answer is whatever the detector happened to produce last time. Generated frames
have an answer key.

Two properties make it useful:

**Deterministic.** The same seed produces byte-identical frames, so a test that
passes today fails tomorrow only because the code changed. The generator is
seeded per frame index rather than carried forward, which means frame 400 is the
same whether it was reached by iterating or by seeking.

**Faultable.** Real sources corrupt frames, stall, and disconnect, and code that
has never seen those does not handle them -- it merely has not been asked to.
Faults are injected by index, so a test can say "the eleventh frame is corrupt"
and assert exactly what the pipeline does about it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from multicam_tracker.clock import ensure_utc
from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.ingest.protocol import BaseVideoSource, SourceKind, SourceProperties
from multicam_tracker.logging_config import get_logger
from multicam_tracker.timesync.derivation import derive_timestamp, fps_as_fraction

__all__ = ["FaultInjection", "SyntheticVideoSource"]

logger = get_logger(__name__)

_KEYFRAME_INTERVAL = 10
"""Every tenth frame is a keyframe, mirroring a typical encoder GOP."""

_RECTANGLE_SIZE = 40


@dataclass(frozen=True)
class FaultInjection:
    """The ways a source can misbehave, on purpose.

    Args:
        corrupt_frame_indices: Frames that decode to garbage. Counted and
            skipped, never fatal -- one bad frame in an hour of footage must not
            end the run.
        stall_at_index: Frame at which the source pauses, to exercise a read
            timeout.
        stall_seconds: How long that pause lasts.
        disconnect_at_index: Frame at which the source drops the connection.
        disconnect_is_permanent: Whether the drop ends the stream or the source
            recovers and continues from the next frame.
    """

    corrupt_frame_indices: frozenset[int] = frozenset()
    stall_at_index: int | None = None
    stall_seconds: float = 0.0
    disconnect_at_index: int | None = None
    disconnect_is_permanent: bool = True


@dataclass
class SyntheticVideoSource(BaseVideoSource):
    """A deterministic generated video source.

    Args:
        camera_id: The camera these frames belong to.
        start_utc: Capture time of frame zero.
        fps: Frame rate. Rational rates are kept exact.
        frame_count: How many frames to produce.
        width: Frame width in pixels.
        height: Frame height in pixels.
        seed: Seeds the generated content.
        clock_offset_ms: Correction applied to produce ``timestamp_utc``, as
            stage 09 would apply it at ingestion.
        moving_rectangle: Draw a rectangle that moves across the frame, so
            motion detection and tracking have something to find.
        faults: Faults to inject.
        source_id: Recorded on every frame. Defaults to the camera id plus a
            suffix.
    """

    camera_id: str = "cam_01"
    start_utc: datetime = field(
        default_factory=lambda: datetime.fromisoformat("2026-08-10T14:00:00+00:00")
    )
    fps: float | Fraction = 10.0
    frame_count: int = 30
    width: int = 160
    height: int = 120
    seed: int = 0
    clock_offset_ms: int = 0
    moving_rectangle: bool = True
    faults: FaultInjection = field(default_factory=FaultInjection)
    source_id: str = ""

    def __post_init__(self) -> None:
        """Validate the configuration and initialise the base source.

        Raises:
            IngestError: If the geometry or frame count is unusable.
        """
        if self.width <= 0 or self.height <= 0:
            raise IngestError(
                "Synthetic frame dimensions must be positive",
                {"width": self.width, "height": self.height},
            )
        if self.frame_count < 0:
            raise IngestError("Frame count cannot be negative", {"frame_count": self.frame_count})

        resolved_source_id = self.source_id or f"{self.camera_id}_synthetic"
        BaseVideoSource.__init__(self, resolved_source_id, self.camera_id)
        self.source_id = resolved_source_id
        self._rate = fps_as_fraction(self.fps)
        self._start = ensure_utc(self.start_utc, field_name="start_utc")
        self.corrupt_frames_skipped = 0
        self.disconnect_count = 0

    # -- content -----------------------------------------------------------

    def render(self, frame_index: int) -> npt.NDArray[np.uint8]:
        """Return the image for one frame index.

        Seeded per index rather than carried forward, so frame 400 is the same
        whether it was reached by iterating or by seeking -- which is what makes
        a seek test meaningful.

        Args:
            frame_index: Which frame to draw.

        Returns:
            A BGR image.
        """
        generator = np.random.default_rng(self.seed * 1_000_003 + frame_index)

        # A per-frame background that changes slowly, so consecutive frames are
        # similar but not identical -- the condition a motion prefilter has to
        # cope with.
        base = np.array(
            [
                60 + (frame_index * 3) % 40,
                70 + (frame_index * 2) % 30,
                80 + frame_index % 20,
            ],
            dtype=np.uint8,
        )
        image = np.broadcast_to(base, (self.height, self.width, 3)).copy()

        # A little noise, seeded, so frames are not perfectly flat.
        noise = generator.integers(0, 6, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self.moving_rectangle and self.frame_count > 0:
            image = self._draw_rectangle(image, frame_index)

        return image

    def _draw_rectangle(
        self, image: npt.NDArray[np.uint8], frame_index: int
    ) -> npt.NDArray[np.uint8]:
        """Draw the moving rectangle onto an image.

        Args:
            image: The background.
            frame_index: Which frame, which sets the position.

        Returns:
            The image with the rectangle drawn.
        """
        size = min(_RECTANGLE_SIZE, self.width // 2, self.height // 2)
        if size <= 0:
            return image

        travel = max(1, self.width - size)
        left = (frame_index * max(1, travel // max(1, self.frame_count))) % travel
        top = (self.height - size) // 2

        image[top : top + size, left : left + size] = np.array([230, 230, 240], dtype=np.uint8)
        return image

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> SourceProperties:
        """Return the configured properties.

        Returns:
            What this source knows about itself, which is everything.
        """
        return SourceProperties(
            fps=self._rate,
            width=self.width,
            height=self.height,
            kind=SourceKind.SYNTHETIC,
            frame_count=self.frame_count,
            duration_sec=float(self.frame_count / self._rate) if self._rate else None,
            codec="synthetic",
        )

    def _frames(self) -> Iterator[Frame]:
        """Yield the configured frames, injecting any configured faults.

        Yields:
            Frames in capture order, with corrupt ones skipped.

        Raises:
            IngestError: When a permanent disconnect is injected.
        """
        for index in range(self.frame_count):
            if self.faults.stall_at_index == index and self.faults.stall_seconds > 0:
                # A real stall is the network going quiet; the read timeout is
                # what a consumer is expected to notice it with.
                logger.warning(
                    "synthetic_source_stalling",
                    source_id=self.source_id,
                    frame_index=index,
                    seconds=self.faults.stall_seconds,
                )
                time.sleep(self.faults.stall_seconds)

            if self.faults.disconnect_at_index == index:
                self.disconnect_count += 1
                logger.warning(
                    "synthetic_source_disconnected",
                    source_id=self.source_id,
                    frame_index=index,
                    permanent=self.faults.disconnect_is_permanent,
                )
                if self.faults.disconnect_is_permanent:
                    raise IngestError(
                        "Synthetic source disconnected",
                        {"source_id": self.source_id, "frame_index": index},
                    )
                # A recoverable drop costs this frame and nothing more.
                continue

            if index in self.faults.corrupt_frame_indices:
                self.corrupt_frames_skipped += 1
                logger.warning(
                    "corrupt_frame_skipped",
                    source_id=self.source_id,
                    frame_index=index,
                    detail="one unreadable frame does not end a run",
                )
                continue

            yield self._frame_at(index)

    def _frame_at(self, frame_index: int) -> Frame:
        """Build the frame for one index.

        Args:
            frame_index: Which frame.

        Returns:
            The frame, with both timestamps set.
        """
        raw = derive_timestamp(self._start, frame_index, self._rate)
        return Frame(
            image=self.render(frame_index),
            frame_index=frame_index,
            raw_timestamp=raw,
            timestamp_utc=raw + timedelta(milliseconds=self.clock_offset_ms),
            source_id=self.source_id,
            camera_id=self.camera_id,
            is_keyframe=frame_index % _KEYFRAME_INTERVAL == 0,
        )
