"""Decoding recorded files, and surviving the ones that are damaged.

Two decoders, because neither is enough on its own. OpenCV is fast, ubiquitous,
and what most of this ecosystem already depends on -- but it reports frame rates
as floats, gives up quietly on some containers, and has no useful notion of a
presentation timestamp. PyAV exposes the container properly: exact rational
rates, per-frame PTS, and a codec name to put in an error message. So OpenCV
runs first and PyAV catches what it drops, and which one served a file is
recorded rather than hidden.

Three kinds of damage, and what each does:

**A corrupt frame mid-file** is counted and skipped. An hour of footage with one
unreadable frame is an hour of usable footage; aborting the run would discard
59 good minutes to punish one bad frame.

**A truncated file** yields what is readable and then stops, with a warning
naming how many frames it managed. This is what a recording that was still being
written when the power went out looks like, and it is common enough that
treating it as an error would mean treating a normal event as a failure.

**An unsupported codec** raises, naming the codec. Nothing else can be done with
it, and the operator's next step is to install a decoder or transcode the file
-- both of which need the codec's name.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from multicam_tracker.clock import ensure_utc
from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.ingest.protocol import BaseVideoSource, SourceKind, SourceProperties
from multicam_tracker.logging_config import get_logger
from multicam_tracker.timesync.derivation import derive_timestamp, fps_as_fraction

__all__ = ["DecoderBackend", "FileVideoSource"]

logger = get_logger(__name__)

_MAX_CONSECUTIVE_FAILURES = 30
"""Consecutive unreadable frames before a file is called truncated.

One bad frame is damage; thirty in a row is the end of the readable data. At a
typical rate that is a second of footage -- long enough not to trip on a short
run of corruption, short enough not to spin at the end of every truncated file.
"""


class DecoderBackend:
    """Which library decoded a file. Recorded on the properties for diagnosis."""

    OPENCV = "opencv"
    PYAV = "pyav"


@dataclass
class FileVideoSource(BaseVideoSource):
    """A recorded video file.

    Args:
        path: The file to decode.
        camera_id: The camera these frames belong to.
        start_utc: Capture time of frame zero. Supplied by stage 09's source
            selection, which knows whether it came from the container, the
            filename, or an operator.
        clock_offset_ms: The camera's clock correction, applied to produce
            ``timestamp_utc``.
        start_frame: Seek to this frame before reading.
        max_frames: Stop after this many frames are delivered.
        duration_sec: Stop once this much source time has elapsed.
        prefer_pyav: Use PyAV first. Off by default; it is the fallback, and it
            is slower on the common case.
        source_id: Recorded on every frame. Defaults to the file name.
    """

    path: Path = Path()
    camera_id: str = "cam_01"
    start_utc: datetime | None = None
    clock_offset_ms: int = 0
    start_frame: int = 0
    max_frames: int | None = None
    duration_sec: float | None = None
    prefer_pyav: bool = False
    source_id: str = ""

    _capture: Any = field(default=None, init=False, repr=False)
    _container: Any = field(default=None, init=False, repr=False)
    _backend: str = field(default=DecoderBackend.OPENCV, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the path and initialise the base source.

        Raises:
            IngestError: If the file does not exist, or the seek is negative.
        """
        self.path = Path(self.path)
        if not self.path.exists():
            raise IngestError("Video file does not exist", {"path": str(self.path)})
        if self.start_frame < 0:
            raise IngestError("Start frame cannot be negative", {"start_frame": self.start_frame})

        resolved = self.source_id or self.path.name
        BaseVideoSource.__init__(self, resolved, self.camera_id)
        self.source_id = resolved
        self.corrupt_frames_skipped = 0
        self.was_truncated = False

    @property
    def backend(self) -> str:
        """Return which decoder is serving this file."""
        return self._backend

    # -- opening -----------------------------------------------------------

    def _open(self) -> SourceProperties:
        """Open the file with OpenCV, falling back to PyAV.

        Returns:
            The file's properties.

        Raises:
            IngestError: If neither decoder can open it.
        """
        # PyAV identifies; OpenCV decodes. FFmpeg's raw-format guesser will
        # happily "open" an arbitrary byte stream and report a plausible
        # resolution for it -- a file of ASCII text came back as 640x64 at 25
        # fps -- so OpenCV's willingness to open something is not evidence that
        # it is a video. The header probe is what makes the unsupported case an
        # error rather than a page of noise frames.
        identified = self._probe_codec()
        if identified is None:
            raise IngestError(
                "This file contains no decodable video stream",
                {"path": str(self.path), "codec": "unknown"},
            )

        if not self.prefer_pyav:
            properties = self._try_opencv()
            if properties is not None:
                return properties
            logger.warning(
                "opencv_decode_failed",
                path=str(self.path),
                detail="falling back to PyAV, which handles containers OpenCV drops",
            )

        properties = self._try_pyav()
        if properties is not None:
            return properties

        raise IngestError(
            "Neither OpenCV nor PyAV could decode this file; its codec is unsupported",
            {"path": str(self.path), "codec": identified},
        )

    def _try_opencv(self) -> SourceProperties | None:
        """Attempt to open the file with OpenCV.

        Returns:
            The properties, or ``None`` when OpenCV cannot read it.
        """
        try:
            import cv2
        except ImportError:
            return None

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            return None

        fps_value = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0 or fps_value <= 0:
            capture.release()
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self._capture = capture
        self._backend = DecoderBackend.OPENCV
        # OpenCV reports a float; fps_as_fraction maps the broadcast rates back
        # to the rationals they really are.
        rate = fps_as_fraction(round(fps_value, 3))
        return SourceProperties(
            fps=rate,
            width=width,
            height=height,
            kind=SourceKind.FILE,
            frame_count=frame_count if frame_count > 0 else None,
            duration_sec=float(frame_count / rate) if frame_count > 0 and rate else None,
            codec=DecoderBackend.OPENCV,
        )

    def _try_pyav(self) -> SourceProperties | None:
        """Attempt to open the file with PyAV.

        Returns:
            The properties, or ``None`` when PyAV cannot read it.
        """
        try:
            import av
        except ImportError:
            return None

        try:
            container = av.open(str(self.path))
            stream = container.streams.video[0]
        except Exception:  # pragma: no cover - depends on the file
            return None

        self._container = container
        self._backend = DecoderBackend.PYAV
        # PyAV gives the rate as an exact Fraction, which is the whole reason it
        # is worth falling back to: 30000/1001 stays 30000/1001.
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(25, 1)
        return SourceProperties(
            fps=rate,
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            kind=SourceKind.FILE,
            frame_count=int(stream.frames) if stream.frames else None,
            duration_sec=(
                float(stream.duration * stream.time_base)
                if stream.duration and stream.time_base
                else None
            ),
            codec=str(stream.codec_context.name),
        )

    def _probe_codec(self) -> str | None:
        """Identify the file's video codec from its header.

        Args:
            None.

        Returns:
            The codec name, ``None`` when the file carries no video stream, or
            ``"unidentified"`` when PyAV is not installed -- in which case
            OpenCV is trusted on its own, which is a weaker guarantee and worth
            being explicit about.
        """
        try:
            import av
        except ImportError:
            return "unidentified"

        try:
            with av.open(str(self.path)) as container:
                streams = container.streams.video
                identified = str(streams[0].codec_context.name) if streams else None
        except Exception:
            return None
        return identified

    # -- reading -----------------------------------------------------------

    def _frames(self) -> Iterator[Frame]:
        """Yield frames from whichever decoder opened the file.

        Yields:
            Frames in capture order.
        """
        if self._backend == DecoderBackend.OPENCV:
            yield from self._frames_opencv()
        else:
            yield from self._frames_pyav()

    def _anchor(self) -> datetime:
        """Return the capture time of frame zero.

        Returns:
            The configured start, or the file's modification time when none was
            supplied -- which is a guess, and the reason stage 09 prefers to be
            told.
        """
        if self.start_utc is not None:
            return ensure_utc(self.start_utc, field_name="start_utc")
        from datetime import UTC

        return datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC)

    def _within_limits(self, delivered: int, offset_sec: float) -> bool:
        """Return whether another frame is still wanted.

        Args:
            delivered: How many frames have been yielded.
            offset_sec: Seconds of source time elapsed.

        Returns:
            ``True`` while both configured limits allow more.
        """
        if self.max_frames is not None and delivered >= self.max_frames:
            return False
        return not (self.duration_sec is not None and offset_sec > self.duration_sec)

    def _build(
        self, image: npt.NDArray[np.uint8], frame_index: int, raw: datetime, is_keyframe: bool
    ) -> Frame:
        """Assemble one frame.

        Args:
            image: The decoded image.
            frame_index: Its position in the file.
            raw: Its capture time before correction.
            is_keyframe: Whether the encoder marked it a keyframe.

        Returns:
            The frame.
        """
        return Frame(
            image=image,
            frame_index=frame_index,
            raw_timestamp=raw,
            timestamp_utc=raw + timedelta(milliseconds=self.clock_offset_ms),
            source_id=self.source_id,
            camera_id=self.camera_id,
            is_keyframe=is_keyframe,
        )

    def _frames_opencv(self) -> Iterator[Frame]:
        """Yield frames using OpenCV.

        Yields:
            Frames in capture order, skipping ones that fail to decode.
        """
        anchor = self._anchor()
        rate = self.properties.fps
        expected = self.properties.frame_count
        index = self.start_frame
        delivered = 0
        consecutive_failures = 0

        while True:
            ok, image = self._capture.read()
            if not ok:
                # OpenCV reports a clean end of file and a damaged one
                # identically: read() simply returns False. The container's
                # frame count is what separates them, and without it every
                # complete file would be reported as truncated -- which is how
                # a warning becomes noise nobody reads.
                if expected is not None and index >= expected:
                    break
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    self.was_truncated = expected is None or index < expected
                    break
                # A single unreadable frame is damage, not the end: advance past
                # it and keep going.
                index += 1
                self.corrupt_frames_skipped += 1
                continue

            consecutive_failures = 0
            raw = derive_timestamp(anchor, index, rate)
            offset_sec = float(index - self.start_frame) / float(rate)
            if not self._within_limits(delivered, offset_sec):
                break

            # OpenCV does not expose keyframe flags; a fixed interval is the
            # honest approximation, and KeyframeOnlySampler documents that it is
            # only exact on the PyAV path.
            yield self._build(image, index, raw, is_keyframe=index % 10 == 0)
            index += 1
            delivered += 1

        if self.was_truncated:
            logger.warning(
                "video_file_truncated",
                path=str(self.path),
                frames_read=delivered,
                frames_expected=expected,
                detail="the file ends mid-stream; what was readable has been delivered",
            )

        if self.corrupt_frames_skipped:
            logger.warning(
                "corrupt_frames_skipped",
                path=str(self.path),
                count=self.corrupt_frames_skipped,
            )

    def _frames_pyav(self) -> Iterator[Frame]:
        """Yield frames using PyAV, preferring per-frame presentation times.

        Yields:
            Frames in capture order.
        """
        anchor = self._anchor()
        rate = self.properties.fps
        stream = self._container.streams.video[0]
        index = 0
        delivered = 0

        try:
            for packet_frame in self._container.decode(stream):
                if index < self.start_frame:
                    index += 1
                    continue

                # PTS is what makes a variable-rate file come out right: the
                # nominal rate would place every frame after the first drop at
                # the wrong instant.
                if packet_frame.pts is not None and stream.time_base is not None:
                    offset_sec = float(packet_frame.pts * stream.time_base)
                    raw = anchor + timedelta(seconds=offset_sec)
                else:
                    raw = derive_timestamp(anchor, index, rate)
                    offset_sec = float(index) / float(rate)

                if not self._within_limits(delivered, offset_sec):
                    break

                image = packet_frame.to_ndarray(format="bgr24")
                yield self._build(
                    image, index, raw, is_keyframe=bool(getattr(packet_frame, "key_frame", False))
                )
                index += 1
                delivered += 1
        except Exception as exc:
            self.was_truncated = True
            logger.warning(
                "video_file_truncated",
                path=str(self.path),
                frames_read=delivered,
                error=str(exc),
            )

    def _close(self) -> None:
        """Release the decoder held by whichever backend opened the file."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._container is not None:
            self._container.close()
            self._container = None

    def _on_iteration_finished(self) -> None:
        """Release the decoder when iteration ends.

        A finished file has nothing more to give, and holding its handle until
        the caller remembers to close is how a batch run exhausts its file
        descriptors half way through.
        """
        self.close()
