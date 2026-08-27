"""Live streams, on the assumption that the network is unreliable. It is.

An RTSP camera drops its connection for reasons that have nothing to do with
this system: a switch reboots, a lease expires, someone unplugs the wrong cable.
Code that treats a dropped connection as a fatal error stops tracking every time
the network hiccups, which in practice means it stops tracking.

Three decisions carry this module.

**Latency beats completeness.** For live tracking a frame from four seconds ago
is worth less than the current one -- the operator is watching for a vehicle
*now*. So the buffer is bounded and drops the **oldest** frames when it fills.
Dropping the newest would keep a tidy record of an increasingly stale past.
Every drop is counted, because a system quietly discarding half its input while
reporting healthy is worse than one that falls behind visibly.

**The decoder is never blocked.** It runs on its own thread and pushes into a
bounded queue that never waits: a slow consumer costs frames, not a stalled
decoder. A blocked decoder means the socket's receive buffer fills, the camera's
transmit buffer fills, and the connection is eventually reset by the far end --
turning a slow consumer into a disconnection.

**Reconnection backs off with jitter.** Exponential backoff stops a dead camera
being hammered; the jitter stops forty cameras that lost the same switch from
reconnecting in lockstep and knocking it over again.
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Any

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.ingest.protocol import (
    BaseVideoSource,
    ConnectionState,
    SourceKind,
    SourceProperties,
)
from multicam_tracker.logging_config import get_logger

__all__ = ["BackoffPolicy", "LiveStreamSource", "StreamMetrics"]

logger = get_logger(__name__)

_DEFAULT_FPS = Fraction(25, 1)
"""Assumed rate when a stream does not declare one. Used only for metadata; the
frames' timestamps come from arrival time, not from this."""


@dataclass(frozen=True)
class BackoffPolicy:
    """How long to wait between reconnection attempts.

    Args:
        initial_sec: Delay before the first retry.
        multiplier: Growth factor per attempt.
        max_sec: Ceiling, so a long outage does not push the retry interval
            into hours.
        jitter_ratio: Fraction of the delay to randomise by. Without it, every
            camera behind a failed switch retries in lockstep and knocks it over
            again the moment it recovers.
        max_attempts: Retries before the source gives up and reports a terminal
            state. ``None`` retries forever, which is what an unattended
            deployment usually wants.
    """

    initial_sec: float = 1.0
    multiplier: float = 2.0
    max_sec: float = 30.0
    jitter_ratio: float = 0.2
    max_attempts: int | None = 5

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Return how long to wait before one retry.

        Args:
            attempt: Which retry this is, from one.
            rng: Seeded generator, so a test can assert the sequence.

        Returns:
            Seconds to wait, jittered.
        """
        raw = min(self.initial_sec * (self.multiplier ** (attempt - 1)), self.max_sec)
        jitter = raw * self.jitter_ratio
        return max(0.0, raw + rng.uniform(-jitter, jitter))


@dataclass
class StreamMetrics:
    """What the stream did, so a quiet failure cannot look like a quiet street."""

    frames_delivered: int = 0
    frames_dropped: int = 0
    reconnect_attempts: int = 0
    reconnects_succeeded: int = 0
    read_timeouts: int = 0

    @property
    def drop_ratio(self) -> float:
        """Return the share of received frames that were dropped."""
        total = self.frames_delivered + self.frames_dropped
        return 0.0 if total == 0 else self.frames_dropped / total

    def as_dict(self) -> dict[str, float]:
        """Return the counters for a metrics exporter.

        Returns:
            A flat mapping.
        """
        return {
            "frames_delivered": self.frames_delivered,
            "frames_dropped": self.frames_dropped,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnects_succeeded": self.reconnects_succeeded,
            "read_timeouts": self.read_timeouts,
            "drop_ratio": round(self.drop_ratio, 4),
        }


class LiveStreamSource(BaseVideoSource):
    """An RTSP or HTTP stream, with reconnection and a bounded buffer.

    Args:
        url: The stream to read.
        camera_id: The camera these frames belong to.
        clock_offset_ms: The camera's clock correction.
        buffer_size: Frames held between the decoder and the consumer. Small on
            purpose: a large buffer trades the latency this source exists to
            preserve for a completeness nobody watching live can use.
        read_timeout_sec: How long a read may block before the connection is
            treated as dead.
        backoff: Reconnection policy.
        source_id: Recorded on every frame. Defaults to the camera id.
        seed: Seeds the backoff jitter, so a test can assert the sequence.
        opener: Builds the underlying capture. Injected so the reconnection and
            buffering logic can be tested without a network -- the behaviour
            under test is what happens when a stream misbehaves, and a real
            server makes that harder to arrange, not easier.
    """

    def __init__(
        self,
        url: str,
        camera_id: str = "cam_01",
        *,
        clock_offset_ms: int = 0,
        buffer_size: int = 8,
        read_timeout_sec: float = 10.0,
        backoff: BackoffPolicy | None = None,
        source_id: str = "",
        seed: int = 0,
        opener: Any = None,
    ) -> None:
        if buffer_size < 1:
            raise IngestError(
                "Buffer size must be at least one frame", {"buffer_size": buffer_size}
            )

        super().__init__(source_id or camera_id, camera_id)
        self.url = url
        self.clock_offset_ms = clock_offset_ms
        self.buffer_size = buffer_size
        self.read_timeout_sec = read_timeout_sec
        self.backoff = backoff or BackoffPolicy()
        self.metrics = StreamMetrics()

        self._state = ConnectionState.IDLE
        self._rng = random.Random(seed)
        self._opener = opener
        self._capture: Any = None
        self._buffer: deque[Frame] = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._frame_available = threading.Condition(self._buffer_lock)
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._frame_index = 0

    @property
    def state(self) -> ConnectionState:
        """Return what the connection is currently doing."""
        return self._state

    @property
    def buffered_frames(self) -> int:
        """Return how many frames are waiting to be consumed."""
        with self._buffer_lock:
            return len(self._buffer)

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> SourceProperties:
        """Connect and start the reader thread.

        Returns:
            What the stream declares about itself.

        Raises:
            IngestError: If the first connection fails outright.
        """
        self._state = ConnectionState.CONNECTING
        capture = self._connect()
        if capture is None:
            self._state = ConnectionState.FAILED
            raise IngestError("Could not connect to the stream", {"url": self.url})

        self._capture = capture
        self._state = ConnectionState.STREAMING

        self._stop.clear()
        self._reader = threading.Thread(
            target=self._read_loop, name=f"live-{self.camera_id}", daemon=True
        )
        self._reader.start()

        return SourceProperties(
            fps=_DEFAULT_FPS,
            width=int(getattr(capture, "width", 0)) or 0,
            height=int(getattr(capture, "height", 0)) or 0,
            kind=SourceKind.LIVE,
            frame_count=None,
            codec="live",
        )

    def _connect(self) -> Any:
        """Open the underlying capture.

        Returns:
            The capture object, or ``None`` when the connection failed.
        """
        if self._opener is not None:
            try:
                return self._opener(self.url)
            except Exception as exc:
                logger.warning("stream_connect_failed", url=self.url, error=str(exc))
                return None

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise IngestError(
                "Live streaming needs OpenCV; install the 'vision' extra",
                {"url": self.url},
            ) from exc

        capture = cv2.VideoCapture(self.url)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    # -- reading -----------------------------------------------------------

    def _read_loop(self) -> None:
        """Decode frames into the bounded buffer until stopped.

        Runs on its own thread. Never blocks on the consumer: pushing into a
        bounded deque discards the oldest frame instead of waiting, which is
        what keeps a slow consumer from turning into a stalled decoder and then
        into a reset connection.
        """
        attempt = 0

        while not self._stop.is_set():
            image = self._read_one()

            if image is None:
                if self._stop.is_set():
                    break
                attempt += 1
                if not self._should_retry(attempt):
                    self._state = ConnectionState.FAILED
                    logger.error(
                        "stream_gave_up",
                        url=self.url,
                        attempts=attempt - 1,
                        detail="retries exhausted; the source is terminal until reopened",
                    )
                    break
                self._reconnect(attempt)
                continue

            attempt = 0
            self._push(image)

        with self._frame_available:
            self._frame_available.notify_all()

    def _read_one(self) -> Any:
        """Read one image from the capture.

        Returns:
            The image, or ``None`` when the read failed or timed out.
        """
        if self._capture is None:
            return None

        started = time.monotonic()
        try:
            ok, image = self._capture.read()
        except Exception as exc:
            logger.warning("stream_read_failed", url=self.url, error=str(exc))
            return None

        if time.monotonic() - started > self.read_timeout_sec:
            self.metrics.read_timeouts += 1
            logger.warning(
                "stream_read_timeout",
                url=self.url,
                timeout_sec=self.read_timeout_sec,
                detail="the stream went quiet; treating the connection as dead",
            )
            return None

        return image if ok else None

    def _should_retry(self, attempt: int) -> bool:
        """Return whether another reconnection is allowed.

        Args:
            attempt: Which retry this would be.

        Returns:
            ``True`` while attempts remain.
        """
        if self.backoff.max_attempts is None:
            return True
        return attempt <= self.backoff.max_attempts

    def _reconnect(self, attempt: int) -> None:
        """Wait, then try to reconnect.

        Args:
            attempt: Which retry this is.
        """
        self._state = ConnectionState.RECONNECTING
        self.metrics.reconnect_attempts += 1
        delay = self.backoff.delay_for(attempt, self._rng)

        logger.warning(
            "stream_reconnecting", url=self.url, attempt=attempt, delay_sec=round(delay, 3)
        )
        # Waiting on the stop event rather than sleeping, so shutdown during a
        # thirty-second backoff does not wait thirty seconds.
        if self._stop.wait(timeout=delay):
            return

        if self._capture is not None:
            with contextlib.suppress(Exception):
                self._capture.release()
            self._capture = None

        capture = self._connect()
        if capture is not None:
            self._capture = capture
            self._state = ConnectionState.STREAMING
            self.metrics.reconnects_succeeded += 1
            logger.info("stream_reconnected", url=self.url, attempt=attempt)

    def _push(self, image: Any) -> None:
        """Add one decoded image to the buffer, dropping the oldest if full.

        Args:
            image: The decoded image.
        """
        arrived = datetime.now(UTC)
        frame = Frame(
            image=image,
            frame_index=self._frame_index,
            raw_timestamp=arrived,
            timestamp_utc=arrived + timedelta(milliseconds=self.clock_offset_ms),
            source_id=self.source_id,
            camera_id=self.camera_id,
            is_keyframe=False,
        )
        self._frame_index += 1

        with self._frame_available:
            if len(self._buffer) == self._buffer.maxlen:
                # deque with maxlen discards the oldest on append. Counting it
                # here is the whole reason this is not a bare append: a drop
                # nobody counts is indistinguishable from a quiet street.
                self.metrics.frames_dropped += 1
            self._buffer.append(frame)
            self._frame_available.notify()

    def _frames(self) -> Iterator[Frame]:
        """Yield buffered frames as they arrive.

        Yields:
            Frames in arrival order, oldest first.
        """
        while True:
            with self._frame_available:
                while not self._buffer and not self._stop.is_set() and self._is_running():
                    self._frame_available.wait(timeout=0.1)

                if self._buffer:
                    frame = self._buffer.popleft()
                elif self._stop.is_set() or not self._is_running():
                    return
                else:
                    continue

            self.metrics.frames_delivered += 1
            yield frame

    def _is_running(self) -> bool:
        """Return whether the reader thread is still alive and not terminal."""
        if self._state is ConnectionState.FAILED:
            return False
        return self._reader is not None and self._reader.is_alive()

    def _on_iteration_finished(self) -> None:
        """Keep the stream open when a consumer stops reading.

        Unlike a file, a live source has more to give: a consumer that pauses,
        or a reader that breaks out of the loop to handle something, expects the
        stream to still be there. Closing is explicit, or via the context
        manager.
        """

    def _close(self) -> None:
        """Stop the reader thread and release the capture."""
        self._stop.set()
        with self._frame_available:
            self._frame_available.notify_all()

        if self._reader is not None:
            self._reader.join(timeout=self.read_timeout_sec)
            self._reader = None

        if self._capture is not None:
            with contextlib.suppress(Exception):
                self._capture.release()
            self._capture = None

        self._state = ConnectionState.CLOSED
        logger.info("stream_closed", url=self.url, **self.metrics.as_dict())
