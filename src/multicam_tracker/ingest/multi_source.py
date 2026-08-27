"""Reading several cameras at once, where one of them is always broken.

A deployment has forty cameras and at any moment one is rebooting, one has a
damaged file, and one is behind a switch someone is replacing. The requirement
that matters is therefore not throughput -- it is that **a failing source never
stalls the others**. A reader that propagates one source's exception stops
tracking on thirty-nine working cameras because of the fortieth.

**Threads rather than asyncio**, and the reason is the decoders. OpenCV and
PyAV are blocking C extensions with no async interface; driving them from an
event loop means running them in a thread pool anyway, with an event loop's
complexity on top and none of its benefit. They release the GIL while decoding,
so threads genuinely run in parallel here.

Frames arrive tagged with their camera and are merged into one bounded queue.
Per-source health is exposed rather than inferred: an operator needs to tell a
camera that is quiet from one that is broken, and only the reader knows which.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from multicam_tracker.ingest.frame import Frame
from multicam_tracker.ingest.protocol import VideoSource
from multicam_tracker.logging_config import get_logger

__all__ = ["MultiSourceReader", "SourceHealth", "SourceStatus"]

logger = get_logger(__name__)

_POLL_INTERVAL_SEC = 0.05
"""How long a queue read waits before checking whether the run is finished.

Short enough that shutdown is prompt, long enough that an idle reader is not
spinning."""


class SourceStatus:
    """What one source is doing, as an operator would describe it."""

    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    """Ran to completion -- a file reaching its end, which is success."""

    FAILED = "failed"
    """Raised and stopped. The other sources continue."""


@dataclass
class SourceHealth:
    """Per-source state, exposed so a quiet camera is not mistaken for a dead one."""

    camera_id: str
    source_id: str
    status: str = SourceStatus.STARTING
    frames_read: int = 0
    error: str | None = None
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None

    @property
    def is_healthy(self) -> bool:
        """Return whether this source is running or finished cleanly."""
        return self.status in {SourceStatus.STARTING, SourceStatus.RUNNING, SourceStatus.FINISHED}

    def as_dict(self) -> dict[str, object]:
        """Return the status for a monitoring endpoint.

        Returns:
            A JSON-native mapping.
        """
        return {
            "camera_id": self.camera_id,
            "source_id": self.source_id,
            "status": self.status,
            "frames_read": self.frames_read,
            "error": self.error,
        }


@dataclass
class MultiSourceReader:
    """Runs several sources concurrently and merges their frames.

    Args:
        sources: The sources to read. Each gets its own thread.
        queue_size: How many frames may wait in the merged queue. Bounds memory:
            the queue holds decoded images, and an unbounded one is a memory
            leak with a slow consumer attached.
        stop_on_error: Whether one source failing ends the whole run. Off by
            default, which is the entire point of the class.
    """

    sources: list[VideoSource]
    queue_size: int = 64
    stop_on_error: bool = False

    health: dict[str, SourceHealth] = field(default_factory=dict, init=False)
    _queue: queue.Queue[Frame | None] = field(init=False, repr=False)
    _threads: list[threading.Thread] = field(default_factory=list, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Prepare the queue and the per-source health records."""
        self._queue = queue.Queue(maxsize=self.queue_size)
        self.health = {
            source.source_id: SourceHealth(camera_id=source.camera_id, source_id=source.source_id)
            for source in self.sources
        }

    @property
    def frames_dropped(self) -> int:
        """Return how many frames were discarded because the queue was full."""
        return self._dropped

    def frames(self) -> Iterator[Frame]:
        """Yield frames from every source, tagged with their camera.

        Yields:
            Frames as they are decoded, interleaved across sources. Ordering
            *within* a source is preserved; ordering *between* sources is not,
            and must not be relied on -- two cameras have no shared clock at
            this layer, which is what stage 09's corrected timestamps are for.
        """
        self._start()
        finished = 0

        try:
            while finished < len(self.sources):
                try:
                    item = self._queue.get(timeout=_POLL_INTERVAL_SEC)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue

                if item is None:
                    finished += 1
                    continue

                yield item
        finally:
            self.shutdown()

    def _start(self) -> None:
        """Launch one reader thread per source."""
        self._stop.clear()
        for source in self.sources:
            thread = threading.Thread(
                target=self._pump,
                args=(source,),
                name=f"ingest-{source.camera_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _pump(self, source: VideoSource) -> None:
        """Read one source into the merged queue.

        Every failure is caught and recorded rather than propagated. A thread
        that raises takes down only itself, and the point of this class is that
        it does not take down anything else either.

        Args:
            source: The source to read.
        """
        health = self.health[source.source_id]
        health.status = SourceStatus.RUNNING
        health.started_at_utc = datetime.now(UTC)

        try:
            source.open()
            for frame in source.frames():
                if self._stop.is_set():
                    break
                self._offer(frame)
                health.frames_read += 1
            health.status = SourceStatus.FINISHED
        except Exception as exc:
            health.status = SourceStatus.FAILED
            health.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "source_failed",
                source_id=source.source_id,
                camera_id=source.camera_id,
                error=health.error,
                detail="the other sources continue",
            )
            if self.stop_on_error:
                self._stop.set()
        finally:
            health.finished_at_utc = datetime.now(UTC)
            with contextlib.suppress(Exception):
                source.close()
            self._queue.put(None)

    def _offer(self, frame: Frame) -> None:
        """Put one frame on the queue, dropping it if the consumer is behind.

        Blocking here would stall this camera's decoder because a *different*
        consumer is slow, which is the coupling this class exists to avoid.

        Args:
            frame: The frame to enqueue.
        """
        try:
            self._queue.put(frame, timeout=_POLL_INTERVAL_SEC)
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "merged_queue_full",
                camera_id=frame.camera_id,
                dropped_total=self._dropped,
                detail="consumer is behind; the frame was dropped rather than "
                "stalling this camera's decoder",
            )

    def shutdown(self, timeout_sec: float = 5.0) -> None:
        """Stop every source and wait for its thread.

        Args:
            timeout_sec: How long to wait per thread before giving up on it.
                A daemon thread stuck in a blocking decode call cannot be
                interrupted, so waiting forever would hang the process on
                exactly the failure this class exists to survive.
        """
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout_sec)
        self._threads.clear()

        for source in self.sources:
            with contextlib.suppress(Exception):
                source.close()

    def health_report(self) -> list[dict[str, object]]:
        """Return every source's status.

        Returns:
            One entry per source, in configuration order.
        """
        return [self.health[source.source_id].as_dict() for source in self.sources]
