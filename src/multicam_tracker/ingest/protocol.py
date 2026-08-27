"""What every video source must offer, and what it guarantees.

One interface covers a recorded file, a live RTSP stream, and a generated test
pattern. That uniformity is the point: stages 11-13 consume frames without
knowing where they came from, so their tests run against generated frames in CI
with no media files and no network.

**Resources are released on every exit path.** A decoder holds a file handle or
a socket, and Python's garbage collector is not a resource manager -- a source
abandoned mid-iteration must still let go. :class:`BaseVideoSource` implements
the context-manager protocol and closes on exception, on normal exit, and on
generator abandonment, and the tests exercise all three.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame
from multicam_tracker.logging_config import get_logger

__all__ = [
    "BaseVideoSource",
    "ConnectionState",
    "SourceKind",
    "SourceProperties",
    "VideoSource",
]

logger = get_logger(__name__)


class SourceKind(StrEnum):
    """Where a source's frames come from."""

    FILE = "file"
    LIVE = "live"
    SYNTHETIC = "synthetic"


class ConnectionState(StrEnum):
    """What a live source is currently doing.

    Exposed for monitoring: an operator watching a wall of cameras needs to
    distinguish "quiet street" from "this camera stopped talking to us an hour
    ago", and only the source knows which.
    """

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    """Terminal: retries were exhausted and the source gave up."""

    CLOSED = "closed"


@dataclass(frozen=True)
class SourceProperties:
    """What a source knows about itself once it is open."""

    fps: Fraction
    """Exact rational where the container provides one. 30000/1001 is not 29.97,
    and stage 09 needs the difference to place frames correctly."""

    width: int
    height: int
    kind: SourceKind
    frame_count: int | None = None
    """``None`` when unknown -- a live stream has no end, and some containers
    lie about it rather than admitting they do not know."""

    duration_sec: float | None = None
    codec: str | None = None

    @property
    def is_live(self) -> bool:
        """Return whether the source runs indefinitely."""
        return self.kind is SourceKind.LIVE

    @property
    def resolution(self) -> tuple[int, int]:
        """Return ``(width, height)``."""
        return (self.width, self.height)


@runtime_checkable
class VideoSource(Protocol):
    """A source of timestamped frames."""

    @property
    def source_id(self) -> str:
        """Return the identifier recorded on every frame this source emits."""
        ...

    @property
    def camera_id(self) -> str:
        """Return the camera these frames belong to."""
        ...

    @property
    def is_live(self) -> bool:
        """Return whether this source runs indefinitely."""
        ...

    @property
    def properties(self) -> SourceProperties:
        """Return what the source knows about itself.

        Raises:
            IngestError: If the source is not open. The properties come from the
                container or the stream, and inventing them before opening would
                be guessing.
        """
        ...

    def open(self) -> None:
        """Acquire the decoder, file handle, or socket.

        Raises:
            IngestError: If the source cannot be opened.
        """
        ...

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted or closed.

        Yields:
            Frames in capture order.

        Raises:
            IngestError: If decoding fails unrecoverably.
        """
        ...

    def close(self) -> None:
        """Release everything. Safe to call more than once."""
        ...


class BaseVideoSource(AbstractContextManager["BaseVideoSource"]):
    """Shared lifecycle for every source.

    Subclasses implement :meth:`_open`, :meth:`_frames`, and :meth:`_close`;
    this class handles the parts that are easy to get subtly wrong -- opening
    once, closing exactly once on every path, and refusing to yield frames from
    a source that is not open.

    Args:
        source_id: Recorded on every frame.
        camera_id: The camera these frames belong to.
    """

    def __init__(self, source_id: str, camera_id: str) -> None:
        self._source_id = source_id
        self._camera_id = camera_id
        self._properties: SourceProperties | None = None
        self._is_open = False
        self._closed = False

    # -- identity ----------------------------------------------------------

    @property
    def source_id(self) -> str:
        """Return the identifier recorded on every frame this source emits."""
        return self._source_id

    @property
    def camera_id(self) -> str:
        """Return the camera these frames belong to."""
        return self._camera_id

    @property
    def is_open(self) -> bool:
        """Return whether the decoder is currently held."""
        return self._is_open

    @property
    def is_live(self) -> bool:
        """Return whether this source runs indefinitely."""
        return self.properties.is_live

    @property
    def properties(self) -> SourceProperties:
        """Return what the source knows about itself.

        Returns:
            The properties read when the source was opened.

        Raises:
            IngestError: If the source has not been opened.
        """
        if self._properties is None:
            raise IngestError(
                "Source properties are not known until the source is opened",
                {"source_id": self._source_id},
            )
        return self._properties

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Acquire the decoder and read the source's properties.

        Opening an already-open source is a no-op rather than an error: a
        caller using the context manager after an explicit ``open()`` is doing
        something reasonable.

        Raises:
            IngestError: If the source cannot be opened, or has been closed.
        """
        if self._closed:
            raise IngestError(
                "This source has been closed and cannot be reopened; construct a new one",
                {"source_id": self._source_id},
            )
        if self._is_open:
            return

        self._properties = self._open()
        self._is_open = True
        logger.info(
            "video_source_opened",
            source_id=self._source_id,
            camera_id=self._camera_id,
            kind=self._properties.kind.value,
            resolution=f"{self._properties.width}x{self._properties.height}",
            fps=str(self._properties.fps),
        )

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted or closed.

        Yields:
            Frames in capture order.

        Raises:
            IngestError: If the source is not open.
        """
        if not self._is_open:
            raise IngestError(
                "Cannot read frames from a source that is not open",
                {"source_id": self._source_id},
            )
        # The try/finally is what survives generator abandonment: when a caller
        # breaks out of the loop, Python closes the generator, which raises
        # GeneratorExit here and runs the cleanup. Without it an abandoned
        # source holds its decoder until the collector happens to run.
        try:
            yield from self._frames()
        finally:
            self._on_iteration_finished()

    def close(self) -> None:
        """Release everything. Safe to call more than once."""
        if not self._is_open and self._closed:
            return
        try:
            self._close()
        finally:
            self._is_open = False
            self._closed = True
            logger.info("video_source_closed", source_id=self._source_id, camera_id=self._camera_id)

    def __enter__(self) -> Self:
        """Open the source.

        Returns:
            The opened source.
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the source on any exit path, including an exception.

        Args:
            exc_type: Exception class, if the body raised.
            exc: The exception, if the body raised.
            traceback: The traceback, if the body raised.
        """
        self.close()

    # -- subclass hooks ----------------------------------------------------

    def _open(self) -> SourceProperties:
        """Acquire the decoder and return the source's properties.

        Returns:
            What the source knows about itself.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def _frames(self) -> Iterator[Frame]:
        """Yield frames from the open decoder.

        Yields:
            Frames in capture order.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def _close(self) -> None:
        """Release the decoder. Called at most once."""

    def _on_iteration_finished(self) -> None:
        """Release the decoder when iteration ends.

        Closing by default rather than on request, because the failure this
        prevents is silent: a caller that breaks out of a frame loop leaves a
        suspended generator, and without this the decoder is held until the
        garbage collector happens to run. On a batch of ten thousand files that
        is far too late, and the symptom is a run that dies half way through
        having exhausted its file descriptors.

        A live source overrides this: it may be iterated again after a
        reconnection, so a pause in reading is not the end of the stream.
        """
        self.close()
