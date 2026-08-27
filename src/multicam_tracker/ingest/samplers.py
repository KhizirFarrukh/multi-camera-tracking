"""Deciding which frames are worth looking at.

A camera at 30 fps produces 108,000 frames an hour, and a detector that ran on
all of them would spend most of its time confirming that the same car is still
in the same place. Detection at 5 fps is normally sufficient to see every vehicle
that passes, so sampling is the first and cheapest saving in the pipeline.

**Sampling never touches a timestamp.** A sampled frame carries the capture time
of the frame that was actually decoded -- never an interpolated one, and never a
renumbered index. This is the rule the whole stage turns on: a trajectory is
built from those timestamps, and a sampler that "smoothed" them to a regular
cadence would move a vehicle to a moment nobody observed. The frame index is
preserved for the same reason: renumbering makes every later reference to
"frame 30" mean something else.

Four strategies, and the trade each makes:

``EveryNthFrame``
    Predictable and source-dependent. Every 6th frame is 5 fps on a 30 fps
    camera and 4 fps on a 25 fps one, which is why it is not the default.

``TargetFpsSampler``
    The default. Asks for a cadence in real time and gets it regardless of the
    source rate, so one configuration works across a mixed estate.

``KeyframeOnlySampler``
    Cheapest of all -- keyframes decode independently -- at the cost of a
    cadence the encoder chose rather than one anybody wanted.

``AdaptiveSampler``
    Spends frames where something is happening. Recovers detail during an event
    and idles cheaply between them, at the cost of being harder to reason about
    when reading a trajectory afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Protocol, runtime_checkable

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame

__all__ = [
    "AdaptiveSampler",
    "EveryNthFrame",
    "FrameSampler",
    "KeyframeOnlySampler",
    "TargetFpsSampler",
]


@runtime_checkable
class FrameSampler(Protocol):
    """Decides which frames reach the detector."""

    def sample(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Filter a stream of frames.

        Args:
            frames: Every decoded frame, in capture order.

        Yields:
            The subset worth processing, unchanged -- same image, same index,
            same timestamps.
        """
        ...

    def reset(self) -> None:
        """Forget any state carried between frames."""
        ...


@dataclass
class EveryNthFrame:
    """Take one frame in every ``n``.

    Args:
        n: The interval. ``1`` takes everything.

    Raises:
        IngestError: If ``n`` is not positive.
    """

    n: int = 6

    def __post_init__(self) -> None:
        """Validate the interval.

        Raises:
            IngestError: If ``n`` is not positive.
        """
        if self.n < 1:
            raise IngestError("Sampling interval must be at least 1", {"n": self.n})

    def sample(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Yield every ``n``-th frame.

        Counted by *arrival*, not by frame index: a source that already dropped
        a corrupt frame should still deliver a regular cadence, and keying off
        the index would leave a hole whenever the decoder did.

        Args:
            frames: Every decoded frame.

        Yields:
            Every ``n``-th one.
        """
        for position, frame in enumerate(frames):
            if position % self.n == 0:
                yield frame

    def reset(self) -> None:
        """No state to forget."""


@dataclass
class TargetFpsSampler:
    """Sample to approximate a target rate in real time.

    Works off the frames' own timestamps rather than a count, so it produces the
    requested cadence whatever the source rate is -- and keeps producing it when
    the source rate varies, which real cameras do.

    Args:
        target_fps: Frames per second to aim for.
        source_fps: The source's nominal rate, used only to notice when the
            target is unreachable.

    Raises:
        IngestError: If the target rate is not positive.
    """

    target_fps: float | Fraction = 5.0
    source_fps: float | Fraction | None = None
    _last_emitted: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the target rate.

        Raises:
            IngestError: If it is not positive.
        """
        if self.target_fps <= 0:
            raise IngestError(
                "Target frame rate must be positive", {"target_fps": str(self.target_fps)}
            )
        self._interval_sec = 1.0 / float(self.target_fps)

    def sample(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Yield frames spaced at roughly the target interval.

        The first frame always passes: a source's opening frame is the anchor
        everything else is measured from, and skipping it would delay the first
        detection by up to a whole interval.

        Args:
            frames: Every decoded frame.

        Yields:
            Frames whose capture times are at least one interval apart.
        """
        for frame in frames:
            if self._last_emitted is None:
                self._last_emitted = frame.timestamp_utc
                yield frame
                continue

            elapsed = (frame.timestamp_utc - self._last_emitted).total_seconds()
            # A small tolerance, because a source at exactly the target rate
            # would otherwise drop every other frame to floating-point noise.
            if elapsed >= self._interval_sec - 1e-6:
                self._last_emitted = frame.timestamp_utc
                yield frame

    def reset(self) -> None:
        """Forget the last emitted timestamp."""
        self._last_emitted = None


@dataclass
class KeyframeOnlySampler:
    """Take only frames the encoder marked as keyframes.

    The cheapest option: a keyframe decodes without reference to its neighbours.
    The cadence is whatever the encoder chose, which on a typical two-second GOP
    is 0.5 fps -- fast traffic can cross the frame between two of them.
    """

    def sample(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Yield only keyframes.

        Args:
            frames: Every decoded frame.

        Yields:
            The frames flagged ``is_keyframe``.
        """
        for frame in frames:
            if frame.is_keyframe:
                yield frame

    def reset(self) -> None:
        """No state to forget."""


@dataclass
class AdaptiveSampler:
    """Sample faster while something is happening.

    Starts at the idle rate and switches to the active rate when a detection is
    reported, decaying back after a configured quiet period. The caller drives
    it: :meth:`note_detection` is called by whatever consumes the frames, since
    the sampler cannot know what the detector found.

    Args:
        idle_fps: Rate when nothing has been seen recently.
        active_fps: Rate after a detection.
        decay_sec: How long a detection keeps the rate elevated.

    Raises:
        IngestError: If either rate is not positive, or the active rate is below
            the idle rate -- which would make a detection *reduce* the sampling
            rate, the opposite of the point.
    """

    idle_fps: float = 2.0
    active_fps: float = 10.0
    decay_sec: float = 5.0
    _last_emitted: datetime | None = field(default=None, init=False, repr=False)
    _active_until: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the rates.

        Raises:
            IngestError: If a rate is not positive or the rates are inverted.
        """
        if self.idle_fps <= 0 or self.active_fps <= 0:
            raise IngestError(
                "Adaptive sampler rates must be positive",
                {"idle_fps": self.idle_fps, "active_fps": self.active_fps},
            )
        if self.active_fps < self.idle_fps:
            raise IngestError(
                "The active rate must be at least the idle rate; otherwise a detection "
                "would slow sampling down, which is the opposite of the intent",
                {"idle_fps": self.idle_fps, "active_fps": self.active_fps},
            )

    @property
    def is_active(self) -> bool:
        """Return whether the sampler is currently in its elevated state."""
        return self._active_until is not None

    def note_detection(self, at_utc: datetime) -> None:
        """Record that something was found, and raise the rate.

        Args:
            at_utc: When the detection happened, in the frames' own time base --
                not wall-clock time, so a recorded file replays exactly as it
                would have run live.
        """
        from datetime import timedelta

        self._active_until = at_utc + timedelta(seconds=self.decay_sec)

    def sample(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        """Yield frames at whichever rate currently applies.

        Args:
            frames: Every decoded frame.

        Yields:
            Frames spaced at the active or idle interval.
        """
        for frame in frames:
            if self._active_until is not None and frame.timestamp_utc > self._active_until:
                self._active_until = None

            rate = self.active_fps if self._active_until is not None else self.idle_fps
            interval = 1.0 / rate

            if self._last_emitted is None:
                self._last_emitted = frame.timestamp_utc
                yield frame
                continue

            if (frame.timestamp_utc - self._last_emitted).total_seconds() >= interval - 1e-6:
                self._last_emitted = frame.timestamp_utc
                yield frame

    def reset(self) -> None:
        """Return to the idle rate and forget the last emitted frame."""
        self._last_emitted = None
        self._active_until = None
