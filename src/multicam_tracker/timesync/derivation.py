"""Turning a frame index into an instant, exactly.

The naive version is ``start + frame_index / fps`` in floating point, and it is
wrong in a way that hides for a long time. At 29.97 fps -- which is really
30000/1001, not 29.97 -- a float accumulation drifts by seconds over a long
recording. Seconds are exactly the scale at which this system's travel-time
windows operate, so that error does not merely blur a timestamp; it moves a
sighting into or out of a plausible hop.

So the arithmetic here is rational throughout: :class:`fractions.Fraction` holds
30000/1001 exactly, the multiplication by the frame index is exact, and only the
final conversion to a ``timedelta`` rounds -- once, to the millisecond the
Sighting contract stores.

**Presentation timestamps win.** When the container supplies per-frame PTS, they
are used and the nominal frame rate is ignored entirely. Real recordings drop
frames and vary their rate; a frame index multiplied by a nominal fps assumes
neither ever happens, and silently shifts every timestamp after the first drop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from fractions import Fraction

from multicam_tracker.clock import ensure_utc
from multicam_tracker.exceptions import IngestError

__all__ = ["FrameTiming", "derive_timestamp", "fps_as_fraction"]

_MICROSECONDS_PER_SECOND = 1_000_000

COMMON_RATIONAL_FPS: dict[float, Fraction] = {
    29.97: Fraction(30000, 1001),
    59.94: Fraction(60000, 1001),
    23.976: Fraction(24000, 1001),
    119.88: Fraction(120000, 1001),
}
"""Broadcast frame rates whose decimal form is a rounding of a rational.

A caller passing ``29.97`` almost certainly means 30000/1001; taking the decimal
literally accumulates roughly 3 ms per 100 frames, which is 100 seconds over an
hour of footage. The mapping is explicit rather than inferred so the assumption
is visible and can be overridden by passing the ``Fraction`` directly.
"""


def fps_as_fraction(fps: float | Fraction) -> Fraction:
    """Return an exact rational frame rate.

    Args:
        fps: Frames per second. A ``Fraction`` is used as given; a float is
            mapped through :data:`COMMON_RATIONAL_FPS` when it matches a known
            broadcast rate, and converted exactly otherwise.

    Returns:
        The frame rate as a fraction.

    Raises:
        IngestError: If the rate is zero, negative, or not finite. A zero rate
            would divide by zero one line later, and a negative one would run
            time backwards.
    """
    if isinstance(fps, Fraction):
        rate = fps
    else:
        if fps != fps or fps in {float("inf"), float("-inf")}:  # NaN or infinity
            raise IngestError("Frame rate must be a finite number", {"fps": str(fps)})
        rate = COMMON_RATIONAL_FPS.get(
            round(float(fps), 3), Fraction(fps).limit_denominator(100000)
        )

    if rate <= 0:
        raise IngestError(
            "Frame rate must be positive; a zero or negative rate cannot produce timestamps",
            {"fps": str(fps)},
        )
    return rate


class FrameTiming:
    """Per-frame presentation timestamps, when the container supplies them.

    Args:
        pts_seconds: Presentation time of each frame, in seconds from the start
            of the stream, indexed by frame position in the decoded order.

    Raises:
        IngestError: If the timestamps are empty or run backwards. A stream
            whose PTS decrease has been demuxed wrongly, and deriving instants
            from it would produce a trajectory that goes back in time.
    """

    def __init__(self, pts_seconds: list[float]) -> None:
        if not pts_seconds:
            raise IngestError("Frame timing requires at least one presentation timestamp", {})
        for index in range(1, len(pts_seconds)):
            if pts_seconds[index] < pts_seconds[index - 1]:
                raise IngestError(
                    "Presentation timestamps must be non-decreasing",
                    {
                        "frame_index": index,
                        "previous_pts": pts_seconds[index - 1],
                        "pts": pts_seconds[index],
                    },
                )
        self._pts = list(pts_seconds)

    def __len__(self) -> int:
        """Return how many frames have presentation timestamps."""
        return len(self._pts)

    def offset_seconds(self, frame_index: int) -> Fraction:
        """Return the presentation offset of one frame.

        Args:
            frame_index: Position in the decoded order.

        Returns:
            Seconds from the start of the stream, as an exact fraction.

        Raises:
            IngestError: If the index is outside the recorded range. Falling
                back to a nominal frame rate here would mix two timing models
                within one video, which is worse than refusing.
        """
        if frame_index < 0 or frame_index >= len(self._pts):
            raise IngestError(
                "Frame index is outside the recorded presentation timestamps",
                {"frame_index": frame_index, "frames_available": len(self._pts)},
            )
        return Fraction(self._pts[frame_index]).limit_denominator(1_000_000)


def derive_timestamp(
    source_start: datetime,
    frame_index: int,
    fps: float | Fraction | None = None,
    *,
    timing: FrameTiming | None = None,
) -> datetime:
    """Return the instant at which one frame was captured.

    Args:
        source_start: When the stream or file began, timezone-aware.
        frame_index: Zero-based position of the frame. Frame 0 is the start
            instant exactly.
        fps: Nominal frame rate, used only when ``timing`` is absent.
        timing: Per-frame presentation timestamps. When supplied these are used
            and ``fps`` is ignored, because a real recording drops frames and
            varies its rate and the nominal rate assumes neither happens.

    Returns:
        The capture instant in UTC, rounded once to the millisecond the Sighting
        contract stores.

    Raises:
        IngestError: If the frame index is negative, if neither a frame rate nor
            presentation timestamps were supplied, or if the rate is invalid.
        ValidationError: If ``source_start`` is naive.
    """
    start = ensure_utc(source_start, field_name="source_start")

    if frame_index < 0:
        raise IngestError("Frame index must be non-negative", {"frame_index": frame_index})

    if timing is not None:
        offset = timing.offset_seconds(frame_index)
    elif fps is not None:
        offset = Fraction(frame_index) / fps_as_fraction(fps)
    else:
        raise IngestError(
            "Deriving a timestamp needs either a frame rate or presentation timestamps",
            {"frame_index": frame_index},
        )

    # One rounding, at the end, to the millisecond -- rather than a rounding per
    # frame, which is what accumulates into the drift this module exists to
    # avoid.
    microseconds = round(offset * _MICROSECONDS_PER_SECOND)
    return start + timedelta(microseconds=int(microseconds))
