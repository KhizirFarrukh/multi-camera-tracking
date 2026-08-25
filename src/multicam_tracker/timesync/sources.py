"""Where a timestamp came from, and how much that is worth.

A timestamp read from a synchronised RTSP stream and one parsed out of a
filename are both ``datetime`` objects, and treating them as equally
trustworthy is how a system ends up confidently wrong. Every source therefore
declares a reliability tier, and the tier travels with the timestamp into the
temporal-integrity gate and from there onto the trajectory an operator reads.

The tiers, and what separates them:

``HIGH``
    The device's own clock, reported by a protocol that exposes it, on a camera
    whose offset has been verified. Wrong only if the camera's clock is wrong,
    which drift detection is watching for.

``MEDIUM``
    Derived from something the recording chain wrote down -- container creation
    time plus a frame offset. Trustworthy to within whatever the muxer did, and
    silently wrong if the file was copied by a tool that rewrote its metadata.

``LOW``
    Parsed from a name a human or a script chose, or read by OCR from pixels
    burned into the image. Both are guesses about a convention, and both fail
    silently: a filename pattern that stops matching yields a plausible wrong
    date, and OCR misreads a digit without saying so.

Nothing here decides what to *do* about a low-reliability source; that is the
gate's job in :mod:`~multicam_tracker.timesync.integrity`. This module's job is
to make the distinction impossible to lose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Protocol, runtime_checkable

from multicam_tracker.clock import ensure_utc
from multicam_tracker.exceptions import IngestError
from multicam_tracker.timesync.derivation import FrameTiming, derive_timestamp
from multicam_tracker.timesync.timezones import resolve_local_time

__all__ = [
    "FileMetadataSource",
    "FilenameSource",
    "ManualOffsetSource",
    "OverlayOcrSource",
    "ReliabilityTier",
    "StreamClockSource",
    "TimestampSource",
    "select_source",
]


class ReliabilityTier(StrEnum):
    """How much trust a timestamp source has earned."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Return an ordering rank, higher being more trustworthy.

        Returns:
            ``2`` for high, ``1`` for medium, ``0`` for low.
        """
        return {"high": 2, "medium": 1, "low": 0}[self.value]


@runtime_checkable
class TimestampSource(Protocol):
    """Something that can say when a frame was captured."""

    @property
    def tier(self) -> ReliabilityTier:
        """Return how much this source's timestamps are worth."""
        ...

    @property
    def description(self) -> str:
        """Return a short phrase naming the source, for operator-facing text."""
        ...

    def timestamp_for(self, frame_index: int) -> datetime:
        """Return the capture instant of one frame.

        Args:
            frame_index: Zero-based position within the source.

        Returns:
            A timezone-aware UTC instant.
        """
        ...


@dataclass(frozen=True)
class _FrameDerivedSource:
    """Shared behaviour for sources anchored at a start instant.

    Args:
        start_utc: When the source began.
        fps: Nominal frame rate, used when no presentation timestamps exist.
        timing: Per-frame presentation timestamps, preferred when present.
    """

    start_utc: datetime
    fps: float | Fraction | None = None
    timing: FrameTiming | None = None

    def timestamp_for(self, frame_index: int) -> datetime:
        """Return the capture instant of one frame.

        Args:
            frame_index: Zero-based position within the source.

        Returns:
            A timezone-aware UTC instant.

        Raises:
            IngestError: If the frame index or timing information is unusable.
        """
        return derive_timestamp(self.start_utc, frame_index, self.fps, timing=self.timing)


@dataclass(frozen=True)
class StreamClockSource(_FrameDerivedSource):
    """Timestamps from a live stream's own clock.

    The best case: the device says when it saw the frame, and the only thing
    that can be wrong is the device's clock -- which is precisely what offset
    estimation and drift detection are for.
    """

    @property
    def tier(self) -> ReliabilityTier:
        """Return :attr:`ReliabilityTier.HIGH`."""
        return ReliabilityTier.HIGH

    @property
    def description(self) -> str:
        """Return a short phrase naming the source."""
        return "stream clock"


@dataclass(frozen=True)
class FileMetadataSource(_FrameDerivedSource):
    """Timestamps from a container's creation time plus a frame offset.

    Medium trust: the recording chain wrote it down, but a copy, a remux, or a
    transcode can rewrite it without saying so, and the failure looks like a
    perfectly ordinary date.
    """

    @property
    def tier(self) -> ReliabilityTier:
        """Return :attr:`ReliabilityTier.MEDIUM`."""
        return ReliabilityTier.MEDIUM

    @property
    def description(self) -> str:
        """Return a short phrase naming the source."""
        return "container metadata"


@dataclass(frozen=True)
class ManualOffsetSource(_FrameDerivedSource):
    """Timestamps anchored at a start instant an operator supplied.

    Medium trust rather than low: a human stating "this recording starts at
    14:00" is making a claim they can be held to, unlike a pattern that silently
    stops matching. It is not high, because nothing verifies it.
    """

    @property
    def tier(self) -> ReliabilityTier:
        """Return :attr:`ReliabilityTier.MEDIUM`."""
        return ReliabilityTier.MEDIUM

    @property
    def description(self) -> str:
        """Return a short phrase naming the source."""
        return "operator-supplied start time"


@dataclass(frozen=True)
class OverlayOcrSource:
    """Timestamps read by OCR from a burned-in overlay.

    Low trust, and the reason is specific: OCR misreads a digit without any
    signal that it did, so a 2 read as a 7 moves a sighting five hours and
    nothing downstream can tell. Stage 11 supplies the reader; this holds what
    it produced.

    Args:
        readings: Per-frame instants the OCR produced, by frame index.
    """

    readings: dict[int, datetime]

    @property
    def tier(self) -> ReliabilityTier:
        """Return :attr:`ReliabilityTier.LOW`."""
        return ReliabilityTier.LOW

    @property
    def description(self) -> str:
        """Return a short phrase naming the source."""
        return "burned-in overlay OCR"

    def timestamp_for(self, frame_index: int) -> datetime:
        """Return the instant read from one frame.

        Args:
            frame_index: Zero-based position within the source.

        Returns:
            A timezone-aware UTC instant.

        Raises:
            IngestError: If no reading exists for that frame. Interpolating
                between neighbouring reads would manufacture a timestamp that
                looks exactly as trustworthy as a real one.
        """
        reading = self.readings.get(frame_index)
        if reading is None:
            raise IngestError(
                "No overlay reading for this frame; overlay timestamps are not interpolated",
                {"frame_index": frame_index, "frames_read": len(self.readings)},
            )
        return ensure_utc(reading, field_name="overlay_reading")


DEFAULT_FILENAME_PATTERNS: tuple[str, ...] = (
    r"(?P<year>\d{4})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})"
    r"[-_T]?(?P<hour>\d{2})[-_:]?(?P<minute>\d{2})[-_:]?(?P<second>\d{2})",
)
"""Patterns tried in order against a filename.

One pattern by default, deliberately permissive about separators, because the
same camera vendor writes ``2026-08-10T14-22-11`` and ``20260810_142211`` in
different firmware versions. A deployment adds its own rather than editing this.
"""


@dataclass(frozen=True)
class FilenameSource:
    """Timestamps parsed out of the file's name.

    Low trust. The pattern encodes an assumption about a naming convention
    nobody validates, and when the convention changes the parse either fails
    loudly -- which is fine -- or matches the wrong digits and produces a
    plausible wrong date, which is not.

    Args:
        path: The file whose name carries the timestamp.
        fps: Nominal frame rate for deriving later frames.
        timezone_name: Zone the name is written in. Required, because a bare
            ``20260810_142211`` is not an instant until someone says where.
        patterns: Regular expressions tried in order.
        timing: Per-frame presentation timestamps, preferred when present.
    """

    path: Path
    fps: float | Fraction | None = None
    timezone_name: str = "UTC"
    patterns: tuple[str, ...] = DEFAULT_FILENAME_PATTERNS
    timing: FrameTiming | None = None

    @property
    def tier(self) -> ReliabilityTier:
        """Return :attr:`ReliabilityTier.LOW`."""
        return ReliabilityTier.LOW

    @property
    def description(self) -> str:
        """Return a short phrase naming the source."""
        return f"filename pattern ({self.path.name})"

    @property
    def start_utc(self) -> datetime:
        """Return the instant parsed from the filename.

        Returns:
            The start instant in UTC.

        Raises:
            IngestError: If no pattern matches, or the matched fields are not a
                real date.
            ValidationError: If the parsed local time is ambiguous or does not
                exist in the configured zone.
        """
        name = self.path.name
        for pattern in self.patterns:
            match = re.search(pattern, name)
            if match is None:
                continue
            fields = match.groupdict()
            try:
                local = datetime(
                    year=int(fields["year"]),
                    month=int(fields["month"]),
                    day=int(fields["day"]),
                    hour=int(fields["hour"]),
                    minute=int(fields["minute"]),
                    second=int(fields["second"]),
                )
            except (KeyError, ValueError) as exc:
                raise IngestError(
                    "Filename matched a timestamp pattern but the fields are not a real date",
                    {"filename": name, "pattern": pattern, "error": str(exc)},
                ) from exc

            if self.timezone_name.upper() == "UTC":
                return local.replace(tzinfo=UTC)
            return resolve_local_time(local, self.timezone_name)

        raise IngestError(
            "No timestamp pattern matched this filename; the naming convention may have "
            "changed, and guessing at it would produce a plausible wrong date",
            {"filename": name, "patterns_tried": list(self.patterns)},
        )

    def timestamp_for(self, frame_index: int) -> datetime:
        """Return the capture instant of one frame.

        Args:
            frame_index: Zero-based position within the source.

        Returns:
            A timezone-aware UTC instant.

        Raises:
            IngestError: If the filename cannot be parsed or the frame index is
                unusable.
        """
        return derive_timestamp(self.start_utc, frame_index, self.fps, timing=self.timing)


def select_source(
    *,
    stream_start_utc: datetime | None = None,
    container_start_utc: datetime | None = None,
    path: Path | None = None,
    manual_start_utc: datetime | None = None,
    fps: float | Fraction | None = None,
    timing: FrameTiming | None = None,
    timezone_name: str = "UTC",
) -> TimestampSource:
    """Return the most trustworthy timestamp source available for one recording.

    Preference order is the reliability order, and it is deliberate: a file
    whose container metadata was stripped by a copy falls back to its filename
    and **says so**, by returning a low-tier source whose tier reaches the
    integrity gate and then the trajectory. Silently falling back would leave an
    operator reading a route with no idea that its times came from a naming
    convention.

    Args:
        stream_start_utc: Start instant from the stream's own clock.
        container_start_utc: Creation time read from the container.
        path: File whose name may carry a timestamp.
        manual_start_utc: Start instant an operator supplied.
        fps: Nominal frame rate.
        timing: Per-frame presentation timestamps, preferred over ``fps``.
        timezone_name: Zone a filename timestamp is written in.

    Returns:
        The highest-tier source that can answer.

    Raises:
        IngestError: If nothing at all is available. A recording whose capture
            time is unknown cannot contribute to a trajectory, and inventing one
            would put a vehicle somewhere at a time nobody observed.
    """
    if stream_start_utc is not None:
        return StreamClockSource(start_utc=stream_start_utc, fps=fps, timing=timing)
    if container_start_utc is not None:
        return FileMetadataSource(start_utc=container_start_utc, fps=fps, timing=timing)
    if manual_start_utc is not None:
        return ManualOffsetSource(start_utc=manual_start_utc, fps=fps, timing=timing)
    if path is not None:
        return FilenameSource(path=path, fps=fps, timezone_name=timezone_name, timing=timing)

    raise IngestError(
        "No timestamp source is available for this recording; its capture time is "
        "unknown and cannot be invented",
        {},
    )
