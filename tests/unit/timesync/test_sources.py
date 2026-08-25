"""Unit tests for timestamp sources and their reliability tiers.

The tiers are the point. A timestamp parsed from a filename and one from a
synchronised stream are both ``datetime`` objects, and a system that forgets the
difference between them is a system that will one day present a guess as a
measurement.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from multicam_tracker.exceptions import IngestError, ValidationError
from multicam_tracker.timesync import (
    FileMetadataSource,
    FilenameSource,
    ManualOffsetSource,
    OverlayOcrSource,
    ReliabilityTier,
    StreamClockSource,
    TimestampSource,
)

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 22, 11, tzinfo=UTC)


def test_every_source__satisfies_the_protocol() -> None:
    """The pipeline holds them interchangeably, so all five must fit."""
    sources = [
        StreamClockSource(start_utc=START, fps=30.0),
        FileMetadataSource(start_utc=START, fps=30.0),
        ManualOffsetSource(start_utc=START, fps=30.0),
        FilenameSource(path=Path("20260810_142211.mp4"), fps=30.0),
        OverlayOcrSource(readings={0: START}),
    ]

    assert all(isinstance(source, TimestampSource) for source in sources)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (StreamClockSource(start_utc=START, fps=30.0), ReliabilityTier.HIGH),
        (FileMetadataSource(start_utc=START, fps=30.0), ReliabilityTier.MEDIUM),
        (ManualOffsetSource(start_utc=START, fps=30.0), ReliabilityTier.MEDIUM),
        (FilenameSource(path=Path("20260810_142211.mp4"), fps=30.0), ReliabilityTier.LOW),
        (OverlayOcrSource(readings={0: START}), ReliabilityTier.LOW),
    ],
    ids=["stream", "container", "manual", "filename", "overlay"],
)
def test_each_source__declares_the_documented_tier(
    source: TimestampSource, expected: ReliabilityTier
) -> None:
    """The tier travels into the integrity gate and onto the trajectory."""
    assert source.tier is expected


def test_the_tiers__are_ordered_so_the_weakest_source_can_be_found() -> None:
    """A query's reliability is that of its worst camera, not its average."""
    assert ReliabilityTier.HIGH.rank > ReliabilityTier.MEDIUM.rank > ReliabilityTier.LOW.rank


def test_every_source__names_itself_for_operator_facing_text() -> None:
    """The caveat on a trajectory says which source it distrusts."""
    assert StreamClockSource(start_utc=START, fps=30.0).description == "stream clock"
    assert "20260810" in FilenameSource(path=Path("20260810_142211.mp4")).description


# ---------------------------------------------------------------------------
# Frame-derived sources
# ---------------------------------------------------------------------------


def test_a_frame_derived_source__anchors_frame_zero_at_its_start() -> None:
    """Shared behaviour across the three sources that carry a start instant."""
    for source in (
        StreamClockSource(start_utc=START, fps=30.0),
        FileMetadataSource(start_utc=START, fps=30.0),
        ManualOffsetSource(start_utc=START, fps=30.0),
    ):
        assert source.timestamp_for(0) == START


def test_a_frame_derived_source__advances_with_the_frame_rate() -> None:
    """Frame 60 at 30 fps is two seconds in."""
    assert StreamClockSource(start_utc=START, fps=30.0).timestamp_for(60) == START + timedelta(
        seconds=2
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "20260810_142211.mp4",
        "cam_03-2026-08-10T14-22-11.mkv",
        "recording_2026_08_10_14_22_11.avi",
    ],
)
def test_the_documented_patterns__parse_to_the_expected_instant(name: str) -> None:
    """One vendor writes all three of these across firmware versions."""
    source = FilenameSource(path=Path(name), fps=30.0)

    assert source.start_utc == START


def test_a_non_matching_filename__raises_rather_than_guessing() -> None:
    """A convention that changed must fail loudly, not produce a plausible date."""
    source = FilenameSource(path=Path("front_door_clip.mp4"), fps=30.0)

    with pytest.raises(IngestError, match="No timestamp pattern matched"):
        source.timestamp_for(0)


def test_a_filename_matching_the_shape_but_not_a_real_date__raises() -> None:
    """The 13th month parses as digits and is not a date."""
    source = FilenameSource(path=Path("20261310_142211.mp4"), fps=30.0)

    with pytest.raises(IngestError, match="not a real date"):
        source.timestamp_for(0)


def test_a_filename_in_a_local_timezone__converts_to_utc() -> None:
    """A bare ``20260810_152211`` is not an instant until someone says where."""
    source = FilenameSource(
        path=Path("20260810_152211.mp4"), fps=30.0, timezone_name="Europe/London"
    )

    assert source.start_utc == START


def test_a_filename_in_the_dst_gap__raises_rather_than_guessing() -> None:
    """The naming convention cannot express which reading was meant."""
    source = FilenameSource(
        path=Path("20261025_013000.mp4"), fps=30.0, timezone_name="Europe/London"
    )

    with pytest.raises(ValidationError, match="occurs twice"):
        source.timestamp_for(0)


# ---------------------------------------------------------------------------
# Overlay OCR
# ---------------------------------------------------------------------------


def test_an_overlay_source__returns_the_reading_for_that_frame() -> None:
    """Each frame carries its own burned-in time; nothing is derived."""
    source = OverlayOcrSource(readings={0: START, 5: START + timedelta(seconds=1)})

    assert source.timestamp_for(5) == START + timedelta(seconds=1)


def test_an_overlay_source__does_not_interpolate_a_missing_frame() -> None:
    """An interpolated timestamp looks exactly as trustworthy as a real one."""
    source = OverlayOcrSource(readings={0: START})

    with pytest.raises(IngestError, match="not interpolated"):
        source.timestamp_for(3)


def test_an_overlay_reading_without_a_timezone__is_rejected() -> None:
    """OCR of a naive clock face is not an instant."""
    source = OverlayOcrSource(readings={0: datetime(2026, 8, 10, 14, 22, 11)})

    with pytest.raises(ValidationError, match="Naive datetime"):
        source.timestamp_for(0)
