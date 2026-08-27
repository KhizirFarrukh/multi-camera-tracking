"""Deriving a recording's start instant from what the file actually offers.

Against committed files on disk rather than constructed paths, because the
failure this guards against is a naming convention changing under a deployment
-- and a convention lives in filenames, not in string literals.

The container path is exercised through its seam: stage 10 owns the demuxer, so
the creation time is supplied here as a stage-10 reader will supply it, and
``None`` stands for metadata a copy stripped. See the README beside the
fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from multicam_tracker.exceptions import IngestError, ValidationError
from multicam_tracker.timesync import ReliabilityTier, select_source

pytestmark = pytest.mark.integration

MEDIA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "media"
EXPECTED = datetime(2026, 8, 10, 14, 22, 11, tzinfo=UTC)


@pytest.mark.parametrize(
    "filename",
    [
        "cam_01_20260810_142211.mp4",
        "cam_03-2026-08-10T14-22-11.mkv",
        "recording_2026_08_10_14_22_11.avi",
    ],
)
def test_the_documented_naming_conventions__all_parse_to_the_same_instant(
    filename: str,
) -> None:
    """One vendor writes all three of these across firmware versions."""
    source = select_source(path=MEDIA_DIR / filename, fps=30.0)

    assert source.timestamp_for(0) == EXPECTED
    assert source.tier is ReliabilityTier.LOW


def test_a_file_whose_name_carries_no_timestamp__fails_loudly() -> None:
    """A convention that changed must not produce a plausible wrong date."""
    source = select_source(path=MEDIA_DIR / "front_door_clip.mp4", fps=30.0)

    with pytest.raises(IngestError, match="No timestamp pattern matched"):
        source.timestamp_for(0)


def test_a_filename_inside_the_dst_fall_back_hour__refuses_to_pick_a_reading() -> None:
    """The name cannot express which 01:30 was meant, so nothing here may decide."""
    source = select_source(
        path=MEDIA_DIR / "cam_07_20261025_013000.mp4", fps=30.0, timezone_name="Europe/London"
    )

    with pytest.raises(ValidationError, match="occurs twice"):
        source.timestamp_for(0)


def test_every_filename_fixture__is_covered_by_a_test() -> None:
    """A fixture nobody reads is a fixture that quietly stops being right.

    Scoped to the ``cam_*`` names: stage 10 shares this directory for its
    decodable sample clips, which are covered by their own suite.
    """
    committed = {
        path.name
        for path in MEDIA_DIR.iterdir()
        if path.suffix != ".md" and path.name.startswith("cam_")
    }

    assert committed == {
        "cam_01_20260810_142211.mp4",
        "cam_03-2026-08-10T14-22-11.mkv",
        "cam_07_20261025_013000.mp4",
    }


def test_the_non_prefixed_fixtures__are_the_ones_this_suite_names() -> None:
    """The two that carry no camera prefix are still named here explicitly."""
    committed = {path.name for path in MEDIA_DIR.iterdir() if path.suffix != ".md"}

    assert "recording_2026_08_10_14_22_11.avi" in committed
    assert "front_door_clip.mp4" in committed


# ---------------------------------------------------------------------------
# Choosing between what is available
# ---------------------------------------------------------------------------


def test_the_stream_clock__wins_when_it_is_available() -> None:
    """The device's own clock is the only source nothing else has to stand in for."""
    source = select_source(
        stream_start_utc=EXPECTED,
        container_start_utc=EXPECTED,
        path=MEDIA_DIR / "cam_01_20260810_142211.mp4",
        fps=30.0,
    )

    assert source.tier is ReliabilityTier.HIGH
    assert source.description == "stream clock"


def test_container_metadata__is_used_when_there_is_no_stream_clock() -> None:
    """The ordinary case for a file dropped into a batch import."""
    source = select_source(
        container_start_utc=EXPECTED, path=MEDIA_DIR / "cam_01_20260810_142211.mp4", fps=30.0
    )

    assert source.tier is ReliabilityTier.MEDIUM
    assert source.timestamp_for(0) == EXPECTED


def test_missing_metadata__falls_back_to_the_filename_and_reports_lower_reliability() -> None:
    """A copy that stripped the metadata must not look like one that kept it.

    The fallback is fine; the fallback happening silently is not. The tier is
    what carries that fact into the integrity gate and onto the trajectory.
    """
    with_metadata = select_source(container_start_utc=EXPECTED, fps=30.0)
    without = select_source(path=MEDIA_DIR / "cam_01_20260810_142211.mp4", fps=30.0)

    assert with_metadata.tier is ReliabilityTier.MEDIUM
    assert without.tier is ReliabilityTier.LOW
    assert without.tier.rank < with_metadata.tier.rank
    assert without.timestamp_for(0) == with_metadata.timestamp_for(0)


def test_an_operator_supplied_start__outranks_a_filename() -> None:
    """A human stating a start time is a claim they can be held to.

    A naming convention is a claim nobody made on purpose.
    """
    source = select_source(
        manual_start_utc=EXPECTED, path=MEDIA_DIR / "front_door_clip.mp4", fps=30.0
    )

    assert source.tier is ReliabilityTier.MEDIUM
    assert source.timestamp_for(0) == EXPECTED


def test_a_recording_with_no_usable_source__is_refused() -> None:
    """A capture time that is unknown cannot be invented.

    Inventing one would put a vehicle somewhere at a time nobody observed, which
    is exactly the confident wrong answer this stage exists to prevent.
    """
    with pytest.raises(IngestError, match="capture time is unknown"):
        select_source(fps=30.0)
