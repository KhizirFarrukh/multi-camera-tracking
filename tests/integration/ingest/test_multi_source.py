"""Several real decoders running at once, and one of them broken.

The unit tests cover the merging logic against generated sources. These run it
against real files and a real damaged one, which is the arrangement a deployment
actually has: a handful of cameras, one of which is a problem today.

Also here: the end-to-end timestamp check. Two files with different configured
clock offsets have to be comparable in UTC, because that comparability is the
premise the whole system rests on.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from multicam_tracker.ingest import (
    FileVideoSource,
    MultiSourceReader,
    SourceStatus,
    SyntheticVideoSource,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MEDIA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "media"
CLEAN = MEDIA_DIR / "sample_clean.mp4"
UNDECODABLE = MEDIA_DIR / "sample_undecodable.mp4"
START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)
EXPECTED_FRAMES = 20


def _has_opencv() -> bool:
    """Return whether OpenCV is importable.

    Returns:
        ``True`` when it imports.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


requires_opencv = pytest.mark.skipif(not _has_opencv(), reason="OpenCV is not installed")


def _file_source(camera_id: str, **overrides: object) -> FileVideoSource:
    """Build a file source over the clean clip.

    Args:
        camera_id: The camera.
        **overrides: Field overrides.

    Returns:
        The source.
    """
    fields: dict[str, object] = {
        "path": CLEAN,
        "camera_id": camera_id,
        "start_utc": START,
        "source_id": f"{camera_id}_clip",
    }
    fields.update(overrides)
    return FileVideoSource(**fields)  # type: ignore[arg-type]


@requires_opencv
def test_three_real_files__are_decoded_concurrently_and_tagged_correctly() -> None:
    """The ordinary deployment shape, against real decoders."""
    reader = MultiSourceReader(sources=[_file_source(f"cam_0{n}") for n in (1, 2, 3)])

    tally: dict[str, int] = {}
    for frame in reader.frames():
        tally[frame.camera_id] = tally.get(frame.camera_id, 0) + 1

    assert tally == {f"cam_0{n}": EXPECTED_FRAMES for n in (1, 2, 3)}


@requires_opencv
def test_one_unreadable_file__does_not_stop_the_readable_ones() -> None:
    """A damaged file in a batch must cost that file, and nothing else."""
    broken = FileVideoSource(
        path=UNDECODABLE, camera_id="cam_09", start_utc=START, source_id="cam_09_clip"
    )
    reader = MultiSourceReader(sources=[_file_source("cam_01"), broken, _file_source("cam_02")])

    tally: dict[str, int] = {}
    for frame in reader.frames():
        tally[frame.camera_id] = tally.get(frame.camera_id, 0) + 1

    assert tally["cam_01"] == EXPECTED_FRAMES
    assert tally["cam_02"] == EXPECTED_FRAMES
    assert "cam_09" not in tally
    assert reader.health["cam_09_clip"].status == SourceStatus.FAILED


@requires_opencv
def test_the_health_report__names_the_failing_camera_and_the_reason() -> None:
    """An operator reads this to know which camera to go and look at."""
    broken = FileVideoSource(
        path=UNDECODABLE, camera_id="cam_09", start_utc=START, source_id="cam_09_clip"
    )
    reader = MultiSourceReader(sources=[_file_source("cam_01"), broken])

    list(reader.frames())
    report = {entry["camera_id"]: entry for entry in reader.health_report()}

    assert report["cam_01"]["status"] == SourceStatus.FINISHED
    assert report["cam_09"]["status"] == SourceStatus.FAILED
    assert "IngestError" in str(report["cam_09"]["error"])


@requires_opencv
def test_shutdown__releases_every_real_decoder() -> None:
    """Three decoders left open per run is how a long batch runs out of handles."""
    sources = [_file_source(f"cam_0{n}") for n in (1, 2, 3)]
    reader = MultiSourceReader(sources=sources)

    list(reader.frames())

    assert not any(source.is_open for source in sources)


@pytest.mark.slow
def test_three_sources_concurrently__beat_reading_them_one_after_another() -> None:
    """The reason for threads at all.

    Synthetic sources with an injected stall stand in for decode latency: the
    property under test is that the three waits overlap, and a real decoder
    would measure the machine rather than the concurrency.
    """
    from multicam_tracker.ingest import FaultInjection

    def build(camera_id: str) -> SyntheticVideoSource:
        return SyntheticVideoSource(
            camera_id=camera_id,
            start_utc=START,
            frame_count=4,
            width=32,
            height=24,
            faults=FaultInjection(stall_at_index=1, stall_seconds=0.3),
        )

    started = time.monotonic()
    for camera_id in ("cam_01", "cam_02", "cam_03"):
        with build(camera_id) as source:
            list(source.frames())
    sequential_sec = time.monotonic() - started

    started = time.monotonic()
    reader = MultiSourceReader(sources=[build(f"cam_0{n}") for n in (1, 2, 3)])
    list(reader.frames())
    concurrent_sec = time.monotonic() - started

    assert concurrent_sec < sequential_sec


# ---------------------------------------------------------------------------
# Timestamps end to end
# ---------------------------------------------------------------------------


@requires_opencv
def test_a_file_with_a_known_start_and_offset__carries_the_exact_expected_time() -> None:
    """Every layer between the container and the frame has to be arithmetic-clean.

    Frame 7 of a 10 fps clip starting at 14:00:00, on a camera 45 seconds fast,
    is 14:00:00.700 minus 45 seconds. No tolerance: this is exact or it is wrong.
    """
    with _file_source("cam_01", clock_offset_ms=-45_000) as source:
        frames = list(source.frames())

    frame = frames[7]
    assert frame.raw_timestamp == START + timedelta(milliseconds=700)
    assert frame.timestamp_utc == START + timedelta(milliseconds=700) - timedelta(seconds=45)


@requires_opencv
def test_two_cameras_with_different_offsets__are_comparable_in_utc() -> None:
    """The premise the entire system rests on.

    Both cameras saw the same instant; their raw clocks disagree by a minute;
    after correction their frames line up.
    """
    ahead = _file_source("cam_01", start_utc=START + timedelta(seconds=60), clock_offset_ms=-60_000)
    on_time = _file_source("cam_02", start_utc=START, clock_offset_ms=0)

    with ahead as first, on_time as second:
        first_frames = list(first.frames())
        second_frames = list(second.frames())

    assert first_frames[3].raw_timestamp != second_frames[3].raw_timestamp
    assert first_frames[3].timestamp_utc == second_frames[3].timestamp_utc
