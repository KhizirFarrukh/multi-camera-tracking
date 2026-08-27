"""Unit tests for the multi-source reader.

The requirement that matters is not throughput. It is that a deployment with
forty cameras keeps working when one of them is broken -- which is always.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import (
    FaultInjection,
    MultiSourceReader,
    SourceStatus,
    SyntheticVideoSource,
)

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)


def _source(camera_id: str, **overrides: object) -> SyntheticVideoSource:
    """Build a small synthetic source.

    Args:
        camera_id: The camera.
        **overrides: Field overrides.

    Returns:
        The source.
    """
    fields: dict[str, object] = {
        "camera_id": camera_id,
        "start_utc": START,
        "fps": 10.0,
        "frame_count": 6,
        "width": 32,
        "height": 24,
    }
    fields.update(overrides)
    return SyntheticVideoSource(**fields)  # type: ignore[arg-type]


def test_frames_from_three_sources__arrive_tagged_with_the_right_camera() -> None:
    """The merge must not lose track of which camera saw what.

    A frame attributed to the wrong camera puts a vehicle on the wrong street.
    """
    reader = MultiSourceReader(sources=[_source(f"cam_0{n}") for n in (1, 2, 3)])

    tally: dict[str, int] = {}
    for frame in reader.frames():
        tally[frame.camera_id] = tally.get(frame.camera_id, 0) + 1

    assert tally == {"cam_01": 6, "cam_02": 6, "cam_03": 6}


def test_frames_from_one_source__keep_their_order() -> None:
    """Ordering within a camera is meaningful; ordering between them is not."""
    reader = MultiSourceReader(sources=[_source(f"cam_0{n}") for n in (1, 2)])

    per_camera: dict[str, list[int]] = {}
    for frame in reader.frames():
        per_camera.setdefault(frame.camera_id, []).append(frame.frame_index)

    for indices in per_camera.values():
        assert indices == sorted(indices)


def test_one_source_failing__does_not_stop_the_others() -> None:
    """The whole point of the class.

    Thirty-nine working cameras must not stop because of the fortieth.
    """
    broken = _source("cam_02", faults=FaultInjection(disconnect_at_index=2))
    reader = MultiSourceReader(sources=[_source("cam_01"), broken, _source("cam_03")])

    tally: dict[str, int] = {}
    for frame in reader.frames():
        tally[frame.camera_id] = tally.get(frame.camera_id, 0) + 1

    assert tally["cam_01"] == 6
    assert tally["cam_03"] == 6
    assert tally.get("cam_02", 0) == 2


def test_a_failed_source__is_reported_in_the_health_status() -> None:
    """An operator needs to know *which* camera stopped, and why."""
    broken = _source("cam_02", faults=FaultInjection(disconnect_at_index=1))
    reader = MultiSourceReader(sources=[_source("cam_01"), broken])

    list(reader.frames())

    statuses = {entry["camera_id"]: entry for entry in reader.health_report()}
    assert statuses["cam_01"]["status"] == SourceStatus.FINISHED
    assert statuses["cam_02"]["status"] == SourceStatus.FAILED
    assert "IngestError" in str(statuses["cam_02"]["error"])


def test_a_healthy_source__reports_finished_rather_than_failed() -> None:
    """A file reaching its end is success, and must not read as a fault."""
    reader = MultiSourceReader(sources=[_source("cam_01")])

    list(reader.frames())

    assert reader.health["cam_01_synthetic"].status == SourceStatus.FINISHED
    assert reader.health["cam_01_synthetic"].is_healthy
    assert reader.health["cam_01_synthetic"].frames_read == 6


def test_shutdown__drains_without_losing_buffered_frames() -> None:
    """Frames already decoded are evidence; discarding them at shutdown loses it."""
    reader = MultiSourceReader(sources=[_source(f"cam_0{n}") for n in (1, 2)], queue_size=64)

    delivered = list(reader.frames())

    assert len(delivered) == 12
    assert reader.frames_dropped == 0


def test_shutdown__closes_every_source() -> None:
    """A reader that leaks decoders is a reader nobody can run twice."""
    sources = [_source(f"cam_0{n}") for n in (1, 2, 3)]
    reader = MultiSourceReader(sources=sources)

    list(reader.frames())

    assert not any(source.is_open for source in sources)


def test_stop_on_error__ends_the_run_when_asked() -> None:
    """Available for a batch job where a partial answer is worse than none."""
    broken = _source("cam_02", faults=FaultInjection(disconnect_at_index=0))
    reader = MultiSourceReader(
        sources=[_source("cam_01", frame_count=200), broken], stop_on_error=True
    )

    delivered = list(reader.frames())

    assert len(delivered) < 200


def test_the_health_report__lists_every_source_in_configuration_order() -> None:
    """What a monitoring endpoint serves."""
    reader = MultiSourceReader(sources=[_source(f"cam_0{n}") for n in (3, 1, 2)])

    report = reader.health_report()

    assert [entry["camera_id"] for entry in report] == ["cam_03", "cam_01", "cam_02"]


def test_an_empty_reader__yields_nothing_without_hanging() -> None:
    """Boundary: no configured cameras is a configuration state, not a deadlock."""
    reader = MultiSourceReader(sources=[])

    assert list(reader.frames()) == []


def test_a_source_that_fails_to_open__is_recorded_rather_than_raised() -> None:
    """A missing file must not take down the cameras that are working."""

    class Unopenable(SyntheticVideoSource):
        """A source whose open always fails."""

        def _open(self) -> object:
            """Fail to open.

            Raises:
                IngestError: Always.
            """
            msg = "camera unplugged"
            raise IngestError(msg, {})

    reader = MultiSourceReader(sources=[Unopenable(camera_id="cam_09"), _source("cam_01")])

    frames = list(reader.frames())

    assert len(frames) == 6
    assert reader.health["cam_09_synthetic"].status == SourceStatus.FAILED
