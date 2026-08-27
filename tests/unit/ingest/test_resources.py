"""Unit tests for frame ownership and resource release.

Two failures this guards against, both of which are invisible until they are
catastrophic: a consumer writing into a decoder's buffer, and a source that
holds its decoder after the caller has stopped reading. The second one takes
down a batch run half way through, on the machine with the most files.
"""

from __future__ import annotations

import gc
from datetime import UTC, datetime

import numpy as np
import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import Frame, SyntheticVideoSource
from multicam_tracker.ingest.protocol import BaseVideoSource, SourceKind, SourceProperties

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)


def _frame(image: np.ndarray | None = None) -> Frame:
    """Build a frame.

    Args:
        image: The image, or a default one.

    Returns:
        The frame.
    """
    return Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8) if image is None else image,
        frame_index=0,
        raw_timestamp=START,
        timestamp_utc=START,
        source_id="src",
        camera_id="cam_01",
    )


# ---------------------------------------------------------------------------
# Memory ownership
# ---------------------------------------------------------------------------


def test_a_frames_image__is_read_only_to_consumers() -> None:
    """A decoder reuses its buffer; a consumer that writes corrupts the next frame.

    Marked non-writeable so the mistake raises instead of producing images that
    are subtly wrong.
    """
    frame = _frame()

    assert not frame.image.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        frame.image[0, 0] = 255


def test_owned_copy__is_writeable_and_independent() -> None:
    """The one supported way to get an array a consumer may keep."""
    frame = _frame()

    copy = frame.owned_copy()
    copy[0, 0] = (1, 2, 3)

    assert copy.flags.writeable
    assert frame.image[0, 0].tolist() == [0, 0, 0]


def test_replacing_the_image__keeps_every_other_field() -> None:
    """Preprocessing changes pixels and nothing else."""
    frame = _frame()

    replaced = frame.replacing_image(np.ones((4, 4, 3), dtype=np.uint8))

    assert replaced.frame_index == frame.frame_index
    assert replaced.timestamp_utc == frame.timestamp_utc
    assert replaced.shape == (4, 4)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.zeros((8, 8), dtype=np.uint8), "three-channel"),
        (np.zeros((8, 8, 4), dtype=np.uint8), "three-channel"),
        (np.zeros((8, 8, 3), dtype=np.float32), "8-bit"),
    ],
)
def test_an_unusable_image__is_rejected_at_the_boundary(image: np.ndarray, match: str) -> None:
    """A grayscale or float array reaching a detector yields a confident wrong
    answer rather than an error, so it is refused here."""
    with pytest.raises(IngestError, match=match):
        _frame(image)


def test_the_representation__does_not_print_the_whole_image() -> None:
    """A frame is logged thousands of times a minute."""
    text = repr(_frame())

    assert "cam_01" in text
    assert len(text) < 200


# ---------------------------------------------------------------------------
# Release on every exit path
# ---------------------------------------------------------------------------


class _TrackingSource(BaseVideoSource):
    """A source that records how many times it was opened and closed.

    Args:
        frame_count: How many frames to yield before finishing.
    """

    def __init__(self, frame_count: int = 5) -> None:
        super().__init__("tracking", "cam_01")
        self.frame_count = frame_count
        self.opened = 0
        self.closed = 0

    def _open(self) -> SourceProperties:
        """Record the open and return trivial properties.

        Returns:
            The properties.
        """
        self.opened += 1
        from fractions import Fraction

        return SourceProperties(fps=Fraction(10, 1), width=8, height=8, kind=SourceKind.SYNTHETIC)

    def _frames(self):  # type: ignore[no-untyped-def]
        """Yield the configured number of frames.

        Yields:
            Frames.
        """
        for _ in range(self.frame_count):
            yield _frame()

    def _close(self) -> None:
        """Record the close."""
        self.closed += 1


def test_the_context_manager__releases_on_normal_exit() -> None:
    """The ordinary path."""
    source = _TrackingSource()

    with source:
        list(source.frames())

    assert source.closed == 1


def test_the_context_manager__releases_when_the_body_raises() -> None:
    """An exception must not leak a decoder handle."""
    source = _TrackingSource()

    with pytest.raises(RuntimeError), source:
        msg = "something went wrong downstream"
        raise RuntimeError(msg)

    assert source.closed == 1


def test_abandoning_the_generator__still_releases_the_decoder() -> None:
    """The subtle one.

    Breaking out of a frame loop leaves a suspended generator; without the
    try/finally around the yield, the decoder is held until the collector
    happens to run -- which on a batch of ten thousand files is far too late.
    """
    source = _TrackingSource(frame_count=1000)
    source.open()

    for _ in source.frames():
        break
    gc.collect()

    assert source.closed == 1


def test_closing_twice__is_harmless() -> None:
    """Shutdown paths overlap; a double close must not raise or double-count."""
    source = _TrackingSource()
    source.open()

    source.close()
    source.close()

    assert source.closed == 1


def test_reading_from_an_unopened_source__is_refused() -> None:
    """Better than yielding nothing, which reads as an empty camera."""
    with pytest.raises(IngestError, match="not open"):
        list(_TrackingSource().frames())


def test_properties_before_opening__are_refused() -> None:
    """They come from the container; inventing them would be guessing."""
    with pytest.raises(IngestError, match="not known until"):
        _ = _TrackingSource().properties


def test_a_closed_source__cannot_be_reopened() -> None:
    """A decoder handle is not resurrectable; construct a new source instead."""
    source = _TrackingSource()
    source.open()
    source.close()

    with pytest.raises(IngestError, match="cannot be reopened"):
        source.open()


def test_opening_twice__does_not_acquire_twice() -> None:
    """A caller using the context manager after an explicit open is reasonable."""
    source = _TrackingSource()

    source.open()
    source.open()

    assert source.opened == 1


def test_two_hundred_sources_in_sequence__leave_nothing_open() -> None:
    """The batch-run case: file descriptors are finite.

    Counted rather than measured with a descriptor probe, because that count is
    the property under test and a probe would be measuring the platform.
    """
    sources = []
    for _ in range(200):
        source = SyntheticVideoSource(
            camera_id="cam_01", start_utc=START, frame_count=2, width=16, height=12
        )
        with source:
            list(source.frames())
        sources.append(source)

    assert not any(source.is_open for source in sources)
