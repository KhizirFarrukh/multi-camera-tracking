"""Unit tests for the generated source.

This is the source every later vision stage is tested against, so its own
guarantees have to hold exactly: the same seed produces the same pixels, the
timestamps advance by exact rational steps, and every injected fault does what
it says.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import numpy as np
import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import FaultInjection, SourceKind, SyntheticVideoSource

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)


def _source(**overrides: object) -> SyntheticVideoSource:
    """Build a source with test-friendly defaults.

    Args:
        **overrides: Field overrides.

    Returns:
        The configured source.
    """
    fields: dict[str, object] = {
        "camera_id": "cam_01",
        "start_utc": START,
        "fps": 10.0,
        "frame_count": 10,
        "width": 64,
        "height": 48,
    }
    fields.update(overrides)
    return SyntheticVideoSource(**fields)  # type: ignore[arg-type]


def test_yields_exactly_the_configured_number_of_frames() -> None:
    """The count is the contract; a source that over-delivers breaks every count."""
    with _source(frame_count=17) as source:
        assert len(list(source.frames())) == 17


def test_frame_indices_are_sequential_from_zero() -> None:
    """Index zero is the anchor every timestamp is derived from."""
    with _source(frame_count=6) as source:
        assert [frame.frame_index for frame in source.frames()] == [0, 1, 2, 3, 4, 5]


def test_timestamps_advance_by_exactly_one_over_fps() -> None:
    """Hand-computable: 10 fps is 100 ms per frame, exactly."""
    with _source(fps=10.0, frame_count=4) as source:
        stamps = [frame.timestamp_utc for frame in source.frames()]

    assert stamps == [START + timedelta(milliseconds=100 * index) for index in range(4)]


def test_a_rational_frame_rate__stays_exact_across_many_frames() -> None:
    """29.97 is 30000/1001, and the difference accumulates.

    Stage 09 owns that arithmetic; this asserts the source actually uses it
    rather than rounding to a float somewhere on the way.
    """
    rate = Fraction(30000, 1001)
    with _source(fps=rate, frame_count=1) as source:
        anchor = next(iter(source.frames())).timestamp_utc

    with _source(fps=rate, frame_count=1000) as long_source:
        last = list(long_source.frames())[-1]

    expected = float(Fraction(999) / rate)
    assert abs((last.timestamp_utc - anchor).total_seconds() - expected) < 0.001


def test_the_same_seed__produces_identical_pixels() -> None:
    """Determinism is what makes a failing test mean the code changed."""
    first = _source(seed=7).render(5)
    second = _source(seed=7).render(5)

    assert np.array_equal(first, second)


def test_a_different_seed__produces_different_pixels() -> None:
    """The control: the seed has to do something."""
    assert not np.array_equal(_source(seed=1).render(5), _source(seed=2).render(5))


def test_a_frame__is_the_same_whether_reached_by_iterating_or_directly() -> None:
    """Seeded per index, not carried forward -- which is what makes a seek honest."""
    source = _source(seed=3, frame_count=20)
    with source:
        iterated = next(frame for frame in source.frames() if frame.frame_index == 12)

    assert np.array_equal(iterated.image, _source(seed=3, frame_count=20).render(12))


def test_properties_match_the_configuration() -> None:
    """What the source says about itself has to match what it does."""
    with _source(fps=15.0, frame_count=45, width=320, height=240) as source:
        properties = source.properties

    assert properties.fps == Fraction(15, 1)
    assert properties.resolution == (320, 240)
    assert properties.frame_count == 45
    assert properties.kind is SourceKind.SYNTHETIC
    assert not properties.is_live


def test_every_tenth_frame__is_marked_a_keyframe() -> None:
    """Mirrors a typical encoder GOP, so the keyframe sampler has something to select."""
    with _source(frame_count=25) as source:
        keyframes = [frame.frame_index for frame in source.frames() if frame.is_keyframe]

    assert keyframes == [0, 10, 20]


def test_the_clock_offset__is_applied_to_the_corrected_timestamp_only() -> None:
    """The raw timestamp is what the source reported; only the corrected one moves."""
    with _source(clock_offset_ms=-45_000, frame_count=1) as source:
        frame = next(iter(source.frames()))

    assert frame.raw_timestamp == START
    assert frame.timestamp_utc == START - timedelta(seconds=45)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def test_a_corrupt_frame__is_skipped_and_counted_without_aborting() -> None:
    """One unreadable frame in an hour of footage is still an hour of footage."""
    source = _source(frame_count=10, faults=FaultInjection(corrupt_frame_indices=frozenset({3, 4})))

    with source:
        indices = [frame.frame_index for frame in source.frames()]

    assert 3 not in indices
    assert 4 not in indices
    assert len(indices) == 8
    assert source.corrupt_frames_skipped == 2


def test_a_permanent_disconnect__ends_the_stream_with_an_error() -> None:
    """The caller has to be able to tell a finished source from a broken one."""
    source = _source(frame_count=10, faults=FaultInjection(disconnect_at_index=4))

    with source, pytest.raises(IngestError, match="disconnected"):
        list(source.frames())

    assert source.disconnect_count == 1


def test_a_recoverable_disconnect__costs_one_frame_and_continues() -> None:
    """A blip is not an outage, and treating it as one loses the rest of the run."""
    source = _source(
        frame_count=8,
        faults=FaultInjection(disconnect_at_index=4, disconnect_is_permanent=False),
    )

    with source:
        indices = [frame.frame_index for frame in source.frames()]

    assert indices == [0, 1, 2, 3, 5, 6, 7]
    assert source.disconnect_count == 1


def test_an_injected_stall__actually_delays_the_stream() -> None:
    """What a read timeout is meant to notice.

    Measured against an unstalled run rather than against the wall clock:
    ``time.sleep`` returns early by up to a scheduler tick on Windows, so
    asserting the absolute duration makes the test fail on timer granularity
    rather than on behaviour.
    """
    stalled = _source(frame_count=4, faults=FaultInjection(stall_at_index=2, stall_seconds=0.2))

    started = time.monotonic()
    with _source(frame_count=4) as baseline:
        list(baseline.frames())
    unstalled_sec = time.monotonic() - started

    started = time.monotonic()
    with stalled:
        list(stalled.frames())
    stalled_sec = time.monotonic() - started

    assert stalled_sec > unstalled_sec + 0.1


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("width", "height"), [(0, 48), (64, 0), (-10, 48)])
def test_non_positive_dimensions__are_rejected(width: int, height: int) -> None:
    """A zero-sized frame fails far downstream, in the detector, for no clear reason."""
    with pytest.raises(IngestError, match="dimensions must be positive"):
        _source(width=width, height=height)


def test_a_negative_frame_count__is_rejected() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="cannot be negative"):
        _source(frame_count=-1)


def test_a_zero_frame_source__yields_nothing_without_raising() -> None:
    """Boundary: an empty source is a legitimate configuration."""
    with _source(frame_count=0) as source:
        assert list(source.frames()) == []
