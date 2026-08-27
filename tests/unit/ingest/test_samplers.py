"""Unit tests for the frame samplers.

The rule every one of them has to keep: a sampled frame carries the timestamp
and index of the frame that was actually decoded. A sampler that renumbered or
interpolated would move a vehicle to a moment nobody observed, and the
trajectory built from it would be confidently wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import (
    AdaptiveSampler,
    EveryNthFrame,
    Frame,
    FrameSampler,
    KeyframeOnlySampler,
    SyntheticVideoSource,
    TargetFpsSampler,
)

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 10, 14, 0, 0, tzinfo=UTC)


def _frames(count: int = 30, fps: float = 30.0) -> list[Frame]:
    """Return generated frames.

    Args:
        count: How many.
        fps: Source rate.

    Returns:
        The frames.
    """
    source = SyntheticVideoSource(
        camera_id="cam_01", start_utc=START, fps=fps, frame_count=count, width=32, height=24
    )
    with source:
        return list(source.frames())


def test_every_sampler__satisfies_the_protocol() -> None:
    """They are held interchangeably by the pipeline, so all four must fit."""
    samplers = [
        EveryNthFrame(n=2),
        TargetFpsSampler(target_fps=5),
        KeyframeOnlySampler(),
        AdaptiveSampler(),
    ]

    assert all(isinstance(sampler, FrameSampler) for sampler in samplers)


# ---------------------------------------------------------------------------
# EveryNthFrame
# ---------------------------------------------------------------------------


def test_every_nth__yields_exactly_every_nth_frame() -> None:
    """Predictable and countable, which is what it is for."""
    sampled = list(EveryNthFrame(n=5).sample(_frames(20)))

    assert [frame.frame_index for frame in sampled] == [0, 5, 10, 15]


def test_every_nth__with_n_of_one__yields_everything() -> None:
    """Boundary: the identity case must not drop anything."""
    frames = _frames(7)

    assert len(list(EveryNthFrame(n=1).sample(frames))) == 7


def test_every_nth__with_n_beyond_the_frame_count__yields_exactly_one() -> None:
    """Boundary: the first frame always passes, so a short clip still produces one."""
    sampled = list(EveryNthFrame(n=500).sample(_frames(10)))

    assert [frame.frame_index for frame in sampled] == [0]


def test_every_nth__rejects_a_non_positive_interval() -> None:
    """Zero would divide by zero; negative is meaningless."""
    with pytest.raises(IngestError, match="at least 1"):
        EveryNthFrame(n=0)


# ---------------------------------------------------------------------------
# TargetFpsSampler
# ---------------------------------------------------------------------------


def test_target_fps__at_half_the_source_rate__yields_about_half_the_frames() -> None:
    """The default sampler, doing the thing it is the default for."""
    sampled = list(TargetFpsSampler(target_fps=15.0).sample(_frames(60, fps=30.0)))

    assert 28 <= len(sampled) <= 32


def test_target_fps__above_the_source_rate__yields_every_frame_and_duplicates_none() -> None:
    """Asking for more than exists gets everything, not repeats.

    A sampler that padded to reach its target would invent observations.
    """
    frames = _frames(20, fps=10.0)
    sampled = list(TargetFpsSampler(target_fps=60.0).sample(frames))

    assert len(sampled) == len(frames)
    assert len({frame.frame_index for frame in sampled}) == len(frames)


def test_target_fps__is_independent_of_the_source_rate() -> None:
    """One configuration across a mixed estate, which is why this is the default."""
    slow = list(TargetFpsSampler(target_fps=5.0).sample(_frames(30, fps=10.0)))
    fast = list(TargetFpsSampler(target_fps=5.0).sample(_frames(90, fps=30.0)))

    # Both cover three seconds of footage at five frames a second.
    assert abs(len(slow) - len(fast)) <= 1


def test_target_fps__rejects_a_non_positive_rate() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="must be positive"):
        TargetFpsSampler(target_fps=0)


# ---------------------------------------------------------------------------
# The rule they all keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler",
    [EveryNthFrame(n=4), TargetFpsSampler(target_fps=5.0), KeyframeOnlySampler()],
    ids=["every-nth", "target-fps", "keyframe"],
)
def test_sampled_frames__keep_their_original_timestamps_and_indices(
    sampler: FrameSampler,
) -> None:
    """The rule the whole stage turns on.

    Every sampled frame is checked against the source frame with the same index:
    same instant, same raw timestamp, same position.
    """
    frames = _frames(40, fps=20.0)
    by_index = {frame.frame_index: frame for frame in frames}

    for sampled in sampler.sample(frames):
        original = by_index[sampled.frame_index]
        assert sampled.timestamp_utc == original.timestamp_utc
        assert sampled.raw_timestamp == original.raw_timestamp
        assert sampled.frame_index == original.frame_index


@pytest.mark.parametrize(
    "sampler",
    [
        EveryNthFrame(n=3),
        TargetFpsSampler(target_fps=5.0),
        KeyframeOnlySampler(),
        AdaptiveSampler(),
    ],
    ids=["every-nth", "target-fps", "keyframe", "adaptive"],
)
def test_every_sampler__handles_an_empty_source_without_raising(
    sampler: FrameSampler,
) -> None:
    """A camera that produced nothing is an ordinary case, not an error."""
    assert list(sampler.sample([])) == []


# ---------------------------------------------------------------------------
# KeyframeOnlySampler
# ---------------------------------------------------------------------------


def test_keyframe_only__yields_only_keyframes() -> None:
    """The cheapest sampler: keyframes decode without their neighbours."""
    sampled = list(KeyframeOnlySampler().sample(_frames(35)))

    assert [frame.frame_index for frame in sampled] == [0, 10, 20, 30]
    assert all(frame.is_keyframe for frame in sampled)


# ---------------------------------------------------------------------------
# AdaptiveSampler
# ---------------------------------------------------------------------------


def test_adaptive__samples_faster_after_a_detection() -> None:
    """Spend frames where something is happening."""
    frames = _frames(60, fps=20.0)

    idle = AdaptiveSampler(idle_fps=2.0, active_fps=10.0, decay_sec=5.0)
    quiet = list(idle.sample(frames))

    busy_sampler = AdaptiveSampler(idle_fps=2.0, active_fps=10.0, decay_sec=5.0)
    busy_sampler.note_detection(START)
    busy = list(busy_sampler.sample(frames))

    assert len(busy) > len(quiet)


def test_adaptive__decays_back_to_the_idle_rate() -> None:
    """An event does not keep a camera at full rate forever."""
    sampler = AdaptiveSampler(idle_fps=2.0, active_fps=10.0, decay_sec=0.5)
    sampler.note_detection(START)

    frames = _frames(80, fps=20.0)
    sampled = list(sampler.sample(frames))

    early = [f for f in sampled if (f.timestamp_utc - START) < timedelta(seconds=0.5)]
    late = [f for f in sampled if (f.timestamp_utc - START) >= timedelta(seconds=2)]

    # Ten a second while active, two a second once it has decayed.
    assert len(early) >= 4
    assert len(late) <= (len(frames) / 20) * 2 + 2
    assert not sampler.is_active


def test_adaptive__rejects_an_active_rate_below_the_idle_rate() -> None:
    """That configuration would make a detection *slow* sampling down."""
    with pytest.raises(IngestError, match="at least the idle rate"):
        AdaptiveSampler(idle_fps=10.0, active_fps=2.0)


def test_adaptive__rejects_a_non_positive_rate() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="must be positive"):
        AdaptiveSampler(idle_fps=0.0, active_fps=5.0)


def test_reset__returns_a_sampler_to_its_starting_state() -> None:
    """A source that reconnected starts again; a stale cursor would skip frames."""
    sampler = TargetFpsSampler(target_fps=5.0)
    frames = _frames(20, fps=10.0)

    first_pass = list(sampler.sample(frames))
    sampler.reset()
    second_pass = list(sampler.sample(frames))

    assert [f.frame_index for f in first_pass] == [f.frame_index for f in second_pass]
