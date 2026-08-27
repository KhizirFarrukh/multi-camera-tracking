"""Unit tests for the live stream source.

Driven through an injected opener rather than a real server, because the
behaviour under test is what happens when a stream *misbehaves* -- drops at
frame four, stalls, never comes back -- and arranging that with a real server is
harder and less precise than arranging it with a fake capture.

The three properties that matter: the buffer drops the oldest frames rather than
the newest, the decoder never blocks on a slow consumer, and reconnection backs
off with jitter and eventually gives up.
"""

from __future__ import annotations

import itertools
import random
import threading
import time

import numpy as np
import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import BackoffPolicy, ConnectionState, LiveStreamSource

pytestmark = pytest.mark.unit


class FakeCapture:
    """A capture that yields frames and then misbehaves as configured.

    Args:
        frames: How many frames to produce before ending.
        fail_after: Frame at which reads start failing.
        delay_sec: How long each read takes.
    """

    def __init__(
        self, frames: int = 5, fail_after: int | None = None, delay_sec: float = 0.0
    ) -> None:
        self.frames = frames
        self.fail_after = fail_after
        self.delay_sec = delay_sec
        self.index = 0
        self.released = False
        self.width = 32
        self.height = 24

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the next frame, or a failure.

        Returns:
            ``(ok, image)``.
        """
        if self.delay_sec:
            time.sleep(self.delay_sec)
        if self.fail_after is not None and self.index >= self.fail_after:
            return (False, None)
        if self.index >= self.frames:
            return (False, None)
        self.index += 1
        return (True, np.full((24, 32, 3), self.index % 255, dtype=np.uint8))

    def release(self) -> None:
        """Record the release."""
        self.released = True


def _source(opener: object, **overrides: object) -> LiveStreamSource:
    """Build a live source over a fake opener.

    Args:
        opener: Builds the capture.
        **overrides: Constructor overrides.

    Returns:
        The source.
    """
    fields: dict[str, object] = {
        "camera_id": "cam_01",
        "opener": opener,
        "buffer_size": 4,
        "backoff": BackoffPolicy(initial_sec=0.01, max_sec=0.05, max_attempts=2),
        "seed": 1,
    }
    fields.update(overrides)
    return LiveStreamSource("rtsp://camera/stream", **fields)  # type: ignore[arg-type]


def test_frames_arrive_from_the_stream() -> None:
    """The ordinary case."""
    source = _source(lambda _url: FakeCapture(frames=4))

    with source:
        frames = list(itertools.islice(source.frames(), 4))

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert all(frame.camera_id == "cam_01" for frame in frames)


def test_a_failed_first_connection__raises_rather_than_pretending() -> None:
    """A camera that was never reachable is a configuration problem, not a quiet one."""
    with pytest.raises(IngestError, match="Could not connect"):
        _source(lambda _url: None).open()


def test_the_source_reconnects_after_a_drop() -> None:
    """A switch reboot must not end tracking on that camera."""
    attempts = itertools.count()

    def opener(_url: str) -> FakeCapture:
        return FakeCapture(frames=3, fail_after=3 if next(attempts) == 0 else None)

    source = _source(opener)
    with source:
        frames = list(itertools.islice(source.frames(), 5))

    assert len(frames) == 5
    assert source.metrics.reconnects_succeeded >= 1


def test_it_gives_up_after_the_configured_number_of_attempts() -> None:
    """A camera that is genuinely gone must reach a terminal state.

    Retrying forever hides a dead camera behind a permanently "connecting" one.
    """
    source = _source(
        lambda _url: FakeCapture(frames=1, fail_after=1),
        backoff=BackoffPolicy(initial_sec=0.001, max_sec=0.002, max_attempts=2),
    )

    with source:
        list(itertools.islice(source.frames(), 10))
        deadline = time.monotonic() + 2.0
        while source.state is not ConnectionState.FAILED and time.monotonic() < deadline:
            time.sleep(0.01)

    assert source.metrics.reconnect_attempts >= 2


def test_the_buffer_drops_the_oldest_frames_and_counts_them() -> None:
    """Latency beats completeness: the newest frame is the one worth having.

    A drop nobody counts is indistinguishable from a quiet street, which is why
    the counter matters as much as the policy.
    """
    source = _source(lambda _url: FakeCapture(frames=50), buffer_size=3)

    source.open()
    # Let the decoder run ahead of the (absent) consumer.
    deadline = time.monotonic() + 2.0
    while source.metrics.frames_dropped == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    buffered = source.buffered_frames
    first = next(iter(source.frames()))
    source.close()

    assert source.metrics.frames_dropped > 0
    assert buffered <= 3
    # The oldest survivor, not frame zero: the early ones were dropped.
    assert first.frame_index > 0


def test_a_slow_consumer__does_not_stall_the_decoder() -> None:
    """A blocked decoder becomes a full socket buffer and then a reset connection.

    The decoder keeps reading and drops frames instead, which is visible in the
    counters rather than in a dead stream.
    """
    capture = FakeCapture(frames=200)
    source = _source(lambda _url: capture, buffer_size=2)

    source.open()
    time.sleep(0.3)
    read_while_idle = capture.index
    source.close()

    assert read_while_idle > 2


def test_a_read_timeout__is_counted() -> None:
    """A stream that goes quiet is dead, and the timeout is how that is noticed."""
    source = _source(
        lambda _url: FakeCapture(frames=5, delay_sec=0.05),
        read_timeout_sec=0.01,
        backoff=BackoffPolicy(initial_sec=0.001, max_sec=0.002, max_attempts=1),
    )

    with source:
        deadline = time.monotonic() + 2.0
        while source.metrics.read_timeouts == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert source.metrics.read_timeouts > 0


def test_closing__releases_the_capture_and_stops_the_thread() -> None:
    """A leaked reader thread holds a socket for the life of the process."""
    capture = FakeCapture(frames=1000)
    source = _source(lambda _url: capture)

    source.open()
    time.sleep(0.05)
    source.close()

    assert capture.released
    assert source.state is ConnectionState.CLOSED
    assert not any(thread.name.startswith("live-cam_01") for thread in threading.enumerate())


def test_the_connection_state__is_exposed_for_monitoring() -> None:
    """An operator has to tell a quiet street from a camera that stopped talking.

    A long capture, so the reader is still streaming when the assertion runs
    rather than having already reached the end of a five-frame fake.
    """
    source = _source(lambda _url: FakeCapture(frames=100_000))

    assert source.state is ConnectionState.IDLE
    source.open()
    assert source.state is ConnectionState.STREAMING
    source.close()
    assert source.state is ConnectionState.CLOSED


def test_a_zero_sized_buffer__is_rejected() -> None:
    """Boundary: a buffer that holds nothing drops everything."""
    with pytest.raises(IngestError, match="at least one frame"):
        _source(lambda _url: FakeCapture(), buffer_size=0)


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def test_backoff_delays__grow_with_each_attempt() -> None:
    """Exponential, so a dead camera is not hammered."""
    policy = BackoffPolicy(initial_sec=1.0, multiplier=2.0, max_sec=60.0, jitter_ratio=0.0)
    rng = random.Random(0)

    delays = [policy.delay_for(attempt, rng) for attempt in range(1, 5)]

    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_backoff_delays__are_capped() -> None:
    """A long outage must not push the retry interval into hours."""
    policy = BackoffPolicy(initial_sec=1.0, multiplier=10.0, max_sec=30.0, jitter_ratio=0.0)

    assert policy.delay_for(10, random.Random(0)) == 30.0


def test_backoff_delays__include_jitter() -> None:
    """Forty cameras behind one switch must not retry in lockstep.

    Without jitter they reconnect together and knock the switch over again the
    moment it recovers.
    """
    policy = BackoffPolicy(initial_sec=1.0, multiplier=2.0, jitter_ratio=0.5)

    delays = {policy.delay_for(3, random.Random(seed)) for seed in range(5)}

    assert len(delays) > 1
    assert all(2.0 <= delay <= 6.0 for delay in delays)


def test_the_metrics__report_the_drop_ratio() -> None:
    """What a monitoring dashboard reads to see a camera falling behind."""
    source = _source(lambda _url: FakeCapture(frames=3))

    with source:
        list(itertools.islice(source.frames(), 3))

    metrics = source.metrics.as_dict()
    assert metrics["frames_delivered"] == 3
    assert metrics["drop_ratio"] == 0.0
