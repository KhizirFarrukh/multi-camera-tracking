"""The live source against a real server on loopback.

The unit tests drive the reconnection logic through an injected capture, which
is where the precise fault cases belong. These prove the same code works against
a real socket and a real decoder -- the part a fake capture cannot vouch for.

The server is local and committed (``tests/fixtures/local_stream_server.py``):
depending on an external endpoint would make the suite flaky and the failures
someone else's to fix.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import pytest

from multicam_tracker.ingest import BackoffPolicy, ConnectionState, LiveStreamSource
from tests.fixtures.local_stream_server import serving_clip

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MEDIA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "media"
CLIP = MEDIA_DIR / "sample_clean.mp4"


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


@requires_opencv
def test_it_connects_to_a_local_server_and_yields_frames() -> None:
    """The basic proof that the live path works over a real socket."""
    with serving_clip(CLIP) as server:
        source = LiveStreamSource(
            server.url,
            "cam_01",
            buffer_size=8,
            backoff=BackoffPolicy(initial_sec=0.01, max_attempts=1),
        )
        with source:
            frames = list(itertools.islice(source.frames(), 3))

    assert len(frames) == 3
    assert all(frame.camera_id == "cam_01" for frame in frames)
    assert frames[0].shape == (64, 48)


@requires_opencv
def test_frames_carry_arrival_timestamps_in_order() -> None:
    """A live frame's time is when it arrived; there is no container to ask.

    Ordering is what downstream tracking depends on, so it is asserted even
    though the absolute instants are wall-clock.
    """
    with serving_clip(CLIP) as server:
        source = LiveStreamSource(
            server.url, "cam_01", backoff=BackoffPolicy(initial_sec=0.01, max_attempts=1)
        )
        with source:
            frames = list(itertools.islice(source.frames(), 4))

    stamps = [frame.timestamp_utc for frame in frames]
    assert stamps == sorted(stamps)


@requires_opencv
def test_the_source_reconnects_after_the_server_drops_it() -> None:
    """The failure this module exists for, against a server that really drops.

    The server serves one request and then closes without responding, which is
    what a camera does when its session expires.
    """
    with serving_clip(CLIP, fail_after_requests=1) as server:
        source = LiveStreamSource(
            server.url,
            "cam_01",
            buffer_size=4,
            read_timeout_sec=2.0,
            backoff=BackoffPolicy(initial_sec=0.05, max_sec=0.2, max_attempts=3),
        )
        with source:
            list(itertools.islice(source.frames(), 2))
            deadline = time.monotonic() + 5.0
            while source.metrics.reconnect_attempts == 0 and time.monotonic() < deadline:
                time.sleep(0.05)

    # The retry itself is the property under test. Whether it reaches the
    # server as another GET depends on where FFmpeg gives up -- on a connection
    # closed without a response it often fails before sending anything -- and
    # asserting on the server's request count would be testing FFmpeg's
    # internals rather than this module's behaviour.
    assert source.metrics.reconnect_attempts >= 1
    assert server.requests_served >= 1


@requires_opencv
def test_it_reaches_a_terminal_state_when_the_server_never_recovers() -> None:
    """A camera that is genuinely gone must stop being reported as connecting."""
    with serving_clip(CLIP, fail_after_requests=1) as server:
        source = LiveStreamSource(
            server.url,
            "cam_01",
            read_timeout_sec=1.0,
            backoff=BackoffPolicy(initial_sec=0.01, max_sec=0.05, max_attempts=2),
        )
        with source:
            list(itertools.islice(source.frames(), 1))
            deadline = time.monotonic() + 8.0
            while source.state is not ConnectionState.FAILED and time.monotonic() < deadline:
                time.sleep(0.05)
            terminal = source.state

    assert terminal is ConnectionState.FAILED
    assert server.requests_served >= 1


@requires_opencv
def test_shutdown_terminates_within_the_timeout() -> None:
    """A shutdown that hangs on a blocking decode is a process that needs killing."""
    with serving_clip(CLIP) as server:
        source = LiveStreamSource(
            server.url,
            "cam_01",
            read_timeout_sec=2.0,
            backoff=BackoffPolicy(initial_sec=0.01, max_attempts=1),
        )
        source.open()
        time.sleep(0.1)

        started = time.monotonic()
        source.close()
        elapsed = time.monotonic() - started

    assert elapsed < 5.0
    assert source.state is ConnectionState.CLOSED


@requires_opencv
def test_a_slow_server__does_not_stall_the_consumer_indefinitely() -> None:
    """Latency beats completeness: the reader keeps moving or drops frames."""
    with serving_clip(CLIP, chunk_delay_sec=0.2) as server:
        source = LiveStreamSource(
            server.url,
            "cam_01",
            buffer_size=2,
            read_timeout_sec=3.0,
            backoff=BackoffPolicy(initial_sec=0.01, max_attempts=1),
        )
        started = time.monotonic()
        with source:
            list(itertools.islice(source.frames(), 1))
        elapsed = time.monotonic() - started

    assert elapsed < 10.0
