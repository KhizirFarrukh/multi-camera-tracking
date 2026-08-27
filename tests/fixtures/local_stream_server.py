"""A local HTTP server that serves a sample clip, and misbehaves on request.

Live-source tests need a server that drops connections, stalls, and comes back
-- which is exactly what no public endpoint will reliably do for you, and what
depending on an external network would make flaky rather than deterministic.
So the fixture serves a committed clip over loopback and can be told to fail.

HTTP rather than RTSP deliberately: an RTSP server means an RTSP stack, which is
a dependency this stage does not otherwise need, and the behaviour under test --
connect, read, drop, reconnect, back off -- is identical over either. The
protocol is a detail of the capture layer; the reconnection logic is not.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["LocalStreamServer", "serving_clip"]


@dataclass
class LocalStreamServer:
    """Serves one file over loopback, with injectable failures.

    Args:
        path: The clip to serve.
        fail_after_requests: Requests to serve before refusing. ``None`` serves
            forever.
        chunk_delay_sec: Delay inserted between chunks, to make a slow server.
    """

    path: Path
    fail_after_requests: int | None = None
    chunk_delay_sec: float = 0.0

    requests_served: int = field(default=0, init=False)
    _server: http.server.ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def url(self) -> str:
        """Return the URL the clip is served at.

        Returns:
            A loopback URL.

        Raises:
            RuntimeError: If the server is not running.
        """
        if self._server is None:
            msg = "the server is not running"
            raise RuntimeError(msg)
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/clip"

    def start(self) -> None:
        """Bind to a free loopback port and start serving."""
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            """Serves the clip, or refuses once the failure point is reached."""

            def do_GET(self) -> None:
                """Serve the clip bytes, or close the connection."""
                fixture.requests_served += 1
                if (
                    fixture.fail_after_requests is not None
                    and fixture.requests_served > fixture.fail_after_requests
                ):
                    # Closing without a response is what a camera does when its
                    # session expires: not an error page, just silence.
                    self.close_connection = True
                    return

                payload = fixture.path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()

                if fixture.chunk_delay_sec:
                    import time

                    half = len(payload) // 2
                    self.wfile.write(payload[:half])
                    time.sleep(fixture.chunk_delay_sec)
                    self.wfile.write(payload[half:])
                else:
                    self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                """Silence the default per-request logging."""

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="local-stream-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut the server down and join its thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@contextmanager
def serving_clip(
    path: Path, *, fail_after_requests: int | None = None, chunk_delay_sec: float = 0.0
) -> Iterator[LocalStreamServer]:
    """Serve a clip for the duration of a block.

    Args:
        path: The clip to serve.
        fail_after_requests: Requests to serve before refusing.
        chunk_delay_sec: Delay between chunks.

    Yields:
        The running server.
    """
    server = LocalStreamServer(
        path=path, fail_after_requests=fail_after_requests, chunk_delay_sec=chunk_delay_sec
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()
