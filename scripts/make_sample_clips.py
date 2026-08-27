"""Generate the tiny video fixtures the ingestion tests decode.

Committed rather than generated at test time, because a fixture that is rebuilt
on every run is not a fixture -- it drifts with whatever FFmpeg build the
machine happens to have, and a test that fails only on someone else's laptop is
worse than no test.

They are regenerated deliberately, with this script, and they are tiny: the
whole set is a few kilobytes at 64x48.

Four files, each for a specific failure or feature:

``sample_clean.mp4``
    The ordinary case. Constant rate, complete container.

``sample_truncated.ts``
    A recording cut off mid-write. MPEG-TS on purpose: an MP4 keeps its index
    at the end, so a truncated one cannot be opened at all, while a transport
    stream is designed to be decodable from any point -- which is what a camera
    writing when the power failed actually leaves behind.

``sample_vfr.mp4``
    Variable frame rate, authored through PyAV with explicit presentation
    timestamps. The nominal rate is a lie here, and a decoder that trusts it
    places every frame after the first gap at the wrong instant.

``sample_undecodable.mp4``
    Bytes that are not a valid MP4. Exercises the error path for a file that
    cannot be decoded.

    A file with a genuinely *unsupported codec* cannot be authored here -- with
    FFmpeg present essentially every codec is supported, and an exotic encoder
    is not a reasonable test dependency. This is the adjacent real failure, and
    it exercises the same path: identify, fail, and say what was found.

    Naming matters more than it looks. An earlier version of this fixture used
    a ``.bin`` extension, and FFmpeg cheerfully identified it as ANSI art
    (``bintext``, 640x64 at 25 fps) and handed back a junk frame. That is why
    the source probes the header rather than trusting that a decoder agreed to
    open something.

``sample_corrupt.ts``
    A stream with a hole punched through the middle of its payload. Some frames
    read, some are lost, and the run continues -- a recording damaged on disk
    rather than one cut short.

Usage::

    python scripts/make_sample_clips.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

MEDIA_DIR = REPO_ROOT / "tests" / "fixtures" / "media"
WIDTH = 64
HEIGHT = 48
FPS = 10
FRAME_COUNT = 20


def _pattern(index: int) -> np.ndarray:
    """Return the image for one frame index.

    A moving bar on a shifting background: enough structure that a motion gate
    and a decoder round-trip can both be checked, and deterministic so the
    committed files are byte-stable when regenerated.

    Args:
        index: Which frame.

    Returns:
        A BGR image.
    """
    image = np.full((HEIGHT, WIDTH, 3), 40 + (index * 5) % 60, dtype=np.uint8)
    left = (index * 3) % max(1, WIDTH - 12)
    image[HEIGHT // 3 : 2 * HEIGHT // 3, left : left + 12] = 220
    return image


def write_clean(path: Path) -> int:
    """Write the constant-rate MP4.

    Args:
        path: Where to write it.

    Returns:
        How many frames were written.
    """
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (WIDTH, HEIGHT)
    )
    for index in range(FRAME_COUNT):
        writer.write(_pattern(index))
    writer.release()
    return FRAME_COUNT


def write_truncated(path: Path) -> int:
    """Write a transport stream and cut it off mid-write.

    Args:
        path: Where to write it.

    Returns:
        How many frames were written before truncation.
    """
    import cv2

    complete = path.with_suffix(".complete.ts")
    writer = cv2.VideoWriter(
        str(complete), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (WIDTH, HEIGHT)
    )
    for index in range(FRAME_COUNT * 2):
        writer.write(_pattern(index))
    writer.release()

    data = complete.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    complete.unlink()
    return FRAME_COUNT * 2


def write_vfr(path: Path) -> int:
    """Write a variable-frame-rate MP4 with explicit presentation timestamps.

    The gaps are deliberate and uneven: a decoder that multiplies a frame index
    by the nominal rate gets every frame after the first gap wrong, and the
    error grows.

    Args:
        path: Where to write it.

    Returns:
        How many frames were written.
    """
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1000)

    # Milliseconds from the start. Regular for the first few frames, then
    # increasingly uneven -- a camera dropping frames under load.
    presentation_ms = [0, 100, 200, 300, 700, 800, 1500, 1600, 1700, 3000]

    for index, offset_ms in enumerate(presentation_ms):
        frame = av.VideoFrame.from_ndarray(_pattern(index), format="bgr24")
        frame.pts = offset_ms
        frame.time_base = Fraction(1, 1000)
        container.mux(stream.encode(frame))

    container.mux(stream.encode(None))
    container.close()
    return len(presentation_ms)


def write_undecodable(path: Path) -> int:
    """Write bytes that claim to be an MP4 and are not.

    Args:
        path: Where to write it.

    Returns:
        Zero, since nothing decodable was written.
    """
    path.write_bytes(b"NOTAVIDEO\x00" * 64)
    return 0


def write_corrupt(path: Path) -> int:
    """Write a transport stream with a hole punched through its middle.

    A transport stream rather than an MP4: an MP4 keeps its index in a small
    atom, and damaging the payload in place tends to take the index with it, so
    the file will not open at all -- which is a different failure, already
    covered by ``sample_undecodable.mp4``. A TS decodes from any point, so a
    hole in the middle is exactly what it should be: some frames read, some
    lost, the run continuing.

    Args:
        path: Where to write it.

    Returns:
        How many frames were encoded before the damage.
    """
    import cv2

    complete = path.with_suffix(".complete.ts")
    writer = cv2.VideoWriter(
        str(complete), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (WIDTH, HEIGHT)
    )
    for index in range(FRAME_COUNT * 2):
        writer.write(_pattern(index))
    writer.release()

    data = bytearray(complete.read_bytes())
    start = len(data) // 2
    end = start + len(data) // 6
    data[start:end] = bytes(end - start)
    path.write_bytes(bytes(data))
    complete.unlink()
    return FRAME_COUNT * 2


def main() -> int:
    """Regenerate every fixture.

    Returns:
        ``0`` on success.
    """
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    written = {
        "sample_clean.mp4": write_clean(MEDIA_DIR / "sample_clean.mp4"),
        "sample_truncated.ts": write_truncated(MEDIA_DIR / "sample_truncated.ts"),
        "sample_vfr.mp4": write_vfr(MEDIA_DIR / "sample_vfr.mp4"),
        "sample_corrupt.ts": write_corrupt(MEDIA_DIR / "sample_corrupt.ts"),
        "sample_undecodable.mp4": write_undecodable(MEDIA_DIR / "sample_undecodable.mp4"),
    }

    for name, frames in written.items():
        size = (MEDIA_DIR / name).stat().st_size
        print(f"{name:<24} {size:>7} bytes  ({frames} frames written)")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
