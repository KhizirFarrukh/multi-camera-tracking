"""Record detections from the committed sample clips, so CI needs no weights.

Run once, commit the result, and every downstream stage can be tested against
real detection patterns with no model, no GPU, and no network. Re-run it
deliberately when the detector changes: a diff in the fixture file is a claim
that the detections really changed.

Usage::

    python scripts/record_detection_fixtures.py                 # reference backend
    python scripts/record_detection_fixtures.py --backend yolo  # needs weights

**Which detector recorded the committed file matters, and it is written into
the file.** The stage 11 prompt asks for fixtures recorded from a real model
run. No model weights are installed in this environment, and downloading a
50 MB checkpoint into the repository is not something this stage should decide
on its own -- so the committed recording comes from the reference blob detector
in ``tests/fixtures/vision.py``, which finds the bright rectangle the sample
clips draw.

That is a deviation, and it is recorded in three places: here, in
``docs/DETECTION.md``, and in the ``note`` field of the file itself. Re-record
with ``--backend yolo`` once weights are available; stage 13 or 20 should.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:  # pragma: no cover - script bootstrap
        sys.path.insert(0, str(candidate))

from tests.fixtures.vision import ReferenceBlobDetector  # noqa: E402

from multicam_tracker.clock import SystemClock  # noqa: E402
from multicam_tracker.ingest import FileVideoSource  # noqa: E402
from multicam_tracker.logging_config import configure_logging  # noqa: E402
from multicam_tracker.vision import (  # noqa: E402
    Detector,
    YoloDetector,
    build_fixture_payload,
    write_fixture,
)

MEDIA_DIR = REPO_ROOT / "tests" / "fixtures" / "media"
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "detections" / "sample_clips.json"

CLIPS = ("sample_clean.mp4", "sample_vfr.mp4")
"""What to record. Deliberately not the damaged clips: their decode behaviour is
stage 10's subject, and a recording of it would pin one FFmpeg build's idea of
how much of a corrupt file is readable."""

REFERENCE_NOTE = (
    "Recorded with tests.fixtures.vision.ReferenceBlobDetector, NOT with YOLO. "
    "No model weights were available in the environment where this was recorded. "
    "The blob detector finds the bright rectangle the sample clips draw; it would "
    "find nothing in real footage. Re-record with --backend yolo once weights exist "
    "(see docs/DETECTION.md)."
)

YOLO_NOTE = "Recorded from a real Ultralytics YOLO run over the committed sample clips."


def build_detector(backend: str) -> tuple[Detector, str]:
    """Return the detector to record with and the note describing it.

    Args:
        backend: ``reference`` or ``yolo``.

    Returns:
        ``(detector, note)``.

    Raises:
        SystemExit: If the backend name is not recognised.
    """
    if backend == "reference":
        return ReferenceBlobDetector(), REFERENCE_NOTE
    if backend == "yolo":
        return YoloDetector(REPO_ROOT / "weights" / "vehicle_detector.pt"), YOLO_NOTE

    raise SystemExit(f"unknown backend: {backend}")


def main() -> int:
    """Record detections for every sample clip and write the fixture.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="reference", choices=["reference", "yolo"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)
    detector, note = build_detector(arguments.backend)

    frames_by_source = {}
    for name in CLIPS:
        path = MEDIA_DIR / name
        if not path.is_file():
            print(f"  skipped {name}: not found; run scripts/make_sample_clips.py")
            continue
        with FileVideoSource(path=path, camera_id="cam_01") as source:
            frames = list(source.frames())
        frames_by_source[name] = frames
        print(f"  read {len(frames):3d} frames from {name}")

    if not frames_by_source:
        print("nothing to record")
        return 1

    payload = build_fixture_payload(
        detector,
        frames_by_source,
        recorded_at=SystemClock().now_utc(),
        note=note,
    )
    write_fixture(arguments.output, payload)

    total = sum(
        len(entry["frames"])  # type: ignore[index,arg-type]
        for entry in payload["sources"].values()  # type: ignore[union-attr]
    )
    print(f"\nwrote {arguments.output.relative_to(REPO_ROOT)}")
    print(f"  detector    {payload['model_id']} ({payload['model_version']})")
    print(f"  sources     {len(frames_by_source)}")
    print(f"  frames with detections  {total}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
