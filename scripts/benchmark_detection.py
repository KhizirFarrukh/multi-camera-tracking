"""Measure detection throughput, so an operator can size the hardware.

Prints a per-stage breakdown at several batch sizes. The breakdown is the point:
"14 fps" alone does not distinguish a pipeline bottlenecked on decoding from one
bottlenecked in the model, and those need different machines.

Usage::

    python scripts/benchmark_detection.py
    python scripts/benchmark_detection.py --backend yolo --device cuda --frames 500

**Read the detector line before the numbers.** With the reference backend this
measures the pipeline around the detector -- frame generation, cropping,
association, ranking -- and *not* model inference, which on a real model
dominates everything else by an order of magnitude. Those numbers are a floor on
the overhead, not an estimate of production throughput. Only ``--backend yolo``
on the target hardware answers the sizing question.
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

from multicam_tracker.ingest import Frame, SyntheticVideoSource  # noqa: E402
from multicam_tracker.logging_config import configure_logging  # noqa: E402
from multicam_tracker.vision import (  # noqa: E402
    DetectionFilters,
    DetectionMetrics,
    Detector,
    FilterStats,
    SingleCameraTracker,
    TrackerConfig,
    YoloDetector,
)

BATCH_SIZES = (1, 4, 8, 16)


def build_detector(backend: str, device: str) -> Detector:
    """Return the detector to benchmark.

    Args:
        backend: ``reference`` or ``yolo``.
        device: ``cpu`` or ``cuda``, for the YOLO backend.

    Returns:
        The detector.

    Raises:
        SystemExit: If the backend name is not recognised.
    """
    if backend == "reference":
        return ReferenceBlobDetector()
    if backend == "yolo":
        return YoloDetector(REPO_ROOT / "weights" / "vehicle_detector.pt", device=device)
    raise SystemExit(f"unknown backend: {backend}")


def generate_frames(count: int, width: int, height: int) -> list[Frame]:
    """Return generated frames to run the benchmark over.

    Generated rather than decoded so the measurement is not dominated by one
    machine's disk, and so the benchmark runs in a container with no media.

    Args:
        count: How many frames.
        width: Frame width.
        height: Frame height.

    Returns:
        The frames.
    """
    source = SyntheticVideoSource(
        camera_id="cam_bench", frame_count=count, width=width, height=height
    )
    with source:
        return list(source.frames())


def run_pass(
    detector: Detector, frames: list[Frame], batch_size: int
) -> tuple[DetectionMetrics, SingleCameraTracker]:
    """Run one full detection and tracking pass, timed per stage.

    Args:
        detector: The detector under test.
        frames: The frames to process.
        batch_size: How many frames to hand the detector at once.

    Returns:
        ``(metrics, tracker)``.
    """
    metrics = DetectionMetrics(source_id="cam_bench")
    tracker = SingleCameraTracker("cam_bench", "cam_bench_synthetic", TrackerConfig())
    filters = DetectionFilters(min_area_px=1)
    stats = FilterStats()

    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]

        with metrics.timer.measure("detect"):
            results = detector.detect_batch(batch)

        for frame, detections in zip(batch, results, strict=True):
            metrics.record_frame(ran_detector=True)
            for detection in detections:
                metrics.record_detection(detection.object_class)

            with metrics.timer.measure("filter"):
                kept = filters.apply(
                    detections,
                    frame_width=frame.width,
                    frame_height=frame.height,
                    stats=stats,
                )
            with metrics.timer.measure("track"):
                tracker.update(frame, kept)

    tracker.flush()
    return metrics, tracker


def main() -> int:
    """Run the benchmark across batch sizes and print the table.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="reference", choices=["reference", "yolo"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    arguments = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)
    detector = build_detector(arguments.backend, arguments.device)
    frames = generate_frames(arguments.frames, arguments.width, arguments.height)

    print(f"detector   {arguments.backend} ({detector.model_id})")
    if arguments.backend == "reference":
        print("           NOT a model: this measures pipeline overhead, not inference")
    print(f"device     {arguments.device}")
    print(f"frames     {len(frames)} at {arguments.width}x{arguments.height}\n")

    header = f"{'batch':>6}  {'fps':>8}  {'detect ms':>10}  {'filter ms':>10}  {'track ms':>9}"
    print(header)
    print("-" * len(header))

    for batch_size in BATCH_SIZES:
        metrics, _ = run_pass(detector, frames, batch_size)
        print(
            f"{batch_size:>6}  "
            f"{metrics.frames_per_second:>8.1f}  "
            f"{metrics.timer.total_sec('detect') * 1000 / len(frames):>10.3f}  "
            f"{metrics.timer.total_sec('filter') * 1000 / len(frames):>10.3f}  "
            f"{metrics.timer.total_sec('track') * 1000 / len(frames):>9.3f}"
        )

    print("\nMilliseconds are per frame. Throughput counts every timed stage.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
