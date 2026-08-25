"""Recovering each camera's clock offset from vehicles that drove past it.

Most cameras expose no clock to query. What they do expose is a stream of
sightings, and if a vehicle with a known plate drives a known route, the
observed intervals between cameras can be compared with the intervals the
topology says to expect. A camera that is consistently 45 seconds "early" from
every neighbour does not have fast traffic; it has a fast clock.

The estimator solves for all cameras at once. Each reference pass between two
cameras contributes one equation::

    offset[to] - offset[from] = observed_elapsed - expected_elapsed

which is a least-squares problem over the camera graph. One camera is **pinned**
to zero, because the equations only ever constrain *differences*: adding an hour
to every camera fits exactly as well, and without a pin the solver would return
one of the infinitely many equally good answers.

**Refusing to answer is a feature.** A camera with one reference pass is not
estimated -- it is guessed at, and a guessed offset applied to real data is
worse than none, because it looks like a correction. Below
``min_reference_passes`` the estimator says so and stops.

**One bad pass must not dominate.** A single mismatched vehicle -- a wrong plate
read, a decoy -- produces a wildly wrong equation, and plain least squares
spreads that error across every camera in the graph.

The fix is structural rather than statistical: many vehicles cross the same
link, so each link is summarised by the **median** of its crossings before
anything is solved. A median cannot be moved by one outlier however extreme, and
"the median vehicle took 153 seconds on this link" is the honest reading of a
set of crossings anyway.

An earlier version reweighted iteratively instead, seeded from a plain fit. It
failed exactly where it mattered: a three-hour outlier skewed the first fit so
far that the *good* passes looked like the outliers, and the estimate came back
thirty times too large. Robustness that depends on the first guess being roughly
right is not robustness.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime

from multicam_tracker.exceptions import ValidationError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.topology import Topology

__all__ = [
    "CameraOffsetEstimate",
    "OffsetEstimation",
    "ReferencePass",
    "estimate_offsets",
]

logger = get_logger(__name__)

_OUTLIER_SCALE = 3.0
"""How far from its link's median a crossing may fall before it is reported as
an outlier, in median absolute deviations.

Three MADs is roughly two standard deviations for normal data -- tight enough to
notice a mismatched vehicle, loose enough to ignore an ordinary slow transit.
This only affects what is *reported*: the median already ignores the outlier
when solving."""

_MIN_DEVIATION_SEC = 1.0
"""Floor on the deviation scale, so a link whose crossings agree exactly does not
flag every millisecond of jitter as an outlier."""


@dataclass(frozen=True)
class ReferencePass:
    """One vehicle observed at two cameras, with the topology's expectation.

    Args:
        from_camera_id: Camera it left.
        to_camera_id: Camera it arrived at.
        departure_utc: Reported time at the first camera.
        arrival_utc: Reported time at the second camera.
        expected_sec: Transit time the topology considers typical.
        vehicle_id: Which reference vehicle this was, for diagnostics.
    """

    from_camera_id: str
    to_camera_id: str
    departure_utc: datetime
    arrival_utc: datetime
    expected_sec: float
    vehicle_id: str = ""

    @property
    def observed_sec(self) -> float:
        """Return the elapsed time as the two cameras reported it."""
        return (self.arrival_utc - self.departure_utc).total_seconds()

    @property
    def discrepancy_sec(self) -> float:
        """Return how much longer the transit appeared than expected.

        A positive value means the arriving camera reported a later time than
        the route can explain -- either its clock is fast, or the departing
        camera's clock is slow. One pass cannot tell those apart; the graph can.
        """
        return self.observed_sec - self.expected_sec


@dataclass
class CameraOffsetEstimate:
    """The estimated clock offset for one camera."""

    camera_id: str
    offset_ms: float
    pass_count: int
    residual_ms: list[float] = field(default_factory=list)
    is_reference: bool = False
    """True for the camera pinned to zero, whose offset is a definition rather
    than a measurement."""

    @property
    def residual_rms_ms(self) -> float:
        """Return the root-mean-square residual, or ``0.0`` with no passes."""
        if not self.residual_ms:
            return 0.0
        mean_square = sum(value * value for value in self.residual_ms) / len(self.residual_ms)
        return float(mean_square**0.5)

    @property
    def confidence_interval_ms(self) -> tuple[float, float]:
        """Return a rough 95% interval around the estimate.

        Standard error of the mean times two. Rough on purpose: the point is to
        tell an operator whether an estimate is worth acting on, not to make a
        distributional claim about camera clocks that nothing here justifies.
        """
        if len(self.residual_ms) < 2:
            return (self.offset_ms, self.offset_ms)
        margin = 2.0 * self.residual_rms_ms / (len(self.residual_ms) ** 0.5)
        return (self.offset_ms - margin, self.offset_ms + margin)

    def describe(self) -> str:
        """Return an operator-facing summary.

        Returns:
            A sentence stating the estimate, its spread, and its support.
        """
        if self.is_reference:
            return f"{self.camera_id} is the reference clock; its offset is zero by definition"
        low, high = self.confidence_interval_ms
        return (
            f"{self.camera_id} runs {-self.offset_ms:+.0f} ms away from the reference "
            f"(correction {self.offset_ms:+.0f} ms, 95% interval {low:+.0f} to {high:+.0f} ms) "
            f"from {self.pass_count} reference passes"
        )


@dataclass
class OffsetEstimation:
    """What the estimator concluded across the whole camera graph."""

    reference_camera_id: str
    estimates: dict[str, CameraOffsetEstimate] = field(default_factory=dict)
    excluded_camera_ids: list[str] = field(default_factory=list)
    """Cameras with too few reference passes to estimate. Named rather than
    silently omitted: an operator needs to know which cameras are still
    unverified."""

    outlier_pass_count: int = 0

    def offset_ms_for(self, camera_id: str) -> float | None:
        """Return one camera's estimated offset.

        Args:
            camera_id: The camera.

        Returns:
            The offset in milliseconds, or ``None`` when it was not estimated.
        """
        estimate = self.estimates.get(camera_id)
        return None if estimate is None else estimate.offset_ms

    def describe(self) -> str:
        """Return a multi-line operator-facing summary.

        Returns:
            One line per camera, plus a line naming any camera left unestimated.
        """
        lines = [estimate.describe() for estimate in self.estimates.values()]
        if self.excluded_camera_ids:
            lines.append(
                f"not estimated (too few reference passes): "
                f"{', '.join(sorted(self.excluded_camera_ids))}"
            )
        return "\n".join(lines)


def _median_absolute_deviation(values: list[float]) -> float:
    """Return the median absolute deviation of a sample.

    Args:
        values: The sample.

    Returns:
        The MAD, or ``0.0`` for fewer than two values.
    """
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    return statistics.median([abs(value - median) for value in values])


def _solve(
    edges: list[tuple[str, str, float, int]],
    cameras: list[str],
    reference_camera_id: str,
) -> dict[str, float]:
    """Solve the weighted least-squares system directly.

    Each link is one row of an incidence matrix: ``+1`` at the arriving camera,
    ``-1`` at the departing one, and the link's median discrepancy on the
    right-hand side, weighted by how many crossings supported it. Pinning the
    reference camera is done by deleting its column, which is exactly the
    statement "its offset is zero".

    Solved by direct least-squares factorisation rather than by relaxation. An
    earlier version iterated, and iterating is the wrong tool here: a chain of
    cameras is a bipartite graph, and Jacobi relaxation on a bipartite Laplacian
    oscillates rather than converging. It produced a stable-looking answer that
    was simply wrong, which is the worst failure available to this module.

    Args:
        edges: ``(from_camera, to_camera, median_discrepancy_sec, crossings)``.
        cameras: Every camera to solve for.
        reference_camera_id: The camera pinned to zero.

    Returns:
        Offset in seconds per camera, with the reference at exactly zero.
    """
    import numpy as np

    free = [camera for camera in cameras if camera != reference_camera_id]
    if not free or not edges:
        return dict.fromkeys(cameras, 0.0)

    column_of = {camera: index for index, camera in enumerate(free)}

    matrix = np.zeros((len(edges), len(free)), dtype=np.float64)
    rhs = np.zeros(len(edges), dtype=np.float64)

    for row, (origin, destination, discrepancy, crossings) in enumerate(edges):
        # A link crossed twenty times says more about the clocks than one
        # crossed twice, and the square root is the usual weighting for a mean
        # of that many samples.
        scale = float(crossings) ** 0.5
        if destination in column_of:
            matrix[row, column_of[destination]] += scale
        if origin in column_of:
            matrix[row, column_of[origin]] -= scale
        rhs[row] = discrepancy * scale

    # rcond=None uses the machine-precision cutoff, which is what makes a
    # disconnected camera come back as zero rather than as an arbitrary large
    # number.
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)

    offsets = dict.fromkeys(cameras, 0.0)
    for camera, index in column_of.items():
        offsets[camera] = float(solution[index])
    return offsets


def estimate_offsets(
    reference_passes: list[ReferencePass],
    topology: Topology,
    *,
    reference_camera_id: str | None = None,
    min_reference_passes: int | None = None,
) -> OffsetEstimation:
    """Estimate every camera's clock offset from reference vehicle passes.

    Args:
        reference_passes: Confirmed sightings of vehicles on known routes.
        topology: The camera graph, used to validate the camera names.
        reference_camera_id: Camera pinned to zero. Defaults to the one with the
            most reference passes, which is the best-constrained choice.
        min_reference_passes: Fewest passes a camera needs before it is
            estimated at all. Defaults to config.

    Returns:
        The estimation, including which cameras were left out and why.

    Raises:
        ValidationError: If there are no reference passes at all, if the chosen
            reference camera has none, or if a pass names a camera the topology
            does not contain.
    """
    from multicam_tracker.config import get_settings

    minimum = (
        min_reference_passes
        if min_reference_passes is not None
        else get_settings().timesync.min_reference_passes
    )

    if not reference_passes:
        raise ValidationError(
            "Cannot estimate clock offsets with no reference passes; the system is "
            "entirely underdetermined",
            {},
        )

    counts: dict[str, int] = {}
    for reference_pass in reference_passes:
        for camera_id in (reference_pass.from_camera_id, reference_pass.to_camera_id):
            if camera_id not in topology:
                raise ValidationError(
                    "Reference pass names a camera that is not in the topology",
                    {"camera_id": camera_id},
                )
            counts[camera_id] = counts.get(camera_id, 0) + 1

    pinned = reference_camera_id or max(sorted(counts), key=lambda camera: counts[camera])
    if pinned not in counts:
        raise ValidationError(
            "The reference camera appears in no reference pass, so nothing anchors the "
            "solution to it",
            {"reference_camera_id": pinned, "cameras_seen": sorted(counts)},
        )

    cameras = sorted(counts)

    # One entry per directed link, summarised by its median crossing. Direction
    # is kept: a link is asymmetric in traffic and in travel time, and folding
    # the two directions together would average away a real difference.
    by_edge: dict[tuple[str, str], list[float]] = {}
    for entry in reference_passes:
        by_edge.setdefault((entry.from_camera_id, entry.to_camera_id), []).append(
            entry.discrepancy_sec
        )

    edges = [
        (origin, destination, statistics.median(discrepancies), len(discrepancies))
        for (origin, destination), discrepancies in sorted(by_edge.items())
    ]
    offsets = _solve(edges, cameras, pinned)

    outliers = 0
    for (origin, destination), discrepancies in by_edge.items():
        median = statistics.median(discrepancies)
        scale = max(_median_absolute_deviation(discrepancies), _MIN_DEVIATION_SEC)
        outliers += sum(
            1 for value in discrepancies if abs(value - median) > _OUTLIER_SCALE * scale
        )
        del origin, destination

    residuals_by_camera: dict[str, list[float]] = {camera: [] for camera in cameras}
    for entry in reference_passes:
        residual_ms = (
            offsets[entry.to_camera_id] - offsets[entry.from_camera_id] - entry.discrepancy_sec
        ) * 1000.0
        residuals_by_camera[entry.to_camera_id].append(residual_ms)
        residuals_by_camera[entry.from_camera_id].append(residual_ms)

    estimation = OffsetEstimation(reference_camera_id=pinned, outlier_pass_count=outliers)
    for camera in cameras:
        if camera != pinned and counts[camera] < minimum:
            estimation.excluded_camera_ids.append(camera)
            continue
        estimation.estimates[camera] = CameraOffsetEstimate(
            camera_id=camera,
            # The correction to apply is the negative of the observed error: a
            # camera reporting 45 s late needs 45 s subtracted.
            offset_ms=-offsets[camera] * 1000.0,
            pass_count=counts[camera],
            residual_ms=residuals_by_camera[camera],
            is_reference=camera == pinned,
        )

    logger.info(
        "clock_offsets_estimated",
        reference_camera_id=pinned,
        estimated=len(estimation.estimates),
        excluded=len(estimation.excluded_camera_ids),
        outlier_passes=outliers,
    )
    return estimation
