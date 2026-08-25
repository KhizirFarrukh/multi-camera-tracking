"""Telling a fast clock from fast traffic.

Both look the same in a single observation: a vehicle arrives sooner than the
travel-time window expected. What separates them is *consistency*. Fast driving
is a property of one journey and shows up on one link at one time; a fast clock
is a property of the camera and shows up on **every** link, in **every**
direction, all day.

So drift detection works on the residual series -- observed minus expected,
per pass -- and asks two questions:

* is the mean systematically away from zero? That is a constant offset.
* is the residual growing with time? That is a clock gaining or losing at a
  steady rate, and the fitted slope says how fast.

The distinction matters operationally. A constant offset is corrected once. A
drifting clock will be wrong again next week, so the correction is a stopgap and
the camera needs an NTP client, not another estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import TimeWindow
from multicam_tracker.timesync.estimation import ReferencePass

__all__ = ["DriftAlert", "DriftAnalysis", "detect_drift"]

logger = get_logger(__name__)

_SECONDS_PER_HOUR = 3600.0
_MIN_SAMPLES_FOR_RATE = 3
"""Fewest residuals before a drift *rate* is fitted.

Two points always fit a line perfectly, which would report a confident rate from
no evidence at all."""

_MIN_SPAN_HOURS = 1.0
"""Shortest observation span a rate may be fitted over.

A slope measured across twenty minutes and quoted in milliseconds per hour
multiplies whatever noise was in those twenty minutes by three. Run against the
adversarial scenario -- half an hour of traffic -- an unguarded fit reported
every camera in the network as drifting, which is precisely the alert an
operator learns to ignore."""

_MIN_SIGNAL_TO_NOISE = 1.0
"""How far a bias must exceed the scatter around it before it counts as a clock.

Traffic varies; a mean offset smaller than the spread of the observations it was
averaged from is that variance, not a clock. Requiring the signal to exceed its
own noise is what separates "this camera reads late" from "the traffic was
slow today"."""


@dataclass(frozen=True)
class DriftAlert:
    """A camera whose clock cannot be trusted as it stands."""

    camera_id: str
    kind: str
    """``constant_offset`` or ``linear_drift``."""

    magnitude_ms: float
    """Current offset magnitude, signed: positive when the camera reports late."""

    rate_ms_per_hour: float
    """Fitted drift rate. Zero for a pure constant offset."""

    sample_count: int
    detail: str

    def describe(self) -> str:
        """Return the operator-facing sentence."""
        return self.detail


@dataclass
class DriftAnalysis:
    """What the residual series says about one camera's clock."""

    camera_id: str
    sample_count: int = 0
    mean_offset_ms: float = 0.0
    rate_ms_per_hour: float = 0.0
    residual_spread_ms: float = 0.0
    """Scatter around the fitted line. Large spread with a large mean is
    ordinary traffic variance, not a clock problem."""

    alert: DriftAlert | None = None
    directions_seen: set[str] = field(default_factory=set)
    """Which neighbours contributed. A bias seen from only one neighbour is more
    likely a bad travel-time window on that link than a bad clock."""

    @property
    def has_alert(self) -> bool:
        """Return whether the analysis raised an alert."""
        return self.alert is not None


def _fit_line(times_hours: list[float], values_ms: list[float]) -> tuple[float, float]:
    """Fit ``value = intercept + slope * time`` by least squares.

    Args:
        times_hours: Sample times, in hours from the first sample.
        values_ms: Residuals in milliseconds.

    Returns:
        ``(intercept, slope)``. The slope is zero when the times are all equal,
        where no rate is identifiable.
    """
    count = len(times_hours)
    mean_time = sum(times_hours) / count
    mean_value = sum(values_ms) / count

    covariance = sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times_hours, values_ms, strict=True)
    )
    variance = sum((time - mean_time) ** 2 for time in times_hours)
    if variance <= 0.0:
        return (mean_value, 0.0)

    slope = covariance / variance
    return (mean_value - slope * mean_time, slope)


def detect_drift(
    camera_id: str,
    reference_passes: list[ReferencePass],
    *,
    window: TimeWindow | None = None,
    alert_ms: float | None = None,
    rate_alert_ms_per_hour: float | None = None,
) -> DriftAnalysis:
    """Analyse one camera's timing residuals for a clock problem.

    Args:
        camera_id: The camera under analysis.
        reference_passes: Passes touching any camera; those not involving this
            one are ignored.
        window: Restrict to passes whose arrival falls inside this window.
        alert_ms: Offset magnitude that raises an alert. Defaults to config.
        rate_alert_ms_per_hour: Drift rate that raises an alert. Defaults to
            config.

    Returns:
        The analysis, with an alert attached when either threshold is exceeded.
    """
    from multicam_tracker.config import get_settings

    settings = get_settings().timesync
    magnitude_threshold = alert_ms if alert_ms is not None else settings.drift_alert_ms
    rate_threshold = (
        rate_alert_ms_per_hour
        if rate_alert_ms_per_hour is not None
        else settings.drift_rate_alert_ms_per_hour
    )

    samples: list[tuple[datetime, float]] = []
    analysis = DriftAnalysis(camera_id=camera_id)

    for entry in reference_passes:
        if window is not None and not window.contains(entry.arrival_utc):
            continue
        if entry.to_camera_id == camera_id:
            # Arriving later than the route allows means this camera reads late.
            samples.append((entry.arrival_utc, entry.discrepancy_sec * 1000.0))
            analysis.directions_seen.add(entry.from_camera_id)
        elif entry.from_camera_id == camera_id:
            # The same discrepancy seen from the departure side has the opposite
            # sign: a camera whose clock is late makes the next leg look short.
            samples.append((entry.departure_utc, -entry.discrepancy_sec * 1000.0))
            analysis.directions_seen.add(entry.to_camera_id)

    if not samples:
        return analysis

    samples.sort(key=lambda pair: pair[0])
    origin = samples[0][0]
    times_hours = [(moment - origin).total_seconds() / _SECONDS_PER_HOUR for moment, _ in samples]
    values_ms = [value for _, value in samples]

    analysis.sample_count = len(samples)
    analysis.mean_offset_ms = sum(values_ms) / len(values_ms)

    if len(samples) >= _MIN_SAMPLES_FOR_RATE:
        intercept, slope = _fit_line(times_hours, values_ms)
        analysis.rate_ms_per_hour = slope if times_hours[-1] >= _MIN_SPAN_HOURS else 0.0
        fitted = [intercept + slope * time for time in times_hours]
        analysis.residual_spread_ms = (
            sum(
                (value - prediction) ** 2
                for value, prediction in zip(values_ms, fitted, strict=True)
            )
            / len(values_ms)
        ) ** 0.5
        # The offset that matters operationally is the current one, not the
        # average across a window during which the clock was moving.
        current_offset = intercept + slope * times_hours[-1]
    else:
        analysis.residual_spread_ms = 0.0
        current_offset = analysis.mean_offset_ms

    span_hours = times_hours[-1] - times_hours[0]
    # A bias no larger than the scatter it sits in is traffic variance. The
    # detector has to be able to say "I cannot tell", or its alerts stop being
    # read.
    signal_is_real = abs(current_offset) > analysis.residual_spread_ms * _MIN_SIGNAL_TO_NOISE

    if (
        span_hours >= _MIN_SPAN_HOURS
        and signal_is_real
        and abs(analysis.rate_ms_per_hour) >= rate_threshold
    ):
        analysis.alert = DriftAlert(
            camera_id=camera_id,
            kind="linear_drift",
            magnitude_ms=current_offset,
            rate_ms_per_hour=analysis.rate_ms_per_hour,
            sample_count=analysis.sample_count,
            detail=(
                f"{camera_id} is gaining {analysis.rate_ms_per_hour:+.0f} ms/hour against "
                f"the topology's expectations across {len(analysis.directions_seen)} "
                f"neighbours; correcting the current {current_offset:+.0f} ms offset will "
                f"not hold, and the camera needs a working time client"
            ),
        )
    elif signal_is_real and abs(current_offset) >= magnitude_threshold:
        analysis.alert = DriftAlert(
            camera_id=camera_id,
            kind="constant_offset",
            magnitude_ms=current_offset,
            rate_ms_per_hour=0.0,
            sample_count=analysis.sample_count,
            detail=(
                f"{camera_id} reports {current_offset:+.0f} ms away from what the topology "
                f"expects, consistently across {len(analysis.directions_seen)} neighbours "
                f"and {analysis.sample_count} passes; that is a clock offset, not traffic"
            ),
        )

    if analysis.alert is not None:
        logger.warning(
            "camera_clock_drift_detected",
            camera_id=camera_id,
            kind=analysis.alert.kind,
            magnitude_ms=analysis.alert.magnitude_ms,
            rate_ms_per_hour=analysis.alert.rate_ms_per_hour,
            samples=analysis.sample_count,
            neighbours=len(analysis.directions_seen),
        )

    return analysis
