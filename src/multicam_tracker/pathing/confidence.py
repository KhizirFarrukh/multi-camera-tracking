"""How confident the whole route is, given how confident each hop is.

**A chain is not stronger than its weakest link.** That sentence is the entire
specification, and it rules out the obvious aggregations. An arithmetic mean of
``[1.0, 1.0, 0.1]`` is 0.70, which reads as "probably right" for a route
containing one hop that is barely evidence at all. A geometric mean does better
at 0.46 — but 0.46 still reads as "maybe", when the honest answer is that one
link is 0.10 and everything downstream of it is guesswork.

So the aggregate is a geometric mean **capped at the weakest hop plus a small
allowance**::

    overall = min(geometric_mean(hops), min(hops) + weakest_link_tolerance)

The allowance is not a fudge factor. Corroborating strong hops do carry
information: a 0.1 hop sitting between two independently confirmed sightings is
better supported than a 0.1 hop standing alone. The allowance is what that is
worth, it is configured rather than hardcoded, and it is small.

The guarantee this module offers, asserted directly in its tests: the reported
confidence never exceeds the weakest hop by more than the configured tolerance,
and never leaves ``[0, 1]``.
"""

from __future__ import annotations

import math
from enum import StrEnum

from multicam_tracker.pathing.graph import TrajectoryEdge
from multicam_tracker.pathing.preparation import PreparedCandidate

__all__ = [
    "ConfidenceStrategy",
    "aggregate_confidence",
    "hop_confidence",
]


class ConfidenceStrategy(StrEnum):
    """How per-hop confidences combine into one number."""

    GEOMETRIC_MEAN = "geometric_mean"
    """The default. Penalises a single weak link far more than an average does."""

    MINIMUM = "minimum"
    """The strictest reading of the weakest-link rule, available for a caller
    that wants no allowance at all."""

    # An arithmetic mean was considered and deliberately not offered. It is the
    # aggregation that lets strong links average a weak one away -- and once the
    # weakest-link cap is applied on top, it cannot report anything the default
    # does not, so it would be a configuration option with no reachable
    # behaviour. Dead configuration is worse than none.


def hop_confidence(
    origin: PreparedCandidate,
    destination: PreparedCandidate,
    edge: TrajectoryEdge,
    *,
    implausible_penalty: float | None = None,
) -> float:
    """Score one hop, from the three inputs the contract names.

    Endpoint match confidences and topology plausibility, multiplied; then, when
    the hop crossed unmonitored ground, multiplied again by the configured
    penalty. The penalty is a multiplier rather than a subtraction so it scales
    with the evidence: halving a strong hop's confidence costs more than halving
    a weak one, which is the right ordering.

    Args:
        origin: The earlier candidate.
        destination: The later candidate.
        edge: The transition between them.
        implausible_penalty: Multiplier for a hop the topology cannot fully
            account for -- unmonitored ground, or a violated travel-time window.
            Defaults to config.

    Returns:
        Confidence in ``[0, 1]``.
    """
    from multicam_tracker.config import get_settings

    penalty = (
        implausible_penalty
        if implausible_penalty is not None
        else get_settings().thresholds.hop_implausible_penalty
    )

    score = origin.confidence * destination.confidence * edge.plausibility
    if edge.is_unaccounted:
        score *= penalty
    return max(0.0, min(1.0, score))


def aggregate_confidence(
    hop_confidences: list[float],
    *,
    single_sighting_confidence: float | None = None,
    strategy: ConfidenceStrategy = ConfidenceStrategy.GEOMETRIC_MEAN,
    weakest_link_tolerance: float | None = None,
) -> float:
    """Combine per-hop confidences into the trajectory's overall confidence.

    Args:
        hop_confidences: One value per hop, in path order.
        single_sighting_confidence: What to report when there are no hops at
            all. A one-sighting trajectory is exactly as confident as the match
            that produced it -- there is no chain to be weaker than.
        strategy: How to combine. See :class:`ConfidenceStrategy`.
        weakest_link_tolerance: How far the result may exceed the weakest hop.
            Defaults to config. Ignored by :attr:`ConfidenceStrategy.MINIMUM`,
            which is already the weakest hop.

    Returns:
        A confidence in ``[0, 1]``.

    Raises:
        ValueError: If there are no hops and no single-sighting confidence was
            supplied, which would leave nothing to report.
    """
    from multicam_tracker.config import get_settings

    if not hop_confidences:
        if single_sighting_confidence is None:
            msg = (
                "cannot aggregate confidence with no hops and no "
                "single_sighting_confidence; there is nothing to report"
            )
            raise ValueError(msg)
        return max(0.0, min(1.0, single_sighting_confidence))

    clamped = [max(0.0, min(1.0, value)) for value in hop_confidences]
    weakest = min(clamped)

    if strategy is ConfidenceStrategy.MINIMUM:
        return weakest

    # exp(mean(log)) rather than a product raised to 1/n: the product of forty
    # hops underflows, the sum of forty logarithms does not.
    if any(value == 0.0 for value in clamped):
        combined = 0.0
    else:
        combined = math.exp(sum(math.log(value) for value in clamped) / len(clamped))

    tolerance = (
        weakest_link_tolerance
        if weakest_link_tolerance is not None
        else get_settings().thresholds.path_weakest_link_tolerance
    )
    return max(0.0, min(1.0, min(combined, weakest + tolerance)))
