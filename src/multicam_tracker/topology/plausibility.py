"""Travel-time plausibility: the pruning primitive the whole system rests on.

Stage 08 asks this module one question over and over: *could the vehicle have
got from A to B in this much time?* A wrong "yes" admits a false positive into a
route; a wrong "no" breaks a real one. Both failures look like a confident,
plausible answer, which is why every boundary here is tested at its exact value.

Two outputs, for two different jobs:

* :func:`is_transition_plausible` gives a verdict with a *reason*, for
  explaining a decision to an operator.
* :func:`plausibility_score` gives a graded number, for ranking candidates.
  Graded rather than binary because an implausible transit is not impossible --
  a detour, a stop, a queue -- and the contract requires such a hop to be
  penalised, not discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from multicam_tracker.topology.graph import Topology

__all__ = [
    "PlausibilityReason",
    "PlausibilityResult",
    "is_same_pass",
    "is_transition_plausible",
    "plausibility_score",
]


class PlausibilityReason(StrEnum):
    """Why a transition was or was not plausible.

    ``StrEnum`` rather than the ``(str, Enum)`` used in
    :mod:`multicam_tracker.models.enums`: that base class was specified verbatim
    for the contract's canonical entities, this one is not a contract entity,
    and ``StrEnum`` is what the project's lint rules prefer.
    """

    WITHIN_WINDOW = "within_window"
    TOO_FAST = "too_fast"
    TOO_SLOW = "too_slow"
    NO_LINK = "no_link"
    SAME_CAMERA = "same_camera"


@dataclass(frozen=True)
class PlausibilityResult:
    """The verdict on one candidate transition."""

    plausible: bool
    reason: PlausibilityReason
    margin_sec: float
    """How far outside the window the elapsed time fell, in seconds.

    Zero when inside the window, and zero for ``no_link`` and ``same_camera``,
    where there is no window to be outside of.
    """

    def __str__(self) -> str:
        """Return a short human-readable form."""
        verdict = "plausible" if self.plausible else "implausible"
        return f"{verdict} ({self.reason.value}, margin={self.margin_sec:.1f}s)"


def is_transition_plausible(
    topology: Topology,
    from_camera_id: str,
    to_camera_id: str,
    elapsed_sec: float,
) -> PlausibilityResult:
    """Judge whether a vehicle could have made this transition in this time.

    The window is **inclusive at both ends**: arriving in exactly the minimum
    plausible transit is plausible, and so is arriving in exactly the maximum.
    An exclusive bound would make a hop's verdict depend on floating-point
    equality, which is not a property anyone should have to reason about.

    A **negative** elapsed time means the two sightings are out of order. This
    returns ``too_fast`` rather than raising: path reconstruction evaluates
    thousands of candidate pairs, many of them nonsense, and one bad pair must
    not abort the search. The margin reports how far backwards it ran.

    Args:
        topology: The graph to consult.
        from_camera_id: Camera the vehicle left.
        to_camera_id: Camera it arrived at.
        elapsed_sec: Seconds between the two sightings.

    Returns:
        The verdict, its reason, and how far outside the window it fell.
    """
    if from_camera_id == to_camera_id:
        return PlausibilityResult(
            plausible=False, reason=PlausibilityReason.SAME_CAMERA, margin_sec=0.0
        )

    link = topology.get_link(from_camera_id, to_camera_id)
    if link is None:
        return PlausibilityResult(
            plausible=False, reason=PlausibilityReason.NO_LINK, margin_sec=0.0
        )

    if elapsed_sec < link.min_travel_time_sec:
        return PlausibilityResult(
            plausible=False,
            reason=PlausibilityReason.TOO_FAST,
            margin_sec=link.min_travel_time_sec - elapsed_sec,
        )

    if elapsed_sec > link.max_travel_time_sec:
        return PlausibilityResult(
            plausible=False,
            reason=PlausibilityReason.TOO_SLOW,
            margin_sec=elapsed_sec - link.max_travel_time_sec,
        )

    return PlausibilityResult(
        plausible=True, reason=PlausibilityReason.WITHIN_WINDOW, margin_sec=0.0
    )


def plausibility_score(
    topology: Topology,
    from_camera_id: str,
    to_camera_id: str,
    elapsed_sec: float,
    *,
    half_life_sec: float | None = None,
    unlinked_score: float | None = None,
) -> float:
    """Score a transition's plausibility in ``[0, 1]``.

    Inside the window the score is exactly ``1.0``. Outside it decays
    exponentially with the margin::

        score = 0.5 ** (margin_sec / half_life_sec)

    Exponential rather than linear so the score never reaches zero: a hop 20
    minutes outside its window is very unlikely but not impossible, and a hard
    zero would remove it from consideration rather than rank it last. The decay
    is symmetric -- arriving impossibly early is as suspicious as arriving
    impossibly late.

    Args:
        topology: The graph to consult.
        from_camera_id: Camera the vehicle left.
        to_camera_id: Camera it arrived at.
        elapsed_sec: Seconds between the two sightings.
        half_life_sec: Margin at which the score halves. Defaults to the
            configured value.
        unlinked_score: Score for a pair the topology does not connect, and for
            the same camera twice. Defaults to the configured floor.

    Returns:
        A score in ``[0, 1]``, for any input including negative, zero, and
        extremely large elapsed times.
    """
    from multicam_tracker.config import get_settings

    verdict = is_transition_plausible(topology, from_camera_id, to_camera_id, elapsed_sec)

    if verdict.reason in {PlausibilityReason.NO_LINK, PlausibilityReason.SAME_CAMERA}:
        if unlinked_score is not None:
            return max(0.0, min(1.0, unlinked_score))
        return get_settings().topology.unlinked_plausibility_score

    if verdict.plausible:
        return 1.0

    resolved_half_life = (
        half_life_sec
        if half_life_sec is not None
        else get_settings().topology.plausibility_decay_half_life_sec
    )

    # A very large margin drives the exponent far negative; 0.5 ** -1e9
    # underflows to 0.0 rather than raising, which is the answer we want anyway.
    decayed = float(0.5 ** (verdict.margin_sec / resolved_half_life))
    return max(0.0, min(1.0, decayed))


def is_same_pass(elapsed_sec: float, *, min_redetection_gap_sec: float | None = None) -> bool:
    """Return whether two sightings on one camera are a single pass.

    A camera that sees a vehicle in three consecutive sampled frames has
    observed one transit, not three. A vehicle that circles the block and
    returns four minutes later has made two. The gap between those cases is a
    judgement call, so it is configuration rather than a constant, and every
    later stage reads it from here rather than picking its own.

    The comparison is strict: a gap of exactly ``min_redetection_gap_sec``
    counts as a *separate* transit, matching the half-open convention used for
    time windows everywhere else in the system.

    Args:
        elapsed_sec: Seconds between the two sightings. Negative values are
            treated by magnitude, since order does not change whether two
            observations are the same pass.
        min_redetection_gap_sec: Override for the configured gap.

    Returns:
        ``True`` when the two sightings should be collapsed into one pass.
    """
    from multicam_tracker.config import get_settings

    gap = (
        min_redetection_gap_sec
        if min_redetection_gap_sec is not None
        else get_settings().topology.min_redetection_gap_sec
    )
    return abs(elapsed_sec) < gap
