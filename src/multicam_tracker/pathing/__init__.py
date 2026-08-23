"""Path reconstruction: turning a cloud of matches into a route.

Stages 06 and 07 answer "which sightings might be this vehicle?". This package
answers the question an operator actually asked -- *where did it go?* -- and the
difficulty is that the answer must survive being wrong about one of the inputs.

The design decision the whole package rests on is **global scoring rather than
greedy chaining**. Walking forward in time taking the best next sighting is
fooled by a single false positive in the middle of the timeline: it commits, and
every later decision is made from the wrong place. Scoring whole routes instead
means a false positive is accepted only if the route through it beats the route
around it, which a geographically impossible detour never does.

Everything else follows from the same principle: alternatives are enumerated
because sometimes two routes really do explain the evidence equally well, gaps
are reported because unmonitored ground is not the same as an absence of
movement, and every inclusion and exclusion is recorded because a trajectory a
human cannot interrogate is not usable evidence.

The package is pure. It reads no database and no files; a caller supplies the
candidates, the sightings, and the graph.
"""

from __future__ import annotations

from multicam_tracker.pathing.confidence import (
    ConfidenceStrategy,
    aggregate_confidence,
    hop_confidence,
)
from multicam_tracker.pathing.evaluation import (
    PathingMetrics,
    candidates_from_matches,
    evaluate_pathing,
)
from multicam_tracker.pathing.explanation import (
    ExclusionReason,
    TrajectoryExplanation,
    explain_path,
)
from multicam_tracker.pathing.gaps import GapKind, GapReport, detect_gaps, find_camera_outages
from multicam_tracker.pathing.graph import (
    EdgeKind,
    TrajectoryEdge,
    TrajectoryGraph,
    build_graph,
    edge_weight,
)
from multicam_tracker.pathing.incremental import (
    ExtensionResult,
    IncrementalReconstructor,
    LateArrival,
)
from multicam_tracker.pathing.kbest import AmbiguityReport, k_best_paths
from multicam_tracker.pathing.movement import (
    HopMovement,
    MovementSummary,
    bearing_degrees,
    compass_point,
    summarize_movement,
)
from multicam_tracker.pathing.optimal_path import (
    PathResult,
    best_path,
    path_from_table,
    path_score,
    score_table,
)
from multicam_tracker.pathing.preparation import PreparedCandidate, prepare_candidates
from multicam_tracker.pathing.reconstruction import (
    ReconstructionResult,
    assemble_result,
    reconstruct_trajectory,
)

__all__ = [
    "AmbiguityReport",
    "ConfidenceStrategy",
    "EdgeKind",
    "ExclusionReason",
    "ExtensionResult",
    "GapKind",
    "GapReport",
    "HopMovement",
    "IncrementalReconstructor",
    "LateArrival",
    "MovementSummary",
    "PathResult",
    "PathingMetrics",
    "PreparedCandidate",
    "ReconstructionResult",
    "TrajectoryEdge",
    "TrajectoryExplanation",
    "TrajectoryGraph",
    "aggregate_confidence",
    "assemble_result",
    "bearing_degrees",
    "best_path",
    "build_graph",
    "candidates_from_matches",
    "compass_point",
    "detect_gaps",
    "edge_weight",
    "evaluate_pathing",
    "explain_path",
    "find_camera_outages",
    "hop_confidence",
    "k_best_paths",
    "path_from_table",
    "path_score",
    "prepare_candidates",
    "reconstruct_trajectory",
    "score_table",
    "summarize_movement",
]
