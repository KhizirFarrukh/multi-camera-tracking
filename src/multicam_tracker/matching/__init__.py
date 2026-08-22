"""Plate normalization, fuzzy matching, and evaluation.

The primary identification path. Precision matters more than recall here: a
false positive puts an innocent vehicle on a stolen-car trajectory, and unlike a
miss it leaves no visible gap for an operator to notice. So thresholds are
biased toward precision and the uncertain middle goes to human review.

Everything is pure string logic -- no database, no video, no models -- which is
what lets it be tested exhaustively and measured against stage 05's ground truth
before any of that exists.

Stage 07 adds embedding similarity as the fallback for the sightings this layer
genuinely cannot find, which are the ones whose plate was never readable.
"""

from __future__ import annotations

from multicam_tracker.matching.conflicts import PlateConflict, detect_plate_conflicts
from multicam_tracker.matching.distance import (
    damerau_levenshtein,
    weighted_damerau_levenshtein,
)
from multicam_tracker.matching.evaluation import MatchingMetrics, evaluate_plate_matching
from multicam_tracker.matching.folding import (
    assert_schema_fold_matches,
    confusion_groups,
    fold_ambiguous,
    folding_map,
    same_confusion_group,
)
from multicam_tracker.matching.normalize import (
    RegionalRules,
    load_confusion_config,
    normalize_plate,
    regional_rules,
)
from multicam_tracker.matching.plate_match import (
    PlateMatchMethod,
    PlateMatchResult,
    classify_plate_match,
)
from multicam_tracker.matching.scoring import score_plate_match
from multicam_tracker.matching.search import ScoredMatch, find_plate_matches, score_sighting

__all__ = [
    "MatchingMetrics",
    "PlateConflict",
    "PlateMatchMethod",
    "PlateMatchResult",
    "RegionalRules",
    "ScoredMatch",
    "assert_schema_fold_matches",
    "classify_plate_match",
    "confusion_groups",
    "damerau_levenshtein",
    "detect_plate_conflicts",
    "evaluate_plate_matching",
    "find_plate_matches",
    "fold_ambiguous",
    "folding_map",
    "load_confusion_config",
    "normalize_plate",
    "regional_rules",
    "same_confusion_group",
    "score_plate_match",
    "score_sighting",
    "weighted_damerau_levenshtein",
]
