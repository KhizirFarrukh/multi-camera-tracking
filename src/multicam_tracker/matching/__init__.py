"""Plate normalization, fuzzy matching, and evaluation.

The primary identification path. Precision matters more than recall here: a
false positive puts an innocent vehicle on a stolen-car trajectory, and unlike a
miss it leaves no visible gap for an operator to notice. So thresholds are
biased toward precision and the uncertain middle goes to human review.

Everything is pure string logic -- no database, no video, no models -- which is
what lets it be tested exhaustively and measured against stage 05's ground truth
before any of that exists.

Stage 07 adds embedding similarity as the fallback for the sightings the plate
path genuinely cannot find -- the ones whose plate was never readable. Re-id is
weaker evidence and is treated as such: a visual-only match is capped below every
plate match, is never auto-accepted, and is demoted to human review whenever a
second candidate looks almost as similar.
"""

from __future__ import annotations

from multicam_tracker.matching.aggregation import AggregationStrategy, aggregate_similarity
from multicam_tracker.matching.conflicts import PlateConflict, detect_plate_conflicts
from multicam_tracker.matching.distance import (
    damerau_levenshtein,
    weighted_damerau_levenshtein,
)
from multicam_tracker.matching.embedding_search import (
    EmbeddingMatchResult,
    classify_embedding_matches,
    find_embedding_matches,
)
from multicam_tracker.matching.evaluation import MatchingMetrics, evaluate_plate_matching
from multicam_tracker.matching.evidence import CombinedEvidence, combine_evidence
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
from multicam_tracker.matching.references import (
    add_reference_embedding,
    minimum_pairwise_similarity,
    prune_references,
    select_diverse,
)
from multicam_tracker.matching.reid_evaluation import ReidMetrics, evaluate_reid_matching
from multicam_tracker.matching.scoring import score_plate_match
from multicam_tracker.matching.search import ScoredMatch, find_plate_matches, score_sighting
from multicam_tracker.matching.similarity import (
    SimilarityHit,
    batch_cosine_similarity,
    cosine_similarity,
    top_k_similar,
)
from multicam_tracker.matching.versioning import (
    distinct_model_versions,
    require_same_model_version,
    sightings_for_version,
)

__all__ = [
    "AggregationStrategy",
    "CombinedEvidence",
    "EmbeddingMatchResult",
    "MatchingMetrics",
    "PlateConflict",
    "PlateMatchMethod",
    "PlateMatchResult",
    "RegionalRules",
    "ReidMetrics",
    "ScoredMatch",
    "SimilarityHit",
    "add_reference_embedding",
    "aggregate_similarity",
    "assert_schema_fold_matches",
    "batch_cosine_similarity",
    "classify_embedding_matches",
    "classify_plate_match",
    "combine_evidence",
    "confusion_groups",
    "cosine_similarity",
    "damerau_levenshtein",
    "detect_plate_conflicts",
    "distinct_model_versions",
    "evaluate_plate_matching",
    "evaluate_reid_matching",
    "find_embedding_matches",
    "find_plate_matches",
    "fold_ambiguous",
    "folding_map",
    "load_confusion_config",
    "minimum_pairwise_similarity",
    "normalize_plate",
    "prune_references",
    "regional_rules",
    "require_same_model_version",
    "same_confusion_group",
    "score_plate_match",
    "score_sighting",
    "select_diverse",
    "sightings_for_version",
    "top_k_similar",
    "weighted_damerau_levenshtein",
]
