"""Comparison package for OrthoRehab AI Step 5.

Provides the hybrid comparison engine combining biomechanical comparison, DTW
temporal alignment, speed analysis, deterministic deviations, and an optional
(extensible) learned-motion comparator interface.
"""

from .biomechanical_comparator import BiomechanicalComparator, BiomechanicalResult
from .comparison_service import ComparisonEngine, EngineConfig
from .deviation_engine import DeviationConfig, DeviationEngine
from .dtw_comparator import DTWComparator, dtw_distance, dtw_similarity
from .learned_motion_comparator import (
    LearnedMotionComparator,
    LearnedMotionResult,
    UnavailableLearnedComparator,
)
from .speed_analyzer import (
    SPEED_NORMAL,
    SPEED_TOO_FAST,
    SPEED_TOO_SLOW,
    SpeedAnalyzer,
    SpeedConfig,
)

__all__ = [
    "BiomechanicalComparator",
    "BiomechanicalResult",
    "ComparisonEngine",
    "EngineConfig",
    "DeviationConfig",
    "DeviationEngine",
    "DTWComparator",
    "dtw_distance",
    "dtw_similarity",
    "LearnedMotionComparator",
    "LearnedMotionResult",
    "UnavailableLearnedComparator",
    "SpeedAnalyzer",
    "SpeedConfig",
    "SPEED_NORMAL",
    "SPEED_TOO_FAST",
    "SPEED_TOO_SLOW",
]
