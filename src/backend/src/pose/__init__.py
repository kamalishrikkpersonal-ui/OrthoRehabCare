"""Pose processing package for OrthoRehab AI.

Step 1: reference video -> ordered upper-body landmark sequence
        (MediaPipe detects 33 landmarks internally; OrthoRehab filters to the
         hip-and-above subset).
Step 2: landmark sequence -> joint-angle sequence.
Step 3: angle sequence -> exercise phases + repetition count.
Step 4: phases/reps + angles -> reusable reference exercise profile.
"""

from .angle_engine import (
    DEFAULT_ANGLE_DEFINITIONS,
    DEFAULT_VISIBILITY_THRESHOLD,
    AngleEngine,
    AngleFrame,
    AngleSequence,
    angle_sequence_from_landmark_sequence,
    calculate_angle,
)
from .exercise_config import (
    MOVEMENT_TYPES,
    ExercisePhaseConfig,
    config_from_angle_sequence,
)
from .landmark_extractor import (
    UPPER_BODY_LANDMARK_NAMES,
    extract_landmark_sequence,
    filter_upper_body_landmarks,
)
from .phase_detector import (
    PHASE_BOTTOM,
    PHASE_EXTENDING,
    PHASE_FLEXING,
    PHASE_START,
    PhaseDetector,
    PhaseEvent,
    RepResult,
    RepSummary,
    moving_average,
    smooth_angles,
)
from .phase_rep_counter import (
    ExerciseAnalysis,
    count_reps,
    select_primary_angle,
)
from .reference_profile import (
    ReferenceAngleStatistics,
    ReferenceConfiguration,
    ReferenceExerciseProfile,
    ReferencePhaseEvent,
    ReferencePhaseProfile,
    ReferenceRepetition,
    ReferenceSourceVideo,
    ReferenceTrajectory,
    ReferenceTrajectoryPoint,
    now_iso,
)
from .reference_profile_builder import (
    build_reference_profile,
)
from .video_processor import iter_video_frames

__all__ = [
    # Step 1
    "extract_landmark_sequence",
    "filter_upper_body_landmarks",
    "UPPER_BODY_LANDMARK_NAMES",
    "iter_video_frames",
    # Step 2
    "calculate_angle",
    "AngleEngine",
    "AngleFrame",
    "AngleSequence",
    "angle_sequence_from_landmark_sequence",
    "DEFAULT_ANGLE_DEFINITIONS",
    "DEFAULT_VISIBILITY_THRESHOLD",
    # Step 3 config
    "MOVEMENT_TYPES",
    "ExercisePhaseConfig",
    "config_from_angle_sequence",
    # Step 3 phase detection
    "PHASE_START",
    "PHASE_FLEXING",
    "PHASE_BOTTOM",
    "PHASE_EXTENDING",
    "PhaseDetector",
    "PhaseEvent",
    "RepResult",
    "RepSummary",
    "moving_average",
    "smooth_angles",
    # Step 3 rep counting
    "ExerciseAnalysis",
    "count_reps",
    "select_primary_angle",
    # Step 4 reference profile
    "ReferenceExerciseProfile",
    "ReferenceAngleStatistics",
    "ReferencePhaseEvent",
    "ReferencePhaseProfile",
    "ReferenceRepetition",
    "ReferenceTrajectoryPoint",
    "ReferenceTrajectory",
    "ReferenceSourceVideo",
    "ReferenceConfiguration",
    "build_reference_profile",
    "now_iso",
]
