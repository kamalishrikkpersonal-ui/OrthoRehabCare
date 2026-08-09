"""Schema package for OrthoRehab AI Step 5.

Contains JSON-serializable dataclasses for comparison results, deviations,
repetition-level analysis, feedback events and session summaries. These schemas
follow the project's existing dataclass + ``to_dict()`` / ``from_dict()``
convention (see ``src/pose/reference_profile.py``).
"""

from .comparison import (
    AssessmentDeviation,
    CaptureQualityError,
    ComparisonResult,
    DeviatedRange,
    RepComparison,
    SessionSummary,
    SpeedAssessment,
)
from .feedback import FeedbackEvent, FeedbackMessage

__all__ = [
    "AssessmentDeviation",
    "CaptureQualityError",
    "DeviatedRange",
    "RepComparison",
    "SpeedAssessment",
    "ComparisonResult",
    "SessionSummary",
    "FeedbackEvent",
    "FeedbackMessage",
]
