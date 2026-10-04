"""Comparison result schemas for OrthoRehab AI Step 5.

These dataclasses follow the project's existing convention (visited in
``src/pose/reference_profile.py``): pure-Python dataclasses with
``to_dict()`` / ``from_dict()`` so results can be serialized to JSON now and
persisted later.

Scope: hip-and-above only. No lower-body landmarks are ever introduced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpeedAssessment:
    """Speed analysis for a movement (independent of DTW temporal alignment)."""

    reference_duration: float = 0.0
    patient_duration: float = 0.0
    speed_ratio: float = 1.0  # reference_duration / patient_duration
    speed_status: str = "normal_speed"  # normal_speed | too_fast | too_slow
    threshold_fast: float = 0.0
    threshold_slow: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeedAssessment":
        return cls(**data)


@dataclass
class DeviatedRange:
    """A measured min/max range value for a joint (used by deviation engine)."""

    joint: str
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None
    patient_min: Optional[float] = None
    patient_max: Optional[float] = None
    reference_range: Optional[float] = None
    patient_range: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviatedRange":
        return cls(**data)


@dataclass
class AssessmentDeviation:
    """A deterministic, structured deviation finding.

    ``severity`` is one of "minor" | "moderate" | "major".
    ``type`` is one of: insufficient_flexion, excessive_flexion,
    insufficient_extension, excessive_extension, trajectory_deviation,
    too_fast, too_slow, phase_deviation.
    """

    type: str
    joint: str = ""
    reference_value: Optional[float] = None
    patient_value: Optional[float] = None
    deviation: Optional[float] = None
    severity: str = "minor"
    timestamp: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentDeviation":
        return cls(**data)


@dataclass
class RepComparison:
    """Repetition-level comparison for one validated patient repetition."""

    rep_number: int
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    duration: float = 0.0
    dtw_similarity: Optional[float] = None
    dtw_distance: Optional[float] = None
    learned_motion_similarity: Optional[float] = None
    biomechanical_accuracy: Optional[float] = None
    movement_quality: Optional[float] = None
    speed_status: str = "normal_speed"
    primary_min_angle: Optional[float] = None
    primary_max_angle: Optional[float] = None
    reference_min_angle: Optional[float] = None
    reference_max_angle: Optional[float] = None
    rom_accuracy: Optional[float] = None
    deviations: List[AssessmentDeviation] = field(default_factory=list)
    severity: str = "minor"

    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration": self.duration,
            "dtw_similarity": self.dtw_similarity,
            "dtw_distance": self.dtw_distance,
            "learned_motion_similarity": self.learned_motion_similarity,
            "biomechanical_accuracy": self.biomechanical_accuracy,
            "movement_quality": self.movement_quality,
            "speed_status": self.speed_status,
            "primary_min_angle": self.primary_min_angle,
            "primary_max_angle": self.primary_max_angle,
            "reference_min_angle": self.reference_min_angle,
            "reference_max_angle": self.reference_max_angle,
            "rom_accuracy": self.rom_accuracy,
            "deviations": [d.to_dict() for d in self.deviations],
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepComparison":
        return cls(
            rep_number=data["rep_number"],
            start_timestamp=data.get("start_timestamp", 0.0),
            end_timestamp=data.get("end_timestamp", 0.0),
            duration=data.get("duration", 0.0),
            dtw_similarity=data.get("dtw_similarity"),
            dtw_distance=data.get("dtw_distance"),
            learned_motion_similarity=data.get("learned_motion_similarity"),
            biomechanical_accuracy=data.get("biomechanical_accuracy"),
            movement_quality=data.get("movement_quality"),
            speed_status=data.get("speed_status", "normal_speed"),
            primary_min_angle=data.get("primary_min_angle"),
            primary_max_angle=data.get("primary_max_angle"),
            reference_min_angle=data.get("reference_min_angle"),
            reference_max_angle=data.get("reference_max_angle"),
            rom_accuracy=data.get("rom_accuracy"),
            deviations=[AssessmentDeviation.from_dict(d) for d in data.get("deviations", [])],
            severity=data.get("severity", "minor"),
        )


@dataclass
class ComparisonResult:
    """Overall comparison of a patient exercise against a therapist reference."""

    exercise_id: str = ""
    exercise_name: str = ""
    overall_score: Optional[float] = None
    learned_motion_similarity: Optional[float] = None
    dtw_similarity: Optional[float] = None
    biomechanical_accuracy: Optional[float] = None
    rom_accuracy: Optional[float] = None
    reference_duration: float = 0.0
    patient_duration: float = 0.0
    speed_ratio: float = 1.0
    speed_status: str = "normal_speed"
    primary_angle: str = ""
    repetitions: List[RepComparison] = field(default_factory=list)
    deviations: List[AssessmentDeviation] = field(default_factory=list)
    feedback_events: List[Dict[str, Any]] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    learned_model_available: bool = False

    def to_dict(self) -> dict:
        return {
            "exercise_id": self.exercise_id,
            "exercise_name": self.exercise_name,
            "overall_score": self.overall_score,
            "learned_motion_similarity": self.learned_motion_similarity,
            "dtw_similarity": self.dtw_similarity,
            "biomechanical_accuracy": self.biomechanical_accuracy,
            "rom_accuracy": self.rom_accuracy,
            "reference_duration": self.reference_duration,
            "patient_duration": self.patient_duration,
            "speed_ratio": self.speed_ratio,
            "speed_status": self.speed_status,
            "primary_angle": self.primary_angle,
            "repetitions": [r.to_dict() for r in self.repetitions],
            "deviations": [d.to_dict() for d in self.deviations],
            "feedback_events": list(self.feedback_events),
            "timings_ms": dict(self.timings_ms),
            "learned_model_available": self.learned_model_available,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonResult":
        return cls(
            exercise_id=data.get("exercise_id", ""),
            exercise_name=data.get("exercise_name", ""),
            overall_score=data.get("overall_score"),
            learned_motion_similarity=data.get("learned_motion_similarity"),
            dtw_similarity=data.get("dtw_similarity"),
            biomechanical_accuracy=data.get("biomechanical_accuracy"),
            rom_accuracy=data.get("rom_accuracy"),
            reference_duration=data.get("reference_duration", 0.0),
            patient_duration=data.get("patient_duration", 0.0),
            speed_ratio=data.get("speed_ratio", 1.0),
            speed_status=data.get("speed_status", "normal_speed"),
            primary_angle=data.get("primary_angle", ""),
            repetitions=[RepComparison.from_dict(r) for r in data.get("repetitions", [])],
            deviations=[AssessmentDeviation.from_dict(d) for d in data.get("deviations", [])],
            feedback_events=list(data.get("feedback_events", [])),
            timings_ms=dict(data.get("timings_ms", {})),
            learned_model_available=data.get("learned_model_available", False),
        )

    def summary_dict(self) -> dict:
        """A compact summary for the LLM (only structured findings)."""
        return {
            "exercise": self.exercise_name,
            "learned_motion_similarity": self.learned_motion_similarity,
            "dtw_similarity": self.dtw_similarity,
            "biomechanical_accuracy": self.biomechanical_accuracy,
            "speed_status": self.speed_status,
            "speed_ratio": self.speed_ratio,
            "deviations": [d.to_dict() for d in self.deviations],
        }


@dataclass
class CaptureQualityError:
    """A structured recording-quality failure (see capture-quality validator).

    Returned when the required landmarks cannot be reliably observed, so the
    movement cannot be analyzed. This is distinct from a valid-but-poor
    exercise performance (which is still scored).
    """

    code: str = "INSUFFICIENT_LANDMARK_VISIBILITY"
    message: str = ""
    user_instruction: str = ""
    missing_landmarks: List[str] = field(default_factory=list)
    valid_angle_ratio: float = 0.0
    total_frames: int = 0
    valid_frames: int = 0
    missing_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "user_instruction": self.user_instruction,
            "missing_landmarks": list(self.missing_landmarks),
            "valid_angle_ratio": round(self.valid_angle_ratio, 4),
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "missing_frames": self.missing_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureQualityError":
        return cls(
            code=data.get("code", "INSUFFICIENT_LANDMARK_VISIBILITY"),
            message=data.get("message", ""),
            user_instruction=data.get("user_instruction", ""),
            missing_landmarks=list(data.get("missing_landmarks", [])),
            valid_angle_ratio=data.get("valid_angle_ratio", 0.0),
            total_frames=data.get("total_frames", 0),
            valid_frames=data.get("valid_frames", 0),
            missing_frames=data.get("missing_frames", 0),
        )


@dataclass
class SessionSummary:
    """Session-level summary for patient and therapist reports."""

    exercise_id: str = ""
    exercise_name: str = ""
    total_repetitions: int = 0
    successful_repetitions: int = 0
    overall_score: Optional[float] = None
    learned_motion_similarity: Optional[float] = None
    dtw_similarity: Optional[float] = None
    biomechanical_accuracy: Optional[float] = None
    speed_status: str = "normal_speed"
    major_deviations: List[AssessmentDeviation] = field(default_factory=list)
    patient_feedback: str = ""
    therapist_report: str = ""
    created_at: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "exercise_id": self.exercise_id,
            "exercise_name": self.exercise_name,
            "total_repetitions": self.total_repetitions,
            "successful_repetitions": self.successful_repetitions,
            "overall_score": self.overall_score,
            "learned_motion_similarity": self.learned_motion_similarity,
            "dtw_similarity": self.dtw_similarity,
            "biomechanical_accuracy": self.biomechanical_accuracy,
            "speed_status": self.speed_status,
            "major_deviations": [d.to_dict() for d in self.major_deviations],
            "patient_feedback": self.patient_feedback,
            "therapist_report": self.therapist_report,
            "created_at": self.created_at,
            "timings_ms": dict(self.timings_ms),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSummary":
        return cls(
            exercise_id=data.get("exercise_id", ""),
            exercise_name=data.get("exercise_name", ""),
            total_repetitions=data.get("total_repetitions", 0),
            successful_repetitions=data.get("successful_repetitions", 0),
            overall_score=data.get("overall_score"),
            learned_motion_similarity=data.get("learned_motion_similarity"),
            dtw_similarity=data.get("dtw_similarity"),
            biomechanical_accuracy=data.get("biomechanical_accuracy"),
            speed_status=data.get("speed_status", "normal_speed"),
            major_deviations=[AssessmentDeviation.from_dict(d) for d in data.get("major_deviations", [])],
            patient_feedback=data.get("patient_feedback", ""),
            therapist_report=data.get("therapist_report", ""),
            created_at=data.get("created_at", ""),
            timings_ms=dict(data.get("timings_ms", {})),
        )

@dataclass
class ExerciseDetectionError:
    code: str
    message: str
    user_instruction: str
    detected_reps: int
    valid_frames: int
    min_required_reps: int = 1

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "user_instruction": self.user_instruction,
            "detected_reps": self.detected_reps,
            "valid_frames": self.valid_frames,
            "min_required_reps": self.min_required_reps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExerciseDetectionError":
        return cls(
            code=data["code"],
            message=data["message"],
            user_instruction=data["user_instruction"],
            detected_reps=data["detected_reps"],
            valid_frames=data["valid_frames"],
            min_required_reps=data.get("min_required_reps", 1),
        )