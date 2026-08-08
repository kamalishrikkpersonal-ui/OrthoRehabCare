"""Reference exercise profile data model for OrthoRehab AI Step 4.

Consumes the outputs of Step 1 (LandmarkSequence), Step 2 (AngleSequence) and
Step 3 (ExerciseAnalysis) and packages them into a single, compact, reusable
:class:`ReferenceExerciseProfile`.

Design goals:
* Pure Python dataclasses (no FastAPI / MongoDB / webcam / browser / DTW).
* JSON serializable via ``to_dict()`` / ``from_dict()`` so the profile can be
  written to disk now and later persisted in a MongoDB repository.
* Exercise-specific: the profile stores the primary angle, movement type and
  thresholds that were actually used, so it is not coupled to any one exercise.
* Hip-and-above scope only: required landmarks are drawn from the hip-and-above
  angle definitions (hips, shoulders, elbows, wrists). No knees/ankles/feet.

The most important member is :attr:`ReferenceExerciseProfile.reference_trajectory`:
a normalized representation of the reference movement cycle (progress 0..1 ->
primary angle) that a future DTW / patient-comparison stage can compare
independently of absolute speed. Original frame indices and timestamps are
preserved in the repetition summaries for diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Angle statistics
# ---------------------------------------------------------------------------
@dataclass
class ReferenceAngleStatistics:
    """Statistics for one angle signal (as produced by Step 2)."""

    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    mean_angle: Optional[float] = None
    range: Optional[float] = None
    start_angle: Optional[float] = None
    bottom_angle: Optional[float] = None
    end_angle: Optional[float] = None
    valid_frames: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceAngleStatistics":
        return cls(**data)


# ---------------------------------------------------------------------------
# Phase information
# ---------------------------------------------------------------------------
@dataclass
class ReferencePhaseEvent:
    """A retained Step 3 phase transition (for diagnostics/reuse)."""

    phase: str
    frame_index: int
    timestamp: float
    angle: Optional[float]

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "angle": self.angle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferencePhaseEvent":
        return cls(**data)


@dataclass
class ReferencePhaseProfile:
    """Phase sequence and events retained from Step 3."""

    sequence: List[str] = field(default_factory=list)      # e.g. ["start","flexing",...]
    events: List[ReferencePhaseEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence": list(self.sequence),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferencePhaseProfile":
        return cls(
            sequence=data.get("sequence", []),
            events=[ReferencePhaseEvent.from_dict(e) for e in data.get("events", [])],
        )


# ---------------------------------------------------------------------------
# Repetition information
# ---------------------------------------------------------------------------
@dataclass
class ReferenceRepetition:
    """Summary of one validated complete reference repetition (Step 3 RepResult)."""

    rep_number: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_timestamp: float
    bottom_timestamp: float
    end_timestamp: float
    min_angle: float
    max_angle: float
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceRepetition":
        return cls(**data)


# ---------------------------------------------------------------------------
# Normalized reference trajectory
# ---------------------------------------------------------------------------
@dataclass
class ReferenceTrajectoryPoint:
    """A single point on the normalized reference movement cycle.

    ``progress`` is in [0.0, 1.0] and represents how far through a complete
    repetition the movement has reached, independent of absolute duration.
    ``angle`` is the primary-joint angle (degrees) expected at that progress.
    """

    progress: float
    angle: float

    def to_dict(self) -> dict:
        return {"progress": round(float(self.progress), 4), "angle": round(float(self.angle), 2)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTrajectoryPoint":
        return cls(**data)


@dataclass
class ReferenceTrajectory:
    """A normalized, reusable representation of the reference movement cycle.

    ``points`` is a list of :class:`ReferenceTrajectoryPoint` sampled at evenly
    spaced progress values (default 0%, 5%, ..., 100%). When multiple complete
    repetitions are available, this is a representative (averaged) cycle;
    otherwise (single rep) it is that rep's normalized cycle directly.
    """

    points: List[ReferenceTrajectoryPoint] = field(default_factory=list)
    num_points: int = 0
    source_reps_used: int = 0
    method: str = "linear_interp_percent_100"

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "num_points": self.num_points,
            "source_reps_used": self.source_reps_used,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTrajectory":
        return cls(
            points=[ReferenceTrajectoryPoint.from_dict(p) for p in data.get("points", [])],
            num_points=data.get("num_points", 0),
            source_reps_used=data.get("source_reps_used", 0),
            method=data.get("method", "linear_interp_percent_100"),
        )


# ---------------------------------------------------------------------------
# Source video metadata
# ---------------------------------------------------------------------------
@dataclass
class ReferenceSourceVideo:
    """Metadata about the therapist's reference video (no raw frames stored)."""

    path: str
    fps: float
    frame_count: int
    pose_frames: int
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceSourceVideo":
        return cls(**data)


# ---------------------------------------------------------------------------
# Configuration / thresholds
# ---------------------------------------------------------------------------
@dataclass
class ReferenceConfiguration:
    """The Step 3 configuration used to analyze the reference movement."""

    exercise: str
    primary_angle: str
    movement_type: str
    start_threshold: Optional[float]
    end_threshold: Optional[float]
    bottom_threshold: Optional[float]
    min_rep_duration: float
    min_time_between_reps: float
    min_movement_range: float
    smoothing_window: int
    variants: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceConfiguration":
        return cls(**data)


# ---------------------------------------------------------------------------
# Top-level profile
# ---------------------------------------------------------------------------
@dataclass
class ReferenceExerciseProfile:
    """Compact, reusable template produced from a therapist reference video.

    Stores only useful information for future patient comparison: exercise
    identity, source video metadata, primary angle, angle statistics (primary +
    bilateral), phase info, repetition info, a normalized reference trajectory,
    and the configuration thresholds used. It never embeds the raw video or
    every raw MediaPipe frame.
    """

    exercise_id: str
    exercise_name: str
    created_at: str
    source_video: ReferenceSourceVideo
    primary_angle: str
    movement_type: str
    supported_landmarks: List[str] = field(default_factory=list)
    available_landmarks: List[str] = field(default_factory=list)
    angle_statistics: Dict[str, ReferenceAngleStatistics] = field(default_factory=dict)
    phase_profile: Optional[ReferencePhaseProfile] = None
    repetitions: List[ReferenceRepetition] = field(default_factory=list)
    reference_trajectory: Optional[ReferenceTrajectory] = None
    configuration: Optional[ReferenceConfiguration] = None

    # Metrics for the primary angle (kept high-level for readability).
    primary_min_angle: Optional[float] = None
    primary_max_angle: Optional[float] = None
    primary_mean_angle: Optional[float] = None
    primary_range: Optional[float] = None
    primary_start_angle: Optional[float] = None
    primary_bottom_angle: Optional[float] = None
    primary_end_angle: Optional[float] = None

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "exercise_id": self.exercise_id,
            "exercise_name": self.exercise_name,
            "created_at": self.created_at,
            "source_video": self.source_video.to_dict(),
            "primary_angle": self.primary_angle,
            "movement_type": self.movement_type,
            "supported_landmarks": list(self.supported_landmarks),
            "available_landmarks": list(self.available_landmarks),
            "angle_statistics": {
                k: v.to_dict() for k, v in self.angle_statistics.items()
            },
            "phase_profile": self.phase_profile.to_dict() if self.phase_profile else None,
            "repetitions": [r.to_dict() for r in self.repetitions],
            "reference_trajectory": (
                self.reference_trajectory.to_dict() if self.reference_trajectory else None
            ),
            "configuration": self.configuration.to_dict() if self.configuration else None,
            "primary_min_angle": self.primary_min_angle,
            "primary_max_angle": self.primary_max_angle,
            "primary_mean_angle": self.primary_mean_angle,
            "primary_range": self.primary_range,
            "primary_start_angle": self.primary_start_angle,
            "primary_bottom_angle": self.primary_bottom_angle,
            "primary_end_angle": self.primary_end_angle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceExerciseProfile":
        return cls(
            exercise_id=data["exercise_id"],
            exercise_name=data["exercise_name"],
            created_at=data["created_at"],
            source_video=ReferenceSourceVideo.from_dict(data["source_video"]),
            primary_angle=data["primary_angle"],
            movement_type=data["movement_type"],
            supported_landmarks=list(data.get("supported_landmarks", [])),
            available_landmarks=list(data.get("available_landmarks", [])),
            angle_statistics={
                k: ReferenceAngleStatistics.from_dict(v)
                for k, v in data.get("angle_statistics", {}).items()
            },
            phase_profile=(
                ReferencePhaseProfile.from_dict(data["phase_profile"])
                if data.get("phase_profile")
                else None
            ),
            repetitions=[
                ReferenceRepetition.from_dict(r) for r in data.get("repetitions", [])
            ],
            reference_trajectory=(
                ReferenceTrajectory.from_dict(data["reference_trajectory"])
                if data.get("reference_trajectory")
                else None
            ),
            configuration=(
                ReferenceConfiguration.from_dict(data["configuration"])
                if data.get("configuration")
                else None
            ),
            primary_min_angle=data.get("primary_min_angle"),
            primary_max_angle=data.get("primary_max_angle"),
            primary_mean_angle=data.get("primary_mean_angle"),
            primary_range=data.get("primary_range"),
            primary_start_angle=data.get("primary_start_angle"),
            primary_bottom_angle=data.get("primary_bottom_angle"),
            primary_end_angle=data.get("primary_end_angle"),
        )


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
