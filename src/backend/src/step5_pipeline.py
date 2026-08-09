"""End-to-end Step 5 pipeline for OrthoRehab AI.

Conceptual flow:

    Therapist video -> MediaPipe -> reference representation
    Patient video   -> MediaPipe -> patient representation
    -> capture-quality gate (reject unusable recordings)
    -> comparison engine (biomechanics + DTW + speed + deviations + optional learned)
    -> evidence fusion -> overall score
    -> LLM (or deterministic fallback) -> real-time feedback -> session report

This facade reuses the existing Step 1-4 modules (landmark extraction, angle
engine, phase/rep counter, reference profile builder) and the Step 5 comparison
engine. It does NOT duplicate any existing functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .comparison.comparison_service import ComparisonEngine, EngineConfig
from .feedback.feedback_service import FeedbackService, FeedbackServiceConfig
from .llm.groq_client import GroqClient
from .performance import Timer
from .schemas.comparison import CaptureQualityError, ComparisonResult
from .session_report import SessionReport, build_session_report

# Reuse Steps 1-4.
from .pose.angle_engine import angle_sequence_from_landmark_sequence
from .pose.capture_quality import CaptureQualityValidator
from .pose.landmark_extractor import extract_landmark_sequence
from .pose.phase_rep_counter import count_reps
from .pose.reference_profile_builder import build_reference_profile


@dataclass
class Step5Result:
    """The full output of a Step 5 pipeline run.

    ``status`` is ``"success"`` for a scored comparison, or
    ``"capture_quality_failed"`` when the recording could not be analyzed
    because required landmarks were not reliably visible. When it fails,
    ``comparison`` and ``report`` are None and ``capture_quality_error``
    describes the failure with a re-record instruction.
    """

    comparison: Optional[ComparisonResult] = None
    report: Optional[SessionReport] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    status: str = "success"  # "success" | "capture_quality_failed"
    scored: bool = False
    capture_quality_error: Optional[CaptureQualityError] = None


class Step5Pipeline:
    """End-to-end comparison of a patient video against a reference profile.

    Args:
        reference_profile: a Step 4 :class:`ReferenceExerciseProfile` (built from
            the therapist reference video). If None, it can be built from a
            reference video path at runtime.
        engine_config: optional :class:`EngineConfig` for the comparison engine.
        feedback_config: optional :class:`FeedbackServiceConfig`.
        groq: optional :class:`GroqClient`.
    """

    def __init__(
        self,
        reference_profile=None,
        engine_config: Optional[EngineConfig] = None,
        feedback_config: Optional[FeedbackServiceConfig] = None,
        groq: Optional[GroqClient] = None,
    ) -> None:
        self.reference_profile = reference_profile
        self.engine = ComparisonEngine(config=engine_config)
        self.groq = groq or GroqClient()
        self.feedback = FeedbackService(groq=self.groq, config=feedback_config)
        self.timer = Timer()
        self.capture_validator = CaptureQualityValidator()

    def build_reference_from_video(
        self,
        video_path: str,
        exercise_id: str = "elbow_flexion",
        exercise_name: str = "Elbow Flexion / Extension",
        primary_angle: str = "right_elbow",
        config_overrides: Optional[Dict] = None,
    ):
        """Run Steps 1-4 on a therapist reference video and store the profile."""
        with self.timer.time("reference_mediapipe"):
            lm_seq = extract_landmark_sequence(video_path)
        with self.timer.time("reference_angles"):
            angle_seq = angle_sequence_from_landmark_sequence(lm_seq)
        with self.timer.time("reference_reps"):
            analysis = count_reps(
                angle_seq,
                primary_angle=primary_angle,
                exercise=exercise_id,
                **(config_overrides or {}),
            )
        with self.timer.time("reference_profile"):
            profile = build_reference_profile(
                angle_seq,
                analysis,
                analysis_config(angle_seq, primary_angle, exercise_id, config_overrides),
                landmark_sequence=lm_seq,
                exercise_id=exercise_id,
                exercise_name=exercise_name,
            )
        self.reference_profile = profile
        return profile

    def compare_video(
        self,
        video_path: str,
        primary_angle: Optional[str] = None,
        config_overrides: Optional[Dict] = None,
    ) -> Step5Result:
        """Run the full comparison for a patient video.

        Args:
            video_path: patient video path.
            primary_angle: primary angle (defaults to the reference profile's).
            config_overrides: optional Step 3 config overrides for rep detection.

        Returns:
            A :class:`Step5Result` with comparison + report + timings.
        """
        if self.reference_profile is None:
            raise ValueError("reference_profile is required. Build it first.")

        primary = primary_angle or self.reference_profile.primary_angle

        with self.timer.time("patient_mediapipe"):
            lm_seq = extract_landmark_sequence(video_path)
        with self.timer.time("patient_angles"):
            angle_seq = angle_sequence_from_landmark_sequence(lm_seq)

        # Capture-quality gate: if the required landmarks cannot be reliably
        # observed, the recording cannot be analyzed. This is DISTINCT from a
        # valid recording with poor exercise performance (which is still scored).
        # When quality fails we do NOT run rep counting, DTW, speed, biomech,
        # scoring, or session reporting.
        quality = self.capture_validator.validate(
            angle_seq,
            primary,
            landmark_sequence=lm_seq,
        )
        if not quality.passed:
            err = CaptureQualityError(
                code=quality.error_code,
                message=quality.error_message,
                user_instruction=quality.user_instruction,
                missing_landmarks=quality.missing_landmarks,
                valid_angle_ratio=quality.valid_angle_ratio,
                total_frames=quality.total_frames,
                valid_frames=quality.valid_frames,
                missing_frames=quality.missing_frames,
            )
            return Step5Result(
                comparison=None,
                report=None,
                timings_ms=self.timer.to_dict(),
                status="capture_quality_failed",
                scored=False,
                capture_quality_error=err,
            )

        with self.timer.time("patient_reps"):
            analysis = count_reps(
                angle_seq,
                primary_angle=primary,
                exercise=self.reference_profile.exercise_id,
                **(config_overrides or {}),
            )

        with self.timer.time("comparison"):
            result = self.engine.compare(
                self.reference_profile,
                angle_seq,
                patient_exercise_analysis=analysis,
            )
            result.timings_ms = self.timer.to_dict()

        with self.timer.time("session_report"):
            report = build_session_report(result, groq=self.groq, timings_ms=self.timer.to_dict())

        return Step5Result(
            comparison=result,
            report=report,
            timings_ms=self.timer.to_dict(),
            status="success",
            scored=True,
        )

    def evaluate_feedback(self, findings: Dict, timestamp: float):
        """Evaluate real-time feedback for a single findings update."""
        return self.feedback.evaluate(findings, timestamp)


def analysis_config(angle_seq, primary_angle, exercise_id, overrides: Optional[Dict]):
    """Build the Step 3 config used to derive the reference profile."""
    from .pose.exercise_config import ExercisePhaseConfig, config_from_angle_sequence

    try:
        return config_from_angle_sequence(
            angle_seq,
            exercise=exercise_id,
            primary_angle=primary_angle,
            movement_type="flexion_extension",
            **(overrides or {}),
        )
    except Exception:
        # Fall back to an explicit config if auto-derivation fails.
        return ExercisePhaseConfig(
            exercise=exercise_id,
            primary_angle=primary_angle,
            movement_type="flexion_extension",
            start_threshold=150.0,
            end_threshold=150.0,
            bottom_threshold=85.0,
            min_rep_duration=0.3,
            min_time_between_reps=0.2,
            min_movement_range=30.0,
            smoothing_window=5,
        )
