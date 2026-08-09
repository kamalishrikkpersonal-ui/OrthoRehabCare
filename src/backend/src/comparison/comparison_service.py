"""Hybrid comparison engine (ComparisonEngine) for OrthoRehab AI Step 5.

The engine is the orchestrator that combines:

* BiomechanicalComparator  (reuses Step 2 joint angles / Step 4 reference profile)
* DTWComparator           (temporal alignment of normalized trajectories)
* SpeedAnalyzer           (duration / speed classification, independent of DTW)
* DeviationEngine         (deterministic deviation events)
* LearnedMotionComparator (optional; currently unavailable)

Evidence fusion: the learned similarity (if available) is an ADDITIONAL signal
and never overrides deterministic measurements. The overall score is a
transparent weighted combination of biomechanical accuracy, DTW similarity and
speed quality (and learned similarity only if a validated model exists).

The engine is generic/configurable via reference profile data and configuration
(weights, thresholds). It is not hardcoded to one exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..pose.angle_engine import AngleSequence
from ..pose.reference_profile import ReferenceExerciseProfile
from ..pose.reference_profile_builder import (
    DEFAULT_PROGRESS_SAMPLES,
    smooth_and_normalize_cycle,
)
from ..schemas.comparison import (
    AssessmentDeviation,
    ComparisonResult,
    RepComparison,
)
from .biomechanical_comparator import BiomechanicalComparator, BiomechanicalResult
from .deviation_engine import DEV_TRAJECTORY, DeviationEngine
from .dtw_comparator import DTWComparator
from .learned_motion_comparator import LearnedMotionComparator, UnavailableLearnedComparator
from .speed_analyzer import SpeedAnalyzer

# Default weights for the transparent overall score.
# movement_quality = w_biomech*biomech + w_dtw*dtw + w_speed*speed (+ w_learned*learned if validated)
DEFAULT_WEIGHTS = {
    "biomechanical": 0.45,
    "dtw": 0.35,
    "speed": 0.20,
    "learned": 0.0,  # only used if a validated learned model is present
}


@dataclass
class EngineConfig:
    """Configuration for the comparison engine."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    speed_threshold_fast: float = 1.5
    speed_threshold_slow: float = 0.67
    min_valid_frames: int = 2
    rep_level_compare: bool = True

    def __post_init__(self) -> None:
        for k in ("biomechanical", "dtw", "speed"):
            if k not in self.weights:
                raise ValueError(f"Missing required weight: {k}")
        total = sum(v for k, v in self.weights.items() if v > 0)
        if total <= 0:
            raise ValueError("At least one positive weight is required")


class ComparisonEngine:
    """Orchestrates the hybrid comparison of a patient exercise vs a reference."""

    def __init__(
        self,
        biomech: Optional[BiomechanicalComparator] = None,
        dtw: Optional[DTWComparator] = None,
        speed: Optional[SpeedAnalyzer] = None,
        deviation_engine: Optional[DeviationEngine] = None,
        learned: Optional[LearnedMotionComparator] = None,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.deviation_engine = deviation_engine or DeviationEngine()
        self.biomech = biomech or BiomechanicalComparator(self.deviation_engine)
        self.dtw = dtw or DTWComparator()
        self.speed = speed or SpeedAnalyzer()
        self.learned = learned or UnavailableLearnedComparator()

    # -- reference trajectory helpers -----------------------------------------
    @staticmethod
    def _reference_trajectory_angles(profile: ReferenceExerciseProfile) -> List[Optional[float]]:
        """Extract the reference primary-angle trajectory from the profile."""
        traj = profile.reference_trajectory
        if traj is None or not traj.points:
            return []
        return [p.angle for p in traj.points]

    @staticmethod
    def _patient_trajectory(
        angle_sequence: AngleSequence, primary_angle: str
    ) -> List[Optional[float]]:
        """Return the patient's raw primary-angle series."""
        return [af.angles.get(primary_angle) for af in angle_sequence.frames]

    @staticmethod
    def _resample_to_progress(
        series: Sequence[Optional[float]], num_points: int = 21
    ) -> List[Optional[float]]:
        """Resample a 1-D series to ``num_points`` evenly spaced progress values.

        Mirrors how the reference trajectory is built (see Step 4
        ``_normalize_cycle``): non-None values are interpolated onto a 0..1
        progress axis. Returns a list of ``num_points`` values (or an empty list
        if there are too few valid points).
        """
        import numpy as np

        valid_idx = [i for i, v in enumerate(series) if v is not None]
        if len(valid_idx) < 2:
            return []
        xs = np.array(valid_idx, dtype=float)
        ys = np.array([series[i] for i in valid_idx], dtype=float)
        x_norm = (xs - xs[0]) / (xs[-1] - xs[0]) if xs[-1] > xs[0] else xs * 0.0
        progress = [i / (num_points - 1) for i in range(num_points)]
        return [float(np.interp(p, x_norm, ys)) for p in progress]

    # -- evidence fusion ------------------------------------------------------
    def _overall_score(
        self,
        biomech_accuracy: Optional[float],
        dtw_sim: Optional[float],
        speed_status: str,
        learned_sim: Optional[float],
    ) -> Optional[float]:
        """Compute a transparent weighted overall score in [0, 1]."""
        w = self.config.weights

        # Speed quality: map speed_status to a 0..1 score.
        speed_quality = {
            "normal_speed": 1.0,
            "too_fast": 0.6,
            "too_slow": 0.6,
        }.get(speed_status, 0.5)

        parts = []
        if biomech_accuracy is not None:
            parts.append((w["biomechanical"], biomech_accuracy))
        if dtw_sim is not None:
            parts.append((w["dtw"], dtw_sim))
        parts.append((w["speed"], speed_quality))
        if learned_sim is not None and w.get("learned", 0.0) > 0:
            parts.append((w["learned"], learned_sim))

        if not parts:
            return None
        total_w = sum(p for p, _ in parts)
        if total_w <= 0:
            return None
        return float(round(sum(p * v for p, v in parts) / total_w, 3))

    # -- repetition-level analysis --------------------------------------------
    def _compare_rep(
        self,
        profile: ReferenceExerciseProfile,
        angle_sequence: AngleSequence,
        rep: "object",
        reference_traj: List[Optional[float]],
    ) -> RepComparison:
        """Compare one patient repetition (from Step 3 RepResult) to the reference."""
        primary = profile.primary_angle
        profile_config = profile.configuration
        smoothing_window = profile_config.smoothing_window if profile_config else 5

        # Patient angle series over the rep span (raw, unsmoothed).
        pat_raw = [
            af.angles.get(primary)
            for af in angle_sequence.frames
            if getattr(rep, "start_frame", None) is not None
            and getattr(rep, "end_frame", None) is not None
            and rep.start_frame <= af.frame_index <= rep.end_frame
        ]

        # Normalize the patient cycle with the SAME preprocessing the reference
        # used (smoothing + resampling to the reference progress grid), so DTW
        # compares like-for-like representations.
        pat_resampled = smooth_and_normalize_cycle(
            pat_raw,
            smoothing_window,
            progress_samples=DEFAULT_PROGRESS_SAMPLES[: len(reference_traj)],
        )
        if pat_resampled is None:
            pat_resampled = []
        dtw_sim, dtw_dist = self.dtw.compare(reference_traj, pat_resampled)

        # Speed for this rep.
        ref_dur = profile.repetitions[0].duration if profile.repetitions else 0.0
        pat_dur = getattr(rep, "duration", 0.0)
        speed_assessment = self.speed.analyze(ref_dur, pat_dur)

        # Biomechanical deviations for this rep (min/max within the rep span).
        rep_min = getattr(rep, "min_angle", None)
        rep_max = getattr(rep, "max_angle", None)
        ref_stats = profile.angle_statistics.get(primary)
        ref_min = ref_stats.min_angle if ref_stats else None
        ref_max = ref_stats.max_angle if ref_stats else None
        devs = self.deviation_engine.detect_angle_deviations(
            joint=primary,
            reference_min=ref_min,
            reference_max=ref_max,
            patient_min=rep_min,
            patient_max=rep_max,
            movement_type=profile.movement_type,
        )

        # ROM accuracy for this rep.
        rom_accuracy = None
        if ref_stats and ref_stats.range and rep_min is not None and rep_max is not None:
            pat_range = rep_max - rep_min
            if ref_stats.range > 0:
                ratio = pat_range / ref_stats.range
                rom_accuracy = float(max(0.0, min(1.0, 1.0 - abs(1.0 - ratio))))

        # Biomech accuracy for the rep.
        biomech_accuracy = None
        if rom_accuracy is not None:
            biomech_accuracy = rom_accuracy
            for d in devs:
                if d.severity == "major":
                    biomech_accuracy -= 0.25
                elif d.severity == "moderate":
                    biomech_accuracy -= 0.15
                elif d.severity == "minor":
                    biomech_accuracy -= 0.05
            biomech_accuracy = float(max(0.0, min(1.0, biomech_accuracy)))

        # Learned similarity (optional) at rep level.
        learned_sim = None
        if self.learned.available():
            learned_sim = self.learned.compare(reference_traj, pat_resampled)

        movement_quality = self._overall_score(
            biomech_accuracy, dtw_sim, speed_assessment.speed_status, learned_sim
        )

        # Overall rep severity.
        severity = "minor"
        if any(d.severity == "major" for d in devs) or speed_assessment.speed_status != "normal_speed":
            severity = "major" if any(d.severity == "major" for d in devs) else "moderate"

        return RepComparison(
            rep_number=getattr(rep, "rep_number", 0),
            start_timestamp=getattr(rep, "start_timestamp", 0.0),
            end_timestamp=getattr(rep, "end_timestamp", 0.0),
            duration=pat_dur,
            dtw_similarity=round(dtw_sim, 3) if dtw_sim is not None else None,
            dtw_distance=round(dtw_dist, 3) if dtw_dist is not None else None,
            learned_motion_similarity=round(learned_sim, 3) if learned_sim is not None else None,
            biomechanical_accuracy=biomech_accuracy,
            movement_quality=movement_quality,
            speed_status=speed_assessment.speed_status,
            primary_min_angle=rep_min,
            primary_max_angle=rep_max,
            reference_min_angle=ref_min,
            reference_max_angle=ref_max,
            rom_accuracy=rom_accuracy,
            deviations=devs,
            severity=severity,
        )

    # -- main entry point -----------------------------------------------------
    def compare(
        self,
        reference_profile: ReferenceExerciseProfile,
        patient_angle_sequence: AngleSequence,
        patient_exercise_analysis: Optional["object"] = None,
        reference_duration: Optional[float] = None,
        patient_duration: Optional[float] = None,
    ) -> ComparisonResult:
        """Run the full hybrid comparison.

        Args:
            reference_profile: Step 4 reference profile.
            patient_angle_sequence: Step 2 output for the patient.
            patient_exercise_analysis: optional Step 3 ExerciseAnalysis for the
                patient (used for repetition-level analysis).
            reference_duration: optional reference duration (defaults to profile).
            patient_duration: optional patient duration.

        Returns:
            A :class:`ComparisonResult`.
        """
        primary = reference_profile.primary_angle
        reference_traj = self._reference_trajectory_angles(reference_profile)
        patient_traj = self._patient_trajectory(patient_angle_sequence, primary)

        # Reference / patient durations.
        if reference_duration is None:
            reference_duration = (
                reference_profile.repetitions[0].duration if reference_profile.repetitions else 0.0
            )
        if patient_duration is None:
            patient_duration = (
                patient_exercise_analysis.repetitions[0].duration
                if patient_exercise_analysis and patient_exercise_analysis.repetitions
                else 0.0
            )

        # Speed (separate from DTW).
        speed_assessment = self.speed.analyze(reference_duration, patient_duration)

        # Reference / patient smoothing window (must match Step 4 construction).
        profile_config = reference_profile.configuration
        smoothing_window = profile_config.smoothing_window if profile_config else 5

        # Patient movement cycle over the first detected rep span (raw series).
        # Use the SAME preprocessing as the reference (smoothing + resampling to
        # the reference progress grid) so DTW and trajectory deviation compare
        # like-for-like normalized single cycles, independent of frame rate and
        # absolute speed. If no rep is detected, fall back to the full sequence.
        patient_raw_cycle = patient_traj
        if (
            patient_exercise_analysis is not None
            and patient_exercise_analysis.repetitions
        ):
            first_rep = patient_exercise_analysis.repetitions[0]
            patient_raw_cycle = [
                af.angles.get(primary)
                for af in patient_angle_sequence.frames
                if getattr(first_rep, "start_frame", None) is not None
                and getattr(first_rep, "end_frame", None) is not None
                and first_rep.start_frame <= af.frame_index <= first_rep.end_frame
            ]
        patient_cycle_normalized = smooth_and_normalize_cycle(
            patient_raw_cycle,
            smoothing_window,
            progress_samples=DEFAULT_PROGRESS_SAMPLES[: len(reference_traj)],
        )
        if patient_cycle_normalized is None:
            patient_cycle_normalized = []
        dtw_sim, dtw_dist = self.dtw.compare(reference_traj, patient_cycle_normalized)

        # Biomechanical comparison.
        biomech_result: BiomechanicalResult = self.biomech.compare(
            reference_profile, patient_angle_sequence
        )

        # Learned comparison (optional).
        learned_sim = None
        if self.learned.available():
            learned_sim = self.learned.compare(reference_traj, patient_cycle_normalized)

        # Assemble deviations.
        deviations: List[AssessmentDeviation] = list(biomech_result.deviations)
        speed_dev = self.deviation_engine.speed_deviation(speed_assessment.speed_status)
        if speed_dev is not None:
            deviations.append(speed_dev)

        # Trajectory deviation on the normalized patient cycle (rep-span aware),
        # computed with the SAME representation used for DTW. This avoids the
        # false positive caused by comparing the reference single cycle against
        # the patient's raw full-video sequence.
        if reference_traj and patient_cycle_normalized:
            traj_dev = self.deviation_engine.detect_trajectory_deviation(
                joint=primary,
                reference_trajectory=reference_traj,
                patient_trajectory=patient_cycle_normalized,
            )
            if traj_dev is not None:
                deviations.append(traj_dev)

        # Repetition-level analysis.
        rep_comparisons: List[RepComparison] = []
        if (
            self.config.rep_level_compare
            and patient_exercise_analysis is not None
            and patient_exercise_analysis.repetitions
        ):
            for rep in patient_exercise_analysis.repetitions:
                rep_comparisons.append(
                    self._compare_rep(
                        reference_profile,
                        patient_angle_sequence,
                        rep,
                        reference_traj,
                    )
                )

        # Overall score.
        overall = self._overall_score(
            biomech_result.biomechanical_accuracy, dtw_sim, speed_assessment.speed_status, learned_sim
        )

        return ComparisonResult(
            exercise_id=reference_profile.exercise_id,
            exercise_name=reference_profile.exercise_name,
            overall_score=overall,
            learned_motion_similarity=learned_sim,
            dtw_similarity=round(dtw_sim, 3) if dtw_sim is not None else None,
            biomechanical_accuracy=biomech_result.biomechanical_accuracy,
            rom_accuracy=biomech_result.rom_accuracy,
            reference_duration=round(reference_duration, 3),
            patient_duration=round(patient_duration, 3),
            speed_ratio=speed_assessment.speed_ratio,
            speed_status=speed_assessment.speed_status,
            primary_angle=primary,
            repetitions=rep_comparisons,
            deviations=deviations,
            learned_model_available=self.learned.available(),
        )
