"""Builder for :class:`ReferenceExerciseProfile` — OrthoRehab AI Step 4.

This module turns the outputs of Steps 1-3 into a compact, reusable reference
exercise profile. It reuses the existing modules exactly; it does NOT re-derive
landmarks, angles, phases or reps.

Pipeline:
    Step 1 LandmarkSequence  -> available / supported landmarks + source metadata
    Step 2 AngleSequence     -> primary + bilateral angle statistics
    Step 3 ExerciseAnalysis  -> phase profile + validated repetitions
    Step 4 (this module)     -> normalized reference trajectory + top-level profile

Normalized trajectory:
    For each validated complete repetition we take the smoothed primary-angle
    series (recomputed with the same smoothing window the detector used) over
    the repetition's [start_frame, end_frame] span, resample it to evenly
    spaced progress values (0%, 5%, ..., 100%) via linear interpolation, and
    average the normalized curves across repetitions to produce a single
    representative cycle. If there is only one valid repetition that cycle is
    used directly (no fake averaging). This makes the profile comparable to a
    patient session independently of absolute movement speed (for a future DTW
    stage — not implemented here).

Incomplete repetitions and missing angle values are handled safely: only
validated :class:`RepResult` objects contribute to the trajectory, and only
non-None angle values participate in interpolation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .angle_engine import AngleSequence
from .exercise_config import ExercisePhaseConfig
from .landmark_extractor import LandmarkSequence
from .phase_detector import PHASE_BOTTOM, PHASE_EXTENDING, PHASE_FLEXING, PHASE_START, smooth_angles
from .phase_rep_counter import ExerciseAnalysis
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

# Progress sampling for the normalized trajectory. 0..1 in 0.05 steps -> 21 points.
DEFAULT_PROGRESS_SAMPLES: List[float] = [i / 20.0 for i in range(21)]

# Hip-and-above landmark names used by the default angle definitions. The
# builder only ever exposes upper-body landmarks (shoulders, elbows, wrists,
# hips). Lower-body landmarks are never introduced.
HIP_AND_ABOVE_LANDMARK_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
]


def _angle_stats(values: List[float]) -> ReferenceAngleStatistics:
    """Build :class:`ReferenceAngleStatistics` from a list of valid angles."""
    valid = [v for v in values if v is not None]
    if not valid:
        return ReferenceAngleStatistics(valid_frames=0)
    return ReferenceAngleStatistics(
        min_angle=round(float(min(valid)), 2),
        max_angle=round(float(max(valid)), 2),
        mean_angle=round(float(sum(valid) / len(valid)), 2),
        range=round(float(max(valid) - min(valid)), 2),
        start_angle=round(float(valid[0]), 2),
        end_angle=round(float(valid[-1]), 2),
        valid_frames=len(valid),
    )


def _bottom_angle(stats: ReferenceAngleStatistics, movement_type: str) -> Optional[float]:
    """Report the bottom (flexed) angle given the movement type."""
    if movement_type == "flexion_extension":
        return stats.min_angle
    return stats.max_angle


def _phase_events_from_analysis(analysis: ExerciseAnalysis) -> List[ReferencePhaseEvent]:
    """Convert Step 3 PhaseEvents into profile phase events."""
    return [
        ReferencePhaseEvent(
            phase=p.phase,
            frame_index=p.frame_index,
            timestamp=p.timestamp,
            angle=p.angle,
        )
        for p in analysis.phases
    ]


def _phase_sequence(analysis: ExerciseAnalysis) -> List[str]:
    """Extract the ordered phase labels from Step 3, trimming leading 'start'."""
    seq = [p.phase for p in analysis.phases]
    # Remove a leading start if present (the detector sometimes emits one).
    i = 0
    while i < len(seq) and seq[i] == PHASE_START:
        i += 1
    return seq[i:]


def _smooth_series_for_rep(
    angle_sequence: AngleSequence,
    primary_angle: str,
    start_frame: int,
    end_frame: int,
    smoothing_window: int,
) -> List[Optional[float]]:
    """Return the smoothed primary-angle series over a rep's frame span."""
    raw = [
        af.angles.get(primary_angle)
        for af in angle_sequence.frames
        if start_frame <= af.frame_index <= end_frame
    ]
    return smooth_angles(raw, smoothing_window)


def _normalize_cycle(
    smoothed: List[Optional[float]],
    progress_samples: Optional[List[float]] = None,
) -> Optional[List[float]]:
    """Resample a smoothed angle series to evenly spaced progress values.

    Returns a list of interpolated angles aligned to ``progress_samples``
    (default 0..1 in 0.05 steps -> 21 values), or None if there are too few
    valid points to interpolate.
    """
    if progress_samples is None:
        progress_samples = DEFAULT_PROGRESS_SAMPLES

    valid_idx = [i for i, v in enumerate(smoothed) if v is not None]
    if len(valid_idx) < 2:
        return None

    xs = np.array(valid_idx, dtype=float)
    ys = np.array([smoothed[i] for i in valid_idx], dtype=float)
    # Normalize index positions to [0, 1].
    x_norm = (xs - xs[0]) / (xs[-1] - xs[0]) if xs[-1] > xs[0] else xs * 0.0

    out: List[float] = []
    for p in progress_samples:
        # Clip to the valid range and interpolate.
        out.append(float(np.interp(p, x_norm, ys)))
    return out


def _build_trajectory(
    angle_sequence: AngleSequence,
    primary_angle: str,
    repetitions: List[ReferenceRepetition],
    smoothing_window: int,
    progress_samples: Optional[List[float]] = None,
) -> Optional[ReferenceTrajectory]:
    """Build a representative normalized trajectory from validated reps."""
    if progress_samples is None:
        progress_samples = DEFAULT_PROGRESS_SAMPLES

    cycles: List[List[float]] = []
    for rep in repetitions:
        smoothed = _smooth_series_for_rep(
            angle_sequence, primary_angle, rep.start_frame, rep.end_frame, smoothing_window
        )
        cycle = _normalize_cycle(smoothed, progress_samples)
        if cycle is not None:
            cycles.append(cycle)

    if not cycles:
        return None

    # Average the normalized cycles point-wise for a representative trajectory.
    arr = np.array(cycles, dtype=float)  # shape (n_reps, n_points)
    mean_cycle = arr.mean(axis=0)

    points = [
        ReferenceTrajectoryPoint(progress=p, angle=round(float(a), 2))
        for p, a in zip(progress_samples, mean_cycle)
    ]
    return ReferenceTrajectory(
        points=points,
        num_points=len(points),
        source_reps_used=len(cycles),
        method="linear_interp_percent_100" + ("_avg" if len(cycles) > 1 else ""),
    )


def _required_landmarks(
    primary_angle: str,
    angle_definitions: Dict[str, List[str]],
) -> List[str]:
    """Return the hip-and-above landmarks needed to compute ``primary_angle``."""
    names = angle_definitions.get(primary_angle, [])
    return [n for n in names if n in HIP_AND_ABOVE_LANDMARK_NAMES]


def _available_landmarks(landmark_sequence: Optional[LandmarkSequence]) -> List[str]:
    """Return the set of landmark names present in the reference sequence."""
    if landmark_sequence is None or not landmark_sequence.frames:
        return []
    seen = set()
    for pf in landmark_sequence.frames:
        for lm in pf.landmarks:
            seen.add(lm.name)
    return sorted(seen)


def build_reference_profile(
    angle_sequence: AngleSequence,
    analysis: ExerciseAnalysis,
    config: ExercisePhaseConfig,
    landmark_sequence: Optional[LandmarkSequence] = None,
    exercise_id: Optional[str] = None,
    exercise_name: Optional[str] = None,
    source_video_metadata: Optional[Dict[str, float]] = None,
    angle_definitions: Optional[Dict[str, List[str]]] = None,
    progress_samples: Optional[List[float]] = None,
) -> ReferenceExerciseProfile:
    """Build a :class:`ReferenceExerciseProfile` from Steps 1-3 outputs.

    Args:
        angle_sequence: Step 2 output (needed for stats + trajectory).
        analysis: Step 3 output (phases + validated repetitions).
        config: the Step 3 :class:`ExercisePhaseConfig` used.
        landmark_sequence: optional Step 1 output (for available landmarks).
        exercise_id: optional stable id (defaults to config.exercise + hash).
        exercise_name: optional human name (defaults to config.exercise).
        source_video_metadata: optional dict with width/height/frame_count etc.
        angle_definitions: optional angle-definition map used by Step 2 (defaults
            to the engine defaults) — used to derive required landmarks.
        progress_samples: optional progress values for the normalized trajectory.

    Returns:
        A populated :class:`ReferenceExerciseProfile`.
    """
    primary = config.primary_angle

    # --- angle statistics (primary + bilateral) ----------------------------
    per_angle: Dict[str, List[float]] = {}
    for af in angle_sequence.frames:
        for name, val in af.angles.items():
            if val is not None:
                per_angle.setdefault(name, []).append(val)

    stats_map: Dict[str, ReferenceAngleStatistics] = {
        name: _angle_stats(vals) for name, vals in per_angle.items()
    }

    primary_stats = stats_map.get(primary, ReferenceAngleStatistics())

    # --- repetitions --------------------------------------------------------
    repetitions = [
        ReferenceRepetition(
            rep_number=r.rep_number,
            start_frame=r.start_frame,
            bottom_frame=r.bottom_frame,
            end_frame=r.end_frame,
            start_timestamp=r.start_timestamp,
            bottom_timestamp=r.bottom_timestamp,
            end_timestamp=r.end_timestamp,
            min_angle=r.min_angle,
            max_angle=r.max_angle,
            duration=r.duration,
        )
        for r in analysis.repetitions
    ]

    # --- phase profile ------------------------------------------------------
    phase_profile = ReferencePhaseProfile(
        sequence=_phase_sequence(analysis),
        events=_phase_events_from_analysis(analysis),
    )

    # --- trajectory ---------------------------------------------------------
    trajectory = _build_trajectory(
        angle_sequence,
        primary,
        repetitions,
        config.smoothing_window,
        progress_samples,
    )

    # --- source video metadata ---------------------------------------------
    if landmark_sequence is not None:
        src = ReferenceSourceVideo(
            path=landmark_sequence.video_path,
            fps=landmark_sequence.fps,
            frame_count=landmark_sequence.frame_count,
            pose_frames=landmark_sequence.pose_frames,
            width=int(source_video_metadata.get("width", 0)) if source_video_metadata else 0,
            height=int(source_video_metadata.get("height", 0)) if source_video_metadata else 0,
        )
    else:
        src = ReferenceSourceVideo(
            path=source_video_metadata.get("path", "") if source_video_metadata else "",
            fps=float(source_video_metadata.get("fps", 0.0)) if source_video_metadata else 0.0,
            frame_count=int(source_video_metadata.get("frame_count", 0)) if source_video_metadata else 0,
            pose_frames=int(source_video_metadata.get("pose_frames", 0)) if source_video_metadata else 0,
            width=int(source_video_metadata.get("width", 0)) if source_video_metadata else 0,
            height=int(source_video_metadata.get("height", 0)) if source_video_metadata else 0,
        )

    # --- ids ----------------------------------------------------------------
    if exercise_id is None:
        exercise_id = config.exercise
    if exercise_name is None:
        exercise_name = config.exercise

    # --- configuration ------------------------------------------------------
    configuration = ReferenceConfiguration(
        exercise=config.exercise,
        primary_angle=primary,
        movement_type=config.movement_type,
        start_threshold=config.start_threshold,
        end_threshold=config.end_threshold,
        bottom_threshold=config.bottom_threshold,
        min_rep_duration=config.min_rep_duration,
        min_time_between_reps=config.min_time_between_reps,
        min_movement_range=config.min_movement_range,
        smoothing_window=config.smoothing_window,
        variants=list(config.variants),
    )

    # --- required / available landmarks -------------------------------------
    if angle_definitions is None:
        from .angle_engine import DEFAULT_ANGLE_DEFINITIONS
        angle_definitions = DEFAULT_ANGLE_DEFINITIONS
    required = _required_landmarks(primary, angle_definitions)
    available = _available_landmarks(landmark_sequence)

    # --- build profile ------------------------------------------------------
    bottom_angle = _bottom_angle(primary_stats, config.movement_type)

    return ReferenceExerciseProfile(
        exercise_id=exercise_id,
        exercise_name=exercise_name,
        created_at=now_iso(),
        source_video=src,
        primary_angle=primary,
        movement_type=config.movement_type,
        supported_landmarks=required,
        available_landmarks=available,
        angle_statistics=stats_map,
        phase_profile=phase_profile,
        repetitions=repetitions,
        reference_trajectory=trajectory,
        configuration=configuration,
        primary_min_angle=primary_stats.min_angle,
        primary_max_angle=primary_stats.max_angle,
        primary_mean_angle=primary_stats.mean_angle,
        primary_range=primary_stats.range,
        primary_start_angle=primary_stats.start_angle,
        primary_bottom_angle=bottom_angle,
        primary_end_angle=primary_stats.end_angle,
    )
