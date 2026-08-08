"""Unit tests for the OrthoRehab AI Step 4 reference exercise profile builder.

These tests use synthetic angle sequences and exercise analyses (no video / no
MediaPipe required) and cover the required behaviours:

  1. profile creation from a valid AngleSequence
  2. profile contains the correct primary joint
  3. correct angle statistics
  4. correct phase information
  5. correct repetition information
  6. normalized trajectory generation
  7. incomplete repetition is not treated as a reference repetition
  8. missing angle values are handled safely
  9. JSON serialization/deserialization preserves the profile
  10. bilateral information is preserved where available

The builder is deterministic and reuses Steps 1-3; no ML model is used.
"""

from __future__ import annotations

import os
import sys
import json
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleFrame, AngleSequence
from src.pose.exercise_config import ExercisePhaseConfig
from src.pose.phase_rep_counter import ExerciseAnalysis
from src.pose.reference_profile_builder import build_reference_profile


def make_angle_sequence(
    values: List[Optional[float]],
    fps: float = 30.0,
    angle_name: str = "left_elbow",
    with_bilateral: bool = False,
) -> AngleSequence:
    """Build an AngleSequence from a flat list of primary-angle values."""
    frames = []
    for i, v in enumerate(values):
        angles = {angle_name: v}
        if with_bilateral:
            angles["right_elbow"] = (v - 5.0) if v is not None else None
        frames.append(AngleFrame(frame_index=i, timestamp=i / fps, angles=angles))
    return AngleSequence(fps=fps, frame_count=len(frames), frames=frames)


def full_rep():
    """A ~1.0s synthetic elbow flexion/extension cycle (start->bottom->start)."""
    vals = (
        [170.0] * 5
        + [160.0, 150.0, 130.0, 110.0, 95.0, 85.0, 80.0]
        + [80.0] * 5
        + [90.0, 110.0, 130.0, 150.0, 160.0, 168.0]
        + [168.0] * 3
    )
    return vals


def default_config(**overrides) -> ExercisePhaseConfig:
    params = dict(
        exercise="elbow_flexion",
        primary_angle="left_elbow",
        movement_type="flexion_extension",
        start_threshold=160.0,
        end_threshold=160.0,
        bottom_threshold=90.0,
        min_rep_duration=0.2,
        min_time_between_reps=0.1,
        min_movement_range=30.0,
        smoothing_window=1,
    )
    params.update(overrides)
    return ExercisePhaseConfig(**params)


def build_analysis(values, config=None, **cfg_overrides):
    """Run the Step 3 detector directly to build an ExerciseAnalysis."""
    if config is None:
        config = default_config(**cfg_overrides)
    seq = make_angle_sequence(values)
    from src.pose.phase_detector import PhaseDetector
    summary = PhaseDetector(config).detect(seq)
    return ExerciseAnalysis(
        exercise=config.exercise,
        primary_angle=config.primary_angle,
        total_reps=len(summary.repetitions),
        phases=summary.phases,
        repetitions=summary.repetitions,
        min_angle=summary.min_angle,
        max_angle=summary.max_angle,
        valid_frames=summary.valid_frames,
    ), seq


def run_test(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")


# --- 1. profile creation from a valid AngleSequence ------------------------
def t_profile_creation():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    assert profile is not None
    assert profile.exercise_name == "elbow_flexion"
    assert profile.primary_angle == "left_elbow"
    assert profile.created_at
    assert profile.source_video is not None


# --- 2. profile contains correct primary joint -----------------------------
def t_primary_joint():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    assert profile.primary_angle == "left_elbow"
    stats = profile.angle_statistics.get("left_elbow")
    assert stats is not None, "primary angle must appear in angle_statistics"


# --- 3. correct angle statistics -------------------------------------------
def t_angle_stats():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    stats = profile.angle_statistics["left_elbow"]
    vals = [v for v in full_rep() if v is not None]
    assert stats.min_angle == min(vals)
    assert stats.max_angle == max(vals)
    assert stats.mean_angle == round(sum(vals) / len(vals), 2)
    assert round(stats.range, 2) == round(max(vals) - min(vals), 2)
    assert stats.start_angle == vals[0]
    assert stats.end_angle == vals[-1]


# --- 4. correct phase information ------------------------------------------
def t_phase_info():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    pp = profile.phase_profile
    assert pp is not None
    from src.pose.phase_detector import (
        PHASE_FLEXING, PHASE_BOTTOM, PHASE_EXTENDING, PHASE_START,
    )
    seq_phases = pp.sequence
    # The cycle should contain the main phases in order.
    assert PHASE_FLEXING in seq_phases
    assert PHASE_BOTTOM in seq_phases
    assert PHASE_EXTENDING in seq_phases
    assert seq_phases.index(PHASE_BOTTOM) < seq_phases.index(PHASE_EXTENDING)
    # Events should align with the profile's expected phase transitions.
    assert len(pp.events) == len(analysis.phases)


# --- 5. correct repetition information -------------------------------------
def t_rep_info():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    assert len(profile.repetitions) == 1
    rep = profile.repetitions[0]
    assert rep.rep_number == 1
    assert rep.min_angle <= 90.0
    assert rep.max_angle >= 160.0
    assert rep.start_timestamp < rep.bottom_timestamp < rep.end_timestamp


# --- 6. normalized trajectory generation -----------------------------------
def t_trajectory():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    traj = profile.reference_trajectory
    assert traj is not None
    assert traj.source_reps_used == 1
    assert traj.num_points == 21  # 0%, 5%, ..., 100%
    # First point at progress 0, last at progress 1.
    assert traj.points[0].progress == 0.0
    assert abs(traj.points[-1].progress - 1.0) < 1e-6
    # Progress strictly increasing.
    ps = [p.progress for p in traj.points]
    assert all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))


# --- 7. incomplete repetition is not a reference rep -----------------------
def t_incomplete_ignored():
    cfg = default_config()
    incomplete = full_rep()[:-7]  # ends ~130, never returns to start
    analysis, seq = build_analysis(incomplete, config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    # Step 3 reports 0 reps; profile must reflect that.
    assert analysis.total_reps == 0
    assert len(profile.repetitions) == 0
    assert profile.reference_trajectory is None


# --- 8. missing angle values handled safely --------------------------------
def t_missing_values():
    cfg = default_config()
    vals = full_rep()
    vals[2] = None
    vals[8] = None
    vals[16] = None
    analysis, seq = build_analysis(vals, config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    # A valid rep is still detected; trajectory should be generated.
    assert len(profile.repetitions) == 1
    assert profile.reference_trajectory is not None


# --- 9. JSON serialization/deserialization preserves profile ---------------
def t_json_roundtrip():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg)
    profile = build_reference_profile(seq, analysis, cfg)
    data = profile.to_dict()
    s = json.dumps(data)
    loaded = json.loads(s)
    restored = profile.__class__.from_dict(loaded)
    assert restored.to_dict() == profile.to_dict()
    assert restored.primary_angle == profile.primary_angle
    assert restored.repetitions[0].rep_number == profile.repetitions[0].rep_number
    assert restored.reference_trajectory.source_reps_used == 1


# --- 10. bilateral information preserved -----------------------------------
def t_bilateral():
    cfg = default_config()
    analysis, seq = build_analysis(full_rep(), config=cfg, )
    # Add bilateral right_elbow to the sequence.
    seq2 = make_angle_sequence(full_rep(), with_bilateral=True)
    analysis2 = ExerciseAnalysis(
        exercise=analysis.exercise,
        primary_angle=analysis.primary_angle,
        total_reps=analysis.total_reps,
        phases=analysis.phases,
        repetitions=analysis.repetitions,
        min_angle=analysis.min_angle,
        max_angle=analysis.max_angle,
        valid_frames=analysis.valid_frames,
    )
    profile = build_reference_profile(seq2, analysis2, cfg)
    # Both left and right elbows should appear in angle_statistics.
    assert "left_elbow" in profile.angle_statistics
    assert "right_elbow" in profile.angle_statistics


def run_all() -> None:
    print("STEP 4 REFERENCE PROFILE TESTS")
    print("=" * 60)
    run_test("profile creation from AngleSequence", t_profile_creation)
    run_test("correct primary joint", t_primary_joint)
    run_test("correct angle statistics", t_angle_stats)
    run_test("correct phase information", t_phase_info)
    run_test("correct repetition information", t_rep_info)
    run_test("normalized trajectory generation", t_trajectory)
    run_test("incomplete rep not a reference rep", t_incomplete_ignored)
    run_test("missing angle values handled safely", t_missing_values)
    run_test("JSON serialization roundtrip", t_json_roundtrip)
    run_test("bilateral information preserved", t_bilateral)
    print("=" * 60)


if __name__ == "__main__":
    run_all()

