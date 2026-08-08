"""Unit tests for the OrthoRehab AI Step 3 exercise phase/rep detector.

These tests use synthetic angle sequences (no video / no MediaPipe required) and
cover the required behaviours:

  1. valid complete repetition
  2. incomplete movement should not count
  3. noisy movement should not count multiple reps
  4. minimum duration
  5. missing angle values
  6. correct phase ordering
  7. correct timestamp preservation
  8. minimum movement range
  9. bilateral angle selection (if supported)

The detector is deterministic and config-driven; no ML model is used.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleFrame, AngleSequence
from src.pose.exercise_config import ExercisePhaseConfig
from src.pose.phase_detector import PhaseDetector, PHASE_START, PHASE_FLEXING, PHASE_BOTTOM, PHASE_EXTENDING
from src.pose.phase_rep_counter import count_reps, select_primary_angle


def make_angle_sequence(
    values: List[Optional[float]],
    fps: float = 30.0,
    angle_name: str = "left_elbow",
) -> AngleSequence:
    """Build an AngleSequence from a flat list of primary-angle values."""
    frames = [
        AngleFrame(
            frame_index=i,
            timestamp=i / fps,
            angles={angle_name: v},
        )
        for i, v in enumerate(values)
    ]
    return AngleSequence(fps=fps, frame_count=len(frames), frames=frames)


def run_test(name: str, fn) -> None:
    """Run a test function and report PASS/FAIL."""
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")


def default_config(**overrides) -> ExercisePhaseConfig:
    """A config for elbow flexion/extension tuned for synthetic tests."""
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


def full_rep():
    """A ~1.0s synthetic elbow flexion/extension cycle (start->bottom->start)."""
    # hold extended, descend to bottom, hold bottom, ascend back to extended.
    vals = (
        [170.0] * 5
        + [160.0, 150.0, 130.0, 110.0, 95.0, 85.0, 80.0]
        + [80.0] * 5
        + [90.0, 110.0, 130.0, 150.0, 160.0, 168.0]
        + [168.0] * 3
    )
    return vals


# --- 1. Valid complete repetition -------------------------------------------
def t_valid_rep():
    cfg = default_config()
    seq = make_angle_sequence(full_rep())
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 1, f"expected 1 rep, got {analysis.total_reps}"
    rep = analysis.repetitions[0]
    assert rep.rep_number == 1
    assert rep.start_timestamp < rep.bottom_timestamp < rep.end_timestamp
    assert rep.min_angle <= 90.0
    assert rep.max_angle >= 160.0


# --- 2. Incomplete movement should not count --------------------------------
def t_incomplete():
    cfg = default_config()
# Extends but never returns all the way to start (ends at 130 < 160).
    vals = full_rep()[:-7]  # drop the final return; ends around 130
    seq = make_angle_sequence(vals)
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 0, f"expected 0 reps, got {analysis.total_reps}"


# --- 3. Noisy movement should not count multiple reps -----------------------
def t_noisy():
    cfg = default_config()
    # A single dip with slight jitter around the bottom should count exactly once.
    vals = [170.0] * 4 + [160.0, 150.0, 120.0, 95.0, 85.0, 88.0, 82.0, 84.0, 90.0,
                          110.0, 130.0, 150.0, 162.0] + [168.0] * 3
    seq = make_angle_sequence(vals)
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 1, f"expected 1 rep, got {analysis.total_reps}"


# --- 4. Minimum duration -----------------------------------------------------
def t_min_duration():
    cfg = default_config(min_rep_duration=2.0)  # rep must be >= 2s
    seq = make_angle_sequence(full_rep())  # ~1s rep -> too short under 2s
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 0, f"expected 0 reps (too short), got {analysis.total_reps}"


# --- 5. Missing angle values ------------------------------------------------
def t_missing():
    cfg = default_config()
    # Some frames missing (None). The rep should still be detected but the
    # missing frames must simply be skipped (not invented).
    vals = full_rep()
    vals[2] = None
    vals[8] = None
    vals[16] = None
    seq = make_angle_sequence(vals)
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 1, f"expected 1 rep despite missing values, got {analysis.total_reps}"


# --- 6. Correct phase ordering ----------------------------------------------
def t_phase_order():
    cfg = default_config()
    seq = make_angle_sequence(full_rep())
    analysis = count_reps(seq, config=cfg)
    phases = [p.phase for p in analysis.phases]
    # Expect at least: flexing -> bottom -> extending -> start
    assert PHASE_FLEXING in phases
    assert PHASE_BOTTOM in phases
    assert PHASE_EXTENDING in phases
    assert phases[0] == PHASE_FLEXING
    b_idx = phases.index(PHASE_BOTTOM)
    e_idx = phases.index(PHASE_EXTENDING)
    assert b_idx < e_idx, "bottom must precede extending"


# --- 7. Correct timestamp preservation --------------------------------------
def t_timestamps():
    cfg = default_config()
    fps = 30.0
    # full_rep values: [170]*5, then 160,150,130,110,95,85,80, then [80]*5,
    # then 90,110,130,150,160,168, then [168]*3
    seq = make_angle_sequence(full_rep(), fps=fps)
    analysis = count_reps(seq, config=cfg)
    rep = analysis.repetitions[0]
    # start = first flex crossing (value 85 <= 90) at index 10
    assert rep.start_frame == 10
    assert abs(rep.start_timestamp - 10 / fps) < 1e-6
    # bottom = first 80 at index 11
    assert rep.bottom_frame == 11
    assert abs(rep.bottom_timestamp - 11 / fps) < 1e-6
    # end = return to extended (value 160 >= 160) at index 21
    assert rep.end_frame == 21
    assert abs(rep.end_timestamp - 21 / fps) < 1e-6
    assert rep.start_timestamp < rep.bottom_timestamp < rep.end_timestamp


# --- 8. Minimum movement range ----------------------------------------------
def t_min_range():
    cfg = default_config(min_movement_range=100.0)  # must traverse 100 deg
    # full_rep spans ~170-80 = 90 deg -> below requirement -> rejected.
    seq = make_angle_sequence(full_rep())
    analysis = count_reps(seq, config=cfg)
    assert analysis.total_reps == 0, f"expected 0 reps (range too small), got {analysis.total_reps}"


# --- 9. Bilateral angle selection -------------------------------------------
def t_bilateral_selection():
    # left_elbow has many valid frames; right_elbow has sparse valid frames.
    frames = []
    for i in range(20):
        left = 170 if i < 10 else 80
        right = 170 if i % 4 == 0 else None  # mostly missing
        frames.append(
            AngleFrame(frame_index=i, timestamp=i / 30.0,
                       angles={"left_elbow": left, "right_elbow": right})
        )
    seq = AngleSequence(fps=30.0, frame_count=20, frames=frames)
    chosen = select_primary_angle(seq, preferred="left_elbow", candidates=["left_elbow", "right_elbow"])
    assert chosen == "left_elbow", f"expected left_elbow, got {chosen}"


def run_all() -> None:
    print("STEP 3 PHASE/REP DETECTOR TESTS")
    print("=" * 50)
    run_test("valid complete repetition", t_valid_rep)
    run_test("incomplete movement not counted", t_incomplete)
    run_test("noisy movement counts once", t_noisy)
    run_test("minimum duration", t_min_duration)
    run_test("missing angle values", t_missing)
    run_test("correct phase ordering", t_phase_order)
    run_test("correct timestamp preservation", t_timestamps)
    run_test("minimum movement range", t_min_range)
    run_test("bilateral angle selection", t_bilateral_selection)
    print("=" * 50)


if __name__ == "__main__":
    run_all()
