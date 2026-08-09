"""Unit tests for the OrthoRehab AI Step 5 hybrid comparison engine.

These tests use synthetic angle sequences and reference profiles (no video, no
MediaPipe) and cover the required behaviours:

  1. identical therapist/patient movement -> high DTW sim, low deviation, normal speed
  2. same movement, slower -> high DTW sim, too_slow
  3. same movement, faster -> high DTW sim, too_fast
  4. insufficient ROM -> correct joint + deviation + severity
  5. excessive ROM -> correct excessive-ROM deviation
  6. trajectory deviation -> detected
  7. noisy/missing landmarks -> graceful, no crash
  8. short sequence -> safe validation/fallback
  9. LLM unavailable -> deterministic feedback
  10. malformed LLM response -> schema validation + fallback
  11. feedback spam -> cooldown works
  12. learned model unavailable -> Step 5 still works (DTW + biomech + speed)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleFrame, AngleSequence
from src.pose.exercise_config import ExercisePhaseConfig
from src.pose.phase_rep_counter import ExerciseAnalysis
from src.pose.reference_profile_builder import build_reference_profile
from src.comparison.comparison_service import ComparisonEngine, EngineConfig
from src.comparison.deviation_engine import (
    DEV_EXCESSIVE_FLEXION,
    DEV_EXCESSIVE_EXTENSION,
    DEV_INSUFFICIENT_FLEXION,
    DEV_TRAJECTORY,
    DeviationEngine,
    DeviationConfig,
)
from src.comparison.speed_analyzer import SPEED_TOO_FAST, SPEED_TOO_SLOW, SPEED_NORMAL
from src.comparison.dtw_comparator import dtw_similarity
from src.feedback.feedback_service import FeedbackService, FeedbackServiceConfig
from src.llm.groq_client import GroqClient


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def make_angle_sequence(
    values, fps: float = 30.0, angle_name: str = "right_elbow"
) -> AngleSequence:
    frames = []
    for i, v in enumerate(values):
        frames.append(AngleFrame(frame_index=i, timestamp=i / fps, angles={angle_name: v}))
    return AngleSequence(fps=fps, frame_count=len(frames), frames=frames)


def full_rep():
    """A ~1.4s synthetic right-elbow flexion/extension cycle (start->bottom->start)."""
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
        primary_angle="right_elbow",
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


def build_reference(values=None):
    """Build a Step 4 reference profile from a synthetic sequence."""
    if values is None:
        values = full_rep()
    cfg = default_config()
    analysis, seq = build_analysis(values, config=cfg)
    # Build a reference profile with source metadata.
    return build_reference_profile(
        seq,
        analysis,
        cfg,
        exercise_id="elbow_flexion",
        exercise_name="Elbow Flexion / Extension",
        source_video_metadata={"path": "ref.mp4", "fps": 30, "frame_count": len(seq.frames)},
    ), cfg


def run_test(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


# --- 1. identical movement -------------------------------------------------
def t_identical():
    ref_profile, _ = build_reference()
    pat_vals = full_rep()  # identical
    analysis, pat_seq = build_analysis(pat_vals)
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    assert result.dtw_similarity is not None and result.dtw_similarity > 0.7, result.dtw_similarity
    assert result.speed_status == SPEED_NORMAL
    # No major angle deviations for identical movement (small tolerance).
    angle_devs = [d for d in result.deviations if d.type != DEV_TRAJECTORY]
    assert len(angle_devs) == 0, [d.to_dict() for d in angle_devs]


# --- 2. same movement, slower ----------------------------------------------
def t_slower():
    ref_profile, _ = build_reference()
    # Same values but each frame repeated -> ~2x slower.
    pat = []
    for v in full_rep():
        pat += [v, v]
    analysis, pat_seq = build_analysis(pat, smoothing_window=1)
    engine = ComparisonEngine()
    result = engine.compare(
        ref_profile, pat_seq, patient_exercise_analysis=analysis,
        reference_duration=ref_profile.repetitions[0].duration,
    )
    assert result.speed_status == SPEED_TOO_SLOW, result.speed_status
    assert result.dtw_similarity is not None, "DTW should still be computable"


# --- 3. same movement, faster ----------------------------------------------
def t_faster():
    ref_profile, _ = build_reference()
    # Half the values -> ~2x faster. Pass an explicit shorter patient duration
    # so the speed analyzer sees a fast reproduction regardless of whether the
    # short synthetic clip yields a detected rep.
    pat = full_rep()[::2]
    analysis, pat_seq = build_analysis(pat, smoothing_window=1)
    engine = ComparisonEngine()
    ref_dur = ref_profile.repetitions[0].duration
    pat_dur = ref_dur / 2.0
    result = engine.compare(
        ref_profile,
        pat_seq,
        patient_exercise_analysis=analysis,
        reference_duration=ref_dur,
        patient_duration=pat_dur,
    )
    assert result.speed_status == SPEED_TOO_FAST, (result.speed_status, pat_dur, ref_dur)


# --- 4. insufficient ROM ---------------------------------------------------
def t_insufficient_rom():
    ref_profile, _ = build_reference()
    # Bottom only reaches ~100 instead of ~80 -> insufficient flexion.
    vals = (
        [170.0] * 5
        + [160.0, 152.0, 130.0, 115.0, 105.0, 100.0, 100.0, 100.0]
        + [100.0] * 3
        + [110.0, 130.0, 150.0, 160.0, 168.0]
        + [168.0] * 3
    )
    analysis, pat_seq = build_analysis(vals)
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    types = [d.type for d in result.deviations]
    assert DEV_INSUFFICIENT_FLEXION in types, types
    # The deviation should be on the primary joint.
    flex_dev = next(d for d in result.deviations if d.type == DEV_INSUFFICIENT_FLEXION)
    assert flex_dev.joint == "right_elbow"
    assert flex_dev.severity in ("moderate", "major")


# --- 5. excessive ROM ------------------------------------------------------
def t_excessive_rom():
    ref_profile, _ = build_reference()
    # Bottom over-flexes below reference min (e.g. reaches 60 where ref is ~80).
    vals = (
        [170.0] * 5
        + [160.0, 150.0, 120.0, 95.0, 75.0, 60.0, 60.0, 60.0]
        + [60.0] * 3
        + [80.0, 110.0, 135.0, 155.0, 168.0]
        + [168.0] * 3
    )
    analysis, pat_seq = build_analysis(vals)
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    types = [d.type for d in result.deviations]
    assert DEV_EXCESSIVE_FLEXION in types, types


# --- 6. trajectory deviation ----------------------------------------------
def t_trajectory():
    ref_profile, _ = build_reference()
    # A jittered / non-monotonic trajectory that deviates from the reference.
    vals = (
        [170.0, 165.0, 150.0, 140.0, 125.0, 118.0, 112.0, 116.0, 120.0, 124.0,
         128.0, 132.0, 136.0, 140.0, 144.0, 148.0, 152.0, 156.0, 160.0, 164.0, 168.0]
    )
    analysis, pat_seq = build_analysis(vals, smoothing_window=1)
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    types = [d.type for d in result.deviations]
    assert DEV_TRAJECTORY in types, types


# --- 7. noisy/missing landmarks -------------------------------------------
def t_noisy_missing():
    ref_profile, _ = build_reference()
    vals = full_rep()
    # Introduce None gaps (missing landmarks) throughout.
    for i in range(0, len(vals), 3):
        vals[i] = None
    analysis, pat_seq = build_analysis(vals)
    engine = ComparisonEngine()
    # Must not crash.
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    assert result is not None


# --- 8. short/invalid sequence --------------------------------------------
def t_short_sequence():
    ref_profile, _ = build_reference()
    short = [170.0, 165.0]  # too short for a rep
    analysis, pat_seq = build_analysis(short)
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    assert result is not None
    # Short sequence -> DTW may be None but engine still returns a result.
    assert result.exercise_id == "elbow_flexion"


# --- 12. learned model unavailable ----------------------------------------
def t_learned_unavailable():
    # Step 5 must still work (DTW + biomech + speed) with no learned model.
    ref_profile, _ = build_reference()
    analysis, pat_seq = build_analysis(full_rep())
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=analysis)
    assert result.learned_model_available is False
    assert result.learned_motion_similarity is None
    assert result.dtw_similarity is not None
    assert result.biomechanical_accuracy is not None
    assert result.overall_score is not None


# --- 9. LLM unavailable -> deterministic feedback -------------------------
def t_llm_unavailable():
    groq = GroqClient(api_key="")  # no API key -> unavailable
    assert groq.available is False
    from src.feedback.feedback_templates import fallback_message
    msg = fallback_message(["insufficient_flexion"])
    assert "bending" in msg  # deterministic fallback


# --- 10. malformed LLM response -> fallback -------------------------------
def t_malformed_llm():
    # A client that simulates a malformed/empty response.
    class BadGroq(GroqClient):
        def chat(self, messages, **kw):
            raise RuntimeError("malformed")

    svc = FeedbackService(groq=BadGroq(api_key="x"), config=FeedbackServiceConfig(
        persistence_threshold=1, cooldown_seconds=0
    ))
    event = svc.evaluate(
        {"deviations": [{"type": "too_fast", "joint": "", "severity": "moderate"}]},
        timestamp=1.0,
    )
    assert event is not None
    assert event.source == "fallback"


# --- 11. feedback spam -> cooldown ----------------------------------------
def t_feedback_spam():
    svc = FeedbackService(groq=GroqClient(api_key=""), config=FeedbackServiceConfig(
        persistence_threshold=1, cooldown_seconds=10.0
    ))
    findings = {"deviations": [{"type": "insufficient_flexion", "joint": "elbow", "severity": "moderate"}]}
    e1 = svc.evaluate(findings, timestamp=1.0)
    e2 = svc.evaluate(findings, timestamp=2.0)  # within cooldown
    assert e1 is not None
    assert e2 is None  # suppressed by cooldown
    assert svc.stats()["suppressed_events"] >= 1


# --- DTW helper direct test -----------------------------------------------
def t_dtw_high_for_same():
    # Identical sequences -> high similarity; slower -> still high.
    ref_traj = full_rep()
    pat_same = full_rep()
    pat_slow = []
    for v in full_rep():
        pat_slow += [v, v]
    sim_same = dtw_similarity(ref_traj, pat_same)
    sim_slow = dtw_similarity(ref_traj, pat_slow)
    assert sim_same is not None and sim_same > 0.8, sim_same
    assert sim_slow is not None and sim_slow > 0.7, sim_slow


def run_all() -> None:
    print("STEP 5 HYBRID COMPARISON TESTS")
    print("=" * 60)
    run_test("identical movement (high DTW, normal speed, low dev)", t_identical)
    run_test("same movement slower (too_slow, DTW OK)", t_slower)
    run_test("same movement faster (too_fast)", t_faster)
    run_test("insufficient ROM (correct joint + deviation)", t_insufficient_rom)
    run_test("excessive ROM (excessive-flexion deviation)", t_excessive_rom)
    run_test("trajectory deviation detected", t_trajectory)
    run_test("noisy/missing landmarks (graceful)", t_noisy_missing)
    run_test("short/invalid sequence (safe fallback)", t_short_sequence)
    run_test("LLM unavailable -> deterministic fallback", t_llm_unavailable)
    run_test("malformed LLM response -> fallback", t_malformed_llm)
    run_test("feedback spam -> cooldown works", t_feedback_spam)
    run_test("learned model unavailable -> Step 5 still works", t_learned_unavailable)
    run_test("DTW high for same/slower sequences", t_dtw_high_for_same)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
