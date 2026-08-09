"""End-to-end (synthetic) tests for the OrthoRehab AI Step 5 pipeline.

These tests exercise the full chain with synthetic angle sequences (no video,
no MediaPipe): build reference -> run comparison engine -> generate session
report -> verify JSON round-trip of the result schema.

Also verifies the LearnedMotionComparator extension point exists and that the
pipeline works without a learned model.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleFrame, AngleSequence
from src.pose.exercise_config import ExercisePhaseConfig
from src.pose.phase_rep_counter import ExerciseAnalysis
from src.pose.reference_profile_builder import build_reference_profile
from src.comparison.comparison_service import ComparisonEngine
from src.comparison.learned_motion_comparator import (
    LearnedMotionComparator,
    UnavailableLearnedComparator,
)
from src.session_report import build_session_report


def make_angle_sequence(values, fps=30.0, angle_name="right_elbow"):
    frames = []
    for i, v in enumerate(values):
        frames.append(AngleFrame(frame_index=i, timestamp=i / fps, angles={angle_name: v}))
    return AngleSequence(fps=fps, frame_count=len(frames), frames=frames)


def full_rep():
    return (
        [170.0] * 5
        + [160.0, 150.0, 130.0, 110.0, 95.0, 85.0, 80.0]
        + [80.0] * 5
        + [90.0, 110.0, 130.0, 150.0, 160.0, 168.0]
        + [168.0] * 3
    )


def default_config():
    return ExercisePhaseConfig(
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


def build_analysis(values):
    cfg = default_config()
    seq = make_angle_sequence(values)
    from src.pose.phase_detector import PhaseDetector

    summary = PhaseDetector(cfg).detect(seq)
    return ExerciseAnalysis(
        exercise=cfg.exercise,
        primary_angle=cfg.primary_angle,
        total_reps=len(summary.repetitions),
        phases=summary.phases,
        repetitions=summary.repetitions,
        min_angle=summary.min_angle,
        max_angle=summary.max_angle,
        valid_frames=summary.valid_frames,
    ), seq, cfg


def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def t_full_pipeline():
    analysis, seq, cfg = build_analysis(full_rep())
    ref_profile = build_reference_profile(
        seq, analysis, cfg,
        exercise_id="elbow_flexion",
        exercise_name="Elbow Flexion / Extension",
        source_video_metadata={"path": "ref.mp4", "fps": 30, "frame_count": len(seq.frames)},
    )
    # Patient identical.
    pat_analysis, pat_seq, _ = build_analysis(full_rep())
    engine = ComparisonEngine()
    result = engine.compare(ref_profile, pat_seq, patient_exercise_analysis=pat_analysis)
    report = build_session_report(result, groq=None, timings_ms={"total": 5.0})

    assert result.overall_score is not None
    assert report.summary.total_repetitions == 1
    assert report.summary.patient_feedback  # deterministic fallback since no LLM key
    assert report.summary.therapist_report  # therapist report generated
    assert result.learned_model_available is False


def t_therapist_report_llm_fallback():
    analysis, seq, cfg = build_analysis(full_rep())
    ref_profile = build_reference_profile(seq, analysis, cfg)
    pat_analysis, pat_seq, _ = build_analysis(full_rep())
    result = ComparisonEngine().compare(ref_profile, pat_seq, patient_exercise_analysis=pat_analysis)

    class FixedGroq(GroqClient):
        def __init__(self):
            super().__init__(api_key="x")

        def chat(self, messages, **kwargs):
            return "The patient moved quickly and under-extended the elbow; slow down and increase extension."

    report = build_session_report(result, groq=FixedGroq(), timings_ms={"total": 5.0})
    assert "slow down" in report.summary.therapist_report


def t_json_roundtrip():
    analysis, seq, cfg = build_analysis(full_rep())
    ref_profile = build_reference_profile(seq, analysis, cfg)
    pat_analysis, pat_seq, _ = build_analysis(full_rep())
    result = ComparisonEngine().compare(ref_profile, pat_seq, patient_exercise_analysis=pat_analysis)
    data = result.to_dict()
    s = json.dumps(data, default=str)
    loaded = json.loads(s)
    restored = result.__class__.from_dict(loaded)
    assert restored.to_dict() == result.to_dict()
    assert restored.exercise_id == "elbow_flexion"


def t_learned_interface_exists():
    # The extension point must exist and be importable.
    assert LearnedMotionComparator is not None
    u = UnavailableLearnedComparator()
    assert u.available() is False
    assert u.compare([1, 2], [3, 4]) is None
    assert u.result().available is False


def t_no_lower_body_introduced():
    # Verify the supported landmarks in the reference profile are hip-and-above only.
    analysis, seq, cfg = build_analysis(full_rep())
    ref_profile = build_reference_profile(seq, analysis, cfg)
    lower = {"knee", "ankle", "heel", "foot", "shin", "calf"}
    supported = set(ref_profile.supported_landmarks)
    assert not (supported & lower), f"Lower-body landmarks found: {supported & lower}"
    # The only landmarks referenced by the default angle definitions are upper-body.
    from src.pose.angle_engine import DEFAULT_ANGLE_DEFINITIONS
    for angle, names in DEFAULT_ANGLE_DEFINITIONS.items():
        for n in names:
            assert "knee" not in n and "ankle" not in n and "foot" not in n and "heel" not in n


def run_all():
    print("STEP 5 END-TO-END (SYNTHETIC) TESTS")
    print("=" * 60)
    run_test("full pipeline (reference -> compare -> report)", t_full_pipeline)
    run_test("JSON round-trip of comparison result", t_json_roundtrip)
    run_test("LearnedMotionComparator extension point exists", t_learned_interface_exists)
    run_test("no lower-body landmarks introduced", t_no_lower_body_introduced)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
