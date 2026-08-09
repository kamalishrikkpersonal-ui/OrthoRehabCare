"""Real-video validation tool for the OrthoRehab AI Step 5 comparison.

Runs the complete pipeline end-to-end on a therapist reference video and a
patient video:

    reference_video.mp4
      -> Step 1: extract_landmark_sequence()       (existing)
      -> Step 2: AngleEngine.sequence_angles()     (existing)
      -> Step 3: select_primary_angle + count_reps (existing)
      -> Step 4: build_reference_profile()         (existing)
      -> Step 5: Step5Pipeline.compare_video()     (new)

It writes a compact JSON result to
``backend/output/step5_comparison_result.json`` and prints a readable summary
showing the transparent component metrics (DTW similarity, biomechanical
accuracy, ROM accuracy, speed status, deviations and the overall score).

If the recording cannot be analyzed (capture-quality failure), it reports the
structured capture-quality error instead of a misleading low score.

The comparison is clinically honest: it compares the patient's movement against
the therapist's demonstrated reference. It does NOT diagnose or claim clinical
correctness.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.step5_pipeline import Step5Pipeline

DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output", "step5_comparison_result.json")
)


def run(reference_video: str, patient_video: str, output_path: str = None) -> int:
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    print("=" * 72)
    print("ORTHOREHAB AI - STEP 5: HYBRID COMPARISON VALIDATION")
    print("=" * 72)
    print(f"Reference video : {reference_video}")
    print(f"Patient video   : {patient_video}")
    print(f"Output JSON     : {output_path}")

    # Build the reference profile from the therapist video (Steps 1-4, existing).
    print("\n[1/2] Building reference profile from therapist video (Steps 1-4)...")
    pipeline = Step5Pipeline()
    profile = pipeline.build_reference_from_video(reference_video)
    print(f"  exercise={profile.exercise_name}, primary={profile.primary_angle}, "
          f"reps={len(profile.repetitions)}")

    # Compare the patient video (Step 5).
    print("\n[2/2] Comparing patient video (Step 5)...")
    result = pipeline.compare_video(patient_video)

    timings = result.timings_ms

    # Handle a capture-quality failure: the recording cannot be analyzed.
    if result.status == "capture_quality_failed":
        err = result.capture_quality_error
        print("\n  status            : capture_quality_failed")
        print(f"  scored            : {result.scored}")
        if err is not None:
            print(f"  code              : {err.code}")
            print(f"  message           : {err.message}")
            print(f"  user_instruction  : {err.user_instruction}")
            print(f"  missing_landmarks : {err.missing_landmarks}")
            print(f"  valid_angle_ratio : {err.valid_angle_ratio}")
            print(f"  frames            : {err.valid_frames}/{err.total_frames} usable")
        print("\n  Timings (ms):")
        for k, v in sorted(timings.items()):
            print(f"    {k}: {v:.2f}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": result.status,
                    "scored": result.scored,
                    "capture_quality_error": err.to_dict() if err else None,
                    "timings_ms": timings,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n  wrote {output_path}")
        print("\n" + "=" * 72)
        print("STEP 5 RESULT: CAPTURE_QUALITY_FAILED (recording not analyzable)")
        print("=" * 72)
        return 0

    print(f"\n  overall_score           : {result.comparison.overall_score}")
    print(f"  learned_motion_similarity : {result.comparison.learned_motion_similarity}")
    print(f"  dtw_similarity           : {result.comparison.dtw_similarity}")
    print(f"  biomechanical_accuracy   : {result.comparison.biomechanical_accuracy}")
    print(f"  rom_accuracy             : {result.comparison.rom_accuracy}")
    print(f"  reference_duration       : {result.comparison.reference_duration}s")
    print(f"  patient_duration         : {result.comparison.patient_duration}s")
    print(f"  speed_ratio              : {result.comparison.speed_ratio}")
    print(f"  speed_status             : {result.comparison.speed_status}")
    print(f"  repetitions              : {len(result.comparison.repetitions)}")

    print("\n  Deviations:")
    if result.comparison.deviations:
        for d in result.comparison.deviations:
            print(f"    - {d.type} ({d.joint}) ref={d.reference_value} "
                  f"pat={d.patient_value} dev={d.deviation} sev={d.severity}")
    else:
        print("    - none")

    print("\n  Patient feedback:", result.report.summary.patient_feedback)

    print("\n  Timings (ms):")
    for k, v in sorted(timings.items()):
        print(f"    {k}: {v:.2f}")

    # Persist.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": result.status,
                "scored": result.scored,
                "comparison": result.comparison.to_dict(),
                "report": result.report.to_dict(),
                "timings_ms": timings,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  wrote {output_path}")

    print("\n" + "=" * 72)
    print("STEP 5 RESULT: SUCCESS (hybrid comparison completed)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python -m tools.validate_step5_comparison <reference.mp4> <patient.mp4>")
        sys.exit(2)
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    raise SystemExit(run(sys.argv[1], sys.argv[2], out_path))
