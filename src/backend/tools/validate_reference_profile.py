"""Real-video validation tool for the OrthoRehab AI Step 4 reference profile.

Runs the complete pipeline on the therapist's reference video:

    reference_video.mp4
      -> Step 1: extract_landmark_sequence()       (existing)
      -> Step 2: AngleEngine.sequence_angles()     (existing)
      -> Step 3: select_primary_angle + count_reps (existing)
      -> Step 4: build_reference_profile()         (new)

It writes a compact, reusable JSON profile to
``backend/output/reference_exercise_profile.json`` and prints a readable summary
showing: exercise, primary joint, movement type, reference repetitions, angle
statistics, phase sequence, normalized trajectory, required landmarks and source
video metadata.

The original reference video is never modified. The profile stores only useful
information (no raw video / no raw MediaPipe frames).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleEngine
from src.pose.exercise_config import config_from_angle_sequence
from src.pose.landmark_extractor import extract_landmark_sequence
from src.pose.phase_rep_counter import count_reps, select_primary_angle
from src.pose.reference_profile_builder import build_reference_profile
from src.pose.video_processor import read_video_info

# Primary-movement candidate angles for this exercise, in preference order.
# Elbow angles are preferred because shoulder angles are sparse in the cropped
# reference video (matches the Step 3 validation tool).
CANDIDATE_ANGLES = ["left_elbow", "right_elbow"]

DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output", "reference_exercise_profile.json")
)


def run(input_video: str, output_path: str = None) -> int:
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    print("=" * 72)
    print("ORTHOREHAB AI - STEP 4: REFERENCE EXERCISE PROFILE VALIDATION")
    print("=" * 72)
    print(f"Input video : {input_video}")
    print(f"Output JSON : {output_path}")

    # ---------- Step 1 ----------
    print("\n[1/5] Running Step 1 landmark extraction (existing)...")
    seq = extract_landmark_sequence(input_video)
    print(f"  fps={seq.fps:.2f}, frame_count={seq.frame_count}, pose_frames={seq.pose_frames}")
    if not seq.frames:
        print("ERROR: no valid pose frames.")
        return 1
    info = read_video_info(input_video)

    # ---------- Step 2 ----------
    print("\n[2/5] Running Step 2 angle engine (existing)...")
    engine = AngleEngine()
    angle_seq = engine.sequence_angles(seq)
    print(f"  frames: {angle_seq.frame_count}")

    primary = select_primary_angle(angle_seq, preferred="left_elbow",
                                   candidates=CANDIDATE_ANGLES)
    print(f"  primary angle chosen: {primary}")

    # ---------- Step 3 ----------
    print("\n[3/5] Running Step 3 phase/rep counting (existing)...")
    config = config_from_angle_sequence(
        angle_seq,
        exercise="elbow_flexion",
        primary_angle=primary,
        movement_type="flexion_extension",
    )
    analysis = count_reps(angle_seq, config=config)
    print(f"  exercise={config.exercise}, movement={config.movement_type}")
    print(f"  total_reps={analysis.total_reps}")

    # ---------- Step 4 ----------
    print("\n[4/5] Building reference exercise profile (new)...")
    profile = build_reference_profile(
        angle_sequence=angle_seq,
        analysis=analysis,
        config=config,
        landmark_sequence=seq,
        exercise_id="elbow_flexion",
        exercise_name="Elbow Flexion / Extension",
        source_video_metadata={
            "path": info.path,
            "fps": info.fps,
            "frame_count": seq.frame_count,
            "pose_frames": seq.pose_frames,
            "width": info.width,
            "height": info.height,
        },
    )

    # ---------- persist ----------
    print("\n[5/5] Writing JSON profile...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"  wrote {output_path}")

    # ---------- readable summary ----------
    print("\n" + "=" * 72)
    print("REFERENCE EXERCISE PROFILE SUMMARY")
    print("=" * 72)
    print(f"Exercise            : {profile.exercise_name} ({profile.exercise_id})")
    print(f"Primary joint       : {profile.primary_angle}")
    print(f"Movement type       : {profile.movement_type}")
    print(f"Reference repetitions : {len(profile.repetitions)}")
    if profile.primary_min_angle is not None:
        print(f"Angle statistics    : min={profile.primary_min_angle}, "
              f"max={profile.primary_max_angle}, mean={profile.primary_mean_angle}, "
              f"range={profile.primary_range}")

    if profile.phase_profile:
        print(f"Phase sequence      : {' -> '.join(profile.phase_profile.sequence)}")
        print(f"  phase events      : {len(profile.phase_profile.events)}")

    print("\nRepetitions:")
    for r in profile.repetitions:
        print(f"  rep {r.rep_number}: start t={r.start_timestamp:.3f}s f{r.start_frame}, "
              f"bottom t={r.bottom_timestamp:.3f}s f{r.bottom_frame}, "
              f"end t={r.end_timestamp:.3f}s f{r.end_frame}, "
              f"min={r.min_angle:.1f} max={r.max_angle:.1f}, dur={r.duration:.3f}s")

    traj = profile.reference_trajectory
    if traj:
        print("\nNormalized trajectory "
              f"(source_reps_used={traj.source_reps_used}, {traj.num_points} points):")
        # Print a compact subset of every 5th point for readability.
        for i, p in enumerate(traj.points):
            if i % 5 == 0 or i == traj.num_points - 1:
                print(f"  {int(round(p.progress * 100)):>3}% -> {p.angle:.2f} deg")

    print(f"\nRequired landmarks : {profile.supported_landmarks}")
    print(f"Available landmarks: {profile.available_landmarks}")

    sv = profile.source_video
    print("\nSource video metadata:")
    print(f"  path        : {sv.path}")
    print(f"  fps         : {sv.fps}")
    print(f"  frame_count : {sv.frame_count}")
    print(f"  pose_frames : {sv.pose_frames}")
    print(f"  resolution  : {sv.width}x{sv.height}")

    print("\n" + "=" * 72)
    ok = len(profile.repetitions) > 0 and traj is not None and profile.primary_angle
    if ok:
        print("STEP 4 RESULT: SUCCESS (valid reusable reference profile generated)")
    else:
        print("STEP 4 RESULT: FAILED (profile missing repetitions or trajectory)")
    print("=" * 72)
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(run(in_path, out_path))
