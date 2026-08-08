"""Real-video validation tool for the OrthoRehab AI step 3 phase/rep counter.

Runs the complete pipeline on a real reference video:

    reference_video.mp4
      -> Step 1: extract_landmark_sequence()       (existing, upper-body filter)
      -> Step 2: AngleEngine.sequence_angles()     (existing)
      -> Step 3: count_reps() on the AngleSequence (new)

It reports:
  * video FPS, total frames, valid angle frames
  * selected primary angle + why
  * min/max of the primary angle
  * detected phases (frame + timestamp + angle)
  * number of repetitions
  * each rep's start/bottom/end timestamps and min/max angle
  * an explicit success/failure

It also writes an annotated diagnostic video
(``backend/output/reference_phase_rep_annotated.mp4``) showing:

  * upper-body landmarks + skeleton
  * primary joint angle
  * current phase
  * current repetition count

The original reference video is never modified.
"""

from __future__ import annotations

import os
import sys

import cv2

# Make package importable when run from the backend directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleEngine
from src.pose.exercise_config import config_from_angle_sequence
from src.pose.landmark_extractor import extract_landmark_sequence
from src.pose.phase_rep_counter import count_reps, select_primary_angle
from src.pose.video_processor import iter_video_frames, read_video_info

# Primary-movement candidate angles for this exercise, in preference order.
# Shoulder angles are sparse in this cropped video, so elbow angles are preferred.
CANDIDATE_ANGLES = ["left_elbow", "right_elbow"]

# Upper-body connection graph for the skeleton overlay (no legs/feet).
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

SHORT_NAMES = {
    "nose": "nose", "left_eye": "Leye", "right_eye": "Reye",
    "left_ear": "Lear", "right_ear": "Rear", "left_shoulder": "Lsh",
    "right_shoulder": "Rsh", "left_elbow": "Lel", "right_elbow": "Rel",
    "left_wrist": "Lwr", "right_wrist": "Rwr", "left_hip": "Lhip",
    "right_hip": "Rhip",
}


def draw_annotations(frame, pixel_map, angle_val, phase, rep, frame_idx, timestamp):
    """Draw landmarks, skeleton, and a phase/rep/angle panel on a BGR frame."""
    vis = frame.copy()

    for a, b in CONNECTIONS:
        if a in pixel_map and b in pixel_map:
            xa, ya, _ = pixel_map[a]
            xb, yb, _ = pixel_map[b]
            cv2.line(vis, (xa, ya), (xb, yb), (0, 255, 0), 1)
    for name, (px, py, _v) in pixel_map.items():
        cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
        short = SHORT_NAMES.get(name, name)
        cv2.putText(vis, short, (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-left info panel
    y0 = 20
    cv2.putText(vis, f"frame={frame_idx}  t={timestamp:.2f}s", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y0 += 22
    angle_text = f"ANGLE: {angle_val:.1f}" if angle_val is not None else "ANGLE: n/a"
    cv2.putText(vis, angle_text, (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    y0 += 22
    cv2.putText(vis, f"PHASE: {phase.upper()}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
    y0 += 22
    cv2.putText(vis, f"REP: {rep}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def per_frame_phase_and_rep(angle_seq, analysis):
    """Build two maps frame_index -> current phase, current rep number."""
    phase_map = {}
    rep_map = {}
    events = sorted(analysis.phases, key=lambda e: e.frame_index)

    current_phase = "start"
    ev_idx = 0
    for af in angle_seq.frames:
        while ev_idx < len(events) and events[ev_idx].frame_index <= af.frame_index:
            current_phase = events[ev_idx].phase
            ev_idx += 1
        current_rep = sum(1 for r in analysis.repetitions if r.end_frame <= af.frame_index)
        phase_map[af.frame_index] = current_phase
        rep_map[af.frame_index] = current_rep
    return phase_map, rep_map


def create_annotated_video(input_path, output_path, angle_seq, norm_cache, analysis):
    """Write an annotated copy of the input video using cached landmarks."""
    info = read_video_info(input_path)
    fps = float(info.fps) if info.fps > 0 else 30.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    size = (info.width, info.height)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    phase_at, rep_at = per_frame_phase_and_rep(angle_seq, analysis)
    angles = {af.frame_index: af.angles for af in angle_seq.frames}

    n_frames = 0
    for vf in iter_video_frames(info.path):
        norm = norm_cache.get(vf.frame_index, {})
        pixel_map = normalized_to_pixel(norm, info.width, info.height)
        angle_val = angles.get(vf.frame_index, {}).get(analysis.primary_angle)
        phase = phase_at.get(vf.frame_index, "start")
        rep = rep_at.get(vf.frame_index, 0)
        annotated = draw_annotations(vf.image, pixel_map, angle_val, phase, rep,
                                     vf.frame_index, vf.timestamp)
        writer.write(annotated)
        n_frames += 1
    writer.release()
    print(f"Annotated phase/rep video written to: {output_path}  ({n_frames} frames)")


def normalized_to_pixel(norm_map, width, height):
    """Convert {name: (x_norm, y_norm, vis)} to {name: (x_px, y_px, vis)}."""
    out = {}
    for name, (nx, ny, vis) in norm_map.items():
        px = max(0, min(width - 1, int(nx * width)))
        py = max(0, min(height - 1, int(ny * height)))
        out[name] = (px, py, vis)
    return out


def run(input_video, output_video=None) -> int:
    if output_video is None:
        output_video = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output",
                         "reference_phase_rep_annotated.mp4")
        )

    print("=" * 70)
    print("ORTHOREHAB AI - STEP 3: REAL-VIDEO PHASE/REP COUNTER VALIDATION")
    print("=" * 70)
    print(f"Input video : {input_video}")
    print(f"Output video: {output_video}")

    # --- Step 1 ---
    print("\n[1/4] Running Step 1 landmark extraction (existing)...")
    seq = extract_landmark_sequence(input_video)
    print(f"  fps={seq.fps:.2f}, frame_count={seq.frame_count}, pose_frames={seq.pose_frames}")
    if not seq.frames:
        print("ERROR: no valid pose frames.")
        return 1

    norm_cache = {}
    for pf in seq.frames:
        norm_cache[pf.frame_index] = {lm.name: (lm.x, lm.y, lm.visibility)
                                      for lm in pf.landmarks}

    # --- Step 2 ---
    print("\n[2/4] Running Step 2 angle engine (existing)...")
    engine = AngleEngine()
    angle_seq = engine.sequence_angles(seq)
    print(f"  frames: {angle_seq.frame_count}")
    primary = select_primary_angle(angle_seq, preferred="left_elbow",
                                   candidates=CANDIDATE_ANGLES)
    print(f"  primary angle chosen: {primary}")

    vals = [af.angles[primary] for af in angle_seq.frames
            if af.angles.get(primary) is not None]
    valid = len(vals)
    min_a = min(vals) if vals else None
    max_a = max(vals) if vals else None
    print(f"  valid frames for {primary}: {valid}")
    if vals:
        print(f"  {primary} min={min_a:.2f}  max={max_a:.2f}")
    else:
        print(f"  {primary}: no valid values")

    # --- Step 3 (auto-derive thresholds from this video's angle range) ---
    print("\n[3/4] Running Step 3 phase/rep counting...")
    config = config_from_angle_sequence(
        angle_seq,
        exercise="elbow_flexion",
        primary_angle=primary,
        movement_type="flexion_extension",
    )
    print(f"  exercise={config.exercise} movement={config.movement_type}")
    print(f"  thresholds -> start(end)={config.start_threshold:.2f}, "
          f"bottom={config.bottom_threshold:.2f}")
    print(f"  smoothing_window={config.smoothing_window}, "
          f"min_rep_duration={config.min_rep_duration}s, "
          f"min_range={config.min_movement_range}deg")

    analysis = count_reps(angle_seq, config=config)

    # --- report ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  video fps          : {seq.fps}")
    print(f"  total frames       : {seq.frame_count}")
    print(f"  valid angle frames : {analysis.valid_frames}")
    if analysis.min_angle is not None:
        print(f"  primary angle      : {analysis.primary_angle}  "
              f"min={analysis.min_angle:.2f} max={analysis.max_angle:.2f}")
    else:
        print(f"  primary angle      : {analysis.primary_angle} (no valid)")
    print(f"  total reps         : {analysis.total_reps}")

    print("\n  Detected phases:")
    for p in analysis.phases:
        print(f"    {p.phase:<10} frame={p.frame_index:4d} "
              f"t={p.timestamp:7.3f}s angle={p.angle}")

    print("\n  Repetitions:")
    for r in analysis.repetitions:
        print(f"    rep {r.rep_number}: start t={r.start_timestamp:.3f}s "
              f"(f{r.start_frame}), bottom t={r.bottom_timestamp:.3f}s "
              f"(f{r.bottom_frame}), end t={r.end_timestamp:.3f}s (f{r.end_frame}), "
              f"min={r.min_angle:.1f} max={r.max_angle:.1f}, dur={r.duration:.3f}s")

    # --- annotated video ---
    print("\n[4/4] Generating annotated diagnostic video...")
    create_annotated_video(input_video, output_video, angle_seq, norm_cache, analysis)

    print("\n" + "=" * 70)
    ok = analysis.total_reps > 0 and valid > 0
    if ok:
        print("STEP 3 RESULT: SUCCESS")
    else:
        print("STEP 3 RESULT: FAILED (no reps detected or no valid angles)")
    print("=" * 70)
    return 0 if ok else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(run(in_path, out_path))
