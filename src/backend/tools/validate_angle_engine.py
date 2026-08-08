"""Real-video validation tool for the OrthoRehab AI angle engine (Step 2).

Pipeline (reuses existing modules):
    reference_video.mp4
      -> existing extract_landmark_sequence()      (Step 1, upper-body filter)
      -> existing AngleEngine.sequence_angles()    (Step 2)
      -> real AngleSequence

What it does:
  1. Processes a real reference video with the existing Step 1 extractor.
  2. Prints ALL 13 supported upper-body landmarks of the first valid frame
     (both left and right) with name, x, y, z, visibility.
  3. Runs the EXISTING angle engine on the real landmark sequence.
  4. Reports frames processed, frames with valid angles, exact angle names,
     angle values at frame 0 and several later frames, and min/max per joint.
  5. Verifies left & right landmarks are both handled and that low-visibility
     landmarks are rejected per the existing rules.
  6. Writes an annotated output video for visual debugging.

Usage:
    python -m tools.validate_angle_engine <input_video> [output_video]
"""

from __future__ import annotations

import os
import sys

import cv2

# Make package importable when run from the backend directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import AngleEngine
from src.pose.landmark_extractor import (
    UPPER_BODY_LANDMARK_NAMES,
    extract_landmark_sequence,
)
from src.pose.video_processor import iter_video_frames, read_video_info

# Upper-body connection graph for a minimal skeleton overlay. Legs/feet are
# intentionally never drawn.
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

# Short labels used when drawing on the video.
SHORT_NAMES = {
    "nose": "nose",
    "left_eye": "Leye",
    "right_eye": "Reye",
    "left_ear": "Lear",
    "right_ear": "Rear",
    "left_shoulder": "Lsh",
    "right_shoulder": "Rsh",
    "left_elbow": "Lel",
    "right_elbow": "Rel",
    "left_wrist": "Lwr",
    "right_wrist": "Rwr",
    "left_hip": "Lhip",
    "right_hip": "Rhip",
}


def landmark_row(lm) -> str:
    """Format one landmark for the console report."""
    return (
        f"    {lm.name:<16} id={lm.id:<2} x={lm.x:.4f} y={lm.y:.4f} "
        f"z={lm.z: .4f} visibility={lm.visibility:.4f}"
    )


def print_first_frame_landmarks(seq) -> None:
    """Print all 13 supported landmarks of the first valid frame."""
    first = seq.frames[0]
    print("\n" + "=" * 70)
    print("ALL 13 SUPPORTED UPPER-BODY LANDMARKS (first valid frame)")
    print("=" * 70)
    print(f"frame_index={first.frame_index} timestamp={first.timestamp:.3f}s")
    print(f"Detected count={len(first.landmarks)} (expected {len(UPPER_BODY_LANDMARK_NAMES)})")
    for lm in first.landmarks:
        print(landmark_row(lm))


def normalized_to_pixel(norm_map, width, height):
    """Convert {name: (x_norm, y_norm, vis)} to {name: (x_px, y_px, vis)}."""
    out = {}
    for name, (norm_x, norm_y, vis) in norm_map.items():
        px = max(0, min(width - 1, int(norm_x * width)))
        py = max(0, min(height - 1, int(norm_y * height)))
        out[name] = (px, py, vis)
    return out


def draw_annotations(frame, pixel_map, angle_values):
    """Draw upper-body landmarks, connections, labels and angles on a BGR frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw connections only where both endpoints are present.
    for a, b in CONNECTIONS:
        if a in pixel_map and b in pixel_map:
            xa, ya, _ = pixel_map[a]
            xb, yb, _ = pixel_map[b]
            cv2.line(vis, (xa, ya), (xb, yb), (0, 255, 0), 1)

    # Draw landmarks.
    for name, (px, py, _vis) in pixel_map.items():
        cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
        short = SHORT_NAMES.get(name, name)
        cv2.putText(
            vis, short, (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Draw calculated joint angles (top-left panel).
    y0 = 20
    cv2.putText(vis, "Angles", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y0 += 20
    if not angle_values:
        cv2.putText(vis, "no angles", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    for angle_name, val in angle_values.items():
        if val is None:
            text = f"{angle_name}: n/a"
            color = (0, 165, 255)
        else:
            text = f"{angle_name}: {val:.1f}"
            color = (0, 255, 0)
        cv2.putText(vis, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y0 += 20

    return vis


def create_annotated_video(input_path, output_path, angle_seq, norm_cache):
    """Re-read the input video and write an annotated copy using cached landmarks.

    norm_cache: {frame_index: {name: (x_norm, y_norm, visibility)}} built from
    the Step 1 result so we never re-run MediaPipe.
    """
    info = read_video_info(input_path)
    fps = float(info.fps) if info.fps > 0 else 30.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    size = (info.width, info.height)

    angles_by_frame = {af.frame_index: af.angles for af in angle_seq.frames}

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size,
        )

    n_frames = 0
    for vf in iter_video_frames(info.path):
        norm = norm_cache.get(vf.frame_index, {})
        pixel_map = normalized_to_pixel(norm, info.width, info.height)
        angles = angles_by_frame.get(vf.frame_index, {})
        annotated = draw_annotations(vf.image, pixel_map, angles)
        writer.write(annotated)
        n_frames += 1

    writer.release()
    print(f"Annotated video written to: {output_path}  ({n_frames} frames)")


def run(input_video, output_video=None) -> int:
    """Run the diagnostic and return 0 on success."""
    if output_video is None:
        output_video = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output", "reference_annotated.mp4")
        )

    print("=" * 70)
    print("ORTHOREHAB AI - STEP 2: REAL-VIDEO ANGLE ENGINE VALIDATION")
    print("=" * 70)
    print(f"Input video : {input_video}")
    print(f"Output video: {output_video}")

    # 1) Step 1 extraction (existing, includes upper-body filtering)
    print("\nRunning Step 1 landmark extraction...")
    seq = extract_landmark_sequence(input_video)
    print(f"  fps={seq.fps:.2f}, frame_count={seq.frame_count}, pose_frames={seq.pose_frames}")

    if not seq.frames:
        print("ERROR: no valid pose frames found.")
        return 1

    # Build normalized landmark cache (normalized x,y,visibility) for annotation.
    norm_cache = {}
    for pf in seq.frames:
        norm_cache[pf.frame_index] = {
            lm.name: (lm.x, lm.y, lm.visibility) for lm in pf.landmarks
        }

    # 2) First-frame landmark dump
    print_first_frame_landmarks(seq)

    # 3) Run EXISTING angle engine
    print("\n" + "=" * 70)
    print("ANGLE ENGINE (existing AngleEngine, default definitions)")
    print("=" * 70)
    engine = AngleEngine()
    print(f"  angle definitions: {list(engine.angle_definitions.keys())}")
    print(f"  visibility threshold: {engine.visibility_threshold}")

    angle_seq = engine.sequence_angles(seq)

    # 4) Stats
    print(f"\n  frames processed: {angle_seq.frame_count}")
    valid_frames = sum(
        1 for af in angle_seq.frames if any(v is not None for v in af.angles.values())
    )
    print(f"  frames with at least one valid angle: {valid_frames}")

    print("\n  Per-joint min/max across video (deg):")
    for angle_name in engine.angle_definitions:
        vals = [
            af.angles[angle_name] for af in angle_seq.frames
            if af.angles.get(angle_name) is not None
        ]
        if vals:
            print(f"    {angle_name:<14} min={min(vals):.2f}  max={max(vals):.2f}  "
                  f"(n={len(vals)})")
        else:
            print(f"    {angle_name:<14} min=n/a max=n/a  (no valid frames)")

    # 5) Sample angle values at several spread-out frames
    print("\n  Sample angle values:")
    sample_indices = sorted(set([0, max(1, angle_seq.frame_count // 4),
                                 max(1, angle_seq.frame_count // 2),
                                 angle_seq.frame_count - 1 if angle_seq.frame_count > 1 else 0]))
    for idx in sample_indices:
        af = angle_seq.frames[idx]
        vals = ", ".join(
            f"{k}={v:.1f}" if v is not None else f"{k}=n/a"
            for k, v in af.angles.items()
        )
        print(f"    frame={af.frame_index:4d} (t={af.timestamp:6.3f}s): {vals}")

    # 6) Left/right and low-visibility checks
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    left_names = {"left_shoulder", "left_elbow", "left_wrist", "left_hip"}
    right_names = {"right_shoulder", "right_elbow", "right_wrist", "right_hip"}
    first_names = {lm.name for lm in seq.frames[0].landmarks}
    print(f"  left landmarks present in first frame: {sorted(left_names & first_names)}")
    print(f"  right landmarks present in first frame: {sorted(right_names & first_names)}")

    print("\n  Low-visibility rejection check (per existing rules):")
    for angle_name in engine.angle_definitions:
        a_name, v_name, c_name = engine.angle_definitions[angle_name]
        low_count = 0
        for pf in seq.frames:
            by_name = {lm.name: lm for lm in pf.landmarks}
            needed = [by_name.get(x) for x in (a_name, v_name, c_name)]
            if any(nm is None for nm in needed):
                continue  # missing != visibility issue
            if any(nm.visibility < engine.visibility_threshold for nm in needed):
                low_count += 1
        if low_count:
            print(f"    {angle_name}: {low_count} frame(s) rejected due to low visibility")
        else:
            print(f"    {angle_name}: no low-visibility rejections")

    # 7) Annotated video
    print("\n" + "=" * 70)
    print("ANNOTATED VIDEO")
    print("=" * 70)
    create_annotated_video(input_video, output_video, angle_seq, norm_cache)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(run(in_path, out_path))

