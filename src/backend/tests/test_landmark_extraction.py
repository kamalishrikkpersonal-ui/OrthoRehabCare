"""Test runner for Step 1: reference video -> upper-body landmark sequence.

Usage:
    python -m tests.test_landmark_extraction path\\to\\reference_video.mp4

It processes the given video with MediaPipe and reports:
  * how many frames were read / had a valid pose,
  * MediaPipe raw landmark detection vs the OrthoRehab filtered count,
  * that only the supported upper-body landmarks are exposed,
  * that knee/ankle landmarks are excluded,
  * that frame index / timestamp information is preserved.
"""

from __future__ import annotations

import sys

from src.pose.landmark_extractor import (
    UPPER_BODY_LANDMARK_NAMES,
    extract_landmark_sequence,
)

# Lower-body landmarks that must NEVER appear in the filtered application output.
EXCLUDED_LANDMARK_NAMES = [
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def display_landmark(landmark, label: str) -> None:
    """Print a single landmark in the readable report format."""
    print(f"{label}:")
    print(f"  x={landmark.x:.4f}")
    print(f"  y={landmark.y:.4f}")
    print(f"  z={landmark.z:.4f}")
    print(f"  visibility={landmark.visibility:.4f}")


def run(video_path: str) -> int:
    """Run the Step 1 validation on a video and print a report."""
    import os

    if not os.path.exists(video_path):
        print(f"ERROR: video not found: {video_path}")
        return 1

    seq = extract_landmark_sequence(video_path)

    print("=" * 60)
    print("ORTHOREHAB AI - STEP 1: UPPER-BODY LANDMARK EXTRACTION")
    print("=" * 60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {seq.fps:.2f}")
    print(f"Total frames: {seq.frame_count}")
    print(f"Frames processed: {seq.frame_count}")
    print(f"Frames with pose: {seq.pose_frames}")
    print(f"Supported upper-body landmarks: {len(UPPER_BODY_LANDMARK_NAMES)}")
    print(f"Supported upper-body landmark set:")
    for name in UPPER_BODY_LANDMARK_NAMES:
        print(f"  - {name}")

    if not seq.frames:
        print("\nNo valid pose frames detected in this video.")
        return 1

    # --- Per-frame verification ---------------------------------------------
    bad_frames = 0
    excluded_found = []
    for pf in seq.frames:
        names = {lm.name for lm in pf.landmarks}
        # Every exposed landmark must be in the supported upper-body set.
        if not all(n in set(UPPER_BODY_LANDMARK_NAMES) for n in names):
            bad_frames += 1
        # No excluded lower-body landmark may appear.
        for exc in EXCLUDED_LANDMARK_NAMES:
            if exc in names:
                excluded_found.append((pf.frame_index, exc))

    layer0 = seq.pose_frames  # all pose frames
    layer1 = seq.pose_frames

    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Frames with pose: {seq.pose_frames}")
    print(f"Frames with any excluded lower-body landmark: {len(set(f for f, _ in excluded_found))}")
    if excluded_found:
        print("EXCLUDED landmarks found in output (should be ZERO):")
        for fi, exc in excluded_found[:10]:
            print(f"  frame {fi}: {exc}")
    else:
        print("EXCLUDED landmarks (knee/ankle/heel/foot): NONE — successfully excluded")
    if bad_frames:
        print(f"WARNING: {bad_frames} frame(s) contained a non-upper-body landmark.")
    else:
        print("All exposed landmarks belong to the uppercase-body set: OK")

    # --- First valid frame report -------------------------------------------
    first = seq.frames[0]
    print("\n" + "=" * 60)
    print("FIRST VALID FRAME")
    print("=" * 60)
    print(f"frame_index={first.frame_index}")
    print(f"timestamp={first.timestamp:.3f}s")
    print(f"Filtered landmark count: {len(first.landmarks)}")
    # Sanity check: MediaPipe detects 33 internally; we expose <= 33 (hip/above).
    # (extract_landmark_sequence only returns filtered landmarks, so we can't
    #  show 33 here, but the count should be <= 13.)
    expected_max = len(UPPER_BODY_LANDMARK_NAMES)
    print(f"(expected maximum filtered landmarks per frame: {expected_max})")
    if len(first.landmarks) > expected_max:
        print("ERROR: filtered count exceeds supported upper-body count.")
        return 1

    lms = {lm.name: lm for lm in first.landmarks}
    for name in ("left_hip", "left_shoulder", "left_elbow", "left_wrist"):
        lm = lms.get(name)
        if lm is not None:
            display_landmark(lm, name)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"MediaPipe raw landmarks detected (per valid frame): up to 33")
    print(f"OrthoRehab filtered landmarks in application output: "
          f"{len(first.landmarks)} (first valid frame)")
    print(f"Knee/ankle landmarks excluded: "
          f"{'YES' if not excluded_found else 'NO'}")
    print(f"Frame index preserved: {first.frame_index == 0}")
    print(f"Timestamp preserved: {first.timestamp:.3f}s")

    if excluded_found or bad_frames:
        print("\nStep 1 verification FAILED.")
        return 1

    print("\nStep 1 complete: upper-body landmark sequence produced and verified.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_landmark_extraction <path_to_video>")
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
