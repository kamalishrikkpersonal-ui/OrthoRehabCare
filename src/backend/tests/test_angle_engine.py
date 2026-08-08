"""Unit tests for the OrthoRehab AI Step 2 angle engine.

These tests use synthetic landmark data and do NOT require a real video.
A real-video integration test is provided but conditioned on the existence of
a valid video file (none ships with the project, so it is skipped by default).
"""

from __future__ import annotations

import math
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose.angle_engine import (
    DEFAULT_ANGLE_DEFINITIONS,
    AngleEngine,
    calculate_angle,
    angle_sequence_from_landmark_sequence,
)
from src.pose.landmark_extractor import Landmark, LandmarkSequence, PoseFrame


def make_landmark(idx: int, name: str, x: float, y: float, z: float, visibility: float = 0.9) -> Landmark:
    return Landmark(id=idx, name=name, x=x, y=y, z=z, visibility=visibility)


def run_test(name: str, fn) -> None:
    """Run a test function and report PASS/FAIL."""
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")


def approx(a, b, tol=0.5):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# --- 1. Known 90-degree angle ------------------------------------------------
def t_90_degree():
    # vertex at origin, arms along +x and +y -> exactly 90 degrees
    a = (1.0, 0.0, 0.0)
    v = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    ang = calculate_angle(a, v, c)
    assert approx(ang, 90.0), f"expected 90, got {ang}"


# --- 2. Known ~180-degree straight angle ------------------------------------
def t_180_degree():
    # arms opposite -> straight line
    a = (-1.0, 0.0, 0.0)
    v = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)
    ang = calculate_angle(a, v, c)
    assert approx(ang, 180.0), f"expected 180, got {ang}"


# --- 3. Known ~0-degree small angle (3D, per MathWorld formula) --------------
def t_0_degree():
    # All three points collinear with the vertex BETWEEN the two endpoints on the
    # SAME ray, so both vectors point the same way -> 0 degrees.
    va = (2.0, 0.0, 0.0)
    vv = (1.0, 0.0, 0.0)
    vc = (1.5, 0.0, 0.0)  # same side of vv as va (v2 = +x)
    ang = calculate_angle(va, vv, vc)
    assert approx(ang, 0.0, tol=1.0), f"expected ~0, got {ang}"


# --- 4. Zero-length vector handling ------------------------------------------
def t_zero_length():
    # vertex coincides with point_a -> zero-length vector -> None
    a = (0.0, 0.0, 0.0)
    v = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)
    assert calculate_angle(a, v, c) is None, "expected None for zero-length vector"

    # vertex coincides with point_c
    a = (1.0, 0.0, 0.0)
    v = (0.0, 0.0, 0.0)
    c = (0.0, 0.0, 0.0)
    assert calculate_angle(a, v, c) is None, "expected None for zero-length vector"


# --- 5. Missing landmark handling --------------------------------------------
def t_missing_landmark():
    engine = AngleEngine()
    # Only some landmarks present; left_wrist / right_wrist missing.
    frame = PoseFrame(
        frame_index=0,
        timestamp=0.0,
        landmarks=[
            make_landmark(11, "left_shoulder", 0.1, 0.2, 0.0),
            make_landmark(12, "right_shoulder", 0.9, 0.2, 0.0),
            make_landmark(13, "left_elbow", 0.2, 0.4, 0.0),
            make_landmark(14, "right_elbow", 0.8, 0.4, 0.0),
            make_landmark(23, "left_hip", 0.2, 0.6, 0.0),
            make_landmark(24, "right_hip", 0.8, 0.6, 0.0),
            # left_wrist and right_wrist intentionally missing
        ],
    )
    angles = engine.frame_angles(frame)
    assert angles["left_elbow"] is None, "expected None for missing landmark"
    assert angles["right_elbow"] is None, "expected None for missing landmark"


# --- 6. Visibility threshold handling ----------------------------------------
def t_visibility_threshold():
    engine = AngleEngine(visibility_threshold=0.5)
    # Low visibility on right_shoulder -> right_elbow angle becomes None
    frame = PoseFrame(
        frame_index=0,
        timestamp=0.0,
        landmarks=[
            make_landmark(11, "left_shoulder", 0.1, 0.2, 0.0, 0.9),
            make_landmark(12, "right_shoulder", 0.9, 0.2, 0.0, 0.1),  # low visibility
            make_landmark(13, "left_elbow", 0.2, 0.4, 0.0, 0.9),
            make_landmark(14, "right_elbow", 0.8, 0.4, 0.0, 0.9),
            make_landmark(15, "left_wrist", 0.2, 0.6, 0.0, 0.9),
            make_landmark(16, "right_wrist", 0.8, 0.6, 0.0, 0.9),
            make_landmark(23, "left_hip", 0.2, 0.7, 0.0, 0.9),
            make_landmark(24, "right_hip", 0.8, 0.7, 0.0, 0.9),
        ],
    )
    angles = engine.frame_angles(frame)
    assert angles["right_elbow"] is None, "expected None for low visibility"
    assert angles["left_elbow"] is not None, "expected a valid left_elbow angle"


# --- 7. Named MediaPipe landmark mapping -------------------------------------
def t_landmark_mapping():
    engine = AngleEngine()
    # Build a realistic near-90-degree elbow: shoulder-elbow-wrist.
    # Elbow at origin; shoulder along +y, wrist along +x (vector math -> 90 deg).
    shoulder = make_landmark(11, "left_shoulder", 0.0, 0.5, 0.0, 0.95)
    elbow = make_landmark(13, "left_elbow", 0.0, 0.0, 0.0, 0.95)
    wrist = make_landmark(15, "left_wrist", 0.5, 0.0, 0.0, 0.95)   # perpendicular to shoulder -> 90 deg at elbow
    hip = make_landmark(23, "left_hip", 0.3, 0.3, 0.0, 0.95)
    # right-side mirrored landmarks so definitions are all present
    rshoulder = make_landmark(12, "right_shoulder", 0.0, 0.5, 0.0, 0.95)
    relbow = make_landmark(14, "right_elbow", 0.0, 0.0, 0.0, 0.95)
    rwrist = make_landmark(16, "right_wrist", 0.5, 0.0, 0.0, 0.95)
    rhip = make_landmark(24, "right_hip", -0.3, 0.3, 0.0, 0.95)

    frame = PoseFrame(frame_index=0, timestamp=0.0, landmarks=[shoulder, elbow, wrist, hip, rshoulder, relbow, rwrist, rhip])
    angles = engine.frame_angles(frame)
    # left_elbow measured at elbow (vertex) between shoulder and wrist == 90
    assert approx(angles["left_elbow"], 90.0), f"expected 90, got {angles['left_elbow']}"
    # correct named mapping check
    assert "left_elbow" in angles and "right_elbow" in angles
    assert "left_shoulder" in angles and "right_shoulder" in angles


# --- 8. Full AngleSequence from a synthetic LandmarkSequence -----------------
def t_sequence():
    engine = AngleEngine()
    frames = [
        PoseFrame(
            frame_index=0,
            timestamp=0.0,
            landmarks=[
                make_landmark(11, "left_shoulder", 0.0, 0.5, 0.0),
                make_landmark(12, "right_shoulder", 0.0, 0.5, 0.0),
                make_landmark(13, "left_elbow", 0.0, 0.0, 0.0),
                make_landmark(14, "right_elbow", 0.0, 0.0, 0.0),
                make_landmark(15, "left_wrist", 0.5, 0.0, 0.0),
                make_landmark(16, "right_wrist", 0.5, 0.0, 0.0),
                make_landmark(23, "left_hip", 0.3, 0.3, 0.0),
                make_landmark(24, "right_hip", -0.3, 0.3, 0.0),
            ],
        ),
        PoseFrame(
            frame_index=1,
            timestamp=0.033,
            landmarks=[],  # empty -> every angle None
        ),
    ]
    seq = LandmarkSequence(video_path="synthetic.mp4", fps=30, frame_count=2, pose_frames=2, frames=frames)
    angles_seq = engine.sequence_angles(seq)
    assert angles_seq.fps == 30
    assert angles_seq.frame_count == 2
    assert angles_seq.frames[0].frame_index == 0
    assert angles_seq.frames[0].timestamp == 0.0
    assert approx(angles_seq.frames[0].angles["left_elbow"], 90.0)
    # frame with no landmarks -> all None
    assert all(v is None for v in angles_seq.frames[1].angles.values())

    # convenience function path
    angles_seq2 = angle_sequence_from_landmark_sequence(seq)
    assert len(angles_seq2.frames) == 2

    # to_dict serialization
    d = angles_seq.to_dict()
    assert "fps" in d and "frames" in d and "frame_count" in d


def run_optional_real_video_test() -> None:
    """Optionally validate against a real Step 1 LandmarkSequence if a sample
    video exists. No video ships with the project, so this is reported as skipped."""
    candidate = os.environ.get("ORTHOREHAB_TEST_VIDEO")
    if not candidate:
        print("[SKIP] real-video integration test: no sample video available "
              "(set ORTHOREHAB_TEST_VIDEO=<path> to run it)")
        return
    if not os.path.exists(candidate):
        print("[SKIP] real-video integration test: video not found")
        return
    from src.pose.landmark_extractor import extract_landmark_sequence
    seq = extract_landmark_sequence(candidate)
    angles_seq = angle_sequence_from_landmark_sequence(seq)
    print(f"[PASS] real-video integration: processed {angles_seq.frame_count} frames, "
          f"{len(['e' for f in angles_seq.frames if f.angles])} with angles")


if __name__ == "__main__":
    print("STEP 2 ANGLE ENGINE TESTS")
    print("=" * 40)
    run_test("90-degree angle", t_90_degree)
    run_test("180-degree straight angle", t_180_degree)
    run_test("~0-degree small angle", t_0_degree)
    run_test("zero-length vector handling", t_zero_length)
    run_test("missing landmark handling", t_missing_landmark)
    run_test("visibility threshold handling", t_visibility_threshold)
    run_test("named MediaPipe landmark mapping", t_landmark_mapping)
    run_test("AngleSequence from synthetic LandmarkSequence", t_sequence)
    print("=" * 40)
    run_optional_real_video_test()

