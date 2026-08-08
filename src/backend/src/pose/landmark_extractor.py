"""MediaPipe pose landmark extraction for OrthoRehab AI Step 1.

Pipeline:
    reference video
      -> frames (OpenCV)
      -> MediaPipe Pose
      -> 33 landmarks per valid frame
      -> ordered landmark sequence

This module is adapted from the concepts in FitPose-Detector:
  * FitPose-Detector/main.py       -> MediaPipe initialization + pose processing
  * FitPose-Detector/trained_pose_model/train_model.py -> per-frame landmark extraction

It does NOT use the LSTM classifier, angle math, rep counting, or the OpenCV
webcam UI. It only produces the ordered landmark sequence that becomes the
input to Step 2 (angle engine).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Suppress verbose MediaPipe / TensorFlow logging that is not needed here.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp

from .video_processor import VideoFrame, iter_video_frames, read_video_info

# Standard MediaPipe Pose landmark ordering (index -> canonical name).
# Only the landmarks we will eventually need for the angle engine are called
# out by name, but the full 33-index list is preserved in order.
POSE_LANDMARK_NAMES: List[str] = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]

# OrthoRehab is scoped to HIP-AND-ABOVE exercises. These are the only landmarks
# the application exposes at the application level. MediaPipe still detects all
# 33 internally; this filter drops lower-body landmarks (knees, ankles, feet,
# fingers, mouth details) that are not relevant to upper-body rehab and often
# carry extremely low visibility.
UPPER_BODY_LANDMARK_NAMES: List[str] = [
    "nose",              # 0
    "left_eye",          # 2
    "right_eye",         # 5
    "left_ear",          # 7
    "right_ear",         # 8
    "left_shoulder",     # 11
    "right_shoulder",    # 12
    "left_elbow",        # 13
    "right_elbow",       # 14
    "left_wrist",        # 15
    "right_wrist",       # 16
    "left_hip",          # 23
    "right_hip",         # 24
]

# Named landmarks used by the upcoming angle engine. Kept here as a convenience
# so name lookups are explicit rather than magic numbers.
KEY_LANDMARK_NAMES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class Landmark:
    """A single normalized 3D pose landmark."""

    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseFrame:
    """A frame that produced a valid 33-landmark pose."""

    frame_index: int
    timestamp: float
    landmarks: List[Landmark] = field(default_factory=list)


@dataclass
class LandmarkSequence:
    """Ordered result of processing a reference video."""

    video_path: str
    fps: float
    frame_count: int           # total frames read from the video
    pose_frames: int           # frames that produced a valid pose
    frames: List[PoseFrame] = field(default_factory=list)  # only valid-pose frames

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "pose_frames": self.pose_frames,
            "frames": [asdict(f) for f in self.frames],
        }


def extract_landmarks_from_frame(frame_bgr) -> Optional[List[Landmark]]:
    """Run MediaPipe Pose on one BGR frame and return the 33 landmarks.

    Args:
        frame_bgr: An OpenCV BGR image.

    Returns:
        A list of 33 Landmark objects, or None if no pose was detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return None

    landmarks: List[Landmark] = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        landmarks.append(
            Landmark(
                id=i,
                name=POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"landmark_{i}",
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
        )
    return landmarks


def filter_upper_body_landmarks(landmarks: List[Landmark]) -> List[Landmark]:
    """Keep only the OrthoRehab hip-and-above landmarks.

    This is the explicit upper-body filtering layer applied after MediaPipe
    extraction. It keeps a landmark only if its name is in
    UPPER_BODY_LANDMARK_NAMES, preserving the original id, x/y/z and visibility.
    It never invents or replaces landmarks. The raw 33-landmark list returned by
    `extract_landmarks_from_frame` is still available for debugging.

    Args:
        landmarks: The full 33 MediaPipe landmarks for a pose frame.

    Returns:
        A new list containing only the supported upper-body landmarks.
    """
    allowed = set(UPPER_BODY_LANDMARK_NAMES)
    return [lm for lm in landmarks if lm.name in allowed]


def extract_landmark_sequence(video_path: str) -> LandmarkSequence:
    """Extract an ordered upper-body landmark sequence from a reference video.

    Pipeline:
        open video -> read frames sequentially -> MediaPipe Pose ->
        33 landmarks per valid frame -> filter to hip-and-above ->
        ordered PoseFrame list.

    MediaPipe still detects all 33 landmarks internally
    (`extract_landmarks_from_frame`), but the application-level
    `LandmarkSequence` contains ONLY the supported upper-body landmarks
    (see UPPER_BODY_LANDMARK_NAMES). Visibility values are preserved; low
    visibility landmarks are NOT invented or replaced.

    Frames where no pose is detected are skipped gracefully (not added to
    `frames`) but still counted in `frame_count`, preserving ordering and
    timestamps for the frames that do have a pose.

    Args:
        video_path: Path to the video file.

    Returns:
        A LandmarkSequence with filtered (hip-and-above) valid pose frames.
    """
    info = read_video_info(video_path)
    fps = info.fps if info.fps > 0 else 30.0

    sequence = LandmarkSequence(
        video_path=info.path,
        fps=fps,
        frame_count=0,
        pose_frames=0,
    )

    for vf in iter_video_frames(info.path):
        sequence.frame_count += 1
        raw_landmarks = extract_landmarks_from_frame(vf.image)
        if raw_landmarks is not None:
            # Apply the OrthoRehab upper-body filtering layer.
            upper_body = filter_upper_body_landmarks(raw_landmarks)
            sequence.frames.append(
                PoseFrame(frame_index=vf.frame_index, timestamp=vf.timestamp, landmarks=upper_body)
            )
            sequence.pose_frames += 1

    return sequence
