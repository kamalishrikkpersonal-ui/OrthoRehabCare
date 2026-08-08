"""Joint-angle calculation engine for OrthoRehab AI Step 2.

Consumes the Step 1 landmark sequence (LandmarkSequence) and produces an
AngleSequence with per-frame joint angles.

Pipeline:
    FitPose reference/conventions
      -> Step 1: video -> MediaPipe -> LandmarkSequence
      -> Step 2 (this module): LandmarkSequence -> AngleSequence

Design:
* The engine is generic and data-driven: angle definitions are a config mapping
  of angle-name -> [point_a, vertex_point, point_c] landmark names. Step 3 can
  supply exercise-specific definitions.
* Angles are computed in 3D from x, y, z coordinates using vector math:
      vector_1 = point_a - vertex_point
      vector_2 = point_c - vertex_point
      cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))
      theta      = arccos(cos(theta))   [cos clamped to [-1, 1]]
  returned in degrees.
* Zero-length vectors return None (cannot compute a meaningful angle).
* Landmarks with visibility below a configurable threshold are ignored (angle
  set to None) — coordinates are never invented.
* Missing landmarks set the corresponding angle to None (frame never dropped).

SCOPE: hip-and-above only. This engine's default definitions consume only hips,
shoulders, elbows and wrists. Even though Step 1 extracts all 33 landmarks,
Step 2 ignores knees/ankles/feet per the current OrthoRehab scope.

This module is a pure Python processor. It has no dependency on FastAPI,
MongoDB, React, browser APIs, OpenCV display, the LSTM, or Groq, so an
equivalent TypeScript version can be written later for the patient's browser.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .landmark_extractor import Landmark, LandmarkSequence, PoseFrame

# ---------------------------------------------------------------------------
# Default angle definitions (data/config driven).
# Only hip-and-above landmarks are used. Step 3 may override these per exercise.
# ---------------------------------------------------------------------------
DEFAULT_ANGLE_DEFINITIONS: Dict[str, List[str]] = {
    "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_shoulder": ["left_hip", "left_shoulder", "left_elbow"],
    "right_shoulder": ["right_hip", "right_shoulder", "right_elbow"],
}

# Visibility threshold below which a landmark is considered unreliable.
DEFAULT_VISIBILITY_THRESHOLD: float = 0.5


def calculate_angle(
    point_a,
    vertex_point,
    point_c,
) -> Optional[float]:
    """Compute the angle at `vertex_point` between vectors to `point_a` and `point_c`.

    Inputs may be anything convertible to an (x, y, z) coordinate triple:
    a Landmark, a length-3 sequence of floats, or a numpy (3,) array.

    Returns the angle in degrees, or None if either vector has (near) zero
    length (i.e. the vertex coincides with one of the endpoints).
    """
    va = np.asarray(_point_coords(point_a), dtype=float)
    vv = np.asarray(_point_coords(vertex_point), dtype=float)
    vc = np.asarray(_point_coords(point_c), dtype=float)

    v1 = va - vv
    v2 = vc - vv

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    # Guard against zero-length vectors -> undefined angle.
    if n1 < 1e-12 or n2 < 1e-12:
        return None

    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    # Clamp to avoid arccos domain errors from floating-point noise.
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta = math.degrees(math.acos(cos_theta))
    return float(round(theta, 2))


def _point_coords(point):
    """Return the (x, y, z) triple for a landmark-like point."""
    if isinstance(point, Landmark):
        return (point.x, point.y, point.z)
    # Sequence-like: (x, y, z) or array-like with length >= 3.
    vals = tuple(float(v) for v in point)
    if len(vals) < 3:
        raise ValueError(f"Point must have at least 3 coordinates, got {len(vals)}")
    return vals[:3]


@dataclass
class AngleFrame:
    """Joint angles for a single frame (aligned to Step 1 frame index/timestamp)."""

    frame_index: int
    timestamp: float
    angles: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class AngleSequence:
    """Ordered joint-angle results for a landmark sequence."""

    fps: float
    frame_count: int  # number of valid pose frames processed from the source
    frames: List[AngleFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "frames": [asdict(f) for f in self.frames],
        }


class AngleEngine:
    """Reusable, config-driven joint-angle engine.

    Args:
        angle_definitions: mapping of angle name -> [a_name, vertex_name, c_name]
            using the same landmark names produced by Step 1. Defaults to
            DEFAULT_ANGLE_DEFINITIONS (hip-and-above upper-body angles).
        visibility_threshold: minimum visibility required for every landmark in
            an angle definition; below this the angle is set to None.
    """

    def __init__(
        self,
        angle_definitions: Optional[Dict[str, List[str]]] = None,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        self.angle_definitions = angle_definitions or dict(DEFAULT_ANGLE_DEFINITIONS)
        self.visibility_threshold = float(visibility_threshold)

    # -- core per-frame logic -------------------------------------------------
    def landmark_map(self, landmarks: Sequence[Landmark]) -> Dict[str, Landmark]:
        """Index landmarks by name for fast lookup."""
        return {lm.name: lm for lm in landmarks}

    def frame_angles(self, pose_frame: PoseFrame) -> Dict[str, Optional[float]]:
        """Compute every configured angle for a single PoseFrame.

        Angles whose definition requires a missing or low-visibility landmark are
        set to None (the frame is never dropped).
        """
        by_name = self.landmark_map(pose_frame.landmarks)
        result: Dict[str, Optional[float]] = {}

        for angle_name, (a_name, vertex_name, c_name) in self.angle_definitions.items():
            lm_a = by_name.get(a_name)
            lm_v = by_name.get(vertex_name)
            lm_c = by_name.get(c_name)

            # Missing landmark -> cannot compute.
            if lm_a is None or lm_v is None or lm_c is None:
                result[angle_name] = None
                continue

            # Low visibility -> unreliable, do not invent coordinates.
            if (
                lm_a.visibility < self.visibility_threshold
                or lm_v.visibility < self.visibility_threshold
                or lm_c.visibility < self.visibility_threshold
            ):
                result[angle_name] = None
                continue

            result[angle_name] = calculate_angle(lm_a, lm_v, lm_c)

        return result

    # -- sequence-level API ---------------------------------------------------
    def sequence_angles(self, landmark_sequence: LandmarkSequence) -> AngleSequence:
        """Convert a Step 1 LandmarkSequence into an AngleSequence."""
        angle_frames: List[AngleFrame] = []
        for pf in landmark_sequence.frames:
            angle_frames.append(
                AngleFrame(
                    frame_index=pf.frame_index,
                    timestamp=pf.timestamp,
                    angles=self.frame_angles(pf),
                )
            )
        return AngleSequence(
            fps=landmark_sequence.fps,
            frame_count=len(angle_frames),
            frames=angle_frames,
        )


def angle_sequence_from_landmark_sequence(
    landmark_sequence: LandmarkSequence,
    angle_definitions: Optional[Dict[str, List[str]]] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> AngleSequence:
    """Convenience function: build an AngleSequence from a LandmarkSequence."""
    engine = AngleEngine(
        angle_definitions=angle_definitions,
        visibility_threshold=visibility_threshold,
    )
    return engine.sequence_angles(landmark_sequence)
