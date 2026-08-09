"""Capture-quality validation for OrthoRehab AI Step 5.

This module distinguishes two fundamentally different failure situations:

A. **Recording quality failure** — the required landmarks cannot be reliably
   observed (e.g. the right wrist is missing for a substantial portion of the
   recording). The movement cannot be analyzed, so the recording should be
   rejected and the patient asked to re-record.

B. **Good recording, poor exercise performance** — the required landmarks are
   visible and the movement can be analyzed, but the patient performs the
   exercise incorrectly. This is a *valid* scored comparison with a lower score
   and corrective feedback.

The validator is a gate that runs BEFORE the comparison engine. It uses the
existing :class:`AngleSequence` validity information (an angle is ``None`` when
any required landmark is missing or below the visibility threshold — see the
Step 2 angle engine). It does NOT duplicate angle/visibility logic.

It is generic: required landmarks are derived from the angle definitions for the
chosen primary angle, so future exercises can define their own required
landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .angle_engine import DEFAULT_ANGLE_DEFINITIONS, AngleSequence
from .landmark_extractor import LandmarkSequence

# Engineering starting points (configurable). These are deliberately tolerant:
# a few missing frames should not reject a recording; only a substantial deficit
# should.
DEFAULT_MIN_VALID_ANGLE_RATIO = 0.70
DEFAULT_MAX_MISSING_RATIO = 0.30

# Friendly landmark names used in user-facing messages.
FRIENDLY_LANDMARK_NAMES = {
    "right_wrist": "right wrist",
    "left_wrist": "left wrist",
    "right_elbow": "right elbow",
    "left_elbow": "left elbow",
    "right_shoulder": "right shoulder",
    "left_shoulder": "left shoulder",
    "right_hip": "right hip",
    "left_hip": "left hip",
}


@dataclass
class CaptureQualityConfig:
    """Configurable thresholds for the capture-quality validator.

    ``min_valid_angle_ratio``: minimum fraction of frames with a valid primary
        angle required for the recording to be analyzable (default 0.70).
    ``max_missing_ratio``: maximum fraction of frames allowed to be missing /
        low-visibility before the recording is rejected (default 0.30).
    ``required_landmarks``: optional explicit list of required landmark names.
        If None, they are derived from the angle definitions for the primary
        angle.
    """

    min_valid_angle_ratio: float = DEFAULT_MIN_VALID_ANGLE_RATIO
    max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO
    required_landmarks: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.min_valid_angle_ratio <= 1.0):
            raise ValueError("min_valid_angle_ratio must be in (0, 1]")
        if not (0.0 <= self.max_missing_ratio < 1.0):
            raise ValueError("max_missing_ratio must be in [0, 1)")
        if self.min_valid_angle_ratio + self.max_missing_ratio > 1.0:
            # This is a sanity check: the two boundaries should be consistent.
            raise ValueError(
                "min_valid_angle_ratio + max_missing_ratio must be <= 1.0"
            )


@dataclass
class CaptureQualityResult:
    """Outcome of capture-quality validation.

    If ``passed`` is True, the recording is analyzable and should be scored.
    If ``passed`` is False, ``error`` describes the recording-quality failure.
    """

    passed: bool
    total_frames: int = 0
    valid_frames: int = 0
    missing_frames: int = 0
    valid_angle_ratio: float = 0.0
    missing_ratio: float = 0.0
    primary_angle: str = ""
    missing_landmarks: List[str] = field(default_factory=list)
    error_code: str = ""
    error_message: str = ""
    user_instruction: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "missing_frames": self.missing_frames,
            "valid_angle_ratio": round(self.valid_angle_ratio, 4),
            "missing_ratio": round(self.missing_ratio, 4),
            "primary_angle": self.primary_angle,
            "missing_landmarks": list(self.missing_landmarks),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "user_instruction": self.user_instruction,
        }


def _required_landmarks_for(
    primary_angle: str, angle_definitions: Dict[str, List[str]]
) -> List[str]:
    """Return the ordered landmark names required to compute ``primary_angle``."""
    return list(angle_definitions.get(primary_angle, []))


def _friendly(name: str) -> str:
    """Return a human-friendly name for a landmark (defaults to the raw name)."""
    return FRIENDLY_LANDMARK_NAMES.get(name, name)


class CaptureQualityValidator:
    """Validate that a recording contains enough landmark data to analyze.

    Args:
        config: optional :class:`CaptureQualityConfig`.
        angle_definitions: optional angle-definition map. Defaults to the Step 2
            defaults; used to derive the required landmarks for the primary
            angle.
    """

    def __init__(
        self,
        config: Optional[CaptureQualityConfig] = None,
        angle_definitions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.config = config or CaptureQualityConfig()
        self.angle_definitions = angle_definitions or dict(DEFAULT_ANGLE_DEFINITIONS)

    @staticmethod
    def _angle_validity(
        angle_sequence: AngleSequence, primary_angle: str
    ) -> Dict[str, int]:
        """Count valid vs missing primary-angle frames from the AngleSequence.

        This reuses the Step 2 angle engine's validity decisions: an angle is
        ``None`` when a required landmark is missing or below the visibility
        threshold. No angle/visibility logic is duplicated here.
        """
        total = len(angle_sequence.frames)
        valid = sum(
            1
            for af in angle_sequence.frames
            if af.angles.get(primary_angle) is not None
        )
        missing = total - valid
        return {"total": total, "valid": valid, "missing": missing}

    @staticmethod
    def _missing_landmarks(
        landmark_sequence: Optional[LandmarkSequence],
        required: List[str],
    ) -> List[str]:
        """Identify which required landmarks are most often missing/low-visibility.

        A landmark is counted as "missing" for a frame if it is absent from the
        frame's landmark list. This is a coarse signal derived from the existing
        ``LandmarkSequence`` and is only used to produce a helpful diagnostic in
        the error message. It never replaces the angle-validity gate.
        """
        if landmark_sequence is None or not landmark_sequence.frames:
            return list(required)
        total = len(landmark_sequence.frames)
        missing_counts: Dict[str, int] = {}
        for name in required:
            missing_counts[name] = sum(
                1 for pf in landmark_sequence.frames if name not in {l.name for l in pf.landmarks}
            )
        # Return all required landmarks sorted by missing frequency (desc).
        ranked = sorted(missing_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in ranked if missing_counts[name] > 0]

    def validate(
        self,
        angle_sequence: AngleSequence,
        primary_angle: str,
        landmark_sequence: Optional[LandmarkSequence] = None,
    ) -> CaptureQualityResult:
        """Validate capture quality for a patient angle sequence.

        Args:
            angle_sequence: Step 2 output for the patient.
            primary_angle: the primary joint angle name (e.g. ``"right_elbow"``).
            landmark_sequence: optional Step 1 output, used only to identify the
                likely problematic required landmark for a helpful message.

        Returns:
            A :class:`CaptureQualityResult`. ``passed`` is True when the
            recording contains sufficient valid data to analyze.
        """
        required = self.config.required_landmarks or _required_landmarks_for(
            primary_angle, self.angle_definitions
        )

        counts = self._angle_validity(angle_sequence, primary_angle)
        total = counts["total"]
        valid = counts["valid"]
        missing = counts["missing"]

        if total <= 0:
            return CaptureQualityResult(
                passed=False,
                total_frames=0,
                valid_frames=0,
                missing_frames=0,
                valid_angle_ratio=0.0,
                missing_ratio=0.0,
                primary_angle=primary_angle,
                missing_landmarks=list(required),
                error_code="INSUFFICIENT_LANDMARK_VISIBILITY",
                error_message="No analyzable frames were detected in this recording.",
                user_instruction=(
                    "Please record again while keeping your full exercising arm "
                    "visible in the camera frame."
                ),
            )

        valid_ratio = valid / total
        missing_ratio = missing / total

        # Determine the likely problematic required landmark (for a helpful msg).
        likely_missing = self._missing_landmarks(landmark_sequence, required)
        weak_target = likely_missing[0] if likely_missing else (required[-1] if required else "")

        if valid_ratio < self.config.min_valid_angle_ratio or (
            missing_ratio > self.config.max_missing_ratio
        ):
            target_name = _friendly(weak_target)
            return CaptureQualityResult(
                passed=False,
                total_frames=total,
                valid_frames=valid,
                missing_frames=missing,
                valid_angle_ratio=valid_ratio,
                missing_ratio=missing_ratio,
                primary_angle=primary_angle,
                missing_landmarks=likely_missing or list(required),
                error_code="INSUFFICIENT_LANDMARK_VISIBILITY",
                error_message=(
                    f"We couldn't reliably see your {target_name} during the "
                    f"recording ({valid}/{total} frames usable)."
                ),
                user_instruction=(
                    "Please record again while keeping your full exercising arm "
                    "visible in the camera frame."
                ),
            )

        return CaptureQualityResult(
            passed=True,
            total_frames=total,
            valid_frames=valid,
            missing_frames=missing,
            valid_angle_ratio=valid_ratio,
            missing_ratio=missing_ratio,
            primary_angle=primary_angle,
            missing_landmarks=[],
        )
