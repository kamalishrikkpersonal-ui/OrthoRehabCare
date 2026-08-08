"""Exercise phase/rep configuration for OrthoRehab AI Step 3.

The phase detector must never assume one universal angle threshold. Different
rehabilitation exercises have different ranges and movement patterns, so every
parameter that matters for phase detection / rep counting is configurable via
:class:`ExercisePhaseConfig`.

This module also derives sensible default thresholds from a real
:class:`AngleSequence` (see :func:`config_from_angle_sequence`) so the initial
values match the reference exercise the user already captured.

SCOPE: hip-and-above movement only. Config must reference upper-body landmark
angle names produced by Step 2 (e.g. ``left_elbow``).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# Movement profiles tell the state machine which way is "flexing" versus
# "extending" relative to the chosen primary angle.
#
#   * "flexion_extension"  -> larger angle = extended/start,
#                             smaller angle = flexed/bottom (e.g. elbow/hip/knee).
#   * "extension_flexion"  -> inverted: smaller angle = extended/start,
#                             larger angle = flexed/bottom.
MOVEMENT_TYPES = (
    "flexion_extension",  # start/atl larger angle, bottom at smaller angle
    "extension_flexion",  # start at smaller angle, bottom at larger angle
)


def _resolve_thresholds(
    min_angle: float,
    max_angle: float,
    start_threshold: Optional[float],
    end_threshold: Optional[float],
    bottom_threshold: Optional[float],
    movement_type: str,
) -> Dict[str, float]:
    """Given a measured range, fill in any missing thresholds sensibly.

    For ``flexion_extension``:
      * start/bottom thresholds properly mean: start is near max angle,
        bottom near min angle.
      * If not supplied, derive from the observed range using a configurable
        fraction of the span.
    """
    span = max(1e-6, max_angle - min_angle)
    if movement_type == "flexion_extension":
        # Larger angle = extended. "start" is at the extended end.
        if start_threshold is None:
            start_threshold = max_angle - 0.20 * span
        if bottom_threshold is None:
            bottom_threshold = min_angle + 0.20 * span
        # end_threshold = angle used to confirm we have returned from a bottom.
        if end_threshold is None:
            end_threshold = start_threshold
    elif movement_type == "extension_flexion":
        if start_threshold is None:
            start_threshold = min_angle + 0.20 * span
        if bottom_threshold is None:
            bottom_threshold = max_angle - 0.20 * span
        if end_threshold is None:
            end_threshold = start_threshold
    else:
        raise ValueError(f"Unknown movement_type: {movement_type!r}")

    return {
        "start_threshold": float(start_threshold),
        "end_threshold": float(end_threshold),
        "bottom_threshold": float(bottom_threshold),
    }


@dataclass
class ExercisePhaseConfig:
    """Configuration for the phase/rep detector of a single exercise.

    Args:
        exercise: human-friendly exercise name (e.g. ``"elbow_flexion"``).
        primary_angle: name of the Step 2 angle used as movement signal
            (e.g. ``"left_elbow"``). Should be the most reliable angle for the
            exercise.
        movement_type: one of :data:`MOVEMENT_TYPES`. Tells the state machine
            which direction is flexion (bottom) vs extension (start).
        start_threshold: angle value that marks the extended/start region.
        end_threshold: angle value that confirms a return to start after a rep.
        bottom_threshold: angle value that marks the flexed/bottom region.
        min_rep_duration: minimum seconds a full rep must take (noise guard).
        min_time_between_reps: minimum seconds between consecutive rep starts.
        min_movement_range: minimum angular span (deg) a rep must traverse.
        smoothing_window: number of frames for the moving-average smoother.
        variants: optional sequence of secondary angle names considered for
            bilateral selection when the primary angle is missing. If not empty,
            the detector may fall back to these (same movement profile).
    """

    exercise: str = "exercise"
    primary_angle: str = "left_elbow"
    movement_type: str = "flexion_extension"
    start_threshold: Optional[float] = None
    end_threshold: Optional[float] = None
    bottom_threshold: Optional[float] = None
    min_rep_duration: float = 0.3
    min_time_between_reps: float = 0.2
    min_movement_range: float = 30.0
    smoothing_window: int = 5
    variants: List[str] = None

    def __post_init__(self) -> None:
        if self.movement_type not in MOVEMENT_TYPES:
            raise ValueError(f"movement_type must be one of {MOVEMENT_TYPES}")
        if self.variants is None:
            self.variants = []
        self.smoothing_window = max(1, int(self.smoothing_window))

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration dict."""
        return asdict(self)


def config_from_angle_sequence(
    angle_sequence,
    exercise: str = "elbow_flexion",
    primary_angle: str = "left_elbow",
    movement_type: str = "flexion_extension",
    variants: Optional[List[str]] = None,
    **overrides,
) -> ExercisePhaseConfig:
    """Derive an :class:`ExercisePhaseConfig` from a real :class:`AngleSequence`.

    Uses the observed min/max of the primary angle to auto-fill threshold values
    (start/end/bottom) unless explicitly overridden. All other fields come from
    ``overrides`` or sensible defaults. This keeps the detector from being tuned
    to one hard-coded universal threshold.

    Args:
        angle_sequence: an :class:`AngleSequence` from Step 2.
        exercise, primary_angle, movement_type: see :class:`ExercisePhaseConfig`.
        variants: optional fallback angle names (e.g. ``["right_elbow"]``).
        **overrides: any other field of :class:`ExercisePhaseConfig` to set
            explicitly (e.g. ``smoothing_window=7``).

    Returns:
        A populated :class:`ExercisePhaseConfig`.
    """
    vals = [
        af.angles[primary_angle] for af in angle_sequence.frames
        if af.angles.get(primary_angle) is not None
    ]
    if not vals:
        raise ValueError(f"No valid values found for primary_angle={primary_angle!r} "
                         f"in the given angle sequence.")

    min_angle = min(vals)
    max_angle = max(vals)

    thr = _resolve_thresholds(
        min_angle=min_angle,
        max_angle=max_angle,
        start_threshold=overrides.pop("start_threshold", None),
        end_threshold=overrides.pop("end_threshold", None),
        bottom_threshold=overrides.pop("bottom_threshold", None),
        movement_type=movement_type,
    )

    cfg = ExercisePhaseConfig(
        exercise=exercise,
        primary_angle=primary_angle,
        movement_type=movement_type,
        start_threshold=thr["start_threshold"],
        end_threshold=thr["end_threshold"],
        bottom_threshold=thr["bottom_threshold"],
        variants=list(variants) if variants else [],
    )
    # Apply remaining overrides.
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown ExercisePhaseConfig field: {k!r}")
        setattr(cfg, k, v)
    # Re-run post-init validation for any derived fields.
    cfg.__post_init__()
    return cfg

