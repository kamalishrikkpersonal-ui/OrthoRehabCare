"""High-level rep counting facade for OrthoRehab AI Step 3.

Ties together the deterministic phase detector with an :class:`ExercisePhaseConfig`
and produces an :class:`ExerciseAnalysis` result object. It also provides a small
helper for bilateral primary-angle selection so a future step can combine left and
right signals.

Pipeline (unchanged, each layer independently usable)::

    LandmarkSequence  (Step 1)
      -> AngleSequence (Step 2)
      -> ExerciseAnalysis (Step 3, this module)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .angle_engine import AngleSequence
from .exercise_config import ExercisePhaseConfig, config_from_angle_sequence
from .phase_detector import (
    PhaseDetector,
    PhaseEvent,
    RepResult,
    RepSummary,
)


@dataclass
class ExerciseAnalysis:
    """Result of phase/rep analysis for one exercise over an AngleSequence."""

    exercise: str
    primary_angle: str
    total_reps: int
    phases: List[PhaseEvent] = field(default_factory=list)
    repetitions: List[RepResult] = field(default_factory=list)
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    valid_frames: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "exercise": self.exercise,
            "primary_angle": self.primary_angle,
            "total_reps": self.total_reps,
            "valid_frames": self.valid_frames,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "phases": [p.to_dict() for p in self.phases],
            "repetitions": [r.to_dict() for r in self.repetitions],
        }


def select_primary_angle(
    angle_sequence: AngleSequence,
    preferred: str,
    candidates: Optional[List[str]] = None,
    min_valid_frames: int = 1,
) -> str:
    """Pick the most reliable primary angle among candidates.

    "Reliable" = the angle with the most valid (non-None) frames across the
    sequence, preferring the user's ``preferred`` name when it is usable.

    Args:
        angle_sequence: Step 2 angle sequence.
        preferred: first-choice angle name.
        candidates: optional list of fallback angle names. If None, defaults to
            the preferred angle alone.
        min_valid_frames: minimum number of valid frames required for an angle
            to be considered usable.

    Returns:
        The chosen angle name.
    """
    if candidates is None:
        candidates = [preferred]

    names = [preferred] + [c for c in candidates if c != preferred]
    best_name = preferred
    best_count = -1
    for name in names:
        count = sum(
            1 for af in angle_sequence.frames if af.angles.get(name) is not None
        )
        if count > best_count:
            best_count = count
            best_name = name
    # If even the best is unusable, fall back to preferred anyway (caller will
    # handle the empty result gracefully).
    if best_count < min_valid_frames:
        return preferred
    return best_name


def count_reps(
    angle_sequence: AngleSequence,
    config: Optional[ExercisePhaseConfig] = None,
    exercise: str = "elbow_flexion",
    primary_angle: str = "left_elbow",
    movement_type: str = "flexion_extension",
    auto_derive: bool = True,
    **overrides,
) -> ExerciseAnalysis:
    """Run Step 3 phase/rep counting over an :class:`AngleSequence`.

    Args:
        angle_sequence: output of Step 2.
        config: an explicit :class:`ExercisePhaseConfig`. If provided it wins.
        exercise: exercise name (used when deriving a config).
        primary_angle: primary angle name (used when deriving a config).
        movement_type: movement profile (used when deriving a config).
        auto_derive: if True (default) and no config given, derive thresholds
            from the angle sequence via :func:`config_from_angle_sequence`.
        **overrides: extra fields passed to the config (e.g. ``smoothing_window``).

    Returns:
        An :class:`ExerciseAnalysis`.
    """
    if config is None:
        if auto_derive:
            config = config_from_angle_sequence(
                angle_sequence,
                exercise=exercise,
                primary_angle=primary_angle,
                movement_type=movement_type,
                **overrides,
            )
        else:
            config = ExercisePhaseConfig(
                exercise=exercise,
                primary_angle=primary_angle,
                movement_type=movement_type,
                start_threshold=overrides.pop("start_threshold", None),
                end_threshold=overrides.pop("end_threshold", None),
                bottom_threshold=overrides.pop("bottom_threshold", None),
                **overrides,
            )

    detector = PhaseDetector(config)
    summary: RepSummary = detector.detect(angle_sequence)

    return ExerciseAnalysis(
        exercise=config.exercise,
        primary_angle=config.primary_angle,
        total_reps=len(summary.repetitions),
        phases=summary.phases,
        repetitions=summary.repetitions,
        min_angle=summary.min_angle,
        max_angle=summary.max_angle,
        valid_frames=summary.valid_frames,
    )
