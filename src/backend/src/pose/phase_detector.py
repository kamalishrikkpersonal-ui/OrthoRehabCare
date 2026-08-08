"""Deterministic exercise phase detection + repetition counting for Step 3.

Consumes an :class:`AngleSequence` (Step 2) and produces phase events and
repetitions. The movement direction and all thresholds are supplied by an
:class:`ExercisePhaseConfig` — no universal angle threshold is hard-coded here.

State machine (configurable direction via ``movement_type``):

    START --[flex crossing bottom]--> FLEXING --[crossing bottom]--> BOTTOM
    BOTTOM --[ext crossing bottom]--> EXTENDING --[crossing start]--> START
    START --(return to start completes)--> ONE REP

A rep is only *counted* after a full, valid cycle:

  * the movement must traverse at least ``min_movement_range`` degrees,
  * the rep must last at least ``min_rep_duration`` seconds,
  * consecutive reps must be separated by at least ``min_time_between_reps``,
  * the phase order must be START -> FLEXING -> BOTTOM -> EXTENDING -> START.

Smoothing is a configurable moving average applied to the primary angle series
before the state machine runs. Missing/``None`` primary-angle values are handled
safely: the smoother simply skips them (they do not advance the state machine),
and the detector never fabricates an angle when the underlying landmark(s) are
unavailable.

This module is pure Python + NumPy (used only for the moving average). It has no
dependency on FastAPI, MongoDB, React, browser APIs, OpenCV display, the LSTM, or
Groq, so an equivalent TypeScript port is possible later for the patient's
browser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .angle_engine import AngleSequence, AngleFrame
from .exercise_config import ExercisePhaseConfig

# Public phase names produced by the state machine.
PHASE_START = "start"
PHASE_FLEXING = "flexing"
PHASE_BOTTOM = "bottom"
PHASE_EXTENDING = "extending"


@dataclass
class PhaseEvent:
    """A single phase transition with frame/time context."""

    phase: str
    frame_index: int
    timestamp: float
    angle: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepResult:
    """A validated, fully completed repetition."""

    rep_number: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_timestamp: float
    bottom_timestamp: float
    end_timestamp: float
    min_angle: float
    max_angle: float
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepSummary:
    """Summarized result of phase detection for one angle series."""

    valid_frames: int
    primary_angle: str
    min_angle: Optional[float]
    max_angle: Optional[float]
    phases: List[PhaseEvent] = field(default_factory=list)
    repetitions: List[RepResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid_frames": self.valid_frames,
            "primary_angle": self.primary_angle,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "phases": [p.to_dict() for p in self.phases],
            "repetitions": [r.to_dict() for r in self.repetitions],
        }


def moving_average(window: int) -> "np.typing.NDArray":
    """Return a smoothing kernel (configurable size)."""
    return np.ones(window, dtype=float) / float(window)


def smooth_angles(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Smooth a series of (possibly None) angles with a moving average.

    ``None`` values are left as ``None`` (they simply do not participate in the
    average). The smoothing window is a simple configurable integer; larger
    values smooth more aggressively but also add latency.
    """
    if window <= 1:
        return list(values)
    # Build an array replacing None with NaN for the average, then restore None.
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    n = len(arr)
    if n == 0:
        return list(values)
    kernel = moving_average(window)
    # Use np.apply_along_axis with a valid-mask padding approach:
    # pad with NaN so the moving average ignores missing values.
    padded = np.pad(arr, (window // 2, window // 2), constant_values=np.nan)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        seg = padded[i:i + window]
        valid = ~np.isnan(seg)
        if valid.any():
            out[i] = float(np.nanmean(seg))
    return [None if math.isnan(v) else round(float(v), 2) for v in out]


def _crosses(low: float, high: float, value: Optional[float]) -> Optional[bool]:
    """Return True if value is above high, False if below low, None otherwise."""
    if value is None:
        return None
    if value >= high:
        return True
    if value <= low:
        return False
    return None


class PhaseDetector:
    """Deterministic state machine for phase detection + rep counting.

    Args:
        config: an :class:`ExercisePhaseConfig` fully populated with thresholds.
    """

    def __init__(self, config: ExercisePhaseConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.primary_angle is None or not self.config.primary_angle.strip():
            raise ValueError("ExercisePhaseConfig.primary_angle cannot be empty.")
        if self.config.start_threshold is None or self.config.end_threshold is None \
                or self.config.bottom_threshold is None:
            raise ValueError(
                "ExercisePhaseConfig requires start_threshold, end_threshold and "
                "bottom_threshold to be populated (can derive them with "
                "config_from_angle_sequence)."
            )

    # -- direction helpers ---------------------------------------------------
    def _is_flexing(self, value: Optional[float]) -> bool:
        """True when the angle is moving toward the bottom (flexed) region.

        For ``flexion_extension`` bottom is the *smaller* angle; for
        ``extension_flexion`` bottom is the *larger* angle.
        """
        if value is None:
            return False
        if self.config.movement_type == "flexion_extension":
            return value <= self.config.bottom_threshold
        return value >= self.config.bottom_threshold

    def _is_extended(self, value: Optional[float]) -> bool:
        """True when the angle is back in the extended (start) region."""
        if value is None:
            return False
        if self.config.movement_type == "flexion_extension":
            return value >= self.config.start_threshold
        return value <= self.config.start_threshold

    # -- main API ------------------------------------------------------------
    def detect(self, angle_sequence: AngleSequence) -> RepSummary:
        """Run phase detection + rep counting over an entire AngleSequence."""
        frames = angle_sequence.frames
        primary = self.config.primary_angle

        # Smooth the primary angle series.
        raw = [af.angles.get(primary) for af in frames]
        smoothed = smooth_angles(raw, self.config.smoothing_window)

        valid = [v for v in smoothed if v is not None]
        min_angle = min(valid) if valid else None
        max_angle = max(valid) if valid else None

        phases: List[PhaseEvent] = []
        reps: List[RepResult] = []

        state = PHASE_START
        rep_start_index: Optional[int] = None
        rep_start_time: Optional[float] = None
        flex_start_index: Optional[int] = None
        flex_start_time: Optional[float] = None
        rep_bottom_index: Optional[int] = None
        rep_bottom_time: Optional[float] = None
        rep_bottom_value: Optional[float] = None
        rep_min: Optional[float] = None
        rep_max: Optional[float] = None
        last_rep_end_time: Optional[float] = None

        def frame_of(i: int) -> AngleFrame:
            return frames[i]

        add_phase = phases.append
        for i, val in enumerate(smoothed):
            af = frame_of(i)
            if val is None:
                # Missing angle: do not advance state, do not invent a value.
                continue

            # Track running min/max for the current rep.
            if rep_min is None or val < rep_min:
                rep_min = val
            if rep_max is None or val > rep_max:
                rep_max = val

            if state == PHASE_START:
                # A new rep begins when we leave the extended region toward the
                # bottom (flex). The first time we clearly flex counts as start.
                if self._is_flexing(val):
                    flex_start_index = af.frame_index
                    flex_start_time = af.timestamp
                    rep_start_index = af.frame_index
                    rep_start_time = af.timestamp
                    rep_min = val
                    rep_max = val
                    state = PHASE_FLEXING
                    add_phase(PhaseEvent(PHASE_FLEXING, af.frame_index, af.timestamp, val))
                continue

            if state == PHASE_FLEXING:
                # Still flexing; when we cross the bottom threshold we enter BOTTOM.
                if self._is_extended(val):
                    # Reached start again without ever reaching bottom -> incomplete.
                    # Reset, but keep the flex-start info for a fresh attempt.
                    state = PHASE_START
                    flex_start_index = None
                    flex_start_time = None
                    rep_min = None
                    rep_max = None
                    continue
                if self._is_flexing(val) and val <= self.config.bottom_threshold:
                    rep_bottom_index = af.frame_index
                    rep_bottom_time = af.timestamp
                    rep_bottom_value = val
                    state = PHASE_BOTTOM
                    add_phase(PhaseEvent(PHASE_BOTTOM, af.frame_index, af.timestamp, val))
                continue

            if state == PHASE_BOTTOM:
                # Stay in BOTTOM until we clearly start extending.
                if not self._is_flexing(val):
                    state = PHASE_EXTENDING
                    add_phase(PhaseEvent(PHASE_EXTENDING, af.frame_index, af.timestamp, val))
                continue

            if state == PHASE_EXTENDING:
                # Extending; when we return to the extended/start region we
                # complete a rep (subject to validation).
                if self._is_extended(val):
                    # Validate & potentially finalize the rep.
                    self._finalize_rep(
                        reps=reps,
                        add_phase=add_phase,
                        rep_start_index=rep_start_index,
                        rep_start_time=rep_start_time,
                        rep_bottom_index=rep_bottom_index,
                        rep_bottom_time=rep_bottom_time,
                        rep_bottom_value=rep_bottom_value,
                        rep_min=rep_min,
                        rep_max=rep_max,
                        end_frame=af.frame_index,
                        end_time=af.timestamp,
                        last_rep_end_time=last_rep_end_time,
                    )
                    if reps and reps[-1].end_frame == af.frame_index:
                        last_rep_end_time = af.timestamp
                    # Reset for next rep regardless (even if rejected).
                    state = PHASE_START
                    rep_start_index = None
                    rep_start_time = None
                    flex_start_index = None
                    flex_start_time = None
                    rep_bottom_index = None
                    rep_bottom_time = None
                    rep_bottom_value = None
                    rep_min = None
                    rep_max = None
                    add_phase(PhaseEvent(PHASE_START, af.frame_index, af.timestamp, val))
                continue

        return RepSummary(
            valid_frames=len(valid),
            primary_angle=primary,
            min_angle=min_angle,
            max_angle=max_angle,
            phases=phases,
            repetitions=reps,
        )

    def _finalize_rep(
        self,
        reps: List[RepResult],
        add_phase,
        rep_start_index: Optional[int],
        rep_start_time: Optional[float],
        rep_bottom_index: Optional[int],
        rep_bottom_time: Optional[float],
        rep_bottom_value: Optional[float],
        rep_min: Optional[float],
        rep_max: Optional[float],
        end_frame: int,
        end_time: float,
        last_rep_end_time: Optional[float],
    ) -> None:
        """Validate a candidate rep and append it if it passes all checks."""
        if None in (rep_start_index, rep_bottom_index, rep_start_time, rep_bottom_time,
                    rep_min, rep_max):
            return

        duration = end_time - rep_start_time
        movement_range = (rep_max - rep_min) if rep_max is not None and rep_min is not None else 0.0

        # 1) Minimum movement range.
        if movement_range < self.config.min_movement_range:
            return
        # 2) Minimum rep duration.
        if duration < self.config.min_rep_duration:
            return
        # 3) Minimum time between consecutive reps.
        if last_rep_end_time is not None and \
                (rep_start_time - last_rep_end_time) < self.config.min_time_between_reps:
            return

        reps.append(
            RepResult(
                rep_number=len(reps) + 1,
                start_frame=rep_start_index,
                bottom_frame=rep_bottom_index,
                end_frame=end_frame,
                start_timestamp=rep_start_time,
                bottom_timestamp=rep_bottom_time,
                end_timestamp=end_time,
                min_angle=rep_min,
                max_angle=rep_max,
                duration=round(float(duration), 3),
            )
        )
