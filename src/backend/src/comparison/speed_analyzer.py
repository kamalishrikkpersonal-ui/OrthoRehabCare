"""Speed analysis for OrthoRehab AI Step 5.

Speed analysis is intentionally SEPARATE from DTW. DTW aligns trajectories
allowing different temporal speeds; speed analysis independently measures the
duration ratio and classifies execution as too_fast / normal_speed / too_slow.

Thresholds are configurable (not hardcoded inside the logic). The default
thresholds are expressed as a ratio of reference_duration / patient_duration:

    speed_ratio = reference_duration / patient_duration

* speed_ratio > threshold_fast  -> patient faster than reference -> too_fast
* speed_ratio < threshold_slow  -> patient slower than reference -> too_slow
* otherwise                     -> normal_speed

Example: reference takes 3.0s, patient takes 6.0s -> ratio 0.5 -> too_slow.
Example: reference takes 3.0s, patient takes 1.5s -> ratio 2.0 -> too_fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas.comparison import SpeedAssessment

# Default ratio thresholds (configurable).
# ratio = reference_duration / patient_duration
DEFAULT_THRESHOLD_FAST = 1.5   # ratio >= 1.5 -> patient ~1.5x faster than ref
DEFAULT_THRESHOLD_SLOW = 0.67  # ratio <= 0.67 -> patient ~1.5x slower than ref

# Speed status labels.
SPEED_NORMAL = "normal_speed"
SPEED_TOO_FAST = "too_fast"
SPEED_TOO_SLOW = "too_slow"


@dataclass
class SpeedConfig:
    """Configurable thresholds for speed classification."""

    threshold_fast: float = DEFAULT_THRESHOLD_FAST
    threshold_slow: float = DEFAULT_THRESHOLD_SLOW

    def __post_init__(self) -> None:
        if self.threshold_fast <= 0:
            raise ValueError("threshold_fast must be > 0")
        if self.threshold_slow <= 0:
            raise ValueError("threshold_slow must be > 0")
        if self.threshold_slow > self.threshold_fast:
            raise ValueError("threshold_slow must be <= threshold_fast")


class SpeedAnalyzer:
    """Analyze the speed ratio between reference and patient durations."""

    def __init__(self, config: Optional[SpeedConfig] = None) -> None:
        self.config = config or SpeedConfig()

    def analyze(self, reference_duration: float, patient_duration: float) -> SpeedAssessment:
        """Classify the speed of a patient movement vs a reference.

        Args:
            reference_duration: duration (s) of the therapist reference movement.
            patient_duration: duration (s) of the patient's movement.

        Returns:
            A :class:`SpeedAssessment` with duration, ratio and status.
        """
        if reference_duration <= 0 or patient_duration <= 0:
            # Degenerate durations -> cannot classify; default to normal.
            return SpeedAssessment(
                reference_duration=reference_duration,
                patient_duration=patient_duration,
                speed_ratio=1.0,
                speed_status=SPEED_NORMAL,
                threshold_fast=self.config.threshold_fast,
                threshold_slow=self.config.threshold_slow,
            )

        ratio = reference_duration / patient_duration

        if ratio >= self.config.threshold_fast:
            status = SPEED_TOO_FAST
        elif ratio <= self.config.threshold_slow:
            status = SPEED_TOO_SLOW
        else:
            status = SPEED_NORMAL

        return SpeedAssessment(
            reference_duration=round(reference_duration, 3),
            patient_duration=round(patient_duration, 3),
            speed_ratio=round(ratio, 3),
            speed_status=status,
            threshold_fast=self.config.threshold_fast,
            threshold_slow=self.config.threshold_slow,
        )
