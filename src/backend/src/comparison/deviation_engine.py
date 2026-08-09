"""Deterministic deviation engine for OrthoRehab AI Step 5.

Converts biomechanical measurements into structured, timestamped deviation
events. All thresholds are configurable (not hardcoded in the deviation logic).

Deviation types:
    insufficient_flexion, excessive_flexion,
    insufficient_extension, excessive_extension,
    trajectory_deviation, too_fast, too_slow, phase_deviation

Severity: minor | moderate | major.

The engine is deterministic and does not use AI. It is the source of truth for
deviation findings; the LLM only rephrases them into patient-friendly language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..schemas.comparison import AssessmentDeviation

# Deviation type constants.
DEV_INSUFFICIENT_FLEXION = "insufficient_flexion"
DEV_EXCESSIVE_FLEXION = "excessive_flexion"
DEV_INSUFFICIENT_EXTENSION = "insufficient_extension"
DEV_EXCESSIVE_EXTENSION = "excessive_extension"
DEV_TRAJECTORY = "trajectory_deviation"
DEV_TOO_FAST = "too_fast"
DEV_TOO_SLOW = "too_slow"
DEV_PHASE = "phase_deviation"

# Severity constants.
SEV_MINOR = "minor"
SEV_MODERATE = "moderate"
SEV_MAJOR = "major"


@dataclass
class DeviationConfig:
    """Configurable thresholds for the deviation engine.

    ``flexion_margin_deg`` and ``extension_margin_deg`` are the allowed tolerance
    (degrees) around the reference min/max before a flexion/extension deviation
    is flagged.
    """

    flexion_tolerance_deg: float = 8.0
    extension_tolerance_deg: float = 8.0
    trajectory_tolerance_deg: float = 12.0
    major_deviation_deg: float = 25.0
    moderate_deviation_deg: float = 12.0


def _severity_for(deviation_deg: float, config: DeviationConfig) -> str:
    """Map an angular deviation magnitude to a severity label."""
    ad = abs(deviation_deg)
    if ad >= config.major_deviation_deg:
        return SEV_MAJOR
    if ad >= config.moderate_deviation_deg:
        return SEV_MODERATE
    return SEV_MINOR


class DeviationEngine:
    """Deterministic deviation detection over reference vs patient measurements."""

    def __init__(self, config: Optional[DeviationConfig] = None, timestamp: float = 0.0) -> None:
        self.config = config or DeviationConfig()
        self.timestamp = timestamp

    # -- angle (min/max) deviations -----------------------------------------
    def detect_angle_deviations(
        self,
        joint: str,
        reference_min: Optional[float],
        reference_max: Optional[float],
        patient_min: Optional[float],
        patient_max: Optional[float],
        movement_type: str = "flexion_extension",
    ) -> List[AssessmentDeviation]:
        """Detect flexion/extension deviations from reference vs patient extrema.

        For ``flexion_extension``:
            * flexion (bottom) is the SMALLER angle -> reference_min.
            * extension (start) is the LARGER angle -> reference_max.

        For ``extension_flexion`` (inverted):
            * flexion (bottom) is the LARGER angle -> reference_max.
            * extension is the SMALLER angle -> reference_min.
        """
        deviations: List[AssessmentDeviation] = []

        if movement_type == "flexion_extension":
            ref_flex, ref_ext = reference_min, reference_max
            pat_flex, pat_ext = patient_min, patient_max
        else:  # extension_flexion
            ref_flex, ref_ext = reference_max, reference_min
            pat_flex, pat_ext = patient_min, patient_max

        # --- flexion (bottom) ---
        if ref_flex is not None and pat_flex is not None:
            # For flexion_extension, smaller angle = more flexed.
            # insufficient_flexion: patient did NOT flex enough (patient_flex > ref_flex)
            # excessive_flexion: patient flexed too much (patient_flex < ref_flex)
            if movement_type == "flexion_extension":
                diff = pat_flex - ref_flex  # positive => patient less flexed
                if diff > self.config.flexion_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_INSUFFICIENT_FLEXION,
                            joint=joint,
                            ref=round(ref_flex, 2),
                            pat=round(pat_flex, 2),
                            dev=round(diff, 2),
                        )
                    )
                elif diff < -self.config.flexion_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_EXCESSIVE_FLEXION,
                            joint=joint,
                            ref=round(ref_flex, 2),
                            pat=round(pat_flex, 2),
                            dev=round(diff, 2),
                        )
                    )
            else:
                diff = pat_flex - ref_flex
                if diff < -self.config.flexion_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_INSUFFICIENT_FLEXION,
                            joint=joint,
                            ref=round(ref_flex, 2),
                            pat=round(pat_flex, 2),
                            dev=round(diff, 2),
                        )
                    )
                elif diff > self.config.flexion_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_EXCESSIVE_FLEXION,
                            joint=joint,
                            ref=round(ref_flex, 2),
                            pat=round(pat_flex, 2),
                            dev=round(diff, 2),
                        )
                    )

        # --- extension ---
        if ref_ext is not None and pat_ext is not None:
            if movement_type == "flexion_extension":
                diff = pat_ext - ref_ext  # positive => patient over-extended
                if diff > self.config.extension_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_EXCESSIVE_EXTENSION,
                            joint=joint,
                            ref=round(ref_ext, 2),
                            pat=round(pat_ext, 2),
                            dev=round(diff, 2),
                        )
                    )
                elif diff < -self.config.extension_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_INSUFFICIENT_EXTENSION,
                            joint=joint,
                            ref=round(ref_ext, 2),
                            pat=round(pat_ext, 2),
                            dev=round(diff, 2),
                        )
                    )
            else:
                diff = pat_ext - ref_ext
                if diff > self.config.extension_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_EXCESSIVE_EXTENSION,
                            joint=joint,
                            ref=round(ref_ext, 2),
                            pat=round(pat_ext, 2),
                            dev=round(diff, 2),
                        )
                    )
                elif diff < -self.config.extension_tolerance_deg:
                    deviations.append(
                        self._dev(
                            type=DEV_INSUFFICIENT_EXTENSION,
                            joint=joint,
                            ref=round(ref_ext, 2),
                            pat=round(pat_ext, 2),
                            dev=round(diff, 2),
                        )
                    )

        return deviations

    # -- trajectory deviation ------------------------------------------------
    def detect_trajectory_deviation(
        self,
        joint: str,
        reference_trajectory: Sequence[Optional[float]],
        patient_trajectory: Sequence[Optional[float]],
    ) -> Optional[AssessmentDeviation]:
        """Detect a trajectory deviation from reference vs patient angle series.

        Uses the mean absolute error (MAE) between the two series after aligning
        them by length (both are treated as progress-aligned). If the MAE exceeds
        the trajectory tolerance, a ``trajectory_deviation`` is produced.
        """
        ref = [float(v) for v in reference_trajectory if v is not None]
        pat = [float(v) for v in patient_trajectory if v is not None]
        if len(ref) < 2 or len(pat) < 2:
            return None

        # Align lengths by resampling the patient to the reference length.
        pat_aligned = _resample_to_length(pat, len(ref))
        if pat_aligned is None:
            return None

        mae = sum(abs(r - p) for r, p in zip(ref, pat_aligned)) / len(ref)
        if mae > self.config.trajectory_tolerance_deg:
            return self._dev(
                type=DEV_TRAJECTORY,
                joint=joint,
                ref=round(float(max(ref)), 2),
                pat=round(float(max(pat_aligned)), 2),
                dev=round(mae, 2),
            )
        return None

    # -- speed deviation ------------------------------------------------------
    def speed_deviation(self, speed_status: str) -> Optional[AssessmentDeviation]:
        """Convert a speed status into a deviation event (if abnormal)."""
        if speed_status == "too_fast":
            return self._dev(type=DEV_TOO_FAST, joint="", ref=None, pat=None, dev=None)
        if speed_status == "too_slow":
            return self._dev(type=DEV_TOO_SLOW, joint="", ref=None, pat=None, dev=None)
        return None

    # -- internal helper ------------------------------------------------------
    def _dev(
        self,
        type: str,
        joint: str,
        ref: Optional[float],
        pat: Optional[float],
        dev: Optional[float],
    ) -> AssessmentDeviation:
        sev = _severity_for(dev, self.config) if dev is not None else SEV_MINOR
        return AssessmentDeviation(
            type=type,
            joint=joint,
            reference_value=ref,
            patient_value=pat,
            deviation=dev,
            severity=sev,
            timestamp=self.timestamp,
        )


def _resample_to_length(seq: List[float], n: int) -> Optional[List[float]]:
    """Resample a 1-D sequence to length ``n`` by linear interpolation.

    Returns None if the input is too short.
    """
    if len(seq) < 2 or n < 2:
        return None
    import numpy as np

    xs = np.linspace(0.0, 1.0, len(seq))
    xq = np.linspace(0.0, 1.0, n)
    return list(np.interp(xq, xs, np.array(seq, dtype=float)))
