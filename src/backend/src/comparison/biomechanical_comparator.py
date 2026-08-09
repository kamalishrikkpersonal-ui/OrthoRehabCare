"""Biomechanical comparison for OrthoRehab AI Step 5.

This comparator REUSES the existing Step 2 joint-angle calculations (the
``AngleSequence`` already contains per-frame elbow/shoulder angles). It does NOT
recalculate angles from landmarks. It compares the therapist reference profile
against a patient's angle sequence using:

* minimum angle
* maximum angle
* range of motion (ROM)
* reference trajectory (normalized) vs patient trajectory

All measurements come from the existing Step 1-4 pipeline. Nothing is invented:
if a value is not measurable, it is left as None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..pose.angle_engine import AngleSequence
from ..pose.reference_profile import ReferenceExerciseProfile
from .deviation_engine import DeviationEngine, DeviationConfig
from ..schemas.comparison import AssessmentDeviation, DeviatedRange


@dataclass
class BiomechanicalResult:
    """Outcome of a biomechanical comparison."""

    primary_angle: str
    deviations: List[AssessmentDeviation] = None
    biomechanical_accuracy: Optional[float] = None
    rom_accuracy: Optional[float] = None
    ranges: Dict[str, DeviatedRange] = None

    def __post_init__(self) -> None:
        if self.deviations is None:
            self.deviations = []
        if self.ranges is None:
            self.ranges = {}


class BiomechanicalComparator:
    """Compare reference vs patient joint-angle statistics.

    Args:
        deviation_engine: a :class:`DeviationEngine` used to produce deviations.
    """

    def __init__(self, deviation_engine: Optional[DeviationEngine] = None) -> None:
        self.deviation_engine = deviation_engine or DeviationEngine(DeviationConfig())

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _angle_stats(
        angle_sequence: AngleSequence, angle_name: str
    ) -> Dict[str, Optional[float]]:
        """Compute min/max/mean of a single angle over a patient sequence."""
        vals = [af.angles.get(angle_name) for af in angle_sequence.frames]
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"min": None, "max": None, "mean": None, "n": 0}
        return {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
            "n": len(vals),
        }

    @staticmethod
    def _patient_trajectory(
        angle_sequence: AngleSequence, angle_name: str
    ) -> List[Optional[float]]:
        """Return the raw patient primary-angle series (None preserved)."""
        return [af.angles.get(angle_name) for af in angle_sequence.frames]

    # -- main ----------------------------------------------------------------
    def compare(
        self,
        reference_profile: ReferenceExerciseProfile,
        patient_angle_sequence: AngleSequence,
    ) -> BiomechanicalResult:
        """Compare a reference profile against a patient angle sequence.

        Args:
            reference_profile: the Step 4 profile (contains reference stats and
                normalized trajectory).
            patient_angle_sequence: Step 2 output for the patient.

        Returns:
            A :class:`BiomechanicalResult` with deviations + accuracy metrics.
        """
        primary = reference_profile.primary_angle
        pat_stats = self._angle_stats(patient_angle_sequence, primary)

        # Reference stats from the profile.
        ref_stats = reference_profile.angle_statistics.get(primary)
        ref_min = ref_stats.min_angle if ref_stats else None
        ref_max = ref_stats.max_angle if ref_stats else None
        ref_range = ref_stats.range if ref_stats else None

        pat_min = pat_stats["min"]
        pat_max = pat_stats["max"]
        pat_range = (pat_max - pat_min) if (pat_max is not None and pat_min is not None) else None

        ranges = {
            primary: DeviatedRange(
                joint=primary,
                reference_min=ref_min,
                reference_max=ref_max,
                patient_min=pat_min,
                patient_max=pat_max,
                reference_range=ref_range,
                patient_range=pat_range,
            )
        }

        # Produce deterministic deviations on min/max angle deviations.
        deviations = self.deviation_engine.detect_angle_deviations(
            joint=primary,
            reference_min=ref_min,
            reference_max=ref_max,
            patient_min=pat_min,
            patient_max=pat_max,
        )

        # NOTE: trajectory_deviation is intentionally NOT computed here. It is
        # computed in the ComparisonEngine against the normalized patient cycle
        # (rep-span aware) so that the reference normalized single cycle and the
        # patient normalized single cycle use IDENTICAL preprocessing. Comparing
        # the reference single cycle against the patient's raw full-video sequence
        # here caused a false-positive major trajectory deviation for identical
        # reference/patient videos.

        # ROM accuracy: closeness of patient range to reference range, in [0,1].
        rom_accuracy = None
        if ref_range is not None and pat_range is not None and ref_range > 0:
            ratio = pat_range / ref_range
            # 1.0 => perfect; penalize both under- and over-shooting.
            rom_accuracy = float(max(0.0, min(1.0, 1.0 - abs(1.0 - ratio))))

        # Biomechanical accuracy: derived from deviations and ROM.
        biomech_accuracy = self._biomechanical_accuracy(deviations, rom_accuracy)

        return BiomechanicalResult(
            primary_angle=primary,
            deviations=deviations,
            biomechanical_accuracy=biomech_accuracy,
            rom_accuracy=rom_accuracy,
            ranges=ranges,
        )

    @staticmethod
    def _biomechanical_accuracy(
        deviations: List[AssessmentDeviation], rom_accuracy: Optional[float]
    ) -> Optional[float]:
        """Combine deviation severity and ROM accuracy into a single 0..1 score."""
        if rom_accuracy is not None:
            # Start from ROM accuracy; penalize for each deviation by severity.
            score = rom_accuracy
            for d in deviations:
                if d.severity == "major":
                    score -= 0.25
                elif d.severity == "moderate":
                    score -= 0.15
                elif d.severity == "minor":
                    score -= 0.05
            return float(max(0.0, min(1.0, score)))
        return None
