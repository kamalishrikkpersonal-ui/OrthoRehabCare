"""Learned motion comparator interface for OrthoRehab AI Step 5.

This module defines the future extension point for a learned motion model. It is
an INTERFACE ONLY. No learned model is currently integrated (all evaluated
candidates require full-body skeleton topologies and/or dependency changes that
violate the hip-and-above scope).

The comparison engine depends only on this interface, never on a specific model.
When a valid learned model is available later, a concrete implementation can be
provided that returns a real ``learned_motion_similarity`` in [0, 1].

The interface makes it explicit that the learned component is currently
unavailable/optional: the default implementation returns ``None`` similarity
and never blocks the hybrid comparison engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


class LearnedMotionComparator:
    """Interface for comparing reference and patient motion via a learned model.

    Subclasses should implement :meth:`compare`. The interface accepts
    ``reference_sequence`` and ``patient_sequence`` (any serializable
    representation — e.g. a list of landmark frames or normalized feature
    vectors) and returns a similarity in [0, 1] (or None if unavailable).
    """

    name: str = "base"

    def available(self) -> bool:
        """Whether a learned model is actually available."""
        return False

    def compare(
        self,
        reference_sequence: Sequence[Any],
        patient_sequence: Sequence[Any],
    ) -> Optional[float]:
        """Return a learned motion similarity in [0, 1], or None if unavailable.

        The interface deliberately does not define a specific feature format so a
        future model adapter can map MediaPipe's hip-and-above representation to
        whatever the model needs.

        Args:
            reference_sequence: reference skeleton/motion sequence.
            patient_sequence: patient skeleton/motion sequence.

        Returns:
            Similarity in [0, 1] or None if the component is unavailable.
        """
        raise NotImplementedError(
            "LearnedMotionComparator.compare is not implemented. "
            "A future learned-model adapter should override this method."
        )


@dataclass
class LearnedMotionResult:
    """Result of a learned-motion comparison (if a model is available)."""

    similarity: Optional[float] = None
    model_name: str = ""
    available: bool = False


class UnavailableLearnedComparator(LearnedMotionComparator):
    """Default implementation: no learned model is available.

    It always reports ``available() == False`` and returns ``None`` similarity,
    so the rest of the comparison engine works identically whether or not a
    learned model is present.
    """

    name = "unavailable"

    def __init__(self) -> None:
        self._result = LearnedMotionResult(similarity=None, model_name="unavailable", available=False)

    def available(self) -> bool:
        return False

    def compare(
        self,
        reference_sequence: Sequence[Any],
        patient_sequence: Sequence[Any],
    ) -> Optional[float]:
        """Return None — no learned similarity is available."""
        return None

    def result(self) -> LearnedMotionResult:
        """Return the (empty) learned-motion result."""
        return self._result
