"""DTW (Dynamic Time Warping) comparator for OrthoRehab AI Step 5.

DTW is an ALGORITHM, not AI. It measures how similar two temporal sequences are
after allowing different temporal speeds (warping). We use it on normalized
joint-angle trajectories (not raw RGB frames).

We use the ``dtaidistance`` library. The distance is normalized to a similarity
score in [0, 1] so it can be combined with other signals in evidence fusion.

Important: DTW handles temporal alignment. It does NOT capture speed differences
(that is the job of the SpeedAnalyzer). A slow reproduction of the same movement
will still yield high DTW similarity, and the speed analyzer will independently
report ``too_slow``.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

# dtaidistance is the only new dependency for Step 5.
# Its import prints a C-extension warning when the C libraries were not compiled
# (e.g. a pure-Python wheel). That is benign — dtaidistance falls back to pure
# Python — but it is noisy, so we suppress the import-time stderr chatter.
with contextlib.redirect_stderr(io.StringIO()):
    with contextlib.redirect_stdout(io.StringIO()):
        from dtaidistance import dtw

# Guards against sequences that are too short to reliably DTW.
MIN_SEQUENCE_LENGTH = 2


def _filled(seq: Sequence[Optional[float]]) -> List[float]:
    """Return only non-None values; None values are excluded (never fabricated)."""
    return [float(v) for v in seq if v is not None]


def dtw_distance(
    reference: Sequence[Optional[float]],
    patient: Sequence[Optional[float]],
    use_c: bool = True,
) -> Optional[float]:
    """Compute the DTW distance between two 1-D angle trajectories.

    Args:
        reference: reference trajectory values (None allowed, skipped).
        patient: patient trajectory values (None allowed, skipped).
        use_c: use the C-accelerated dtaidistance implementation (default True).

    Returns:
        The DTW distance, or None if either sequence is too short.
    """
    ref = _filled(reference)
    pat = _filled(patient)
    if len(ref) < MIN_SEQUENCE_LENGTH or len(pat) < MIN_SEQUENCE_LENGTH:
        return None
    try:
        if use_c and dtw.try_import_c():
            from dtaidistance.dtw import distance_fast
            return float(distance_fast(ref, pat))
        return float(dtw.distance(ref, pat))
    except Exception:
        # Fall back to pure-Python on any C extension issue.
        try:
            return float(dtw.distance(ref, pat))
        except Exception:
            return None


def dtw_similarity(
    reference: Sequence[Optional[float]],
    patient: Sequence[Optional[float]],
    use_c: bool = True,
) -> Optional[float]:
    """Return a normalized DTW similarity in [0, 1].

    The distance is normalized against an upper bound derived from the observed
    range of the two series; higher is more similar. Returns None if DTW cannot
    be computed.

    Formula: similarity = clip(1 - distance / (max_range + eps), 0, 1)
    """
    dist = dtw_distance(reference, patient, use_c=use_c)
    if dist is None:
        return None

    ref = _filled(reference)
    pat = _filled(patient)
    combined = ref + pat
    if not combined:
        return None
    data_range = max(combined) - min(combined)
    if data_range < 1e-9:
        # Identical flat sequences -> perfectly similar.
        return 1.0

    eps = 1e-9
    sim = 1.0 - (dist / (data_range + eps))
    return float(max(0.0, min(1.0, sim)))


class DTWComparator:
    """Compute DTW similarity/distance between reference and patient trajectories."""

    def __init__(self, use_c: bool = True) -> None:
        self.use_c = use_c

    def compare(
        self,
        reference_trajectory: Sequence[Optional[float]],
        patient_trajectory: Sequence[Optional[float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(similarity, distance)`` for the two trajectories.

        Returns ``(None, None)`` if DTW cannot be computed.
        """
        sim = dtw_similarity(reference_trajectory, patient_trajectory, use_c=self.use_c)
        dist = dtw_distance(reference_trajectory, patient_trajectory, use_c=self.use_c)
        return sim, dist
