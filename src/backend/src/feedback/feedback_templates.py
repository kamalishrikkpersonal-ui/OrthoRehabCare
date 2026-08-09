"""Deterministic fallback feedback templates for OrthoRehab AI Step 5.

Used when the LLM is unavailable, times out, rate-limited, or returns a
malformed response. These are static, deterministic messages — never computed
from measurements — so the comparison engine always produces feedback.
"""

from __future__ import annotations

from typing import List, Optional

# Deviation-type -> fallback message.
FALLBACK_BY_TYPE = {
    "insufficient_flexion": "Try bending a little further while keeping the movement controlled.",
    "excessive_flexion": "Ease off slightly at the bottom of the movement to protect the joint.",
    "insufficient_extension": "Try straightening a little more at the end of the movement.",
    "excessive_extension": "Ease off slightly at the end of the movement to keep it controlled.",
    "trajectory_deviation": "Try to follow the same path as the demonstration, keeping the movement smooth.",
    "too_fast": "Slow down slightly and keep the movement controlled.",
    "too_slow": "Try a slightly quicker but controlled pace.",
    "phase_deviation": "Try to follow the same rhythm as the demonstration, pausing at the same points.",
}

# Generic fallbacks.
GOOD_FORM_FALLBACK = "Good movement. Keep going."
GENERIC_FALLBACK = "Keep the movement smooth and controlled."


def template_message(deviation_type: Optional[str]) -> str:
    """Return a deterministic fallback message for a deviation type."""
    if deviation_type is None:
        return GENERIC_FALLBACK
    return FALLBACK_BY_TYPE.get(deviation_type, GENERIC_FALLBACK)


def fallback_message(deviation_types: List[str]) -> str:
    """Build a fallback message from a list of deviation types.

    Prefers the most severe/actionable deviation first and appends a generic
    closing if more than one issue is present.
    """
    if not deviation_types:
        return GOOD_FORM_FALLBACK

    primary = deviation_types[0]
    msg = template_message(primary)
    if len(deviation_types) > 1:
        msg += " " + GENERIC_FALLBACK
    return msg
