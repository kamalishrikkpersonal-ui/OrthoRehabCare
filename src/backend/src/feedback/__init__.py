"""Feedback package for OrthoRehab AI Step 5.

Provides deterministic fallback messages and a feedback service with cooldown /
duplicate suppression so the LLM is never called on every frame.
"""

from .feedback_service import FeedbackService, FeedbackServiceConfig
from .feedback_templates import fallback_message, template_message

__all__ = [
    "FeedbackService",
    "FeedbackServiceConfig",
    "fallback_message",
    "template_message",
]
