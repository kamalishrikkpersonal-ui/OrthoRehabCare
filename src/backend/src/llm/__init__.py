"""LLM package for OrthoRehab AI Step 5.

Provides a minimal Groq client and prompt helpers for rehabilitation feedback.
The client is intentionally thin (uses standard-library urllib) and always has a
deterministic fallback so the comparison engine keeps working if the LLM is
unavailable.
"""

from .groq_client import GroqClient, LLMUnavailableError
from .rehabilitation_feedback import build_system_prompt, build_user_prompt

__all__ = [
    "GroqClient",
    "LLMUnavailableError",
    "build_system_prompt",
    "build_user_prompt",
]
