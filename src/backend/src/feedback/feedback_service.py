"""Feedback service for OrthoRehab AI Step 5.

Implements the real-time feedback pipeline:

    Frame -> MediaPipe -> movement features -> comparison -> persistent
    deviation? -> feedback event -> LLM -> patient feedback

This service enforces:
* persistence threshold (a deviation must persist across several evaluations
  before it triggers feedback),
* cooldown (no feedback until ``cooldown_seconds`` have elapsed since the last),
* duplicate suppression (identical deviation sets are not re-emitted).

The LLM is never called on every frame. It is only called when a new, distinct
feedback event is triggered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.groq_client import GroqClient
from ..llm.rehabilitation_feedback import messages_for_findings
from ..schemas.feedback import FeedbackEvent, FeedbackMessage
from .feedback_templates import GOOD_FORM_FALLBACK, fallback_message


@dataclass
class FeedbackServiceConfig:
    """Configuration for the feedback service."""

    persistence_threshold: int = 3      # N consecutive evaluations with same deviation set
    cooldown_seconds: float = 5.0       # min time between feedback events
    enabled: bool = True


class FeedbackService:
    """Real-time feedback with persistence, cooldown and duplicate suppression.

    Args:
        groq: optional :class:`GroqClient`. If None, a client is created from
            the ``GROQ_API_KEY`` env var. If unavailable, fallback messages are
            used.
        config: :class:`FeedbackServiceConfig`.
    """

    def __init__(
        self,
        groq: Optional[GroqClient] = None,
        config: Optional[FeedbackServiceConfig] = None,
    ) -> None:
        self.groq = groq or GroqClient()
        self.config = config or FeedbackServiceConfig()
        self._last_feedback_time: Optional[float] = None
        self._current_deviation_sig: Optional[frozenset] = None
        self._persistence_count: int = 0
        self._suppressed_events: int = 0

    # -- internals -----------------------------------------------------------
    def _deviation_signature(self, deviations: List[Dict[str, Any]]) -> frozenset:
        """A stable signature of the current deviation set (for dedup)."""
        return frozenset(
            (d.get("type", ""), d.get("joint", ""), d.get("severity", ""))
            for d in deviations
        )

    def _is_cooldown(self, now: float) -> bool:
        if self._last_feedback_time is None:
            return False
        return (now - self._last_feedback_time) < self.config.cooldown_seconds

    # -- main API ------------------------------------------------------------
    def evaluate(
        self,
        findings: Dict[str, Any],
        timestamp: float,
    ) -> Optional[FeedbackEvent]:
        """Evaluate structured findings and return a feedback event if warranted.

        Args:
            findings: structured comparison findings for this evaluation.
            timestamp: current timestamp (seconds).

        Returns:
            A :class:`FeedbackEvent` when feedback should be shown, else None.
        """
        if not self.config.enabled:
            return None

        deviations = findings.get("deviations", [])
        sig = self._deviation_signature(deviations)

        # Persistence: only react to a deviation set that persists.
        if sig == self._current_deviation_sig:
            self._persistence_count += 1
        else:
            self._current_deviation_sig = sig
            self._persistence_count = 1

        if self._persistence_count < self.config.persistence_threshold:
            return None

        # Cooldown: suppress if too soon after the last feedback.
        if self._is_cooldown(timestamp):
            self._suppressed_events += 1
            return None

        # Trigger feedback.
        deviation_types = [d.get("type", "") for d in deviations if d.get("type")]
        severity = "major" if any(
            d.get("severity") == "major" for d in deviations
        ) else "moderate" if deviations else "minor"

        # Try LLM, fall back to deterministic template.
        message, source = self._generate_message(findings, deviation_types)

        self._last_feedback_time = timestamp
        return FeedbackEvent(
            timestamp=round(timestamp, 3),
            deviation_types=deviation_types,
            severity=severity,
            message=message,
            source=source,
        )

    def _generate_message(
        self, findings: Dict[str, Any], deviation_types: List[str]
    ) -> tuple:
        """Return ``(message, source)``; source is 'llm' or 'fallback'."""
        if not self.groq.available:
            return fallback_message(deviation_types), "fallback"

        try:
            msgs = messages_for_findings(findings)
            content = self.groq.chat(msgs)
            if content:
                return content, "llm"
        except Exception:  # noqa: BLE001 — LLM failure must never crash feedback
            pass

        return fallback_message(deviation_types), "fallback"

    # -- stats ---------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "suppressed_events": self._suppressed_events,
            "last_feedback_time": self._last_feedback_time,
            "persistence_count": self._persistence_count,
        }

    def reset(self) -> None:
        self._last_feedback_time = None
        self._current_deviation_sig = None
        self._persistence_count = 0
        self._suppressed_events = 0
