"""Feedback schemas for OrthoRehab AI Step 5.

A :class:`FeedbackEvent` is a triggered, persistent feedback signal (after
cooldown/duplicate suppression). A :class:`FeedbackMessage` is the final
patient-facing natural-language message (either LLM-generated or a
deterministic fallback).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackEvent:
    """A single feedback event produced by the real-time feedback service."""

    timestamp: float = 0.0
    deviation_types: List[str] = field(default_factory=list)
    severity: str = "minor"
    message: str = ""
    source: str = "fallback"  # "llm" | "fallback"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEvent":
        return cls(**data)


@dataclass
class FeedbackMessage:
    """A natural-language feedback message for the patient."""

    text: str = ""
    severity: str = "minor"
    source: str = "fallback"
    triggered_by: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackMessage":
        return cls(**data)
