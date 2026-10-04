"""Session report generation for OrthoRehab AI Step 5.

At exercise/session completion, produce a patient-facing summary and a
therapist-facing structured report. The therapist report is generated from the
existing deterministic analysis JSON by an isolated OpenAI reporter (a pure
presentation-layer conversion). If that LLM conversion is unavailable or fails,
a deterministic structured therapist report is used instead. The analysis JSON
is never modified by the LLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .feedback.feedback_templates import fallback_message
from .llm.groq_client import GroqClient
from .llm.openai_therapist_report import (
    OpenAITherapistReportUnavailable,
    OpenAITherapistReporter,
)
from .llm.rehabilitation_feedback import messages_for_findings
from .schemas.comparison import ComparisonResult, SessionSummary


@dataclass
class SessionReport:
    """A complete session report (patient + therapist view)."""

    summary: SessionSummary = field(default_factory=SessionSummary)
    comparison: Optional[ComparisonResult] = None

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(),
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


def _major_deviations(result: ComparisonResult) -> List[Any]:
    """Return only major/moderate deviations for the session summary."""
    return [d for d in result.deviations if d.severity in ("major", "moderate")]


def _patient_feedback(result: ComparisonResult) -> str:
    """Deterministic patient feedback from the comparison result."""
    dev_types = [d.type for d in result.deviations]
    if not dev_types:
        return fallback_message([])
    return fallback_message(dev_types)


def _therapist_report_fallback(result: ComparisonResult) -> str:
    """Return a raw therapist report when LLM-based report generation is unavailable."""
    therapist_lines = [
        f"Exercise: {result.exercise_name} ({result.exercise_id})",
        f"Primary angle: {result.primary_angle}",
        f"Overall score: {result.overall_score}",
        f"Learned motion similarity: {result.learned_motion_similarity}",
        f"DTW similarity: {result.dtw_similarity}",
        f"Biomechanical accuracy: {result.biomechanical_accuracy}",
        f"ROM accuracy: {result.rom_accuracy}",
        f"Reference duration: {result.reference_duration}s",
        f"Patient duration: {result.patient_duration}s",
        f"Speed ratio: {result.speed_ratio}",
        f"Speed status: {result.speed_status}",
        f"Total reps: {len(result.repetitions)}",
    ]
    for i, rep in enumerate(result.repetitions, 1):
        therapist_lines.append(
            f"  Rep {i}: duration={rep.duration}s dtw={rep.dtw_similarity} "
            f"rom_acc={rep.rom_accuracy} speed={rep.speed_status} "
            f"severity={rep.severity}"
        )
    therapist_lines.append("Deviations:")
    if result.deviations:
        for d in result.deviations:
            therapist_lines.append(
                f"  - {d.type} ({d.joint}) ref={d.reference_value} "
                f"pat={d.patient_value} dev={d.deviation} sev={d.severity}"
            )
    else:
        therapist_lines.append("  - none")
    return "\n".join(therapist_lines)


def _therapist_report_openai(
    result: ComparisonResult, openai_reporter: Optional[OpenAITherapistReporter]
) -> Optional[str]:
    """Generate the therapist report via the OpenAI reporter, or None on failure."""
    if openai_reporter is None:
        return None
    try:
        analysis = result.to_dict()  # the deterministic source of truth
        return openai_reporter.generate(analysis)
    except OpenAITherapistReportUnavailable:
        # Expected fallback path: deterministic report is used instead.
        return None
    except Exception:  # noqa: BLE001 - never let an LLM failure break reporting
        return None


def build_session_report(
    result: ComparisonResult,
    groq: Optional[GroqClient] = None,
    timings_ms: Optional[Dict[str, float]] = None,
    openai_reporter: Optional[OpenAITherapistReporter] = None,
) -> SessionReport:
    """Build a :class:`SessionReport` from a completed comparison.

    Args:
        result: the completed :class:`ComparisonResult`.
        groq: optional LLM client. If provided and available, used to generate
            the patient summary; otherwise a deterministic fallback is used.
        timings_ms: optional performance timings to include in the report.
        openai_reporter: optional :class:`OpenAITherapistReporter`. If provided
            and configured, used to generate the therapist report from the
            existing analysis JSON. On any failure the deterministic therapist
            report is used instead.

    Returns:
        A :class:`SessionReport` with patient and therapist views.
    """
    total_reps = len(result.repetitions)
    successful = sum(
        1
        for r in result.repetitions
        if (r.dtw_similarity or 0) >= 0.7
        and r.speed_status == "normal_speed"
        and r.severity == "minor"
    )

    major = _major_deviations(result)

    # Patient feedback: try LLM, fall back to deterministic.
    patient_text = _patient_feedback(result)
    source = "fallback"
    if groq is not None and groq.available:
        try:
            findings = result.summary_dict()
            msgs = messages_for_findings(findings)
            content = groq.chat(msgs)
            if content:
                patient_text = content
                source = "llm"
        except Exception:  # noqa: BLE001
            patient_text = _patient_feedback(result)
            source = "fallback"

    # Therapist report: prefer the isolated OpenAI reporter (presentation layer
    # only). If it is unavailable or fails, fall back to the deterministic
    # structured therapist report. The analysis JSON is never modified.
    therapist_report = _therapist_report_fallback(result)
    openai_report = _therapist_report_openai(result, openai_reporter)
    if openai_report:
        therapist_report = openai_report

    summary = SessionSummary(
        exercise_id=result.exercise_id,
        exercise_name=result.exercise_name,
        total_repetitions=total_reps,
        successful_repetitions=successful,
        overall_score=result.overall_score,
        learned_motion_similarity=result.learned_motion_similarity,
        dtw_similarity=result.dtw_similarity,
        biomechanical_accuracy=result.biomechanical_accuracy,
        speed_status=result.speed_status,
        major_deviations=list(result.deviations),
        patient_feedback=patient_text,
        therapist_report=therapist_report,
        created_at=datetime.now(timezone.utc).isoformat(),
        timings_ms=dict(timings_ms or {}),
    )

    return SessionReport(summary=summary, comparison=result)
