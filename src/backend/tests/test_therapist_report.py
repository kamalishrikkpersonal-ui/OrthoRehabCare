"""Tests for the isolated OpenAI therapist-report generator (presentation layer).

These tests verify that converting the EXISTING deterministic analysis JSON into
a human-readable ``therapist_report`` string:

  1. Calls the OpenAI client exactly once with the analysis JSON (mocked).
  2. Falls back to the deterministic report when the LLM raises.
  3. Falls back to the deterministic report when ``OPENAI_API_KEY`` is missing.
  4. Leaves the numerical analysis values untouched (integrity).
  5. Reports detected deviations (e.g. ``insufficient_flexion``, ``too_fast``)
     without inventing anything.
  6. Keeps the API/report contract unchanged.
  7. Threads through the pipeline so the frontend still reads
     ``report.summary.therapist_report``.

The OpenAI client is ALWAYS mocked. No real network call is made.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.openai_therapist_report import (
    OpenAITherapistReportUnavailable,
    OpenAITherapistReporter,
    THERAPIST_SYSTEM_PROMPT,
)
from src.schemas.comparison import AssessmentDeviation, RepComparison
from src.comparison.deviation_engine import (
    DEV_INSUFFICIENT_FLEXION,
    DEV_TOO_FAST,
)
from src.session_report import build_session_report


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------
class FakeResponses:
    """Fake ``client.responses`` object returning a canned report."""

    def __init__(self, text: str = "AI THERAPIST REPORT", exc: Exception = None):
        self._text = text
        self._exc = exc
        self.calls = []

    def create(self, model=None, instructions=None, input=None, **kwargs):
        self.calls.append({"model": model, "instructions": instructions, "input": input})
        if self._exc is not None:
            raise self._exc
        return _SimpleResponse(self._text)


class _SimpleResponse:
    def __init__(self, text: str):
        self.output_text = text


class FakeClient:
    """Minimal fake OpenAI client exposing only ``responses.create``."""

    def __init__(self, responses=None):
        self.responses = responses or FakeResponses()


def _make_comparison_result():
    """Build a minimal but realistic ComparisonResult for report tests.

    The numerical values mirror a common demo output. We assert these are
    preserved through the report pipeline (numerical integrity).
    """
    from src.schemas.comparison import ComparisonResult

    dev = [
        AssessmentDeviation(
            type=DEV_INSUFFICIENT_FLEXION,
            joint="right_elbow",
            reference_value=80.0,
            patient_value=100.0,
            deviation=20.0,
            severity="major",
            timestamp=0.0,
        ),
        AssessmentDeviation(
            type=DEV_TOO_FAST,
            joint="",
            reference_value=None,
            patient_value=None,
            deviation=None,
            severity="minor",
            timestamp=0.0,
        ),
    ]
    rep = RepComparison(
        rep_number=1,
        start_timestamp=0.0,
        end_timestamp=1.2,
        duration=1.2,
        dtw_similarity=0.72,
        dtw_distance=8.0,
        learned_motion_similarity=None,
        biomechanical_accuracy=0.68,
        movement_quality=0.75,
        speed_status="too_fast",
        primary_min_angle=100.0,
        primary_max_angle=170.0,
        reference_min_angle=80.0,
        reference_max_angle=170.0,
        rom_accuracy=0.81,
        deviations=dev,
        severity="moderate",
    )
    return ComparisonResult(
        exercise_id="elbow_flexion",
        exercise_name="Elbow Flexion / Extension",
        overall_score=0.75,
        learned_motion_similarity=None,
        dtw_similarity=0.72,
        biomechanical_accuracy=0.68,
        rom_accuracy=0.81,
        reference_duration=1.18,
        patient_duration=1.0,
        speed_ratio=1.18,
        speed_status="too_fast",
        primary_angle="right_elbow",
        repetitions=[rep],
        deviations=dev,
        learned_model_available=False,
    )


def run_test(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 1. Successful LLM generation (mock OpenAI)
# ---------------------------------------------------------------------------
def t_llm_generates_report():
    os.environ["OPENAI_API_KEY"] = "test-key"  # presence only; client is mocked
    fake_resp = FakeResponses(text="Professional therapist report text.")
    client = FakeClient(responses=fake_resp)
    reporter = OpenAITherapistReporter(client=client)

    analysis = _make_comparison_result().to_dict()
    report_text = reporter.generate(analysis)

    assert report_text == "Professional therapist report text."
    # Verify the client was called once with the analysis JSON.
    assert len(fake_resp.calls) == 1
    call = fake_resp.calls[0]
    assert call["model"] == reporter.model
    # The analysis JSON must be embedded in the user input.
    user_content = call["input"][0]["content"]
    assert "overall_score" in user_content
    assert "dtw_similarity" in user_content
    assert "insufficient_flexion" in user_content
    assert "too_fast" in user_content


def t_llm_exception_falls_back():
    os.environ["OPENAI_API_KEY"] = "test-key"
    client = FakeClient(responses=FakeResponses(exc=RuntimeError("boom")))
    reporter = OpenAITherapistReporter(client=client)
    try:
        reporter.generate(_make_comparison_result().to_dict())
        assert False, "expected OpenAITherapistReportUnavailable"
    except OpenAITherapistReportUnavailable:
        pass


def t_missing_key_falls_back():
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        reporter = OpenAITherapistReporter(client=FakeClient())
        assert reporter.configured is False
        try:
            reporter.generate(_make_comparison_result().to_dict())
            assert False, "expected OpenAITherapistReportUnavailable"
        except OpenAITherapistReportUnavailable:
            pass
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def t_numerical_integrity():
    # The report pipeline must never alter the analysis numbers.
    result = _make_comparison_result()
    before = result.to_dict()
    report = build_session_report(result, groq=None, openai_reporter=None)
    after = result.to_dict()
    assert before == after
    # The deterministic fallback report still contains the exact metric values.
    assert "0.72" in report.summary.therapist_report
    assert "0.68" in report.summary.therapist_report
    assert "0.81" in report.summary.therapist_report
    assert "0.75" in report.summary.therapist_report
    assert "1.18" in report.summary.therapist_report
    assert "8" in report.summary.therapist_report


def t_deviations_mentioned_without_inventing():
    result = _make_comparison_result()
    report = build_session_report(result, groq=None, openai_reporter=None)
    text = report.summary.therapist_report.lower()
    # Detected deviations must be reflected in the report text.
    assert "insufficient_flexion" in text
    assert "too_fast" in text
    # The report must not invent a value that was never present.
    assert "knee" not in text  # no lower-body claim
    assert "diagnosis" not in text


def t_api_contract_unchanged():
    result = _make_comparison_result()
    report = build_session_report(result, groq=None, openai_reporter=None)
    rd = report.to_dict()
    # Frontend reads report.summary.therapist_report.
    assert rd["summary"]["therapist_report"]
    assert rd["summary"]["patient_feedback"]
    # The comparison payload is unchanged (still a full dict).
    assert rd["comparison"]["dtw_similarity"] == 0.72
    assert rd["comparison"]["overall_score"] == 0.75


def t_frontend_reads_therapist_report():
    # Simulate the frontend contract: result.report.summary.therapist_report.
    result = _make_comparison_result()
    report = build_session_report(result, groq=None, openai_reporter=None)
    therapist_report = report.summary.therapist_report
    assert isinstance(therapist_report, str) and len(therapist_report) > 0
    # The system prompt is strict: use ONLY the JSON, no markdown tables.
    assert "markdown tables" in THERAPIST_SYSTEM_PROMPT
    assert "Return ONLY the report text" in THERAPIST_SYSTEM_PROMPT


def run_all() -> None:
    print("OPENAI THERAPIST-REPORT (PRESENTATION LAYER) TESTS")
    print("=" * 60)
    run_test("LLM generates report from JSON (mock OpenAI)", t_llm_generates_report)
    run_test("LLM exception -> deterministic fallback", t_llm_exception_falls_back)
    run_test("missing key -> deterministic fallback", t_missing_key_falls_back)
    run_test("numerical integrity preserved", t_numerical_integrity)
    run_test("deviations mentioned without inventing", t_deviations_mentioned_without_inventing)
    run_test("API/report contract unchanged", t_api_contract_unchanged)
    run_test("frontend reads report.summary.therapist_report", t_frontend_reads_therapist_report)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
