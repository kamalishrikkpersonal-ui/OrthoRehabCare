"""Prompt building for rehabilitation feedback (OrthoRehab AI Step 5).

The LLM receives ONLY structured findings from the comparison engine. It must
NOT calculate measurements, invent deviations, diagnose, or claim clinical
correctness. The comparison engine is the source of truth.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM_PROMPT = """\
You are an AI rehabilitation assistant that compares a patient's exercises \
against a therapist's demonstrated reference. You are NOT a medical clinician \
and you do NOT diagnose, treat, or claim clinical correctness.

You will receive STRUCTURED findings from the comparison engine. Your only job \
is to convert those findings into concise, encouraging, patient-friendly \
English feedback (1-3 short sentences). 

Rules:
- Use ONLY the numbers and findings you are given. Never calculate or invent \
  measurements, deviations, or diagnoses.
- Do not claim the movement is clinically correct or that the patient is \
  recovering.
- Address the most important issue (e.g. insufficient flexion, too slow, \
  insufficient range of motion) first.
- Keep the tone supportive and actionable.
- Example: "You're following the movement pattern well. Try bending a little \
  further and keep a slightly quicker, controlled pace."
"""


def build_system_prompt() -> str:
    """Return the system prompt for rehabilitation feedback."""
    return SYSTEM_PROMPT


def build_user_prompt(findings: Dict[str, Any]) -> str:
    """Build a user prompt containing ONLY structured findings (as JSON)."""
    payload: Dict[str, Any] = {}

    # Copy only the structured fields we want the LLM to see.
    for key in (
        "exercise",
        "learned_motion_similarity",
        "dtw_similarity",
        "biomechanical_accuracy",
        "speed_status",
        "speed_ratio",
    ):
        if key in findings:
            payload[key] = findings[key]

    deviations = findings.get("deviations", [])
    payload["deviations"] = [
        {
            "type": d.get("type"),
            "joint": d.get("joint"),
            "reference_value": d.get("reference_value"),
            "patient_value": d.get("patient_value"),
            "deviation": d.get("deviation"),
            "severity": d.get("severity"),
        }
        for d in deviations
    ]

    return json.dumps(payload, default=str)


def messages_for_findings(findings: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the full message list (system + user) for a chat request."""
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(findings)},
    ]
