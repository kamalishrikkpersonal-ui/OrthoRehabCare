"""Isolated OpenAI therapist-report generator for OrthoRehab AI Step 5.

This module is deliberately a *presentation layer only*. It receives the
already-computed deterministic analysis JSON (a :class:`ComparisonResult`
serialized via ``to_dict()``) and converts it into a concise, human-readable
therapist report using the OpenAI Responses API.

It NEVER performs exercise analysis. Every factual statement in the generated
report must come from the input JSON. No metric is recalculated, reinterpreted,
or invented. The comparison engine remains the single source of truth.

Failure handling is strict and non-fatal: if ``OPENAI_API_KEY`` is missing, the
request times out, is rate-limited, a network error occurs, or the response is
malformed or empty, this module raises :class:`OpenAITherapistReportUnavailable`
so the caller can fall back to the deterministic report. The rehabilitation
analysis JSON is never blocked by an LLM failure.

The API key is read ONLY from the ``OPENAI_API_KEY`` environment variable. It is
never hardcoded, never logged, and never exposed to the frontend.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Env-only configuration (no hardcoded secrets).
_DEFAULT_MODEL = "gpt-4o"  # an appropriate current OpenAI model; override via OPENAI_MODEL

# The reporter reads OPENAI_API_KEY / OPENAI_MODEL from the environment. To make
# it robust from ANY entry point (API, CLI tool, tests, scripts), we load the
# project's .env file if present. This reuses the existing python-dotenv
# dependency (already in requirements.txt) and is idempotent: if the environment
# was already populated by the caller (e.g. src/api.py also calls load_dotenv),
# we do not override existing values.
_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load the project .env into os.environ exactly once (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv

        # This module lives at <repo>/src/backend/src/llm/.
        here = os.path.dirname(os.path.abspath(__file__))
        # The project .env lives at <repo>/src/backend/.env.
        backend_dir = os.path.abspath(os.path.join(here, "..", ".."))
        load_dotenv(os.path.join(backend_dir, ".env"), override=False)
    except Exception:  # noqa: BLE001 - dotenv is optional; never break reporting
        logger.debug("Could not load .env via python-dotenv")


class OpenAITherapistReportUnavailable(Exception):
    """Raised when an OpenAI therapist report cannot be produced.

    Callers should fall back to the deterministic report. The analysis pipeline
    is never affected by this error.
    """


THERAPIST_SYSTEM_PROMPT = (
    "You are a rehabilitation reporting assistant.\n\n"
    "Convert the provided exercise-analysis JSON into a concise, professional "
    "therapist report.\n\n"
    "Rules:\n"
    "- Use ONLY information present in the JSON.\n"
    "- Do not invent measurements, observations, diagnoses, or patient history.\n"
    "- Do not modify numerical values.\n"
    "- Do not recalculate scores.\n"
    "- Do not reinterpret the comparison algorithm.\n"
    "- Do not claim a medical diagnosis.\n"
    "- Clearly mention repetitions when available.\n"
    "- Clearly mention overall performance when available.\n"
    "- Mention ROM/range-of-motion performance when available.\n"
    "- Mention movement similarity/biomechanical performance when available.\n"
    "- Mention speed deviations when available.\n"
    "- Mention detected deviations and their severity when available.\n"
    "- Identify strengths.\n"
    "- Identify areas requiring improvement.\n"
    "- Give practical exercise-performance guidance based only on detected deviations.\n"
    "- Keep the report understandable to a physiotherapist.\n"
    "- Do not use markdown tables.\n"
    "- Return ONLY the report text."
)


def _model_from_env() -> str:
    """Return the configured OpenAI model, with a sensible default."""
    _ensure_env_loaded()
    return os.environ.get("OPENAI_MODEL", "").strip() or _DEFAULT_MODEL


def _build_user_prompt(analysis_json: Dict[str, Any]) -> str:
    """Build the user prompt with the analysis JSON clearly delimited."""
    payload = json.dumps(analysis_json, default=str, indent=2)
    return (
        "Here is the exercise-analysis JSON (the deterministic source of truth).\n"
        "Convert it into a concise, professional therapist report following the "
        "system rules. Use ONLY the information present in this JSON.\n\n"
        "ANALYSIS_JSON_BEGIN\n"
        f"{payload}\n"
        "ANALYSIS_JSON_END\n\n"
        "Return ONLY the therapist report text."
    )


class OpenAITherapistReporter:
    """Convert an existing analysis JSON into therapist-report text via OpenAI.

    Args:
        client: a compatible OpenAI client. If None, an ``openai.OpenAI`` client
            is created from ``OPENAI_API_KEY`` (the SDK itself raises if the key
            is missing). Tests pass a fake client.
        model: model id. Defaults to ``OPENAI_MODEL`` or ``gpt-4o``.
    """

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None) -> None:
        self.client = client
        self.model = model or _model_from_env()

    @property
    def configured(self) -> bool:
        """Whether an OpenAI API key is available (client can be constructed)."""
        _ensure_env_loaded()
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())

    def generate(self, analysis: Dict[str, Any]) -> str:
        """Return a therapist report string for the given analysis JSON.

        Args:
            analysis: the deterministic analysis result as a dict (e.g. the
                output of ``ComparisonResult.to_dict()``).

        Returns:
            The therapist-report text.

        Raises:
            OpenAITherapistReportUnavailable: if the OpenAI request fails or is
                unusable. Callers must fall back to the deterministic report.
        """
        if not self.configured:
            logger.info("Therapist report source: DETERMINISTIC_FALLBACK (OPENAI_API_KEY not set)")
            raise OpenAITherapistReportUnavailable(
                "OPENAI_API_KEY is not set; falling back to deterministic report"
            )

        # Lazily construct the client so tests can inject a fake and so the SDK
        # is only imported when actually used.
        if self.client is None:
            try:
                from openai import OpenAI  # imported lazily / never in tests
            except Exception as exc:  # noqa: BLE001
                raise OpenAITherapistReportUnavailable(
                    f"OpenAI SDK unavailable: {exc}"
                ) from exc
            try:
                self.client = OpenAI()  # reads OPENAI_API_KEY from env
            except Exception as exc:  # noqa: BLE001
                raise OpenAITherapistReportUnavailable(
                    f"Could not create OpenAI client: {exc}"
                ) from exc

        try:
            # Use the current OpenAI Responses API.
            logger.info("Therapist report source: OPENAI (calling responses.create)")
            response = self.client.responses.create(
                model=self.model,
                instructions=THERAPIST_SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": _build_user_prompt(analysis),
                    }
                ],
            )
            text = _extract_responses_text(response)
        except OpenAITherapistReportUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            # Missing key, timeout, rate limit, network error, invalid response.
            # Log the failure CATEGORY only (never the API key or request payload).
            logger.warning(
                "OpenAI therapist report failed; falling back to deterministic. "
                "category=%s error=%s",
                type(exc).__name__,
                _safe_error_summary(exc),
            )
            raise OpenAITherapistReportUnavailable(str(exc)) from exc

        if not text:
            logger.info("Therapist report source: DETERMINISTIC_FALLBACK (empty OpenAI response)")
            raise OpenAITherapistReportUnavailable("OpenAI returned an empty therapist report")
        logger.info("Therapist report source: OPENAI (report generated)")
        return text


def _safe_error_summary(exc: Exception) -> str:
    """Return a short, safe summary of an exception for logging.

    Never includes the API key or the full request payload. For HTTP errors we
    report only the status/reason, which are safe and useful for diagnosis.
    """
    # OpenAI SDK HTTP errors carry a `status_code` and user-facing message.
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"http_status={status}"
    # Generic: use just the exception class name + a truncated message.
    msg = str(exc)
    if len(msg) > 120:
        msg = msg[:120] + "..."
    return msg


def _extract_responses_text(response: Any) -> str:
    """Extract the assistant text from an OpenAI Responses-API response object.

    Handles both the SDK object (``response.output_text``) and a plain dict.
    """
    if hasattr(response, "output_text"):
        return str(response.output_text or "").strip()
    if isinstance(response, dict):
        # Responses API dict shape: {"output": [{"content": [{"text": ...}]}]}
        output = response.get("output") or []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content") or []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        return str(block.get("text") or "").strip()
        # Fall back to the GPT-4o / chat-completions dict shape.
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            return str(msg.get("content") or "").strip()
    return ""
