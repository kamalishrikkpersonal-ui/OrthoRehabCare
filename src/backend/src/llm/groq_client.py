"""Minimal Groq/OpenAI-compatible LLM client for OrthoRehab AI Step 5.

Uses the standard-library ``urllib`` (no HTTP dependency) to call a chat
completions API. It reads the API key from ``GROQ_API_KEY`` or ``OPENAI_API_KEY``
environment variables. A custom endpoint may also be supplied with
``LLM_API_URL`` or via the client constructor.

The client ALWAYS surfaces a deterministic fallback path: callers catch
:class:`LLMUnavailableError` (or any exception) and use fallback templates. The
LLM is never allowed to crash the comparison engine.

The LLM receives ONLY structured findings and must NOT calculate or invent
measurements.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached or returns an unusable response."""


DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_TIMEOUT = 15.0  # seconds


class GroqClient:
    """A thin, dependency-free Groq/OpenAI chat client.

    Args:
        api_key: API key for Groq or OpenAI. If None, read from ``GROQ_API_KEY``
            or ``OPENAI_API_KEY`` env vars.
        model: model id to use.
        url: API endpoint.
        timeout: request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        self.api_key = api_key or groq_key or openai_key
        self.url = url or os.environ.get("LLM_API_URL", "")
        if not self.url:
            self.url = DEFAULT_OPENAI_URL if openai_key else DEFAULT_GROQ_URL

        env_model = os.environ.get("LLM_MODEL", "")
        if model:
            self.model = model
        elif env_model:
            self.model = env_model
        elif self.url == DEFAULT_OPENAI_URL:
            self.model = DEFAULT_OPENAI_MODEL
        else:
            self.model = DEFAULT_GROQ_MODEL

        self.timeout = timeout

    @property
    def available(self) -> bool:
        """Whether a Groq/OpenAI API key is configured."""
        return bool(self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str:
        """Send a chat request and return the assistant's text.

        Raises:
            LLMUnavailableError: if no API key, network failure, timeout,
                rate limit, malformed JSON, or empty response.
        """
        if not self.api_key:
            raise LLMUnavailableError("GROQ_API_KEY or OPENAI_API_KEY is not set")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # 429 rate limit, 401/403 auth, 500 server error, etc.
            raise LLMUnavailableError(f"Groq HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise LLMUnavailableError(f"Groq network error: {e.reason}") from e
        except TimeoutError as e:
            raise LLMUnavailableError("Groq request timed out") from e
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailableError(f"Groq request failed: {e}") from e

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            raise LLMUnavailableError(f"Malformed Groq response: {e}") from e

        if not content:
            raise LLMUnavailableError("Empty Groq response")

        return content
