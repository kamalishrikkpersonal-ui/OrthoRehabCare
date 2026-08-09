"""Minimal Groq LLM client for OrthoRehab AI Step 5.

Uses the standard-library ``urllib`` (no HTTP dependency) to call the Groq
chat-completions API. It follows the project's configuration convention by
reading the API key from the ``GROQ_API_KEY`` environment variable.

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
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TIMEOUT = 15.0  # seconds


class GroqClient:
    """A thin, dependency-free Groq chat client.

    Args:
        api_key: Groq API key. If None, read from ``GROQ_API_KEY`` env var.
        model: model id to use.
        url: API endpoint.
        timeout: request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_GROQ_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.url = url
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """Whether a Groq API key is configured."""
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
            raise LLMUnavailableError("GROQ_API_KEY is not set")

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
