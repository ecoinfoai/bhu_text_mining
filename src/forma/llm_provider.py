"""LLM provider abstraction layer.

Supports Gemini (default) and Anthropic Claude providers via a unified
interface.  Use ``create_provider()`` factory to instantiate.

Includes exponential backoff retry for transient errors (429, connection).
"""

from __future__ import annotations

import abc
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3
INITIAL_BACKOFF: float = 2.0
DEFAULT_TIMEOUT: float = 60.0


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is transient and should be retried."""
    exc_str = str(exc).lower()
    if "429" in exc_str or "too many requests" in exc_str:
        return True
    if "rate" in exc_str and "limit" in exc_str:
        return True
    # Server errors (500, 502, 503, 504) are transient
    for code in ("500", "502", "503", "504"):
        if code in exc_str:
            return True
    type_name = type(exc).__name__.lower()
    if any(kw in type_name for kw in ("connection", "timeout", "network", "server")):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def _generate_impl(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> str:
        """Provider-specific generation (no retry logic)."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str:
        """Generate a text response with exponential backoff retry.

        Retries on transient errors (429, connection). Does NOT retry
        on parse errors or other non-transient failures.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds (default 60).

        Returns:
            Generated text string.

        Raises:
            Last exception if all retries exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_impl(prompt, max_tokens, temperature)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider via google-genai SDK.

    Requires ``GOOGLE_API_KEY`` environment variable or explicit api_key.
    """

    DEFAULT_MODEL: str = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Google API key required. Set GOOGLE_API_KEY environment "
                "variable or pass api_key= explicitly."
            )
        from google import genai

        self._client = genai.Client(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def _generate_impl(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider.

    Requires ``ANTHROPIC_API_KEY`` environment variable or explicit api_key.
    """

    DEFAULT_MODEL: str = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key= explicitly."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def _generate_impl(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


def create_provider(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider: Provider name — ``"gemini"`` (default) or ``"anthropic"``.
        api_key: API key (falls back to environment variable).
        model: Model ID override (uses provider default if None).

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    provider = provider.lower()
    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    elif provider in ("anthropic", "claude"):
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported: 'gemini', 'anthropic'."
        )
