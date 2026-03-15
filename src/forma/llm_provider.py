"""LLM provider abstraction layer.

Supports Gemini (default) and Anthropic Claude providers via a unified
interface.  Use ``create_provider()`` factory to instantiate.

Includes exponential backoff retry for transient errors (429, connection).
"""

from __future__ import annotations

import abc
import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3
INITIAL_BACKOFF: float = 2.0
DEFAULT_TIMEOUT: float = 60.0

_IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


@dataclass
class LLMFullResponse:
    """Full LLM response including metadata for vision OCR.

    Attributes:
        text: Generated text.
        logprobs_result: Gemini logprobs result object, None for Anthropic.
        usage: Token usage dict with ``input_tokens`` and ``output_tokens``.
        finish_reason: Model stop reason (e.g. ``"STOP"``, ``"end_turn"``).
        safety_ratings: Gemini safety ratings list, None for Anthropic.
    """

    text: str
    logprobs_result: Any | None
    usage: dict[str, int]
    finish_reason: str
    safety_ratings: list[dict] | None


def _read_image(image_path: str) -> tuple[bytes, str]:
    """Read image file and return (bytes, mime_type).

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If extension is not jpg/jpeg/png.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = _IMAGE_MIME_TYPES.get(ext)
    if mime_type is None:
        raise ValueError(
            f"Unsupported image format: {ext!r}. "
            f"Supported: {', '.join(sorted(_IMAGE_MIME_TYPES))}"
        )
    with open(image_path, "rb") as f:
        return f.read(), mime_type


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
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Provider-specific generation (no retry logic)."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate a text response with exponential backoff retry.

        Retries on transient errors (429, connection). Does NOT retry
        on parse errors or other non-transient failures.

        Args:
            prompt: The input prompt (user message).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds (default 60).
            system_instruction: Optional system-level instruction for the
                LLM. Passed as Gemini ``system_instruction`` config or
                Anthropic ``system`` parameter.

        Returns:
            Generated text string.

        Raises:
            Last exception if all retries exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_impl(
                    prompt, max_tokens, temperature, system_instruction
                )
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

    @abc.abstractmethod
    def _generate_with_image_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Provider-specific image generation (no retry logic)."""

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate a text response from prompt + image with retry.

        Args:
            prompt: The input prompt (user message).
            image_path: Path to JPG/JPEG/PNG image file.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            system_instruction: Optional system-level instruction.

        Returns:
            Generated text string.
        """
        image_data, mime_type = _read_image(image_path)
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_with_image_impl(
                    prompt, image_data, mime_type,
                    max_tokens, temperature, system_instruction,
                )
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "LLM vision call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @abc.abstractmethod
    def _generate_with_image_full_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
        response_logprobs: bool = False,
    ) -> LLMFullResponse:
        """Provider-specific full image generation (no retry logic)."""

    def generate_with_image_full(
        self,
        prompt: str,
        image_path: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        system_instruction: Optional[str] = None,
        response_logprobs: bool = False,
    ) -> LLMFullResponse:
        """Generate a full response from prompt + image with retry.

        Returns LLMFullResponse including metadata (logprobs, usage,
        finish_reason, safety_ratings).

        Args:
            prompt: The input prompt (user message).
            image_path: Path to JPG/JPEG/PNG image file.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            system_instruction: Optional system-level instruction.
            response_logprobs: Request logprobs from Gemini (ignored by Anthropic).

        Returns:
            LLMFullResponse with text and metadata.
        """
        image_data, mime_type = _read_image(image_path)
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._generate_with_image_full_impl(
                    prompt, image_data, mime_type,
                    max_tokens, temperature, system_instruction,
                    response_logprobs,
                )
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "LLM vision full call failed (attempt %d/%d): %s. Retrying in %.1fs",
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

    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        from google.genai import types

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    def _generate_with_image_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        from google.genai import types

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        contents = [prompt, image_part]

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    def _generate_with_image_full_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
        response_logprobs: bool = False,
    ) -> LLMFullResponse:
        from google.genai import types

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_logprobs:
            config_kwargs["response_logprobs"] = True

        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        contents = [prompt, image_part]

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        candidate = response.candidates[0]
        usage_meta = response.usage_metadata
        return LLMFullResponse(
            text=response.text,
            logprobs_result=candidate.logprobs_result,
            usage={
                "input_tokens": usage_meta.prompt_token_count,
                "output_tokens": usage_meta.candidates_token_count,
            },
            finish_reason=str(candidate.finish_reason),
            safety_ratings=candidate.safety_ratings,
        )


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

    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        message = self._client.messages.create(**kwargs)
        return message.content[0].text

    def _generate_with_image_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
    ) -> str:
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        message = self._client.messages.create(**kwargs)
        return message.content[0].text

    def _generate_with_image_full_impl(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_instruction: Optional[str] = None,
        response_logprobs: bool = False,
    ) -> LLMFullResponse:
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        message = self._client.messages.create(**kwargs)
        return LLMFullResponse(
            text=message.content[0].text,
            logprobs_result=None,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
            finish_reason=message.stop_reason,
            safety_ratings=None,
        )


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
